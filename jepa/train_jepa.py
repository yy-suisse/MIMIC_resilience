"""JEPA training from a BERT encoder (D1, D2 or D3): predict the embedding of a future window.

Steps:
  1. arrays for (model, min_patients) are built from the master table if they do not exist yet
  2. student = BERT encoder + segment embedding; teacher = EMA copy of the student; predictor Z
  3. each step: context -> c (student), target -> y (teacher), y_hat = Z(c, delta, L_obs),
     loss = ||y_hat - y||^2 + lambda_var * variance term + lambda_cov * covariance term (on c)
  4. evaluation every --eval-every steps on fixed validation batches, with and without the history block
  5. TensorBoard logs and checkpoints (best.pt, last.pt) in <out-dir>/<run-name>

Variants (--variant, see jepa_model.py): base (c = mean of the context tokens), concat_hist (c = [history mean ;
recent per-day mean]), z_transformer (Z reads every context token with a (delta, L) query).

Usage (PowerShell), from the jepa folder:
  python train_jepa.py --init-run D:\\MIMIC\\bert_model\\D3_min50 --run-name JEPA_D3
  python train_jepa.py --init-run D:\\MIMIC\\bert_model\\D1_min50_20261120_xxxx --run-name JEPA_D1
  python train_jepa.py --init-run D:\\MIMIC\\bert_model\\D3_min50 --run-name JEPA_D3 --resume
  python train_jepa.py --init-run D:\\MIMIC\\bert_model\\D2_min50 --variant concat_hist      # JEPA_D2_concat_hist
  python train_jepa.py --init-run D:\\MIMIC\\bert_model\\D2_min50 --variant z_transformer    # JEPA_D2_z_transformer
  tensorboard --logdir D:\\MIMIC\\jepa_model
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from jepa_data import JepaBatches, fixed_batches, load_yaml, prepare_arrays
from jepa_model import (VARIANTS, JepaEncoder, context_inputs, effective_rank, ema_update, load_bert, make_predictor,
                        make_teacher, pool_target, run_predictor, variance_covariance)

HERE = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser(description="JEPA training from a BERT encoder (D1, D2, D3)")
    ap.add_argument("--init-run", required=True, help=r"BERT run folder, e.g. D:\MIMIC\bert_model\D3_min50")
    ap.add_argument("--init-checkpoint", default="best.pt", help="best.pt or last.pt of the BERT run")
    ap.add_argument("--config", default=str(HERE.parent / "config" / "jepa_config.yaml"))
    ap.add_argument("--trajectory-config", default=str(HERE.parent / "bert" / "trajectory_columns.yaml"))
    ap.add_argument("--recent-days", type=int, default=None, help="overrides recent_days of the config")
    ap.add_argument("--out-dir", default=r"D:\MIMIC\jepa_model")
    ap.add_argument("--variant", default="base", choices=VARIANTS,
                    help="how the context reaches Z: base (mean), concat_hist, z_transformer")
    ap.add_argument("--run-name", default=None, help="default: JEPA_<model>_<date> (base) or JEPA_<model>_<variant>")
    ap.add_argument("--resume", action="store_true", help="continue the run --run-name from --resume-checkpoint")
    ap.add_argument("--resume-checkpoint", default="last.pt", help="last.pt, or best.pt to go back to the best step")
    ap.add_argument("--prepare-only", action="store_true", help="only build the arrays, then stop")
    ap.add_argument("--batch-size", type=int, default=32, help="contexts per batch (each with n_pairs targets)")
    ap.add_argument("--lr", type=float, default=3e-4, help="embeddings, segment embedding and predictor Z")
    ap.add_argument("--lr-layers-scale", type=float, default=0.1, help="transformer layers get lr * this")
    ap.add_argument("--min-lr-scale", type=float, default=0.1, help="end of the cosine decay, as a share of lr")
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def lr_factor(step, args):
    """Linear warm-up, then cosine decay to min_lr_scale at max_steps (applied to both parameter groups)."""
    if step < args.warmup:
        return (step + 1) / args.warmup
    t = min(1.0, (step - args.warmup) / max(1, args.max_steps - args.warmup))
    return args.min_lr_scale + 0.5 * (1 - args.min_lr_scale) * (1 + math.cos(math.pi * t))


def ema_momentum(step, args, jcfg):
    """Teacher momentum: cosine from ema.start to ema.end over the run."""
    t = min(1.0, step / max(1, args.max_steps))
    start, end = jcfg["ema"]["start"], jcfg["ema"]["end"]
    return end - (end - start) * (math.cos(math.pi * t) + 1) / 2


def forward(student, teacher, predictor, batch, variant):
    """c, y and y_hat of one batch."""
    c, inputs = context_inputs(variant, student(batch["ctx"]), batch["ctx"], batch["ctx_key"])
    with torch.no_grad():
        y = pool_target(teacher(batch["tgt"]), batch["tgt"], batch["tgt_key"])
    y_hat = run_predictor(variant, predictor, c, inputs, batch["pair_ctx"], batch["delta"], batch["length"])
    return c, y, y_hat.float()


def losses(c, y, y_hat, jcfg):
    pred = ((y_hat - y) ** 2).mean(dim=-1)                       # one value per (context, target) pair
    var_term, cov_term = variance_covariance(c)
    total = pred.mean() + jcfg["lambda_var"] * var_term + jcfg["lambda_cov"] * cov_term
    return total, pred, var_term, cov_term


@torch.no_grad()
def evaluate(student, teacher, predictor, val, val_no_history, jcfg, device, autocast, variant):
    """Prediction loss against 'always the mean target', per gap range, spreads, rank, and without history."""
    student.eval()
    bins = jcfg["delta_bins"]
    names = [f"<{bins[0]}d"] + [f"{a}-{b}d" for a, b in zip(bins[:-1], bins[1:])] + [f">={bins[-1]}d"]
    pred_all, base_all, deltas, cs, ys, yhats = [], [], [], [], [], []
    for b in val:
        b = to_device(b, device)
        with autocast():
            c, y, y_hat = forward(student, teacher, predictor, b, variant)
        pred_all.append(((y_hat - y) ** 2).mean(-1))
        base_all.append(((y - y.mean(0)) ** 2).mean(-1))          # always predict the batch's mean target
        deltas.append(b["delta"])
        cs.append(c.float()); ys.append(y); yhats.append(y_hat)
    pred, base, delta = torch.cat(pred_all), torch.cat(base_all), torch.cat(deltas)
    c, y, y_hat = torch.cat(cs), torch.cat(ys), torch.cat(yhats)
    res = {"pred_loss": pred.mean().item(), "mean_target_loss": base.mean().item()}
    res["pred_ratio"] = res["pred_loss"] / max(res["mean_target_loss"], 1e-8)
    which = torch.bucketize(delta, torch.tensor(bins, device=delta.device, dtype=delta.dtype), right=True)
    for i, name in enumerate(names):
        if (which == i).any():
            res[f"pred_loss_delta_{name}"] = pred[which == i].mean().item()
    res["std_c"] = c.std(0).mean().item()
    res["std_y"] = y.std(0).mean().item()
    res["std_y_hat"] = y_hat.std(0).mean().item()
    res["rank_c"] = effective_rank(c)
    res["rank_y"] = effective_rank(y)
    no_hist = []
    for b in val_no_history:
        b = to_device(b, device)
        with autocast():
            _, y, y_hat = forward(student, teacher, predictor, b, variant)
        no_hist.append(((y_hat - y) ** 2).mean(-1))
    res["pred_loss_no_history"] = torch.cat(no_hist).mean().item()
    student.train()
    return res


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    jcfg, tcfg = load_yaml(args.config), load_yaml(args.trajectory_config)
    if args.recent_days is not None:
        jcfg["recent_days"] = args.recent_days
    device = torch.device(args.device)

    # 1) the BERT encoder decides the model (D1, D2, D3) and the vocabulary cut-off
    bert, bert_cfg = load_bert(args.init_run, args.init_checkpoint, "cpu")
    model, min_patients = bert_cfg["args"]["model"], bert_cfg["min_patients"]
    arrays_dir = Path(tcfg["paths"]["out"]) / "jepa" / "arrays" / f"{model}_min{min_patients}"
    if not (arrays_dir / "meta.json").exists():
        print(f"building JEPA arrays in {arrays_dir} ...", flush=True)
        prepare_arrays(tcfg, model, min_patients, arrays_dir)
    with open(arrays_dir / "meta.json") as f:
        meta = json.load(f)
    if meta["vocab_size"] != bert_cfg["vocab_size"]:
        raise ValueError("vocabulary of the arrays and of the BERT encoder differ: check min_patients")
    print(f"init {model} (min_patients {min_patients}) from {args.init_run}; variant {args.variant}; "
          f"{meta['n_patients']['train']:,} train patients; recent_days {jcfg['recent_days']}", flush=True)
    if args.prepare_only:
        return

    # 2) student, teacher, predictor
    student = JepaEncoder(bert).to(device)
    teacher = make_teacher(student).eval()
    dim = bert.tok.embedding_dim
    predictor = make_predictor(dim, jcfg, args.variant).to(device)
    layers = [p for n, p in student.named_parameters()
              if p.requires_grad and (n.startswith("bert.blocks") or n.startswith("bert.final_norm"))]
    others = [p for n, p in student.named_parameters()
              if p.requires_grad and not (n.startswith("bert.blocks") or n.startswith("bert.final_norm"))]
    opt = torch.optim.AdamW([
        {"params": layers, "base_lr": args.lr * args.lr_layers_scale},
        {"params": others + list(predictor.parameters()), "base_lr": args.lr},
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 \
        else (lambda: torch.autocast(device.type, enabled=False))

    run_name = args.run_name or (f"JEPA_{model}_{time.strftime('%Y%m%d_%H%M')}" if args.variant == "base"
                                 else f"JEPA_{model}_{args.variant}")
    run_dir = Path(args.out_dir) / run_name
    if (run_dir / "last.pt").exists() and not args.resume:     # never overwrite a trained run by accident
        raise FileExistsError(f"{run_dir} already has a checkpoint: add --resume, or choose another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    step, best = 0, float("inf")
    if args.resume and (run_dir / args.resume_checkpoint).exists():
        ck = torch.load(run_dir / args.resume_checkpoint, map_location=device, weights_only=False)
        saved = ck["config"].get("variant", ck["config"]["args"].get("variant", "base"))
        if saved != args.variant:
            raise ValueError(f"{run_dir} was trained with --variant {saved}: resume with the same variant")
        student.load_state_dict(ck["student"])
        teacher.load_state_dict(ck["teacher"])
        predictor.load_state_dict(ck["predictor"])
        opt.load_state_dict(ck["optimizer"])
        step, best = ck["step"], ck["best"]
        print(f"resumed {run_dir} from {args.resume_checkpoint} at step {step}", flush=True)

    config = {"args": vars(args), "jepa": jcfg, "bert_run": str(args.init_run), "bert_config": bert_cfg,
              "model": model, "min_patients": min_patients, "arrays": str(arrays_dir), "dim": dim,
              "variant": args.variant}
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=1, default=str)
    # on resume, curve points logged after the checkpoint's step (e.g. a NaN tail) are hidden in TensorBoard
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), purge_step=step if step > 0 else None)

    def save(name):
        torch.save({"student": student.state_dict(), "teacher": teacher.state_dict(),
                    "predictor": predictor.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "best": best, "config": config}, run_dir / name)

    # 3) validation batches (with and without history), then training
    val_dir = arrays_dir / "val"
    val = fixed_batches(val_dir, meta, jcfg, model, args.batch_size, args.eval_batches)
    val_no_history = fixed_batches(val_dir, meta, jcfg, model, args.batch_size, args.eval_batches, use_history=False)
    train = JepaBatches(arrays_dir / "train", meta, jcfg, model, args.batch_size, seed=args.seed + step)
    loader = torch.utils.data.DataLoader(train, batch_size=None, num_workers=args.num_workers,
                                         persistent_workers=args.num_workers > 0)
    student.train()
    t0, logged, n_skipped = time.time(), {}, 0
    for batch in loader:
        if step >= args.max_steps:
            break
        factor = lr_factor(step, args)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * factor
        batch = to_device(batch, device)
        with autocast():
            c, y, y_hat = forward(student, teacher, predictor, batch, args.variant)
        total, pred, var_term, cov_term = losses(c, y, y_hat, jcfg)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(total):     # one bad batch must not turn every weight into NaN: skip it
            n_skipped += 1
            print(f"step {step}: loss is {total.item()}, batch skipped ({n_skipped} so far)", flush=True)
            writer.add_scalar("train/skipped_batches", n_skipped, step)
            continue
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(predictor.parameters()), 1.0)
        opt.step()
        momentum = ema_momentum(step, args, jcfg)
        ema_update(teacher, student, momentum)
        step += 1

        for k, v in {"loss_total": total.item(), "pred_loss": pred.mean().item(),
                     "mean_target_loss": ((y - y.mean(0)) ** 2).mean().item(),
                     "var_term": var_term.item(), "cov_term": cov_term.item(),
                     "std_c": c.float().std(0).mean().item(), "ema_momentum": momentum,
                     "pairs_per_context": len(batch["pair_ctx"]) / len(batch["ctx_key"]),
                     "history_per_context": batch["n_history"] / len(batch["ctx_key"])}.items():
            s, n = logged.get(k, (0.0, 0))
            logged[k] = (s + v, n + 1)
        if step % args.log_every == 0:
            for k, (s, n) in logged.items():
                writer.add_scalar(f"train/{k}", s / n, step)
            writer.add_scalar("train/lr", opt.param_groups[1]["lr"], step)
            print(f"step {step:>7}  pred {logged['pred_loss'][0] / logged['pred_loss'][1]:.4f}  "
                  f"mean-target {logged['mean_target_loss'][0] / logged['mean_target_loss'][1]:.4f}  "
                  f"std_c {logged['std_c'][0] / logged['std_c'][1]:.3f}  "
                  f"{args.log_every / (time.time() - t0):.1f} steps/s", flush=True)
            logged, t0 = {}, time.time()

        # 4) evaluation and checkpoints
        if step % args.eval_every == 0 or step == args.max_steps:
            res = evaluate(student, teacher, predictor, val, val_no_history, jcfg, device, autocast, args.variant)
            for k, v in res.items():
                writer.add_scalar(f"val/{k}", v, step)
            print("  eval step {}: ".format(step) + "  ".join(f"{k} {v:.4f}" for k, v in sorted(res.items())),
                  flush=True)
            if res["pred_ratio"] < best:          # the ratio is comparable across steps, the raw loss is not
                best = res["pred_ratio"]
                save("best.pt")
            save("last.pt")
    save("last.pt")
    writer.close()
    print(f"done: best validation pred_ratio {best:.4f}, run saved in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
