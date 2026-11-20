"""BERT pre-training on the MIMIC master trajectories, for D1, D2 or D3.

Steps:
  1. arrays for (model, min_patients) are built from the master table if they do not exist yet
  2. fixed validation batches; baseline of each loss part = loss of the frequency guess on them
  3. training with packed rows, masking from the masking config, loss = sum_k w_k * mean_k / baseline_k
  4. evaluation every --eval-every steps; best.pt and last.pt saved; early stop on the validation loss
  5. TensorBoard logs in <run folder>/tensorboard

Usage (PowerShell), from the bert folder:
  python train_bert.py --model D1 --min-patients 50
  python train_bert.py --model D3 --min-patients 50 --max-steps 200000 --batch-size 16
  tensorboard --logdir D:\\MIMIC\\bert_model
Resume an interrupted run:
  python train_bert.py --model D1 --min-patients 50 --run-name D1_min50 --resume
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

from bert_data import PackedBatches, fixed_batches, load_yaml, prepare_arrays, soft_label_matrix
from bert_model import TrajectoryBert

HERE = Path(__file__).resolve().parent
PARTS = ["concept_event", "concept_measure", "quantile", "unit", "value"]


def parse_args():
    ap = argparse.ArgumentParser(description="BERT pre-training on MIMIC trajectories (D1, D2, D3)")
    ap.add_argument("--model", choices=["D1", "D2", "D3"], required=True, help="masking strategy / trajectory type")
    ap.add_argument("--masking-config", default=str(HERE.parent / "config" / "masking_configs.yaml"))
    ap.add_argument("--trajectory-config", default=str(HERE / "trajectory_columns.yaml"))
    ap.add_argument("--min-patients", type=int, default=None,
                    help="vocabulary cut-off: concepts in fewer patients become [UNK] (default: from the trajectory config)")
    ap.add_argument("--out-dir", default=r"D:\MIMIC\bert_model", help="where runs are saved")
    ap.add_argument("--run-name", default=None, help="folder name of the run (default: <model>_min<N>_<date>)")
    ap.add_argument("--resume", action="store_true", help="continue the run --run-name from its last.pt")
    ap.add_argument("--prepare-only", action="store_true", help="only build the arrays, then stop")
    # model
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    # training
    ap.add_argument("--batch-size", type=int, default=32, help="packed rows per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=150000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8, help="evaluations without improvement before stopping")
    ap.add_argument("--min-delta", type=float, default=0.002, help="relative improvement that counts")
    # Windows: worker processes send batches through shared memory, which can exhaust the paging file
    # (RuntimeError "Couldn't open shared file mapping", error 1455), so the default there is 0
    ap.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2,
                    help="processes preparing batches (0 = main process; default 0 on Windows)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


# ---- loss --------------------------------------------------------------------------------------
def segment_mean(values, key):
    """Mean within each patient segment, then over segments: every segment counts the same."""
    uniq, inv = torch.unique(key, return_inverse=True)
    sums = torch.zeros(len(uniq), device=values.device).scatter_add_(0, inv, values.float())
    counts = torch.zeros(len(uniq), device=values.device).scatter_add_(0, inv, torch.ones_like(values, dtype=torch.float))
    return (sums / counts).mean()


def part_losses(model, batch, soft, huber_delta):
    """Loss (segment mean) and simple metrics of each part present in the batch."""
    h = model(batch)
    B, L = batch["seg"].shape
    seg_key = batch["seg"] + torch.arange(B, device=h.device)[:, None] * (L + 1)   # unique per segment
    losses, metrics = {}, {}

    # concept, split by kind of event: 1 = without a number, 2 = numeric
    m = batch["tgt_concept"] >= 0
    if m.any():
        logits = model.concept_out(model.head(h[m])).float()
        tgt, part, key = batch["tgt_concept"][m], batch["tgt_cpart"][m], seg_key[m]
        ce = F.cross_entropy(logits, tgt, reduction="none")
        top10 = logits.topk(min(10, logits.shape[-1]), dim=-1).indices
        for name, code in (("concept_event", 1), ("concept_measure", 2)):
            sel = part == code
            if sel.any():
                losses[name] = segment_mean(ce[sel], key[sel])
                metrics[f"{name}_acc"] = (logits[sel].argmax(-1) == tgt[sel]).float().mean().item()
                metrics[f"{name}_top10"] = (top10[sel] == tgt[sel, None]).any(-1).float().mean().item()

    # quantile, soft labels on the neighbouring bins
    m = batch["tgt_q"] >= 0
    if m.any():
        logits = model.quantile_out(model.head(h[m])).float()
        tgt = batch["tgt_q"][m]
        loss = -(soft[tgt] * F.log_softmax(logits, dim=-1)).sum(-1)
        losses["quantile"] = segment_mean(loss, seg_key[m])
        pred = logits.argmax(-1)
        metrics["quantile_acc"] = (pred == tgt).float().mean().item()
        metrics["quantile_within1"] = ((pred - tgt).abs() <= 1).float().mean().item()

    # unit
    m = batch["tgt_unit"] >= 0
    if m.any():
        logits = model.unit_out(model.head(h[m])).float()
        tgt = batch["tgt_unit"][m]
        losses["unit"] = segment_mean(F.cross_entropy(logits, tgt, reduction="none"), seg_key[m])
        metrics["unit_acc"] = (logits.argmax(-1) == tgt).float().mean().item()

    # value (D3): Huber on the scaled value
    m = ~torch.isnan(batch["tgt_value"])
    if m.any():
        pred = model.value_out(model.head(h[m])).float().squeeze(-1)
        tgt = batch["tgt_value"][m]
        losses["value"] = segment_mean(F.huber_loss(pred, tgt, delta=huber_delta, reduction="none"), seg_key[m])
        metrics["value_mae"] = (pred - tgt).abs().mean().item()
    return losses, metrics


def total_loss(losses, baselines, weights):
    """Sum over parts of weight * (loss / baseline): each part reads as 'share of the frequency-guess loss left'."""
    return sum(weights[k] * losses[k] / baselines[k] for k in losses if k in baselines)


# ---- baselines: the frequency guess -------------------------------------------------------------
def frequency_baselines(batches, soft, huber_delta):
    """Loss of predicting, for every masked target, the frequency of each answer (no context at all).

    Measured on the validation batches with the same averaging as the real loss, so the two compare directly.
    """
    coll = {k: ([], []) for k in ["concept_event", "concept_measure", "quantile", "unit", "value"]}
    for i, b in enumerate(batches):
        B, L = b["seg"].shape
        key = b["seg"] + (torch.arange(B)[:, None] + i * B) * (L + 1)
        for name, code in (("concept_event", 1), ("concept_measure", 2)):
            m = (b["tgt_concept"] >= 0) & (b["tgt_cpart"] == code)
            coll[name][0].append(b["tgt_concept"][m]); coll[name][1].append(key[m])
        for name, col in (("quantile", "tgt_q"), ("unit", "tgt_unit")):
            m = b[col] >= 0
            coll[name][0].append(b[col][m]); coll[name][1].append(key[m])
        m = ~torch.isnan(b["tgt_value"])
        coll["value"][0].append(b["tgt_value"][m]); coll["value"][1].append(key[m])

    baselines = {}
    for name, (tgts, keys) in coll.items():
        t, k = torch.cat(tgts), torch.cat(keys)
        if len(t) == 0:
            continue
        if name == "value":
            loss = F.huber_loss(torch.zeros_like(t), t, delta=huber_delta, reduction="none")   # always the median
        else:
            p = torch.bincount(t).float()
            p = p / p.sum()
            if name == "quantile":
                p = torch.cat([p, torch.zeros(10 - len(p))]) if len(p) < 10 else p
                loss = -(soft[t] * torch.log(p.clamp_min(1e-12))).sum(-1)
            else:
                loss = -torch.log(p[t])
        baselines[name] = max(segment_mean(loss, k).item(), 1e-6)
    return baselines


# ---- training -----------------------------------------------------------------------------------
def lr_at(step, args):
    """Linear warm-up, then cosine decay to min_lr at max_steps."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = min(1.0, (step - args.warmup) / max(1, args.max_steps - args.warmup))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, batches, soft, mcfg, baselines, device, autocast):
    model.eval()
    sums, counts = {}, {}
    for b in batches:
        with autocast():
            losses, metrics = part_losses(model, to_device(b, device), soft, mcfg["huber_delta"])
        for k, v in {**{f"loss_{k}": v.item() for k, v in losses.items()}, **metrics}.items():
            sums[k], counts[k] = sums.get(k, 0.0) + v, counts.get(k, 0) + 1
    model.train()
    res = {k: sums[k] / counts[k] for k in sums}
    parts = {k: res[f"loss_{k}"] for k in PARTS if f"loss_{k}" in res}
    res["loss_total"] = float(sum(mcfg["loss_weights"][k] * v / baselines[k] for k, v in parts.items() if k in baselines))
    for k, v in parts.items():
        if k in baselines:
            res[f"ratio_{k}"] = v / baselines[k]      # share of the frequency-guess loss left (1 = no context learned)
    return res


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mcfg, tcfg = load_yaml(args.masking_config), load_yaml(args.trajectory_config)
    min_patients = args.min_patients if args.min_patients is not None else tcfg["min_patients"]

    # 1) arrays for this model and cut-off
    arrays_dir = Path(tcfg["paths"]["out"]) / "bert" / "arrays" / f"{args.model}_min{min_patients}"
    if not (arrays_dir / "meta.json").exists():
        print(f"building arrays in {arrays_dir} ...", flush=True)
        prepare_arrays(tcfg, args.model, min_patients, arrays_dir)
    with open(arrays_dir / "meta.json") as f:
        meta = json.load(f)
    print(f"{args.model}, min_patients {min_patients}: vocabulary {meta['vocab_size']:,} tokens, "
          f"{meta['n_patients']['train']:,} train patients, {meta['n_tokens']['train']:,} train tokens", flush=True)
    if args.prepare_only:
        return

    # run folder
    run_name = args.run_name or f"{args.model}_min{min_patients}_{time.strftime('%Y%m%d_%H%M')}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 \
        else (lambda: torch.autocast(device.type, enabled=False))

    # model and optimiser
    model = TrajectoryBert(meta["vocab_size"], len(meta["sources"]), len(meta["unit_ids"]), args.dim, args.layers,
                           args.heads, args.dropout, use_value=(args.model == "D3")).to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "tok" not in n and "src" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "tok" not in n and "src" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))
    soft = soft_label_matrix(mcfg["soft_label_eps"])
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters, device {device}", flush=True)

    # 2) validation batches and baselines
    val = fixed_batches(arrays_dir / "val", mcfg, meta, args.model, args.batch_size, args.eval_batches)
    baselines = frequency_baselines(val[:mcfg["baseline_batches"]], soft, mcfg["huber_delta"])
    print("baselines (loss of the frequency guess):", {k: round(v, 4) for k, v in baselines.items()}, flush=True)
    soft = soft.to(device)

    step, best, bad = 0, float("inf"), 0
    if args.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best, bad, baselines = ck["step"], ck["best"], ck["bad_evals"], ck["baselines"]
        print(f"resumed {run_dir} at step {step}", flush=True)

    config = {"args": vars(args), "min_patients": min_patients, "masking": mcfg, "meta_file": str(arrays_dir / "meta.json"),
              "baselines": baselines, "vocab_size": meta["vocab_size"], "n_sources": len(meta["sources"]),
              "n_units": len(meta["unit_ids"])}
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=1)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    def save(name):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "best": best,
                    "bad_evals": bad, "baselines": baselines, "config": config}, run_dir / name)

    # 3) training
    train = PackedBatches(arrays_dir / "train", mcfg, meta, args.model, args.batch_size, seed=args.seed + step)
    loader = torch.utils.data.DataLoader(train, batch_size=None, num_workers=args.num_workers,
                                         persistent_workers=args.num_workers > 0, pin_memory=device.type == "cuda")
    weights = mcfg["loss_weights"]
    model.train()
    t0, logged = time.time(), {}
    for batch in loader:
        if step >= args.max_steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        batch = to_device(batch, device)
        with autocast():
            losses, metrics = part_losses(model, batch, soft, mcfg["huber_delta"])
        loss = total_loss(losses, baselines, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        # running averages for the training logs
        for k, v in {"loss_total": loss.item(), **{f"loss_{k}": v.item() for k, v in losses.items()}, **metrics,
                     "masked_numeric_share": batch["n_numeric"] / max(1, batch["n_masked"]),
                     "tokens_used": (batch["seg"] > 0).float().mean().item()}.items():
            s, c = logged.get(k, (0.0, 0))
            logged[k] = (s + v, c + 1)
        if step % args.log_every == 0:
            for k, (s, c) in logged.items():
                writer.add_scalar(f"train/{k}", s / c, step)
            writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
            rate = args.log_every / (time.time() - t0)
            print(f"step {step:>7}  loss {logged['loss_total'][0] / logged['loss_total'][1]:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {rate:.1f} steps/s", flush=True)
            logged, t0 = {}, time.time()

        # 4) evaluation, checkpoints, early stop
        if step % args.eval_every == 0 or step == args.max_steps:
            res = evaluate(model, val, soft, mcfg, baselines, device, autocast)
            for k, v in res.items():
                writer.add_scalar(f"val/{k}", v, step)
            print(f"  eval step {step}: " + "  ".join(f"{k} {v:.4f}" for k, v in sorted(res.items())), flush=True)
            if res["loss_total"] < best * (1 - args.min_delta):
                best, bad = res["loss_total"], 0
                save("best.pt")
            else:
                bad += 1
            save("last.pt")
            if bad >= args.patience:
                print(f"no improvement for {bad} evaluations: stop at step {step}", flush=True)
                break
    save("last.pt")
    writer.close()
    print(f"done: best validation loss {best:.4f}, run saved in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
