"""Step R2: one vector per point, for every map compared by the resilience score.

JEPA maps (--run, a JEPA run folder; the variant is read from the run's config.json):
  c                   student encoder on the context at the point's landmark T, built exactly as in training
                      (gender + history + recent window, jepa_data.context_at), pooled as in the variant:
                        base, z_transformer   mean over the context tokens (dim numbers)
                        concat_hist           [history mean ; mean of the recent per-day means], each half
                                              scaled to unit length (2 x dim numbers, see map_vector)
  yhat_g<g>_h<H>      Z(context, delta = g, L = H): the expected future of (T + g, T + g + H], one map per
                      setting in resilience_config.yaml (predictions). Z of the variant: the MLP on c (base,
                      concat_hist + its no-history flag) or the cross-attention query over all context tokens
                      (z_transformer). Every point, reference ones included, goes through the same Z(., g, H);
                      c and y_hat are never mixed in one neighbour search.

Ablations, c only (gender + recent window, no history block):
  --run ... --no-history   the JEPA student without its history block (a test-time ablation: trained with it)
  --bert-run ...           the BERT encoder before JEPA (it never saw a history block), mean over the same tokens
                           BERT vs JEPA without history: what JEPA training adds, with the same input.

Baselines (--baseline):
  bag                 the concepts of the same context (history items once, recent window counted), log counts
                      x idf, reduced to 256 dimensions by a truncated SVD fitted on train points
  demo                age, sex and history depth log(1 + h), standardised on train points (euclidean distance)
  random              random vectors (sanity check: expected AUROC 0.5)

Output: <out>/resilience/embeddings/<name>/
  <map>.npy           float16, one row per point in point_id order
  context.parquet     point_id, history items and recent tokens of each context (JEPA and bag)
  info.json           maps with their distance (cosine or euclidean), run, checkpoint, settings

Usage (PowerShell):
  cd C:\\Users\\yy\\Desktop\\codes_PhD\\jepa_mimic\\resilience
  python embed_points.py --run D:\\MIMIC\\jepa_model\\JEPA_D3
  python embed_points.py --run D:\\MIMIC\\jepa_model\\JEPA_D1
  python embed_points.py --run D:\\MIMIC\\jepa_model\\JEPA_D2
  python embed_points.py --run D:\\MIMIC\\jepa_model\\JEPA_D3 --no-history     # saved as JEPA_D3_nohist
  python embed_points.py --run D:\\MIMIC\\jepa_model\\JEPA_D2_concat_hist     # variant A
  python embed_points.py --run D:\\MIMIC\\jepa_model\\JEPA_D2_z_transformer   # variant B
  python embed_points.py --bert-run D:\\MIMIC\\bert_model\\D3_min50             # saved as BERT_D3
  python embed_points.py --baseline bag
  python embed_points.py --baseline demo
  python embed_points.py --baseline random
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bert"))
sys.path.insert(0, str(HERE.parent / "jepa"))
from bert_data import SPLITS, load_yaml  # noqa: E402
from jepa_data import SEGMENT, JepaArrays, context_at, pack  # noqa: E402
from jepa_model import (JepaEncoder, context_inputs, load_bert, make_predictor, map_vector,  # noqa: E402
                        run_predictor)


# ---- landmarks -----------------------------------------------------------------------------------
def landmark_tokens(a, pts):
    """Index of each point's landmark token in the arrays: the last token of its episode, in its patient."""
    n_lm = np.diff(a.landmark_ptr)
    patient = np.repeat(np.arange(a.n_patients), n_lm)
    episode = np.asarray(a.f["episode"][a.landmark_idx])
    key = patient.astype(np.int64) * 1_000_000 + episode
    order = np.argsort(key)
    want = pts["p"].to_numpy().astype(np.int64) * 1_000_000 + pts["episode"].to_numpy()
    at = np.searchsorted(key[order], want)
    found = (at < len(key)) & (key[order][np.minimum(at, len(key) - 1)] == want)
    if not found.all():
        raise ValueError(f"{(~found).sum()} points have no landmark in the arrays: rebuild points.parquet")
    t_idx = a.landmark_idx[order[at]]
    return t_idx


def context_sizes(point_ids, ctx):
    """History items and recent tokens of each context (a check that the contexts look like training)."""
    return pl.DataFrame({
        "point_id": point_ids,
        "history_items": [int((c["segment"] == SEGMENT["history"]).sum()) for c in ctx],
        "recent_tokens": [int((c["segment"] == SEGMENT["recent"]).sum()) for c in ctx],
    })


def batches_of_contexts(a, pts, jcfg, caps, batch, use_history=True):
    """(point ids, contexts) batch after batch: the contexts of ~750k points do not fit in memory at once."""
    t_idx = landmark_tokens(a, pts)
    p_all, ids = pts["p"].to_numpy(), pts["point_id"].to_numpy()
    for i in range(0, len(ids), batch):
        sl = slice(i, i + batch)
        yield ids[sl], [context_at(a, int(p), int(t), jcfg, caps, use_history) for p, t in zip(p_all[sl], t_idx[sl])]


# ---- JEPA maps ------------------------------------------------------------------------------------
def load_jepa(run_dir, checkpoint, device):
    """Student encoder and predictor Z of a JEPA run, and the run's config (runs before the variants: base)."""
    cfg = json.loads((run_dir / "config.json").read_text())
    cfg.setdefault("variant", "base")
    bert, _ = load_bert(cfg["bert_run"], cfg["args"]["init_checkpoint"], "cpu")   # architecture; weights below
    jcfg = cfg["jepa"]
    student = JepaEncoder(bert)
    predictor = make_predictor(cfg["dim"], jcfg, cfg["variant"])
    ck = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=False)
    student.load_state_dict(ck["student"])
    predictor.load_state_dict(ck["predictor"])
    print(f"loaded {run_dir / checkpoint} (variant {cfg['variant']}, step {ck['step']}, "
          f"best val pred_ratio {ck['best']:.4f})", flush=True)
    return student.to(device).eval(), predictor.to(device).eval(), cfg, ck["step"]


@torch.no_grad()
def embed_split(a, sub, meta, jcfg, caps, encoder, predictor, settings, batch, device, maps, use_history=True,
                variant="base"):
    """c and every y_hat of the points of one split, written into maps (rows = point_id).
    encoder: the JEPA student (forward(b)) or a BERT encoder (same call, no segment embedding, variant base)."""
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    sizes, done, t0 = [], 0, time.time()
    for k, (ids, ctx) in enumerate(batches_of_contexts(a, sub, jcfg, caps, batch, use_history)):
        b, keys = pack(ctx, a, jcfg["row_len"], meta["value_id"])
        b = {name: v.to(device, non_blocking=True) for name, v in b.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            h = encoder(b)
            c, inputs = context_inputs(variant, h, b, keys.to(device))      # as in training (train_jepa.forward)
            maps["c"][ids] = map_vector(variant, c.float()).cpu().numpy()
            one_each = torch.arange(len(c), device=device)                   # one (delta, L) pair per context
            for name, (gap, horizon) in settings.items():
                delta = torch.full((len(c),), float(gap), device=device)
                length = torch.full((len(c),), float(horizon), device=device)
                y_hat = run_predictor(variant, predictor, c, inputs, one_each, delta, length)
                maps[name][ids] = y_hat.float().cpu().numpy()
        sizes.append(context_sizes(ids, ctx))
        done += len(ids)
        if k % 200 == 0:
            print(f"  {done:,}/{sub.height:,} points  {time.time() - t0:.0f}s", flush=True)
    return pl.concat(sizes)


def jepa_maps(args, tcfg, rcfg, pts, device):
    run_dir = Path(args.run)
    student, predictor, cfg, step = load_jepa(run_dir, args.checkpoint, device)
    jcfg, model, arrays_dir = cfg["jepa"], cfg["model"], Path(cfg["arrays"])
    caps = jcfg["models"][model]
    meta = json.loads((arrays_dir / "meta.json").read_text())
    use_history = not args.no_history
    settings = {} if args.no_history else {     # without history: c only (Z was trained on contexts with history)
        f"yhat_g{s['gap']}_h{s['horizon']}": (s["gap"], s["horizon"]) for s in rcfg["predictions"]}
    for name, (gap, horizon) in settings.items():
        if gap > jcfg["delta_days"]["max"] or horizon > jcfg["length_days"]["max"]:
            raise ValueError(f"{name} is outside the trained range of Z (delta <= {jcfg['delta_days']['max']}, "
                             f"L <= {jcfg['length_days']['max']})")

    dim, variant = cfg["dim"], cfg["variant"]
    c_dim = 2 * dim if variant == "concat_hist" else dim
    maps = {"c": np.zeros((pts.height, c_dim), np.float32), **{k: np.zeros((pts.height, dim), np.float32) for k in settings}}
    sizes = []
    for s in SPLITS:
        sub = pts.filter(pl.col("split") == s).sort("p")       # patients in array order: faster reads
        if sub.height == 0:
            continue
        a = JepaArrays(arrays_dir / s, meta, jcfg)
        print(f"{s}: {sub.height:,} points", flush=True)
        sizes.append(embed_split(a, sub, meta, jcfg, caps, student, predictor, settings,
                                 args.batch or rcfg["embed_batch"], device, maps, use_history, variant))
    info = {"kind": "jepa", "variant": variant, "run": str(run_dir), "checkpoint": args.checkpoint, "step": step,
            "model": model,
            "bert_run": cfg["bert_run"], "recent_days": jcfg["recent_days"], "history": use_history,
            "maps": {"c": {"metric": "cosine"},
                     **{k: {"metric": "cosine", "gap": g, "horizon": h} for k, (g, h) in settings.items()}}}
    return maps, pl.concat(sizes).sort("point_id"), info


# ---- BERT maps (before JEPA) ------------------------------------------------------------------------------
def bert_maps(args, tcfg, rcfg, pts, device):
    """c of the BERT encoder on gender + recent window (no history: BERT never saw a history block)."""
    run_dir = Path(args.bert_run)
    bert, bcfg = load_bert(run_dir, args.checkpoint, "cpu")
    bert = bert.to(device).eval()
    model, min_patients = bcfg["args"]["model"], bcfg["min_patients"]
    jcfg = load_yaml(args.jepa_config)                    # recent_days, caps: the same context rules as JEPA
    caps = jcfg["models"][model]
    arrays_dir = Path(tcfg["paths"]["out"]) / "jepa" / "arrays" / f"{model}_min{min_patients}"
    meta = json.loads((arrays_dir / "meta.json").read_text())
    maps = {"c": np.zeros((pts.height, bert.tok.embedding_dim), np.float32)}
    sizes = []
    for s in SPLITS:
        sub = pts.filter(pl.col("split") == s).sort("p")
        if sub.height == 0:
            continue
        a = JepaArrays(arrays_dir / s, meta, jcfg)
        print(f"{s}: {sub.height:,} points", flush=True)
        sizes.append(embed_split(a, sub, meta, jcfg, caps, bert, None, {}, args.batch or rcfg["embed_batch"],
                                 device, maps, use_history=False))
    info = {"kind": "bert", "run": str(run_dir), "checkpoint": args.checkpoint, "model": model,
            "recent_days": jcfg["recent_days"], "history": False, "maps": {"c": {"metric": "cosine"}}}
    return maps, pl.concat(sizes).sort("point_id"), info


# ---- baselines ------------------------------------------------------------------------------------------
def bag_map(args, tcfg, rcfg, pts):
    """Bag of codes of the same context as the JEPA maps (concepts only: the D1 arrays), idf, truncated SVD."""
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD

    jcfg = load_yaml(args.jepa_config)
    model = "D1"
    arrays_dir = Path(tcfg["paths"]["out"]) / "jepa" / "arrays" / f"{model}_min{tcfg['min_patients']}"
    meta = json.loads((arrays_dir / "meta.json").read_text())
    caps = jcfg["models"][model]
    rows_all, sizes = [], []
    for s in SPLITS:
        sub = pts.filter(pl.col("split") == s).sort("p")
        if sub.height == 0:
            continue
        a = JepaArrays(arrays_dir / s, meta, jcfg)
        tokens = a.f["tokens"]
        indptr, indices, data = [0], [], []
        for ids_, ctx in batches_of_contexts(a, sub, jcfg, caps, 2048):
            sizes.append(context_sizes(ids_, ctx))
            for c in ctx:
                ids, counts = np.unique(np.asarray(tokens[c["idx"][1:]]), return_counts=True)   # gender left out
                indices.append(ids)
                data.append(np.log1p(counts).astype(np.float32))
                indptr.append(indptr[-1] + len(ids))
        m = sparse.csr_matrix((np.concatenate(data), np.concatenate(indices), np.array(indptr)),
                              shape=(sub.height, meta["vocab_size"]))
        rows_all.append((sub["point_id"].to_numpy(), m))
        print(f"{s}: {sub.height:,} bags", flush=True)

    point_rows = np.concatenate([r for r, _ in rows_all])
    X = sparse.vstack([m for _, m in rows_all]).tocsr()[np.argsort(point_rows)]   # point_id order

    # idf and SVD fitted on a sample of train points
    rng = np.random.default_rng(args.seed)
    train_rows = pts.filter(pl.col("split") == "train")["point_id"].to_numpy()
    fit = rng.choice(train_rows, size=min(len(train_rows), rcfg["bag"]["fit_points"]), replace=False)
    df = np.asarray((X[fit] > 0).sum(axis=0)).ravel()
    idf = np.log((1 + len(fit)) / (1 + df)) + 1
    X = X @ sparse.diags(idf.astype(np.float32))
    dim = min(rcfg["bag"]["dim"], X.shape[1] - 1)
    svd = TruncatedSVD(n_components=dim, algorithm="randomized", random_state=args.seed).fit(X[fit])
    z = svd.transform(X).astype(np.float32)
    print(f"bag: {X.shape[1]:,} concepts -> {dim} dims, explained variance {svd.explained_variance_ratio_.sum():.3f}")
    info = {"kind": "bag", "arrays": str(arrays_dir), "recent_days": jcfg["recent_days"], "dim": dim,
            "fit_points": int(len(fit)), "explained_variance": float(svd.explained_variance_ratio_.sum()),
            "maps": {"bag": {"metric": "cosine"}}}
    return {"bag": z}, pl.concat(sizes).sort("point_id"), info


def demo_map(pts):
    """Age, sex and log(1 + h), standardised on train points."""
    x = np.stack([pts["age_years"].to_numpy().astype(np.float32),
                  (pts["gender"] == "female").to_numpy().astype(np.float32),
                  np.log1p(pts["h"].to_numpy().astype(np.float32))], axis=1)
    train = (pts["split"] == "train").to_numpy()
    mean, std = x[train].mean(0), x[train].std(0)
    z = (x - mean) / np.where(std > 0, std, 1)
    info = {"kind": "demo", "columns": ["age_years", "female", "log1p_h"], "maps": {"demo": {"metric": "euclidean"}}}
    return {"demo": z.astype(np.float32)}, None, info


def random_map(pts, dim, seed):
    z = np.random.default_rng(seed).standard_normal((pts.height, dim)).astype(np.float32)
    return {"random": z}, None, {"kind": "random", "maps": {"random": {"metric": "cosine"}}}


# ---- main ------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--run", help="JEPA run folder, e.g. D:\\MIMIC\\jepa_model\\JEPA_D3")
    what.add_argument("--bert-run", help="BERT run folder, e.g. D:\\MIMIC\\bert_model\\D3_min50 (c without history)")
    what.add_argument("--baseline", choices=["bag", "demo", "random"])
    ap.add_argument("--no-history", action="store_true", help="JEPA run: contexts without the history block (c only)")
    ap.add_argument("--checkpoint", default="best.pt", help="best.pt or last.pt of the run")
    ap.add_argument("--name", default=None, help="output folder name (default: run folder name, or the baseline)")
    ap.add_argument("--trajectory-config", default=str(HERE.parent / "bert" / "trajectory_columns.yaml"))
    ap.add_argument("--config", default=str(HERE.parent / "config" / "resilience_config.yaml"))
    ap.add_argument("--jepa-config", default=str(HERE.parent / "config" / "jepa_config.yaml"),
                    help="context rules for --bert-run and the bag baseline")
    ap.add_argument("--batch", type=int, default=None, help="contexts per forward pass")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tcfg, rcfg = load_yaml(args.trajectory_config), load_yaml(args.config)
    res_dir = Path(tcfg["paths"]["out"]) / rcfg["out_dir"]
    pts = pl.read_parquet(res_dir / "points.parquet").sort("point_id")
    assert (pts["point_id"].to_numpy() == np.arange(pts.height)).all(), "point_id must be 0 .. n-1"
    t0 = time.time()

    suffix = ("" if args.checkpoint == "best.pt" else "_" + Path(args.checkpoint).stem)
    if args.run:
        name = args.name or (Path(args.run).name + ("_nohist" if args.no_history else "") + suffix)
        maps, sizes, info = jepa_maps(args, tcfg, rcfg, pts, torch.device(args.device))
    elif args.bert_run:
        bcfg = json.loads((Path(args.bert_run) / "config.json").read_text())
        name = args.name or ("BERT_" + bcfg["args"]["model"] + suffix)
        maps, sizes, info = bert_maps(args, tcfg, rcfg, pts, torch.device(args.device))
    elif args.baseline == "bag":
        name = args.name or "bag"
        maps, sizes, info = bag_map(args, tcfg, rcfg, pts)
    elif args.baseline == "demo":
        name = args.name or "demo"
        maps, sizes, info = demo_map(pts)
    else:
        name = args.name or "random"
        maps, sizes, info = random_map(pts, rcfg["random_dim"], args.seed)

    out_dir = res_dir / "embeddings" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in maps.items():
        np.save(out_dir / f"{k}.npy", v.astype(np.float16))
        print(f"  {k}: {v.shape}, mean per-dimension std {v.std(0).mean():.3f}", flush=True)
    if sizes is not None:
        sizes.write_parquet(out_dir / "context.parquet")
        print(sizes.select(pl.col("history_items").mean().round(1), pl.col("recent_tokens").mean().round(1)))
    info.update({"name": name, "n_points": pts.height, "seconds": round(time.time() - t0),
                 "built": time.strftime("%Y-%m-%d %H:%M")})
    (out_dir / "info.json").write_text(json.dumps(info, indent=1))
    print(f"written {out_dir} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
