"""Step R4: neighbour score r, gate and resilience list for every map (HERO plan, sections 4-7).

For each map (c, y_hat_g<g>_h<H>, bag, demo, random):
  1. neighbours   for every point, its K = 100 nearest TRAIN points of OTHER patients, each other patient once
                  (their nearest point). Cosine distance (euclidean for demo). Computed once per map and reused
                  for every disease and horizon: the vectors never see a label.
  2. score        r_i = #positive neighbours / #countable neighbours, countable = positive, tested abnormal,
                  tested negative or untested (death, no follow-up, prevalent, excluded, gap: not countable).
                  Fewer than n_min = 20 countable: no score, the point gets the prevalence pi in the metrics.
  3. gate         on the evaluated points of a split (own status positive / tested / untested), each patient
                  weighted 1 / (its number of evaluated points):
                    pi          prevalence among train points (the reference), patient-weighted
                    AUROC       of r against the own label y, patient-weighted
                    PPV_q       weighted share of positives in the top q of r, and lift = PPV_q / pi
                    coverage    weighted share of points with a score
                  95 % intervals from resampling patients.
  4. list         candidates = top q of r with own status tested abnormal, tested negative or untested:
                  number of patients and the share of each status (patient-weighted).
A y_hat map is scored only with the labels of its own horizon and gap (it predicts that window).
Splits: val chooses (map, q), test is reported once, train is the leakage check (train vs test).

Outputs in <out>/resilience/scores/<embedding>/<map>/:
  neighbours.npy      int32 (points x K) point ids, -1 = fewer neighbours found
  scores.parquet      point_id, disease, horizon, r, n_countable
  metrics.csv         one row per disease, horizon, split and q
and <out>/resilience/scores/metrics_all.csv, all maps together (--collect, also run at the end of every call).

Usage (PowerShell):
  cd C:\\Users\\yy\\Desktop\\codes_PhD\\jepa_mimic\\resilience
  python score.py --embedding JEPA_D3                # every map of the embedding (c and y_hat)
  python score.py --embedding bag demo random
  python score.py --embedding JEPA_D3 --map c
  python score.py --collect                          # only rebuild metrics_all.csv
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
from bert_data import load_yaml  # noqa: E402

COUNTABLE = ["positive", "tested_abnormal", "tested_negative", "untested"]
CANDIDATE = ["tested_abnormal", "tested_negative", "untested"]


# ---- 1. neighbours -------------------------------------------------------------------------------------
class Reference:
    """The train points of a map, held on the device once (normalised for cosine)."""

    def __init__(self, x, ref_rows, metric, device):
        self.metric, self.device = metric, device
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.x = self.prepare(x)
        self.ref = self.x[ref_rows].to(device, self.dtype)
        self.ref_sq = (self.ref.float() ** 2).sum(1) if metric == "euclidean" else None
        self.ref_rows = torch.from_numpy(ref_rows).to(device)

    def prepare(self, x):
        xt = torch.from_numpy(x.astype(np.float32))
        return torch.nn.functional.normalize(xt, dim=1) if self.metric == "cosine" else xt


@torch.no_grad()
def search(R, rows, n, q_block=4096, r_block=65536):
    """For the points rows (a slice of the map), the n most similar reference points, best first."""
    ref, ref_sq, metric = R.ref, R.ref_sq, R.metric
    xt = R.x[rows]
    n = min(n, len(ref))
    out = np.zeros((len(xt), n), np.int64)
    for i in range(0, len(xt), q_block):
        q = xt[i:i + q_block].to(R.device, R.dtype)
        best_s = best_i = None
        for j in range(0, len(ref), r_block):
            s = (q @ ref[j:j + r_block].T).float()
            if metric == "euclidean":                  # minus the squared distance, up to a per-query constant
                s = 2 * s - ref_sq[j:j + r_block][None, :]
            v, ix = s.topk(min(n, s.shape[1]), dim=1)
            ix = ix + j
            if best_s is not None:                     # merge with the best of the previous reference blocks
                v, ix = torch.cat([best_s, v], 1), torch.cat([best_i, ix], 1)
                v, sel = v.topk(min(n, v.shape[1]), dim=1)
                ix = ix.gather(1, sel)
            best_s, best_i = v, ix
        out[i:i + len(q)] = R.ref_rows[best_i].cpu().numpy()
    return out


def one_per_patient(cand, patient, query_rows, k):
    """Keep, in rank order, the first (best) candidate of each other patient; the query's own patient is dropped.
    cand: candidates of the queries query_rows, best first. Returns k neighbours per query (-1 = not enough)."""
    n, m = cand.shape
    cand_pat = patient[cand]
    own = patient[query_rows][:, None]
    rank = np.tile(np.arange(m), n)
    row = np.repeat(np.arange(n), m)
    order = np.lexsort((rank, cand_pat.ravel(), row))          # by row, then patient, then rank
    r_s, p_s = row[order], cand_pat.ravel()[order]
    first = np.ones(len(order), bool)
    first[1:] = (r_s[1:] != r_s[:-1]) | (p_s[1:] != p_s[:-1])
    keep = np.zeros(n * m, bool)
    keep[order] = first
    keep = keep.reshape(n, m) & (cand_pat != own)
    pos = np.cumsum(keep, axis=1) - 1                           # rank among the kept candidates
    nb = np.full((n, k), -1, np.int32)
    rr, cc = np.nonzero(keep & (pos < k))
    nb[rr, pos[rr, cc]] = cand[rr, cc]
    return nb


def neighbours(x, pts, metric, k, n_search, device, block=20_000):
    """K neighbours of every point among the train points of other patients, each other patient once."""
    patient = pts["subject_id"].to_numpy()
    ref_rows = np.nonzero((pts["split"] == "train").to_numpy())[0]
    t0 = time.time()
    R = Reference(x, ref_rows, metric, device)
    nb = np.full((len(x), k), -1, np.int32)
    for i in range(0, len(x), block):              # candidates of a block of points, then one point per patient
        sl = slice(i, min(i + block, len(x)))
        nb[sl] = one_per_patient(search(R, sl, n_search), patient, np.arange(sl.start, sl.stop), k)
        if (i // block) % 10 == 0:
            print(f"  {sl.stop:,}/{len(x):,} points  {time.time() - t0:.0f}s", flush=True)
    print(f"  neighbours: {len(x):,} points, {len(ref_rows):,} train references, "
          f"{(nb >= 0).sum(1).mean():.1f} of {k} found on average  ({time.time() - t0:.0f}s)", flush=True)
    return nb


# ---- 2. score ----------------------------------------------------------------------------------------
def neighbour_score(nb, status_codes, n_min):
    """r = positives / countable neighbours (null below n_min countable). status_codes: 0 not countable,
    1 countable negative, 2 positive, per point."""
    valid = nb >= 0
    s = np.where(valid, status_codes[np.maximum(nb, 0)], 0)
    countable = (s > 0).sum(1)
    pos = (s == 2).sum(1)
    r = np.where(countable >= n_min, pos / np.maximum(countable, 1), np.nan)
    return r, countable


# ---- 3. gate -----------------------------------------------------------------------------------------
def weighted_auroc(y, s, w):
    """AUROC with weights, ties counted half (Mann-Whitney)."""
    order = np.argsort(s, kind="mergesort")
    y, s, w = y[order], s[order], w[order]
    _, start = np.unique(s, return_index=True)
    groups = np.repeat(np.arange(len(start)), np.diff(np.append(start, len(s))))
    w_pos = np.bincount(groups, w * (y == 1))
    w_neg = np.bincount(groups, w * (y == 0))
    neg_below = np.cumsum(w_neg) - w_neg
    total = w_pos.sum() * w_neg.sum()
    return float((w_pos * (neg_below + 0.5 * w_neg)).sum() / total) if total > 0 else np.nan


class Ranked:
    """The evaluated points of one split, sorted once; metrics for any weights (bootstrap) in linear time."""

    def __init__(self, y, s, seed):
        self.y = y
        self.order = np.argsort(s, kind="mergesort")                  # ascending, for the AUROC
        _, start = np.unique(s[self.order], return_index=True)
        self.groups = np.repeat(np.arange(len(start)), np.diff(np.append(start, len(s))))   # tie groups
        self.y_sorted = y[self.order]
        tie = np.random.default_rng(seed).random(len(s))
        self.top_order = np.lexsort((tie, -s))                        # descending, ties at random

    def auroc(self, w):
        wo = w[self.order]
        w_pos = np.bincount(self.groups, wo * (self.y_sorted == 1))
        w_neg = np.bincount(self.groups, wo * (self.y_sorted == 0))
        total = w_pos.sum() * w_neg.sum()
        return float((w_pos * (np.cumsum(w_neg) - w_neg + 0.5 * w_neg)).sum() / total) if total > 0 else np.nan

    def top(self, w, q):
        """Indices of the top q of the score, by weight."""
        cw = np.cumsum(w[self.top_order])
        return self.top_order[: int(np.searchsorted(cw, q * cw[-1], side="left")) + 1]

    def ppv(self, w, q):
        t = self.top(w, q)
        return float((w[t] * (self.y[t] == 1)).sum() / w[t].sum())


def top_share(y, s, w, q, rng_order):
    """Weighted share of positives among the top q (by weight) of the score; ties broken by rng_order."""
    order = np.lexsort((rng_order, -s))
    cw = np.cumsum(w[order])
    top = order[: int(np.searchsorted(cw, q * cw[-1], side="left")) + 1]
    return float((w[top] * (y[top] == 1)).sum() / w[top].sum()), top


def gate(y, r, w, prevalence, patient, top_q, n_boot, seed):
    """AUROC, PPV_q, lift and coverage, with patient-bootstrap 95 % intervals. Returns the metrics and the
    ranking (for the list)."""
    s = np.where(np.isnan(r), prevalence, r)
    rk = Ranked(y, s, seed)
    res = {"coverage": float((w * ~np.isnan(r)).sum() / w.sum()), "auroc": rk.auroc(w)}
    for q in top_q:
        res[f"ppv_{int(q * 100)}"] = rk.ppv(w, q)
    # bootstrap: resample patients; a patient drawn m times has its weights multiplied by m
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(patient, return_inverse=True)
    boots = {k: [] for k in res if k != "coverage"}
    for _ in range(n_boot):
        wb = w * np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        if (wb * (y == 1)).sum() == 0 or (wb * (y == 0)).sum() == 0:
            continue
        boots["auroc"].append(rk.auroc(wb))
        for q in top_q:
            boots[f"ppv_{int(q * 100)}"].append(rk.ppv(wb, q))
    for k, v in boots.items():
        if v:
            res[f"{k}_lo"], res[f"{k}_hi"] = (float(x) for x in np.nanpercentile(v, [2.5, 97.5]))
    for q in top_q:
        res[f"lift_{int(q * 100)}"] = res[f"ppv_{int(q * 100)}"] / prevalence if prevalence > 0 else np.nan
    return res, rk


def patient_weights(patient):
    """1 / (number of points of the patient in this set)."""
    _, inv, counts = np.unique(patient, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


# ---- 4. one map, every disease ---------------------------------------------------------------------------
def score_map(nb, pts, labels, map_info, rcfg, seed):
    """Scores and metrics of one map for every (disease, horizon) the map may be scored with."""
    status_code = {s: 1 for s in COUNTABLE} | {"positive": 2}
    split = pts["split"].to_numpy()
    patient = pts["subject_id"].to_numpy()
    rows, metrics = [], []
    for (disease, horizon, gap), lab in labels.group_by(["disease", "horizon", "gap"], maintain_order=True):
        if "horizon" in map_info and (horizon != map_info["horizon"] or gap != map_info["gap"]):
            continue                                       # y_hat predicts one window: its own labels only
        lab = lab.sort("point_id")
        status = lab["status"].to_numpy()
        codes = np.array([status_code.get(s, 0) for s in status], np.int8)
        r, countable = neighbour_score(nb, codes, rcfg["n_min"])
        rows.append(pl.DataFrame({"point_id": lab["point_id"], "disease": disease, "horizon": horizon,
                                  "r": r, "n_countable": countable.astype(np.int32)}))
        y = lab["y"].to_numpy(allow_copy=True).astype(float)          # 1 / 0 / nan
        known = ~np.isnan(y)
        # prevalence: train points with a known outcome, each patient counted once
        ref = known & (split == "train")
        w_ref = patient_weights(patient[ref])
        prevalence = float((w_ref * (y[ref] == 1)).sum() / w_ref.sum())
        for sp in ["val", "test", "train"]:
            ev = known & (split == sp)
            if ev.sum() == 0 or (y[ev] == 1).sum() == 0:
                continue
            w = patient_weights(patient[ev])
            n_boot = 0 if sp == "train" else rcfg["n_boot"]            # train = leakage check, no interval needed
            g, rk = gate(y[ev], r[ev], w, prevalence, patient[ev], rcfg["top_q"], n_boot, seed)
            row = {"disease": disease, "horizon": horizon, "gap": gap, "split": sp,
                   "points": int(ev.sum()), "patients": int(len(np.unique(patient[ev]))),
                   "prevalence": prevalence, **g}
            # list composition per q: candidates among the top q
            for q in rcfg["top_q"]:
                top = rk.top(w, q)
                st, wt, pt = status[ev][top], w[top], patient[ev][top]
                cand = np.isin(st, CANDIDATE)
                tag = int(q * 100)
                row[f"candidates_{tag}"] = int(len(np.unique(pt[cand])))
                for c in CANDIDATE:
                    row[f"share_{c}_{tag}"] = float(wt[st == c].sum() / wt[cand].sum()) if cand.any() else np.nan
            metrics.append(row)
    return pl.concat(rows), pl.DataFrame(metrics)


def collect(scores_dir):
    files = sorted(scores_dir.glob("*/*/metrics.csv"))
    if not files:
        return None
    allm = pl.concat([pl.read_csv(f).with_columns(embedding=pl.lit(f.parent.parent.name), map=pl.lit(f.parent.name))
                      for f in files], how="diagonal_relaxed")
    first = ["embedding", "map", "disease", "horizon", "split"]
    allm = allm.select(first + [c for c in allm.columns if c not in first]).sort(first)
    allm.write_csv(scores_dir / "metrics_all.csv")
    return allm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", nargs="*", default=[], help="folder names in <out>/resilience/embeddings")
    ap.add_argument("--map", nargs="*", default=None, help="only these maps (default: every map of the embedding)")
    ap.add_argument("--collect", action="store_true", help="only rebuild metrics_all.csv")
    ap.add_argument("--trajectory-config", default=str(HERE.parent / "bert" / "trajectory_columns.yaml"))
    ap.add_argument("--config", default=str(HERE.parent / "config" / "resilience_config.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    tcfg, rcfg = load_yaml(args.trajectory_config), load_yaml(args.config)
    res_dir = Path(tcfg["paths"]["out"]) / rcfg["out_dir"]
    scores_dir = res_dir / "scores"
    device = torch.device(args.device)

    if not args.collect:
        pts = pl.read_parquet(res_dir / "points.parquet").sort("point_id")
        labels = pl.read_parquet(res_dir / "labels.parquet")
        for emb in args.embedding:
            info = json.loads((res_dir / "embeddings" / emb / "info.json").read_text())
            for name, map_info in info["maps"].items():
                if args.map and name not in args.map:
                    continue
                t0 = time.time()
                print(f"{emb} / {name} ({map_info['metric']})", flush=True)
                x = np.load(res_dir / "embeddings" / emb / f"{name}.npy")
                nb = neighbours(x, pts, map_info["metric"], rcfg["k"], rcfg["search"], device)
                out = scores_dir / emb / name
                out.mkdir(parents=True, exist_ok=True)
                np.save(out / "neighbours.npy", nb)
                sc, met = score_map(nb, pts, labels, map_info, rcfg, args.seed)
                sc.write_parquet(out / "scores.parquet")
                met.write_csv(out / "metrics.csv")
                with pl.Config(tbl_rows=50, tbl_cols=12):
                    print(met.filter(pl.col("split") == "val").select(
                        "disease", "horizon", "prevalence", "coverage", "auroc", "ppv_5", "lift_5", "candidates_5"))
                print(f"  done in {time.time() - t0:.0f}s", flush=True)
    allm = collect(scores_dir)
    if allm is not None:
        print(f"written {scores_dir / 'metrics_all.csv'} ({allm.height} rows)")


if __name__ == "__main__":
    main()
