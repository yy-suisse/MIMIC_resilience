"""Step R5: how different are two maps? Neighbour overlap and candidate overlap (reads step R4's outputs).

For a pair of maps A and B (scores/<embedding>/<map>/ of score.py):
  neighbour overlap   per point of the split: |N_A(i) & N_B(i)| / K, the share of its K = 100 neighbours that both
                      maps give; mean with patient weights (1 / number of points of the patient)
                      ~0 = unrelated maps (random), 1 = the same neighbourhoods
  candidate Jaccard   per disease: |A & B| / |A | B| of the candidate PATIENTS of the two lists (top q of r, own status
                      tested abnormal, tested negative or untested), chosen exactly as score.py does
Equal scores with a high overlap: B did not change the map. Equal scores with a low overlap: another map, which
ranks other patients as candidates (what matters for the expert review).

Output: <out>/resilience/scores/compare_maps.csv

Usage (PowerShell):
  cd C:\\Users\\yy\\Desktop\\codes_PhD\\jepa_mimic\\resilience
  python compare_maps.py                                       # default pairs (below), val, H = 365, top 5 %
  python compare_maps.py --pairs BERT_D2/c:JEPA_D2/c JEPA_D2/c:bag/bag --q 0.10
  python compare_maps.py --pairs JEPA_D2/c:JEPA_D2_concat_hist/c JEPA_D2/c:JEPA_D2_z_transformer/c
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bert"))
from bert_data import load_yaml  # noqa: E402
from score import CANDIDATE, Ranked, patient_weights  # noqa: E402

DEFAULT_PAIRS = (
    [f"BERT_D{k}/c:JEPA_D{k}_nohist/c" for k in (1, 2, 3)]     # what JEPA training changed (same input)
    + [f"JEPA_D{k}_nohist/c:JEPA_D{k}/c" for k in (1, 2, 3)]   # what the history block changed
    + [f"BERT_D{k}/c:JEPA_D{k}/c" for k in (1, 2, 3)]          # both together
    + ["BERT_D1/c:BERT_D2/c", "BERT_D2/c:BERT_D3/c",           # what Q bins and values change
       "BERT_D2/c:bag/bag", "JEPA_D2/c:bag/bag",               # two good but unrelated maps
       "BERT_D2/c:random/random"]                              # the floor
    # variants A (concat_hist) and B (z_transformer) against base JEPA (pairs of runs not trained yet are skipped)
    + [f"JEPA_D{k}/c:JEPA_D{k}_{v}/c" for k in (1, 2, 3) for v in ("concat_hist", "z_transformer")]
    + [f"JEPA_D{k}/yhat_g0_h365:JEPA_D{k}_{v}/yhat_g0_h365" for k in (1, 2, 3) for v in ("concat_hist", "z_transformer")]
    + [f"JEPA_D{k}_concat_hist/c:JEPA_D{k}_z_transformer/c" for k in (1, 2, 3)]
    + [f"BERT_D{k}/c:JEPA_D{k}_{v}/c" for k in (1, 2, 3) for v in ("concat_hist", "z_transformer")]
    # variant hist_z against base JEPA and the Z transformer (c, and its query map against their c)
    + [f"JEPA_D{k}/c:JEPA_D{k}_hist_z/{m}" for k in (1, 2, 3) for m in ("c", "query_g0_h365")]
    + [f"JEPA_D{k}_z_transformer/c:JEPA_D{k}_hist_z/{m}" for k in (1, 2, 3) for m in ("c", "query_g0_h365")])


def neighbour_overlap(nb_a, nb_b):
    """Per row: number of neighbours (>= 0) found in both rows. One set intersection for all rows at once."""
    n, k = nb_a.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    big = np.int64(max(nb_a.max(), nb_b.max()) + 1)
    key_a = (rows * big + nb_a.ravel())[nb_a.ravel() >= 0]
    key_b = (rows * big + nb_b.ravel())[nb_b.ravel() >= 0]
    both = np.intersect1d(key_a, key_b, assume_unique=True)
    return np.bincount(both // big, minlength=n)


def candidates(scores, labels, pts, disease, horizon, split, q, seed):
    """Candidate patients of one map's list, as in score.py (top q of r, ties at random with the same seed)."""
    lab = labels.filter((pl.col("disease") == disease) & (pl.col("horizon") == horizon)).sort("point_id")
    sc = scores.filter((pl.col("disease") == disease) & (pl.col("horizon") == horizon)).sort("point_id")
    if sc.height == 0:
        return None
    y = lab["y"].to_numpy(allow_copy=True).astype(float)
    r = sc["r"].to_numpy()
    status = lab["status"].to_numpy()
    patient, sp = pts["subject_id"].to_numpy(), pts["split"].to_numpy()
    known = ~np.isnan(y)
    ref = known & (sp == "train")
    w_ref = patient_weights(patient[ref])
    prevalence = float((w_ref * (y[ref] == 1)).sum() / w_ref.sum())
    ev = known & (sp == split)
    w = patient_weights(patient[ev])
    s = np.where(np.isnan(r[ev]), prevalence, r[ev])
    top = Ranked(y[ev], s, seed).top(w, q)
    cand = np.isin(status[ev][top], CANDIDATE)
    return set(patient[ev][top][cand].tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS, help="embedding/map:embedding/map")
    ap.add_argument("--split", default="val")
    ap.add_argument("--horizon", type=int, default=365)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="same as score.py, so the lists are the same")
    ap.add_argument("--trajectory-config", default=str(HERE.parent / "bert" / "trajectory_columns.yaml"))
    ap.add_argument("--config", default=str(HERE.parent / "config" / "resilience_config.yaml"))
    args = ap.parse_args()
    tcfg, rcfg = load_yaml(args.trajectory_config), load_yaml(args.config)
    res_dir = Path(tcfg["paths"]["out"]) / rcfg["out_dir"]
    scores_dir = res_dir / "scores"

    pts = pl.read_parquet(res_dir / "points.parquet").sort("point_id")
    labels = pl.read_parquet(res_dir / "labels.parquet")
    in_split = (pts["split"] == args.split).to_numpy()
    w = patient_weights(pts["subject_id"].to_numpy()[in_split])
    diseases = labels["disease"].unique().sort().to_list()
    cache = {}

    def load(name):
        if name not in cache:
            folder = scores_dir / name
            if not (folder / "neighbours.npy").exists():
                return None
            cache[name] = (np.load(folder / "neighbours.npy")[in_split], pl.read_parquet(folder / "scores.parquet"))
        return cache[name]

    rows = []
    for pair in args.pairs:
        a, b = pair.split(":")
        A, B = load(a), load(b)
        if A is None or B is None:
            print(f"skipped {pair}: no neighbours.npy for {a if A is None else b}")
            continue
        k = A[0].shape[1]
        ov = neighbour_overlap(A[0], B[0]) / k
        row = {"map_a": a, "map_b": b, "split": args.split, "neighbour_overlap": float((w * ov).sum() / w.sum()),
               "overlap_p10": float(np.percentile(ov, 10)), "overlap_p90": float(np.percentile(ov, 90))}
        for d in diseases:
            ca = candidates(A[1], labels, pts, d, args.horizon, args.split, args.q, args.seed)
            cb = candidates(B[1], labels, pts, d, args.horizon, args.split, args.q, args.seed)
            row[f"jaccard_{d}"] = (len(ca & cb) / len(ca | cb)) if ca is not None and cb is not None and (ca | cb) else None
        rows.append(row)
        print(f"{a:>22} vs {b:<22} neighbours {row['neighbour_overlap']:.3f}  "
              + "  ".join(f"{d} {row[f'jaccard_{d}']:.2f}" for d in diseases if row[f"jaccard_{d}"] is not None))

    out = pl.DataFrame(rows)
    out.write_csv(scores_dir / "compare_maps.csv")
    print(f"written {scores_dir / 'compare_maps.csv'} (candidate lists: top {args.q:.0%}, H = {args.horizon})")


if __name__ == "__main__":
    main()
