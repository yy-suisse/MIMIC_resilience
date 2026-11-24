"""Step R1: the points of the resilience map, one per admission or ED visit (HERO plan, section 1).

A point = a patient's whole history up to a landmark T = the end of an episode:
  admission (episode_id H...)  hospital stay, starting at the ED arrival when the visit led to it
  ED visit (episode_id E...)   ED visit without admission
Outpatient days are not points. All episodes are points, a patient's first one included.

For each point we keep what the next steps need:
  split, p          the patient's split and position in the JEPA arrays of that split (same in D1, D2, D3)
  episode           the episode number in the arrays (the landmark = last token with this number)
  start_day, T_day  first and last day of the episode's tokens (age_days); T_day is the landmark day
  h                 history depth: number of admissions / ED visits ended before this one started
                    (h = 0: a "new point", the first episode in the hospital)
  gender, age_years

The patient order and the episode numbers are computed exactly as jepa/jepa_data.py does, from the same master
files: patients in file order, ascending subject_id within a file; episode = dense rank of the episode key.

Output: <out>/resilience/points.parquet

Usage (PowerShell):
  cd C:\\Users\\yy\\Desktop\\codes_PhD\\jepa_mimic\\resilience
  python build_points.py
"""
import argparse
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bert"))
from bert_data import SPLITS, load_yaml  # noqa: E402


def episode_numbers(rows):
    """Episode number per row, as jepa_data.add_episode (key = episode_id, else one per outpatient day)."""
    age = pl.col("age_days").fill_null(-1).cast(pl.Int32)
    key = pl.coalesce(pl.col("episode_id"), pl.concat_str(pl.lit("O"), age.cast(pl.String)))
    number = (key.rank("dense").over("subject_id") - 1).cast(pl.Int32)
    return rows.with_columns(episode=pl.when(pl.col("type") == "gender").then(-1).otherwise(number).cast(pl.Int32))


def points_of_file(f, kinds):
    """The points of one master file: one row per admission / ED visit of each patient."""
    rows = (pl.scan_parquet(f)
              .select("subject_id", "split", "pos", "type", "label", "age_days", "episode_id")
              .collect()
              .sort(["subject_id", "pos"]))                    # the order of the arrays
    rows = episode_numbers(rows)

    gender = (rows.filter(pl.col("type") == "gender")
                  .select("subject_id", pl.col("label").alias("gender")))
    pts = (rows.filter(pl.col("episode_id").is_not_null() & (pl.col("type") != "gender"))
               .with_columns(prefix=pl.col("episode_id").str.slice(0, 1))
               .filter(pl.col("prefix").is_in(list(kinds)))
               .group_by(["subject_id", "episode_id"], maintain_order=True)
               .agg(pl.col("split").first(),
                    pl.col("prefix").first(),
                    pl.col("episode").first(),
                    pl.col("age_days").min().alias("start_day"),
                    pl.col("age_days").sort_by("pos").last().alias("T_day")))   # day of the landmark token
    patients = (rows.select("subject_id", "split").unique(maintain_order=True))
    return history_depth(pts), gender, patients        # patients never cross files: h per file


def history_depth(pts):
    """h = number of the patient's other episodes that ended (T_day) before this one started (start_day)."""
    ends = pts.select("subject_id", pl.col("T_day").alias("_end")).sort("subject_id", "_end")
    counted = (pts.join(ends, on="subject_id", how="left")
                  .group_by(["subject_id", "episode_id"])
                  .agg((pl.col("_end") < pl.col("start_day")).sum().cast(pl.Int32).alias("h")))
    return pts.join(counted, on=["subject_id", "episode_id"], how="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory-config", default=str(HERE.parent / "bert" / "trajectory_columns.yaml"))
    ap.add_argument("--config", default=str(HERE.parent / "config" / "resilience_config.yaml"))
    args = ap.parse_args()
    tcfg, rcfg = load_yaml(args.trajectory_config), load_yaml(args.config)
    kinds = rcfg["point_kinds"]

    out = Path(tcfg["paths"]["out"])
    master_dir = out / tcfg["paths"]["master_dir"]
    files = sorted(master_dir.glob(f"master_n{tcfg['n_buckets']}_b*.parquet"))   # same files and order as the arrays
    if not files:
        raise FileNotFoundError(f"no master files in {master_dir}")
    out_dir = out / rcfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) points, genders and the patient order, file by file
    all_pts, all_gender, order = [], [], []
    for i, f in enumerate(files):
        pts, gender, patients = points_of_file(f, kinds)
        all_pts.append(pts)
        all_gender.append(gender)
        order.append(patients)
        print(f"points: master file {i + 1}/{len(files)}, {pts.height:,} points", flush=True)

    # 2) position of each patient in the arrays of its split (file order, then subject_id)
    order = pl.concat(order).with_columns(p=pl.int_range(pl.len()).over("split").cast(pl.Int64))

    # 3) one table, with history depth, gender, age and kind
    pts = (pl.concat(all_pts)
           .join(order.select("subject_id", "p"), on="subject_id", how="left")
           .join(pl.concat(all_gender), on="subject_id", how="left")
           .with_columns(kind=pl.col("prefix").replace_strict(kinds),
                         new_point=pl.col("h") == 0,
                         age_years=(pl.col("T_day") / 365.25).round(1))
           .sort(["subject_id", "T_day", "episode"])
           .with_row_index("point_id")
           .with_columns(pl.col("point_id").cast(pl.Int64),
                         n_points=pl.len().over("subject_id").cast(pl.Int32))
           .select("point_id", "subject_id", "split", "p", "episode", "episode_id", "kind",
                   "start_day", "T_day", "h", "new_point", "gender", "age_years", "n_points"))
    pts.write_parquet(out_dir / "points.parquet")

    # 4) summary
    print(pts.group_by("split").agg(pl.len().alias("points"), pl.col("subject_id").n_unique().alias("patients"),
                                    pl.col("new_point").mean().round(3).alias("new_share"))
             .sort("split"))
    print(pts.group_by("kind").len().sort("kind"))
    for s in SPLITS:
        assert pts.filter(pl.col("split") == s)["p"].is_not_null().all()
    print(f"written {out_dir / 'points.parquet'}: {pts.height:,} points of {pts['subject_id'].n_unique():,} patients")


if __name__ == "__main__":
    main()
