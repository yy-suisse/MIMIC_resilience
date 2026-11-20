"""Step 3: drop undated events, then build one point per episode.

Inputs (OUT): events.parquet, episodes.parquet
Outputs (OUT):
  events_clean.parquet   events with a day (days_since_birth not null)
  undated_summary.csv    number of dropped undated events per source table
  points.parquet         one row per point (= one episode)

Point rules (from the HERO plan):
  - one point per episode (admission, ED-only visit, outpatient day)
  - landmark T = episode end (end_days_since_birth, end_age_years)
  - history_count = number of episodes of the same patient that ended before this one started
  - is_new = history_count == 0 (first episode in the data)
  - weight = 1 / number of points of the patient (each patient counts once in total)

Usage (PowerShell):
  cd C:\\Users\\yy\\Desktop\\codes_PhD\\jepa_mimic\\trajectory
  python build_points.py
"""
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path(r"D:\MIMIC\trajectory_out")
M = 1_000_000  # day offset per patient in the sort keys; must exceed any day value


def drop_undated():
    ev = pl.scan_parquet(OUT / "events.parquet")
    summary = (ev.filter(pl.col("days_since_birth").is_null())
                 .group_by("source").agg(pl.len().alias("dropped"))
                 .sort("source").collect(engine="streaming"))
    summary.write_csv(OUT / "undated_summary.csv")
    print("undated events dropped, by source:")
    print(summary)
    (ev.filter(pl.col("days_since_birth").is_not_null())
       .sink_parquet(OUT / "events_clean.parquet", engine="streaming"))


def build_points():
    ep = pl.read_parquet(OUT / "episodes.parquet")
    ep = ep.filter(pl.col("start_days_since_birth").is_not_null()
                   & pl.col("end_days_since_birth").is_not_null())
    sid = ep["subject_id"].cast(pl.Int64).to_numpy()
    start = np.clip(ep["start_days_since_birth"].cast(pl.Int64).to_numpy(), 0, M - 1)
    end = np.clip(ep["end_days_since_birth"].cast(pl.Int64).to_numpy(), 0, M - 1)

    # Count, per episode, the earlier-ending episodes of the same patient.
    # Keys combine patient and day, so one sorted array serves all patients.
    end_keys = np.sort(sid * M + end)
    patient_start = np.searchsorted(end_keys, sid * M, side="left")
    ended_before_start = np.searchsorted(end_keys, sid * M + start, side="left")
    history_count = ended_before_start - patient_start

    points = ep.with_columns(pl.Series("history_count", history_count, dtype=pl.Int64))
    n_points = points.group_by("subject_id").agg(pl.len().alias("n_points"))
    points = (points.join(n_points, on="subject_id", how="left")
              .with_columns(
                  (1.0 / pl.col("n_points")).alias("weight"),
                  (pl.col("history_count") == 0).alias("is_new"))
              .rename({"episode_id": "point_id"})
              .select("point_id", "subject_id", "kind",
                      "start_age_years", "start_days_since_birth",
                      "end_age_years", "end_days_since_birth",
                      "history_count", "is_new", "weight"))
    points.write_parquet(OUT / "points.parquet")

    print(f"points: {points.height:,} for {points['subject_id'].n_unique():,} patients")
    print(points.group_by("kind").len())
    print(f"new points (no earlier episode): {points['is_new'].sum():,}")


if __name__ == "__main__":
    drop_undated()
    build_points()
