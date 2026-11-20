"""Step 7: thin the numeric events to their extremes per 4-hour block (memory-safe version).

Input:  OUT/events_binned.parquet  (from quantiles/assign_bins.py)
Output: OUT/events_thinned_4h.parquet  (same columns as events_binned.parquet)

Rule, for each patient, each 4-hour block, and each measurement (source, ontology_version,
label, code, unit):
  - keep the event with the lowest numeric value
  - keep the event with the highest numeric value
  (one event if the lowest and highest are the same event)

Events that cannot be thinned are kept unchanged:
  - no numeric value
  - no clock time (time_of_day is null, i.e. date-only events)

The 4-hour block is counted from the patient's age origin:
  hours = days_since_birth * 24 + clock time in hours
  block = floor(hours / 4)

Memory plan:
  1. one streaming pass reads only the needed columns and writes the candidate events to disk;
  2. the candidates are split into N_BUCKETS groups by patient, one group at a time, and the
     lowest and highest event of each block are chosen in RAM;
  3. one streaming pass writes the final file: all events that are not candidates, plus the chosen extremes.

Usage (PowerShell):
  cd C:\\Users\\yy\\Desktop\\codes_PhD\\jepa_mimic\\thinning
  python thin_events_4h.py
"""
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vocabulary"))
import build_vocabulary as bv  # noqa: E402

INP = bv.OUT / "events_binned.parquet"
OUT_FILE = bv.OUT / "events_thinned_4h.parquet"
TMP_CAND = bv.OUT / "thinning" / "_candidates_tmp.parquet"
TMP_PICKS = bv.OUT / "thinning" / "_picks_tmp.parquet"
BLOCK_HOURS = 4
N_BUCKETS = 64  # raise it if one bucket still runs out of memory
MEASURE = ["source", "ontology_version", "label", "code", "unit"]
NEEDED = ["subject_id", "event_row", "value", "days_since_birth", "time_of_day", *MEASURE]


def clock_hours():
    t = pl.col("time_of_day")
    return (t.str.slice(0, 2).cast(pl.Float64)
            + t.str.slice(3, 2).cast(pl.Float64) / 60
            + t.str.slice(6, 2).cast(pl.Float64) / 3600)


def candidate_expr(lf):
    """Rows that can be thinned, with the numeric value and the block number."""
    x = pl.col("value").cast(pl.Float64, strict=False)
    hours = pl.col("days_since_birth").cast(pl.Float64) * 24 + clock_hours()
    cand = (x.is_not_null() & hours.is_not_null()).fill_null(False)
    return lf.filter(cand).with_columns(
        x.alias("_x"),
        (hours / BLOCK_HOURS).floor().cast(pl.Int64).alias("_block"),
    )


def main():
    TMP_CAND.parent.mkdir(parents=True, exist_ok=True)

    # 1) candidates only, narrow columns, to disk
    narrow = pl.scan_parquet(INP).select(NEEDED)
    candidate_expr(narrow).select(
        "subject_id", "event_row", "_x", "_block", *MEASURE
    ).sink_parquet(TMP_CAND, engine="streaming")
    print("candidates written", flush=True)

    # 2) choose lowest and highest per (patient, block, measurement), one patient bucket at a time
    picks_parts = []
    for b in range(N_BUCKETS):
        part = (pl.scan_parquet(TMP_CAND)
                .filter((pl.col("subject_id").hash(seed=3) % N_BUCKETS) == b)
                .collect(engine="streaming"))
        if part.height == 0:
            continue
        picks = (part.group_by(["subject_id", "_block", *MEASURE])
                 .agg(
                     pl.col("event_row").sort_by("_x").first().alias("lo"),
                     pl.col("event_row").sort_by("_x").last().alias("hi"),
                 )
                 .select(pl.concat_list(["lo", "hi"]).alias("event_row"))
                 .explode("event_row")
                 .unique())
        picks_parts.append(picks)
        print(f"picks: bucket {b + 1}/{N_BUCKETS} done", flush=True)
    picks_all = pl.concat(picks_parts).unique().with_columns(pl.lit(True).alias("_picked"))
    picks_all.write_parquet(TMP_PICKS)

    # 3) final file: non-candidates, plus the chosen extremes
    ev = pl.scan_parquet(INP)
    x = pl.col("value").cast(pl.Float64, strict=False)
    hours = pl.col("days_since_birth").cast(pl.Float64) * 24 + clock_hours()
    cand = (x.is_not_null() & hours.is_not_null()).fill_null(False)
    (ev.join(pl.scan_parquet(TMP_PICKS), on="event_row", how="left")
       .filter(~cand | pl.col("_picked").fill_null(False))
       .drop("_picked")
       .sink_parquet(OUT_FILE, engine="streaming"))

    TMP_CAND.unlink(missing_ok=True)
    TMP_PICKS.unlink(missing_ok=True)
    before = pl.scan_parquet(INP).select(pl.len()).collect().item()
    after = pl.scan_parquet(OUT_FILE).select(pl.len()).collect().item()
    print(f"events before thinning: {before:,}")
    print(f"events after thinning:  {after:,}")
    print(f"written: {OUT_FILE}")


if __name__ == "__main__":
    main()
