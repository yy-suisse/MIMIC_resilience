"""Step 6: assign each event its quantile bin, Q1 to Q10 or "unknown" (memory-safe version).

Input:  OUT/events.parquet
        OUT/quantiles/quantile_cutpoints.parquet  (from build_cutpoints.py)
Output: OUT/events_binned.parquet = every original event column, plus
          ontology_version  "ICD-9" / "ICD-10" or null
          label             readable name (lower case)
          event_row         unique row number of the event (used by the thinning step)
          quantile_bin      "Q1".."Q10" for a numeric value with cut points,
                            "unknown" for a numeric value without cut points,
                            null for an event with no numeric value

Bin rule (cut_1..cut_9 from build_cutpoints.py):
  Q1 if value <= cut_1; Qk if cut_(k-1) < value <= cut_k; Q10 if value > cut_9.

Memory plan: the events are processed in N_BUCKETS groups of patients, one group at a time,
each written to a small temporary file. The files are then joined into one file in a streaming pass.

Usage (PowerShell):
  cd C:\\Users\\yy\\Desktop\\codes_PhD\\jepa_mimic\\quantiles
  python assign_bins.py
"""
import shutil
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vocabulary"))
import build_vocabulary as bv  # noqa: E402

CUT_COLS = [f"cut_{k}" for k in range(1, 10)]
EVENTS = bv.OUT / "events.parquet"
CUTS = bv.OUT / "quantiles" / "quantile_cutpoints.parquet"
OUT_FILE = bv.OUT / "events_binned.parquet"
PARTS = bv.OUT / "quantiles" / "_binned_parts"
N_BUCKETS = 64  # raise it if one bucket still runs out of memory


def key_with_unit(df):
    """Join key: the vocabulary key plus the unit. Internal only, never written."""
    return df.with_columns(
        pl.concat_str([
            pl.col("source"), pl.lit("\x1f"),
            pl.col("ontology_version").fill_null("\x00"), pl.lit("\x1f"),
            pl.col("label"), pl.lit("\x1f"),
            pl.col("code").fill_null("\x00"), pl.lit("\x1f"),
            pl.col("unit").fill_null("\x00"),
        ]).alias("_ku")
    )


def bin_events(events_bucket, labels, cuts):
    """Add ontology, label, the numeric value and the bin to one bucket of events."""
    ev = (events_bucket
          .with_columns(pl.col("icd_version").fill_null(0).cast(pl.Int64).alias("_ver"))
          .join(labels, on=["source", "code", "_ver"], how="left"))

    ontology = (pl.when(pl.col("icd_version") == 9).then(pl.lit("ICD-9"))
                .when(pl.col("icd_version") == 10).then(pl.lit("ICD-10"))
                .otherwise(pl.lit(None, pl.String)))
    cleaned = (pl.coalesce(pl.col("raw_label"), pl.col("text"), pl.col("code"))
               .str.to_lowercase().str.strip_chars().str.replace_all(r"\s+", " "))
    label = (pl.when(pl.col("event_type") == "death").then(pl.lit("death"))
             .otherwise(cleaned).fill_null(""))

    ev = ev.with_columns(
        ontology.alias("ontology_version"),
        label.alias("label"),
        pl.col("value").cast(pl.Float64, strict=False).alias("x"),
    )
    ev = key_with_unit(ev).join(cuts, on="_ku", how="left")

    above = pl.sum_horizontal([(pl.col("x") > pl.col(c)).cast(pl.Int64) for c in CUT_COLS])
    bin_expr = (pl.when(pl.col("x").is_null()).then(pl.lit(None, pl.String))
                .when(pl.col("cut_1").is_null()).then(pl.lit("unknown"))
                .otherwise(pl.concat_str(pl.lit("Q"), (above + 1).cast(pl.String))))
    return ev.with_columns(bin_expr.alias("quantile_bin"))


def main():
    PARTS.mkdir(parents=True, exist_ok=True)

    labels = bv.dictionary_labels().collect()
    cuts = key_with_unit(pl.scan_parquet(CUTS)).select("_ku", *CUT_COLS).collect()

    events = pl.scan_parquet(EVENTS)
    original_cols = events.collect_schema().names()
    events = events.with_row_index("event_row")
    final_cols = [*original_cols, "ontology_version", "label", "event_row", "quantile_bin"]

    for b in range(N_BUCKETS):
        part = events.filter((pl.col("subject_id").hash(seed=3) % N_BUCKETS) == b)
        out = (bin_events(part, labels.lazy(), cuts.lazy())
               .select(final_cols)
               .collect(engine="streaming"))
        out.write_parquet(PARTS / f"b{b:02d}.parquet")
        print(f"bins: bucket {b + 1}/{N_BUCKETS} done ({out.height:,} events)", flush=True)

    # join the bucket files into one file (no join, streamed)
    (pl.scan_parquet(PARTS / "*.parquet")
       .sink_parquet(OUT_FILE, engine="streaming"))
    shutil.rmtree(PARTS, ignore_errors=True)

    counts = (pl.scan_parquet(OUT_FILE).group_by("quantile_bin").len()
              .sort("quantile_bin").collect(engine="streaming"))
    print(counts)
    print(f"written: {OUT_FILE}")


if __name__ == "__main__":
    main()
