"""Step 5: quantile cut points for every numeric (token, unit) pair, from the whole cohort.

Uses the same tokens as the vocabulary (imported from vocabulary/build_vocabulary.py), so the
keys match vocab.parquet exactly.

Rules:
  - all numeric events (labs, vitals, chart values, ICU inputs and outputs), not only labs
  - one set of cut points per (source, ontology_version, label, code, unit)
  - cut points come from all patients in the cohort (no split)
  - a pair gets cut points only when it has at least MIN_VALUES_UNIT values; otherwise it gets none

Output, OUT/quantiles/quantile_cutpoints.parquet and .csv:
  source, ontology_version, label, code, unit, n, cut_1 ... cut_9
  cut_k is the value below which k/10 of the values lie.
  Bin Q1 holds values <= cut_1, Q2 holds values above cut_1 up to cut_2, ..., Q10 holds values above cut_9.

Usage (PowerShell):
  cd C:\\Users\\yy\\Desktop\\codes_PhD\\jepa_mimic\\quantiles
  python build_cutpoints.py
"""
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vocabulary"))
import build_vocabulary as bv  # noqa: E402

OUT_Q = bv.OUT / "quantiles"
TMP = OUT_Q / "_numeric_tmp.parquet"
MIN_VALUES_UNIT = 100
CUTS = [k / 10 for k in range(1, 10)]
CUT_COLS = [f"cut_{k}" for k in range(1, 10)]


def write_numeric():
    """One streaming pass: numeric values as Float64 with their keys, to a temporary file."""
    tokens = bv.add_key(bv.tokenize(pl.scan_parquet(bv.OUT / "events.parquet"), bv.dictionary_labels()))
    (tokens.filter(pl.col("value").is_not_null())
     .select("_key", *bv.KEY_COLS, "unit",
             pl.col("value").cast(pl.Float64, strict=False).alias("x"))
     .drop_nulls("x")
     .sink_parquet(TMP, engine="streaming"))
    print("numeric values written to temporary file", flush=True)


def cut_points():
    """Cut points per (token, unit), bucket by bucket so memory stays bounded."""
    parts = []
    for b in range(bv.N_BUCKETS):
        part = (pl.scan_parquet(TMP)
                .filter((pl.col("_key").hash(seed=7) % bv.N_BUCKETS) == b)
                .collect(engine="streaming"))
        agg = part.group_by(bv.KEY_COLS + ["unit"]).agg(
            pl.len().alias("n"),
            *[pl.col("x").quantile(q, interpolation="linear").alias(c) for q, c in zip(CUTS, CUT_COLS)],
        )
        parts.append(agg.filter(pl.col("n") >= MIN_VALUES_UNIT))
        print(f"cut points: bucket {b + 1}/{bv.N_BUCKETS} done", flush=True)
    return pl.concat(parts).sort("n", descending=True)


def main():
    OUT_Q.mkdir(parents=True, exist_ok=True)
    write_numeric()
    cuts = cut_points()
    cuts.write_parquet(OUT_Q / "quantile_cutpoints.parquet")
    cuts.write_csv(OUT_Q / "quantile_cutpoints.csv")
    TMP.unlink(missing_ok=True)
    print(f"cut points written for {cuts.height:,} (token, unit) pairs "
          f"with at least {MIN_VALUES_UNIT} values")


if __name__ == "__main__":
    main()
