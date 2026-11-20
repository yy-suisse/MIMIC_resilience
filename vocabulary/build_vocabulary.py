"""Step 4: token vocabulary and value statistics, all patients, no cut-off.

Reads OUT/events.parquet, the patients table and the dictionary tables through config/mimic_columns.csv.

Each token is four separate columns, not one joined string:
  source            the MIMIC table the event comes from (special, patients, labevents, diagnoses_icd, ...)
  ontology_version  "ICD-9" or "ICD-10" for diagnoses and procedures; null when the code has no version
  label             the readable description, lower case (dictionary long title or item label, drug name, ...)
  code              the real reference code, exactly as in the source (itemid, ICD code, NDC, GSN, F/M, ...)

Code and label rules:
  - labevents, chartevents, inputevents, outputevents, procedureevents: label from d_labitems / d_items
  - diagnoses and procedures: label from d_icd_diagnoses / d_icd_procedures, joined on code AND version
  - prescriptions, pyxis, medrecon: label is the drug name; code is ndc (or gsn for pyxis)
  - vitals, OMR, triage, microbiology: no dictionary exists; the code is the measure name, label the same
  - gender: code F or M, label female or male

Token order (token_id):
  0-5     reserved special tokens: [PAD], [UNK], [MASK], [SEP], [CLS], [VALUE]
          ([VALUE] is a placeholder after an event that has a number; the number itself is kept in a separate value channel)
  6-16    quantile bin tokens: Q1 to Q10 (value bins), then "unknown" (no bin: missing value or too few values)
  then    gender tokens (patient-level, one per patient, part of the trajectory)
  then    unit tokens: every unit seen on a numeric event, as written in the data (source = "unit",
          code = label = the unit, spaces stripped, case kept; no conversion between units)
  then    every event token seen in the MIMIC events, most frequent first

Every token is kept. The cut-off by number of patients is decided later, using the n_patients column.
Value statistics and quantile cut points are also computed per (concept, unit).

Writes OUT/vocabulary/:
  vocab.parquet         token_id, source, ontology_version, label, code, n_patients
  vocab.csv             same as vocab.parquet, readable in Excel
  value_stats.parquet   source, ontology_version, label, code, unit, level ("unit" or "token"), n, median, iqr

Memory plan for the value statistics:
  1. one streaming pass writes the numeric values to a temporary file on disk;
  2. the temporary file is split into N_BUCKETS groups of tokens, one group at a time.

Value statistics are computed on all patients, before ICU thinning. They must be recomputed after thinning.

Usage (PowerShell):
  cd C:\\Users\\yy\\Desktop\\codes_PhD\\jepa_mimic\\vocabulary
  python build_vocabulary.py
"""
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "config"))
from check_config import load_config, read_rows  # noqa: E402

ROOTS = {
    "MIMIC-IV 3.1": Path(r"D:\MIMIC\mimic-iv-3.1"),
    "MIMIC-IV-ED 2.2": Path(r"D:\MIMIC\mimic-iv-ed-2.2"),
}
OUT = Path(r"D:\MIMIC\trajectory_out")
VOCAB = OUT / "vocabulary"
TMP_VALUES = VOCAB / "_values_tmp.parquet"

KEY_COLS = ["source", "ontology_version", "label", "code"]
SPECIALS = ["[PAD]", "[UNK]", "[MASK]", "[SEP]", "[CLS]", "[VALUE]"]
QUANTILE_BINS = [f"Q{k}" for k in range(1, 11)] + ["unknown"]  # value bins, then the no-bin token
N_BUCKETS = 64  # raise it if a single bucket still runs out of memory

# source table -> dictionary table that gives the label of its item codes
ITEM_DICT = {
    "labevents": "d_labitems",
    "chartevents": "d_items",
    "inputevents": "d_items",
    "outputevents": "d_items",
    "procedureevents": "d_items",
}
# sources whose codes are ICD codes, and the dictionary that names them
ICD_DX_SOURCES = ["diagnosis", "diagnoses_icd"]
ICD_PROC_SOURCES = ["procedures_icd"]
# sources whose label is a readable name carried in the event (text column)
TEXT_SOURCES = ["prescriptions", "pyxis", "medrecon"]

CFG = load_config()
PATHS = {r["table"]: ROOTS[r["dataset"]] / r["file"] for r in read_rows()}


def read_table(table):
    """Lazily read one table: used columns only, typed from the config."""
    cfg = CFG[table]
    used = cfg["used"]
    schema = {c: getattr(pl, cfg["columns"][c]) for c in used}
    return pl.scan_csv(PATHS[table], schema_overrides=schema).select(used)


def add_key(df):
    """Internal join key only. It is never written and never used as a label."""
    return df.with_columns(
        pl.concat_str([
            pl.col("source"), pl.lit("\x1f"),
            pl.col("ontology_version").fill_null("\x00"), pl.lit("\x1f"),
            pl.col("label"), pl.lit("\x1f"),
            pl.col("code").fill_null("\x00"),
        ]).alias("_key")
    )


# ---- labels ---------------------------------------------------------------------
def dictionary_labels():
    """(source, code, version) -> dictionary description. Version is 0 when the code has none."""
    parts = [
        read_table(dict_table).select(
            pl.lit(src).alias("source"),
            pl.col("itemid").cast(pl.String).alias("code"),
            pl.lit(0).cast(pl.Int64).alias("_ver"),
            pl.col("label").alias("raw_label"),
        )
        for src, dict_table in ITEM_DICT.items()
    ]
    for sources, dict_table in ((ICD_DX_SOURCES, "d_icd_diagnoses"), (ICD_PROC_SOURCES, "d_icd_procedures")):
        d = read_table(dict_table).select(
            pl.col("icd_code").cast(pl.String).alias("code"),
            pl.col("icd_version").cast(pl.Int64).alias("_ver"),
            pl.col("long_title").alias("raw_label"),
        )
        parts += [d.with_columns(pl.lit(src).alias("source")) for src in sources]
    return pl.concat(parts, how="diagonal_relaxed").unique(["source", "code", "_ver"])


def tokenize(events, labels):
    """Return subject_id, source, ontology_version, label, code, unit, value.

    The label is the dictionary description, else the drug name, else the code itself.
    """
    ev = (events
          .with_columns(pl.col("icd_version").fill_null(0).cast(pl.Int64).alias("_ver"))
          .join(labels, on=["source", "code", "_ver"], how="left"))
    ontology = (pl.when(pl.col("icd_version") == 9).then(pl.lit("ICD-9"))
                .when(pl.col("icd_version") == 10).then(pl.lit("ICD-10"))
                .otherwise(pl.lit(None, pl.String)))
    cleaned = (pl.coalesce(pl.col("raw_label"), pl.col("text"), pl.col("code"))
               .str.to_lowercase().str.strip_chars().str.replace_all(r"\s+", " "))
    label = (pl.when(pl.col("event_type") == "death").then(pl.lit("death"))
             .otherwise(cleaned)
             .fill_null(""))
    return ev.select("subject_id", "source", ontology.alias("ontology_version"),
                     label.alias("label"), "code", "unit", "value")


def gender_label(col):
    """Readable gender: F -> female, M -> male, anything else kept in lower case."""
    return (pl.when(col == "F").then(pl.lit("female"))
            .when(col == "M").then(pl.lit("male"))
            .otherwise(col.str.to_lowercase()))


# ---- value statistics -----------------------------------------------------------
STAT_EXPRS = [
    pl.len().alias("n"),
    pl.col("x").median().alias("median"),
    (pl.col("x").quantile(0.75) - pl.col("x").quantile(0.25)).alias("iqr"),
]


def write_numeric_values(tokens):
    """One streaming pass: numeric values with their keys, to a temporary file on disk."""
    (tokens.filter(pl.col("value").is_not_null())
     .select("_key", *KEY_COLS, "unit",
             pl.col("value").cast(pl.Float32, strict=False).alias("x"))
     .drop_nulls("x")
     .sink_parquet(TMP_VALUES, engine="streaming"))
    print("  numeric values written to temporary file", flush=True)


def value_stats():
    """Median and IQR per (token, unit) and per token, bucket by bucket, from the temporary file."""
    parts = []
    for b in range(N_BUCKETS):
        part = (pl.scan_parquet(TMP_VALUES)
                .filter((pl.col("_key").hash(seed=7) % N_BUCKETS) == b)
                .collect(engine="streaming"))
        unit = (part.group_by(KEY_COLS + ["unit"]).agg(STAT_EXPRS)
                .with_columns(pl.lit("unit").alias("level")))
        token = (part.group_by(KEY_COLS).agg(STAT_EXPRS)
                 .with_columns(pl.lit(None, pl.String).alias("unit"), pl.lit("token").alias("level")))
        parts.append(pl.concat([unit, token], how="diagonal_relaxed"))
        print(f"  value statistics: bucket {b + 1}/{N_BUCKETS} done", flush=True)
    return pl.concat(parts, how="diagonal_relaxed")


# ---- main -----------------------------------------------------------------------
VOCAB_SCHEMA = {"source": pl.String, "ontology_version": pl.String, "label": pl.String,
                "code": pl.String, "n_patients": pl.Int64}


def main():
    VOCAB.mkdir(parents=True, exist_ok=True)

    # 0) patient-level gender tokens
    patients = read_table("patients").select("subject_id", "gender").collect()
    gender_tokens = (patients.drop_nulls("gender")
                     .group_by("gender").agg(pl.col("subject_id").n_unique().alias("n_patients"))
                     .sort("n_patients", descending=True)
                     .select(
                         pl.lit("patients").alias("source"),
                         pl.lit(None, pl.String).alias("ontology_version"),
                         gender_label(pl.col("gender")).alias("label"),
                         pl.col("gender").alias("code"),
                         pl.col("n_patients").cast(pl.Int64),
                     ))

    # 1) event tokens seen in the events, with the number of patients who have each
    events = pl.scan_parquet(OUT / "events.parquet")
    tokens = add_key(tokenize(events, dictionary_labels()))
    event_tokens = (tokens.select("subject_id", *KEY_COLS)
                    .unique()
                    .group_by(KEY_COLS).len().rename({"len": "n_patients"})
                    .sort("n_patients", descending=True)
                    .collect(engine="streaming")
                    .with_columns(pl.col("n_patients").cast(pl.Int64)))

    # 1b) unit tokens: units of numeric events, as written in the data
    unit_tokens = (tokens
                   .filter(pl.col("value").cast(pl.Float64, strict=False).is_not_null())
                   .select("subject_id", pl.col("unit").str.strip_chars().alias("unit"))
                   .filter(pl.col("unit").is_not_null() & (pl.col("unit") != ""))
                   .unique()
                   .group_by("unit").len().rename({"len": "n_patients"})
                   .sort("n_patients", descending=True)
                   .collect(engine="streaming")
                   .select(
                       pl.lit("unit").alias("source"),
                       pl.lit(None, pl.String).alias("ontology_version"),
                       pl.col("unit").alias("label"),
                       pl.col("unit").alias("code"),
                       pl.col("n_patients").cast(pl.Int64),
                   ))

    # 2) assemble: specials, quantile bins, gender, units, events
    special_tokens = pl.DataFrame({
        "source": ["special"] * len(SPECIALS),
        "ontology_version": [None] * len(SPECIALS),
        "label": SPECIALS,
        "code": [None] * len(SPECIALS),
        "n_patients": [None] * len(SPECIALS),
    }, schema=VOCAB_SCHEMA)
    gender_tokens = gender_tokens.cast(VOCAB_SCHEMA).select(list(VOCAB_SCHEMA))
    event_tokens = event_tokens.cast(VOCAB_SCHEMA).select(list(VOCAB_SCHEMA))
    bin_tokens = pl.DataFrame({
        "source": ["quantile"] * len(QUANTILE_BINS),
        "ontology_version": [None] * len(QUANTILE_BINS),
        "label": QUANTILE_BINS,
        "code": [None] * len(QUANTILE_BINS),
        "n_patients": [None] * len(QUANTILE_BINS),
    }, schema=VOCAB_SCHEMA)

    unit_tokens = unit_tokens.cast(VOCAB_SCHEMA).select(list(VOCAB_SCHEMA))
    vocab = (pl.concat([special_tokens, bin_tokens, gender_tokens, unit_tokens, event_tokens], how="vertical")
             .with_row_index("token_id"))
    vocab.write_parquet(VOCAB / "vocab.parquet")
    vocab.write_csv(VOCAB / "vocab.csv")
    print(f"vocabulary size: {vocab.height:,} tokens "
          f"({len(SPECIALS)} special, {len(QUANTILE_BINS)} quantile, "
          f"{gender_tokens.height} gender, {unit_tokens.height} unit, {event_tokens.height:,} event)")

    # 3) value statistics: all patients, from disk, in buckets
    write_numeric_values(tokens)
    stats = value_stats()
    stats.write_parquet(VOCAB / "value_stats.parquet")
    TMP_VALUES.unlink(missing_ok=True)
    print(f"value statistics: {stats.height:,} rows written")


if __name__ == "__main__":
    main()
