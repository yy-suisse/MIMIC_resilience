"""Step 2: build the event table (one row per fact) for MIMIC-IV.

Reads only the columns with use=Y in config/mimic_columns.csv (via check_config).

Outputs in OUT:
  events.parquet    subject_id, time, event_type, code, value, unit, source,
                    setting, hadm_id, ed_stay_id, icu_stay_id, episode_id
  episodes.parquet  episode_id, subject_id, kind, start, end

Run (PowerShell):
  cd C:\\Users\\yy\\Desktop\\codes_PhD\\jepa_mimic\\trajectory
  python build_trajectories.py --limit 2000     # quick test on 2000 patients
  python build_trajectories.py                  # all patients
"""
import argparse
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "config"))
from check_config import load_config, read_rows  # noqa: E402

# ---- paths -----------------------------------------------------------------
ROOTS = {
    "MIMIC-IV 3.1": Path(r"D:\MIMIC\mimic-iv-3.1"),
    "MIMIC-IV-ED 2.2": Path(r"D:\MIMIC\mimic-iv-ed-2.2"),
}
OUT = Path(r"D:\MIMIC\trajectory_out")

CFG = load_config()
PATHS = {r["table"]: ROOTS[r["dataset"]] / r["file"] for r in read_rows()}

# ---- table groups and the two different stay ids ----------------------------
ED_TABLES = {"edstays", "triage", "vitalsign", "pyxis", "medrecon", "diagnosis"}
ICU_TABLES = {"icustays", "chartevents", "inputevents", "outputevents", "procedureevents"}


def stay_name(table):
    """ED and ICU both use 'stay_id' in MIMIC; rename them so they never mix."""
    if table in ED_TABLES:
        return "ed_stay_id"
    if table in ICU_TABLES:
        return "icu_stay_id"
    return None


# ---- event rules --------------------------------------------------------------
# table: (time source (table, column), event_type, code column, value columns, unit column)
# time source "self" = the table's own column; otherwise the time is taken from
# the linked table (ED tables -> edstays, diagnoses_icd -> admissions).
RULES = {
    "patients":           (("self", "dod"),            "death",        None,          [],                                     None),
    "omr":                (("self", "chartdate"),      "omr",          "result_name", ["result_value"],                       None),
    "triage":             (("edstays", "intime"),      "vital",        None,          ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "acuity"], None),
    "vitalsign":          (("self", "charttime"),      "vital",        None,          ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"], None),
    "pyxis":              (("self", "charttime"),      "drug",         "name",        [],                                     None),
    "medrecon":           (("self", "charttime"),      "home_med",     "name",        [],                                     None),
    "diagnosis":          (("edstays", "outtime"),     "diagnosis",    "icd_code",    [],                                     None),
    "diagnoses_icd":      (("admissions", "dischtime"), "diagnosis",   "icd_code",    [],                                     None),
    "procedures_icd":     (("self", "chartdate"),      "procedure",    "icd_code",    [],                                     None),
    "prescriptions":      (("self", "starttime"),      "drug",         "drug",        [],                                     None),
    "labevents":          (("self", "charttime"),      "lab",          "itemid",      ["valuenum"],                           "valueuom"),
    "microbiologyevents": (("self", "charttime"),      "micro",        "test_name",   [],                                     None),
    "chartevents":        (("self", "charttime"),      "icu_chart",    "itemid",      ["valuenum"],                           "valueuom"),
    "inputevents":        (("self", "starttime"),      "icu_input",    "itemid",      ["amount"],                             "amountuom"),
    "outputevents":       (("self", "charttime"),      "icu_output",   "itemid",      ["value"],                              "valueuom"),
    "procedureevents":    (("self", "starttime"),      "icu_procedure", "itemid",     ["value"],                              "valueuom"),
}

# Which table gives the setting of each source (refined to ward/outpatient later).
DATE_ONLY_TABLES = {"omr", "procedures_icd", "patients"}  # no clock time in these sources

SETTING = {t:("ED" if t in ED_TABLES else "ICU" if t in ICU_TABLES else "hosp") for t in RULES}


def read(table, ids=None):
    """Lazily read one table: only used columns, typed from the config, patients filtered."""
    cfg = CFG[table]
    used = cfg["used"]
    schema = {c: getattr(pl, cfg["columns"][c]) for c in used}
    lf = pl.scan_csv(PATHS[table], schema_overrides=schema).select(used)
    s = stay_name(table)
    if s and "stay_id" in used:
        lf = lf.rename({"stay_id": s})
    if ids is not None:
        lf = lf.join(ids, on="subject_id", how="semi")
    return lf


def build_events(ids):
    parts = []
    for table, (time_src, etype, code_col, value_cols, unit_col) in RULES.items():
        if table not in CFG:
            continue
        lf = read(table, ids)
        src_table, src_col = time_src

        # 1) time column (own table, or taken from the linked table)
        if table == "microbiologyevents":  # charttime, falling back to chartdate (day only)
            lf = lf.with_columns(
                pl.coalesce(pl.col("charttime"), pl.col("chartdate").cast(pl.Datetime("us"))).alias("_time"),
                pl.col("charttime").is_null().alias("_date_only"))
        elif src_table == "self":
            lf = lf.with_columns(pl.col(src_col).alias("_time"),
                                 pl.lit(table in DATE_ONLY_TABLES).alias("_date_only"))
        else:
            key = "ed_stay_id" if src_table == "edstays" else "hadm_id"
            ref = read(src_table).select(key, src_col).unique(key).rename({src_col: "_time"})
            lf = lf.join(ref, on=key, how="left").with_columns(pl.lit(False).alias("_date_only"))

        # 2) keep only what the event needs
        ids_present = [c for c in ["subject_id", "hadm_id", "ed_stay_id", "icu_stay_id"]
                       if c in lf.collect_schema().names()]
        keep = ids_present + ["_time", "_date_only"] + value_cols
        if code_col:
            keep.append(code_col)
        if unit_col:
            keep.append(unit_col)
        lf = lf.select(list(dict.fromkeys(keep)))

        # 3) one event per row, or one per vital sign (unpivot)
        if len(value_cols) > 1:
            lf = lf.unpivot(on=value_cols, index=ids_present + ["_time", "_date_only"],
                            variable_name="code", value_name="value")
            code = pl.col("code")
            value = pl.col("value").cast(pl.String)
        else:
            code = pl.col(code_col).cast(pl.String) if code_col else pl.lit(None, pl.String)
            value = pl.col(value_cols[0]).cast(pl.String) if value_cols else pl.lit(None, pl.String)
        unit = pl.col(unit_col).cast(pl.String) if unit_col else pl.lit(None, pl.String)

        names = lf.collect_schema().names()

        def int_col(c):
            return pl.col(c).cast(pl.Int64).alias(c) if c in names else pl.lit(None, pl.Int64).alias(c)

        parts.append(lf.select(
            pl.col("subject_id").cast(pl.Int64),
            pl.col("_time").cast(pl.Datetime("us")).alias("time"),
            pl.col("_date_only").alias("date_only"),
            pl.lit(etype).alias("event_type"),
            code.alias("code"),
            value.alias("value"),
            unit.alias("unit"),
            pl.lit(table).alias("source"),
            pl.lit(SETTING[table]).alias("setting"),
            int_col("hadm_id"), int_col("ed_stay_id"), int_col("icu_stay_id"),
        ))
    return pl.concat(parts, how="diagonal_relaxed")


def add_age(df, col, prefix):
    """Add age in years and days since the (approximate) birth year for a datetime column."""
    return df.with_columns(
        (pl.col("anchor_age") + (pl.col(col) - pl.datetime(pl.col("anchor_year"), 1, 1))
            .dt.total_seconds() / (365.25 * 86400)).round(2).alias(f"{prefix}age_years"),
        (pl.col(col) - pl.datetime(pl.col("anchor_year") - pl.col("anchor_age"), 1, 1))
            .dt.total_days().alias(f"{prefix}days_since_birth"),
    )


def build_episodes(ids):
    # An admission starts at the ED arrival when the visit led to it (earliest of the two).
    ed_in = (read("edstays", ids).filter(pl.col("hadm_id").is_not_null())
             .group_by("hadm_id").agg(pl.col("intime").min().alias("ed_in")))
    adm = read("admissions", ids).join(ed_in, on="hadm_id", how="left").select(
        "subject_id", "hadm_id",
        pl.min_horizontal("ed_in", "admittime").alias("start"),
        pl.col("dischtime").alias("end"))
    adm = adm.with_columns(pl.lit("admission").alias("kind"),
                           pl.concat_str(pl.lit("H"), pl.col("hadm_id").cast(pl.String)).alias("episode_id"))

    ed = read("edstays", ids).filter(pl.col("hadm_id").is_null()).select(
        "subject_id", "ed_stay_id", pl.col("intime").alias("start"), pl.col("outtime").alias("end"))
    ed = ed.with_columns(pl.lit("ed_only").alias("kind"),
                         pl.concat_str(pl.lit("E"), pl.col("ed_stay_id").cast(pl.String)).alias("episode_id"))
    return pl.concat([adm.select("episode_id", "subject_id", "kind", "start", "end"),
                      ed.select("episode_id", "subject_id", "kind", "start", "end")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="number of patients (test runs)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ids = None
    if args.limit:
        ids = read("patients").select("subject_id").unique().sort("subject_id").head(args.limit)

    # ED and ICU stay -> admission, to fill hadm_id on rows that lack it
    map_ed = read("edstays").select("ed_stay_id", pl.col("hadm_id").alias("hadm_from_ed")).unique("ed_stay_id")
    map_icu = read("icustays").select("icu_stay_id", pl.col("hadm_id").alias("hadm_from_icu")).unique("icu_stay_id")

    events = build_events(ids)
    events = (events
              .join(map_ed, on="ed_stay_id", how="left")
              .join(map_icu, on="icu_stay_id", how="left")
              .with_columns(pl.coalesce("hadm_id", "hadm_from_ed", "hadm_from_icu").alias("hadm_id"))
              .drop("hadm_from_ed", "hadm_from_icu"))

    events = events.with_columns(
        pl.when(pl.col("hadm_id").is_not_null())
          .then(pl.concat_str(pl.lit("H"), pl.col("hadm_id").cast(pl.String)))
          .when(pl.col("ed_stay_id").is_not_null())
          .then(pl.concat_str(pl.lit("E"), pl.col("ed_stay_id").cast(pl.String)))
          .otherwise(None).alias("episode_id"),
        pl.when(pl.col("setting") == "hosp")
          .then(pl.when(pl.col("hadm_id").is_not_null()).then(pl.lit("ward")).otherwise(pl.lit("outpatient")))
          .otherwise(pl.col("setting")).alias("setting"),
    ).sort(["subject_id", "time"])

    # Outpatient episodes: MIMIC has no outpatient-visit table, so one episode is
    # one patient-day with at least one outpatient event (lab, micro, or omr).
    outp = (events.filter(pl.col("setting") == "outpatient")
            .with_columns(pl.col("time").dt.date().alias("day"))
            .group_by(["subject_id", "day"])
            .agg(pl.col("time").min().alias("start"), pl.col("time").max().alias("end"))
            .with_columns(
                pl.concat_str(pl.lit("O"), pl.col("subject_id").cast(pl.String), pl.lit("_"),
                              pl.col("day").cast(pl.String)).alias("episode_id"),
                pl.lit("outpatient_day").alias("kind"))
            .select("episode_id", "subject_id", "kind", "start", "end"))

    # Replace the clock time with age and day-based time.
    # MIMIC has no birth date, so days_since_birth counts from 1 January of the
    # birth year (anchor_year - anchor_age): the error is up to one year.
    anchors = read("patients", ids).select("subject_id", "anchor_age", "anchor_year").unique("subject_id")
    events = events.join(anchors, on="subject_id", how="left")
    year_start = pl.datetime(pl.col("anchor_year"), 1, 1)
    birth_year_start = pl.datetime(pl.col("anchor_year") - pl.col("anchor_age"), 1, 1)
    date_only = pl.col("date_only")  # True when the source gives only a day
    events = events.with_columns(
        (pl.col("anchor_age") + (pl.col("time") - year_start).dt.total_seconds() / (365.25 * 86400))
            .round(2).alias("age_years"),
        (pl.col("time") - birth_year_start).dt.total_days().alias("days_since_birth"),
        pl.when(date_only).then(pl.lit(None, pl.String))
            .otherwise(pl.col("time").dt.strftime("%H:%M:%S")).alias("time_of_day"),
    ).drop("time", "anchor_age", "anchor_year", "date_only")

    events.sink_parquet(OUT / "events.parquet")

    # Episodes: start and end in the same age-based form as the events.
    eps = pl.concat([build_episodes(ids), outp], how="diagonal_relaxed").join(anchors, on="subject_id", how="left")
    eps = add_age(add_age(eps, "start", "start_"), "end", "end_")
    eps.drop("start", "end", "anchor_age", "anchor_year").collect().write_parquet(OUT / "episodes.parquet")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
