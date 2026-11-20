"""Step 2, low-memory version: build events and episodes for MIMIC-IV.

Each source table is processed on its own and streamed to disk, so memory stays
low. There is no global sort (sort per patient later, when building points).

Time rules:
  - the event's own time when it has one (time_basis = "event")
  - the event's own day when only the day is known (time_basis = "date_only")
  - the visit end (discharge or ED departure) when the event has no time but has a
    visit ID (time_basis = "visit_end"); no leak, since the event happened in the visit
  - no time and no visit ID: the event is left with no time (time_basis empty)
  - death events only for patients with a death date

Code and text rules:
  - icd_version is kept for diagnoses and procedures (ICD-9 or ICD-10), so the vocabulary
    can tell the two ontologies apart
  - text holds the readable name of drugs (prescriptions, pyxis, medrecon); the code is the
    real drug code (ndc for prescriptions and medrecon, gsn for pyxis)

Outputs in OUT:
  events_parts/<table>.parquet      one file per source table (intermediate)
  events.parquet                    all events, age-based time, time_basis column
  episodes.parquet                  admissions, ED-only visits, outpatient days
  outpatient_parts/<table>.parquet  intermediate, one row per patient-day

Usage (PowerShell):
  cd C:\\Users\\yy\\Desktop\\codes_PhD\\jepa_mimic\\trajectory
  python build_trajectories_lowmem.py --limit 2000          # quick test
  python build_trajectories_lowmem.py --tables labevents    # one table
  python build_trajectories_lowmem.py                       # everything
"""
import argparse
import shutil
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "config"))
from check_config import load_config, read_rows  # noqa: E402

# ---- paths ------------------------------------------------------------------
ROOTS = {
    "MIMIC-IV 3.1": Path(r"D:\MIMIC\mimic-iv-3.1"),
    "MIMIC-IV-ED 2.2": Path(r"D:\MIMIC\mimic-iv-ed-2.2"),
}
OUT = Path(r"D:\MIMIC\trajectory_out")
PARTS = OUT / "events_parts"
OUTP = OUT / "outpatient_parts"

CFG = load_config()
PATHS = {r["table"]: ROOTS[r["dataset"]] / r["file"] for r in read_rows()}

# ---- tables -------------------------------------------------------------------
ED_TABLES = {"edstays", "triage", "vitalsign", "pyxis", "medrecon", "diagnosis"}
ICU_TABLES = {"icustays", "chartevents", "inputevents", "outputevents", "procedureevents"}
DATE_ONLY_TABLES = {"omr", "procedures_icd", "patients"}  # sources with no clock time
ICD_TABLES = {"diagnosis", "diagnoses_icd", "procedures_icd"}  # codes that have an ICD version

# table: (time source (table, column), event_type, code column, value columns, unit column)
RULES = {
    "patients":           (("self", "dod"),             "death",         None,          [], None),
    "omr":                (("self", "chartdate"),       "omr",           "result_name", ["result_value"], None),
    "triage":             (("edstays", "intime"),       "vital",         None,          ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "acuity"], None),
    "vitalsign":          (("self", "charttime"),       "vital",         None,          ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"], None),
    "pyxis":              (("self", "charttime"),       "drug",          "gsn",         [], None),
    "medrecon":           (("self", "charttime"),       "home_med",      "ndc",         [], None),
    "diagnosis":          (("edstays", "outtime"),      "diagnosis",     "icd_code",    [], None),
    "diagnoses_icd":      (("admissions", "dischtime"), "diagnosis",     "icd_code",    [], None),
    "procedures_icd":     (("self", "chartdate"),       "procedure",     "icd_code",    [], None),
    "prescriptions":      (("self", "starttime"),       "drug",          "ndc",         [], None),
    "labevents":          (("self", "charttime"),       "lab",           "itemid",      ["valuenum"], "valueuom"),
    "microbiologyevents": (("self", "charttime"),       "micro",         "test_name",   [], None),
    "chartevents":        (("self", "charttime"),       "icu_chart",     "itemid",      ["valuenum"], "valueuom"),
    "inputevents":        (("self", "starttime"),       "icu_input",     "itemid",      ["amount"], "amountuom"),
    "outputevents":       (("self", "charttime"),       "icu_output",    "itemid",      ["value"], "valueuom"),
    "procedureevents":    (("self", "starttime"),       "icu_procedure", "itemid",      ["value"], "valueuom"),
}
# readable name of the item, for tables whose code is not a dictionary key
TEXT_COL = {"pyxis": "name", "medrecon": "name", "prescriptions": "drug"}
SETTING = {t: ("ED" if t in ED_TABLES else "ICU" if t in ICU_TABLES else "hosp") for t in RULES}

EVENT_COLUMNS = ["subject_id", "age_years", "days_since_birth", "time_of_day", "time_basis",
                 "event_type", "code", "icd_version", "text", "value", "unit", "source", "setting",
                 "hadm_id", "ed_stay_id", "icu_stay_id", "episode_id"]
EPISODE_COLUMNS = ["episode_id", "subject_id", "kind",
                   "start_age_years", "start_days_since_birth",
                   "end_age_years", "end_days_since_birth"]


# ---- helpers -------------------------------------------------------------------
def stay_name(table):
    """ED and ICU both use 'stay_id' in MIMIC; rename so they never mix."""
    if table in ED_TABLES:
        return "ed_stay_id"
    if table in ICU_TABLES:
        return "icu_stay_id"
    return None


def read(table, ids=None):
    """Lazily read one table: used columns only, typed from the config, patients filtered."""
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


def add_age(df, col, prefix):
    """Age in years and days since 1 January of the birth year (approximate birth date)."""
    return df.with_columns(
        (pl.col("anchor_age") + (pl.col(col) - pl.datetime(pl.col("anchor_year"), 1, 1))
            .dt.total_seconds() / (365.25 * 86400)).round(2).alias(f"{prefix}age_years"),
        (pl.col(col) - pl.datetime(pl.col("anchor_year") - pl.col("anchor_age"), 1, 1))
            .dt.total_days().alias(f"{prefix}days_since_birth"),
    )


# ---- one source table -> events -----------------------------------------------
def source_events(table, ids, anchors, map_ed, map_icu, adm_end, ed_end):
    time_src, etype, code_col, value_cols, unit_col = RULES[table]
    src_table, src_col = time_src
    text_col = TEXT_COL.get(table)
    has_icd = table in ICD_TABLES
    lf = read(table, ids)
    if table == "patients":  # a death event only exists for patients who died
        lf = lf.filter(pl.col("dod").is_not_null())

    # 1) the time of each row
    if table == "microbiologyevents":  # charttime, falling back to chartdate (day only)
        lf = lf.with_columns(
            pl.coalesce(pl.col("charttime"), pl.col("chartdate").cast(pl.Datetime("us"))).alias("_time"),
            pl.col("charttime").is_null().alias("_date_only"))
    elif src_table == "self":
        lf = lf.with_columns(pl.col(src_col).alias("_time"),
                             pl.lit(table in DATE_ONLY_TABLES).alias("_date_only"))
    else:  # time borrowed from the linked table (ED visit or admission)
        key = "ed_stay_id" if src_table == "edstays" else "hadm_id"
        ref = read(src_table).select(key, src_col).unique(key).rename({src_col: "_time"})
        lf = lf.join(ref, on=key, how="left").with_columns(pl.lit(False).alias("_date_only"))

    # 2) keep only what the event needs
    names = lf.collect_schema().names()
    ids_present = [c for c in ["subject_id", "hadm_id", "ed_stay_id", "icu_stay_id"] if c in names]
    extra = [c for c in (text_col, "icd_version" if has_icd else None) if c]
    keep = ids_present + ["_time", "_date_only"] + value_cols + extra + [c for c in (code_col, unit_col) if c]
    lf = lf.select(list(dict.fromkeys(keep)))

    # 3) one event per row; vitals are split into one event per measurement
    index = ids_present + ["_time", "_date_only"]
    if len(value_cols) > 1:
        lf = lf.unpivot(on=value_cols, index=index, variable_name="code", value_name="value")
        code = pl.col("code")
        value = pl.col("value").cast(pl.String)
    else:
        code = pl.col(code_col).cast(pl.String) if code_col else pl.lit(None, pl.String)
        value = pl.col(value_cols[0]).cast(pl.String) if value_cols else pl.lit(None, pl.String)
    unit = pl.col(unit_col).cast(pl.String) if unit_col else pl.lit(None, pl.String)
    icd = pl.col("icd_version").cast(pl.Int64) if has_icd else pl.lit(None, pl.Int64)
    text = pl.col(text_col).cast(pl.String) if text_col else pl.lit(None, pl.String)

    def int_col(c):
        return pl.col(c).cast(pl.Int64).alias(c) if c in ids_present else pl.lit(None, pl.Int64).alias(c)

    ev = lf.select(
        pl.col("subject_id").cast(pl.Int64),
        pl.col("_time").cast(pl.Datetime("us")).alias("time"),
        pl.col("_date_only").alias("date_only"),
        pl.lit(etype).alias("event_type"),
        code.alias("code"),
        icd.alias("icd_version"),
        text.alias("text"),
        value.alias("value"),
        unit.alias("unit"),
        pl.lit(table).alias("source"),
        pl.lit(SETTING[table]).alias("setting"),
        int_col("hadm_id"), int_col("ed_stay_id"), int_col("icu_stay_id"),
    )

    # 4) admission of the row, from its ED stay or ICU stay when it lacks one
    ev = (ev.join(map_ed, on="ed_stay_id", how="left")
            .join(map_icu, on="icu_stay_id", how="left")
            .with_columns(pl.coalesce("hadm_id", "hadm_from_ed", "hadm_from_icu").alias("hadm_id"))
            .drop("hadm_from_ed", "hadm_from_icu"))

    # 5) episode and setting (hosp rows: ward if admitted, else outpatient)
    ev = ev.with_columns(
        pl.when(pl.col("hadm_id").is_not_null())
          .then(pl.concat_str(pl.lit("H"), pl.col("hadm_id").cast(pl.String)))
          .when(pl.col("ed_stay_id").is_not_null())
          .then(pl.concat_str(pl.lit("E"), pl.col("ed_stay_id").cast(pl.String)))
          .otherwise(None).alias("episode_id"),
        pl.when(pl.col("setting") == "hosp")
          .then(pl.when(pl.col("hadm_id").is_not_null()).then(pl.lit("ward")).otherwise(pl.lit("outpatient")))
          .otherwise(pl.col("setting")).alias("setting"),
    )

    # 6) no clock time but a visit ID: use the visit end (discharge or ED departure)
    ev = ev.join(adm_end, on="hadm_id", how="left").join(ed_end, on="ed_stay_id", how="left")
    has_time = pl.col("time").is_not_null()
    visit_end = pl.coalesce("visit_end_adm", "visit_end_ed")
    ev = ev.with_columns(
        pl.when(has_time).then(pl.col("time")).otherwise(visit_end).alias("time"),
        pl.when(has_time & pl.col("date_only")).then(pl.lit("date_only"))
          .when(has_time).then(pl.lit("event"))
          .when(visit_end.is_not_null()).then(pl.lit("visit_end"))
          .otherwise(pl.lit(None, pl.String)).alias("time_basis"),
    ).drop("visit_end_adm", "visit_end_ed")

    # 7) age-based time instead of the clock time
    ev = ev.join(anchors, on="subject_id", how="left")
    ev = add_age(ev, "time", "")
    ev = ev.with_columns(
        pl.when(pl.col("time_basis") == "date_only").then(pl.lit(None, pl.String))
          .otherwise(pl.col("time").dt.strftime("%H:%M:%S")).alias("time_of_day"))
    return ev.select(EVENT_COLUMNS)


# ---- episodes -------------------------------------------------------------------
def build_episodes(ids, anchors):
    """Admissions (starting at ED arrival when linked) and ED-only visits, in age form."""
    ed_in = (read("edstays", ids).filter(pl.col("hadm_id").is_not_null())
             .group_by("hadm_id").agg(pl.col("intime").min().alias("ed_in")))
    adm = read("admissions", ids).join(ed_in, on="hadm_id", how="left").select(
        "subject_id", "hadm_id",
        pl.min_horizontal("ed_in", "admittime").alias("start"),
        pl.col("dischtime").alias("end"),
    ).with_columns(
        pl.lit("admission").alias("kind"),
        pl.concat_str(pl.lit("H"), pl.col("hadm_id").cast(pl.String)).alias("episode_id"))

    ed = read("edstays", ids).filter(pl.col("hadm_id").is_null()).select(
        "subject_id", "ed_stay_id",
        pl.col("intime").alias("start"), pl.col("outtime").alias("end"),
    ).with_columns(
        pl.lit("ed_only").alias("kind"),
        pl.concat_str(pl.lit("E"), pl.col("ed_stay_id").cast(pl.String)).alias("episode_id"))

    cols = ["episode_id", "subject_id", "kind", "start", "end"]
    raw = pl.concat([adm.select(cols), ed.select(cols)], how="diagonal_relaxed")
    raw = raw.join(anchors, on="subject_id", how="left")
    raw = add_age(add_age(raw, "start", "start_"), "end", "end_")
    return raw.select(EPISODE_COLUMNS)


def outpatient_episodes():
    """One episode per patient-day with outpatient events (summaries written per table)."""
    return (pl.scan_parquet(str(OUTP / "*.parquet"))
            .group_by(["subject_id", "days_since_birth"])
            .agg(pl.col("start_age_years").min(), pl.col("end_age_years").max())
            .select(
                pl.concat_str(pl.lit("O"), pl.col("subject_id").cast(pl.String), pl.lit("_"),
                              pl.col("days_since_birth").cast(pl.String)).alias("episode_id"),
                pl.col("subject_id"),
                pl.lit("outpatient_day").alias("kind"),
                pl.col("start_age_years"),
                pl.col("days_since_birth").alias("start_days_since_birth"),
                pl.col("end_age_years"),
                pl.col("days_since_birth").alias("end_days_since_birth"),
            ))


# ---- main -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="number of patients (test runs)")
    ap.add_argument("--tables", nargs="*", default=None, help="only these source tables")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tables = args.tables or list(RULES)
    if not args.tables:  # full run: start clean
        for d in (PARTS, OUTP):
            shutil.rmtree(d, ignore_errors=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    OUTP.mkdir(parents=True, exist_ok=True)

    # small tables are collected once, so they are held in memory
    ids = None
    if args.limit:
        ids = read("patients").select("subject_id").unique().sort("subject_id").head(args.limit).collect().lazy()
    anchors = (read("patients", ids).select("subject_id", "anchor_age", "anchor_year")
               .unique("subject_id").collect().lazy())
    map_ed = (read("edstays", ids).select("ed_stay_id", pl.col("hadm_id").alias("hadm_from_ed"))
              .unique("ed_stay_id").collect().lazy())
    map_icu = (read("icustays", ids).select("icu_stay_id", pl.col("hadm_id").alias("hadm_from_icu"))
               .unique("icu_stay_id").collect().lazy())
    adm_end = (read("admissions", ids).select("hadm_id", pl.col("dischtime").alias("visit_end_adm"))
               .unique("hadm_id").collect().lazy())
    ed_end = (read("edstays", ids).select("ed_stay_id", pl.col("outtime").alias("visit_end_ed"))
              .unique("ed_stay_id").collect().lazy())

    # 1) one source at a time: events to disk, then the outpatient-day summary from that file
    for table in tables:
        if table not in CFG:
            print(f"skip {table}: not in the config")
            continue
        print(f"processing {table} ...", flush=True)
        part = PARTS / f"{table}.parquet"
        source_events(table, ids, anchors, map_ed, map_icu, adm_end, ed_end).sink_parquet(part, engine="streaming")
        (pl.scan_parquet(part)
           .filter((pl.col("setting") == "outpatient") & pl.col("days_since_birth").is_not_null())
           .group_by(["subject_id", "days_since_birth"])
           .agg(pl.col("age_years").min().alias("start_age_years"),
                pl.col("age_years").max().alias("end_age_years"))
           .sink_parquet(OUTP / f"{table}.parquet", engine="streaming"))
        print(f"  done {table}", flush=True)

    # 2) all events in one file (streamed, not held in memory)
    pl.scan_parquet(str(PARTS / "*.parquet")).sink_parquet(OUT / "events.parquet", engine="streaming")

    # 3) episodes: admissions, ED-only visits and outpatient days
    eps = pl.concat([build_episodes(ids, anchors), outpatient_episodes()], how="diagonal_relaxed")
    eps.sink_parquet(OUT / "episodes.parquet", engine="streaming")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
