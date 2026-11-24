"""Step R3: the status of every point for every disease X and horizon H (HERO plan, section 2).

Time is in days since birth, as the trajectories (age_days). T = the point's landmark day, g = the disease's gap.
Each patient's X onset o = the first time the onset rule is met (codes, lab rule, X-specific drug), d = death,
v = last visit (admission, ED visit or outpatient day in omr; isolated labs do not count).

Status, the first row that applies:
  prevalent          o <= T                                   not at risk (X already met at or before T)
  excluded           a code that makes the rule unreliable, up to T + g + H (type 1 diabetes for T2D)
  confirmed_in_gap   T < o <= T + g                           not at risk (probably under way at T)
  positive           T + g < o <= T + g + H                   anchor (own label y = 1)
  death              d <= T + g + H, no X                     competing event
  no_followup        v < T + g + H, no X                      outcome unknown
  tested_abnormal    tests in (T, T + g + H], one abnormal    possible missed diagnosis (y = 0)
  tested_negative    tests in (T, T + g + H], all normal      candidate pool, core (y = 0)
  untested           no test in (T, T + g + H]                candidate pool, weaker (y = 0)
A point of a patient who became positive earlier is prevalent, since o <= T.

Rules (from the MIMIC pilot; thresholds and choices in resilience_config.yaml):
  T2D     code E11 / 250.x0, 250.x2; HbA1c >= 6.5 %; a non-insulin antidiabetic drug       test HbA1c, abnormal >= 5.7 %
          excluded: type 1 diabetes code E10 / 250.x1, 250.x3
  CKD     code N18 / 585; eGFR (CKD-EPI 2021) < 60 on consecutive values over >= 90 days     test eGFR, abnormal < 60
  AKI     code N17 / 584; KDIGO: creatinine >= 0.3 above the lowest of 48 h, or >= 1.5 x the lowest of 7 days
                                                                     test >= 2 creatinine values, abnormal > 1.2 mg/dL
  sepsis  code A40-A41 / R65.2 / 038 / 995.91-92 / 785.52; simplified Adult Sepsis Event (blood culture,
          >= 4 antibiotic days from culture day - 2, one organ dysfunction within 48 h)  test blood culture,
                                                                                         abnormal: organism grown
A diagnosis code takes the start day of its visit (HERO 2a). Labs by collection time, drugs by start time.

Inputs: raw tables (resilience_config.yaml raw), <out>/events_parts (labs, ICU inputs and procedures),
<out>/episodes.parquet, <out>/resilience/points.parquet.
Output: <out>/resilience/labels.parquet, one row per point, disease and horizon:
  point_id, disease, horizon, gap, status, y (1 / 0 / null), onset_day, n_tests, last_test_day

Usage (PowerShell):
  cd C:\\Users\\yy\\Desktop\\codes_PhD\\jepa_mimic\\resilience
  python build_labels.py
  python build_labels.py --diseases ckd aki       # some diseases only
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bert"))
from bert_data import load_yaml  # noqa: E402

BASE = datetime(1900, 1, 1)             # day 0 of the "days since birth" clock, as a datetime (for rolling windows)
DAY = 86400.0
CREATININE, HBA1C, LACTATE, BILIRUBIN, PLATELETS = 50912, 50852, 50813, 50885, 51265
LAB_ITEMS = (CREATININE, HBA1C, LACTATE, BILIRUBIN, PLATELETS)
VASOPRESSORS = (221906, 221289, 221662, 222315, 221749)   # norepinephrine, epinephrine, dopamine, vasopressin, phenylephrine
VENTILATION = (225792,)                                    # invasive ventilation (procedureevents)
ANTIBIOTICS = (
    "amikacin", "amoxicillin", "ampicillin", "azithromycin", "aztreonam", "bactrim", "cefazolin", "cefepime",
    "cefotaxime", "cefotetan", "cefoxitin", "cefpodoxime", "ceftaroline", "ceftazidime", "ceftriaxone", "cefuroxime",
    "cephalexin", "ciprofloxacin", "clarithromycin", "clindamycin", "colistin", "daptomycin", "dicloxacillin",
    "doxycycline", "ertapenem", "erythromycin", "gentamicin", "imipenem", "levofloxacin", "linezolid", "meropenem",
    "metronidazole", "minocycline", "moxifloxacin", "nafcillin", "nitrofurantoin", "oxacillin", "penicillin",
    "piperacillin", "polymyxin", "rifampin", "sulfamethoxazole", "tigecycline", "tobramycin", "unasyn", "vancomycin",
    "zosyn")
NON_SYSTEMIC_ROUTE = r"(?i)^(TP|OU|OD|OS|AU|AD|AS|IH|IT|IRR|VG)$|EYE|EAR|TOPICAL|OPHTH|INHAL|NEB|VAGINAL|IRRIG|LOCK|DIALYS"
MAX_RX_DAYS = 90        # longer prescriptions are cut, so that wrong stop dates do not explode the day list
STATUSES = ["prevalent", "excluded", "confirmed_in_gap", "positive", "death", "no_followup",
            "tested_abnormal", "tested_negative", "untested"]
OWN_LABEL = {"positive": 1, "tested_abnormal": 0, "tested_negative": 0, "untested": 0}
EV = {"subject_id": pl.Int64, "time": pl.Datetime("us")}


# ---- 1. time: everything in days since birth -------------------------------------------------------------
def days_since_birth(t, anchors):
    """Clock time -> days since 1 January of the birth year (float), as the trajectories."""
    birth = pl.datetime(pl.col("anchor_year") - pl.col("anchor_age"), 1, 1)
    return (t - birth).dt.total_seconds() / DAY


def as_time(days):
    """Days since birth -> a datetime on the BASE clock (rolling windows of 48 h, 7 days need a datetime)."""
    return pl.lit(BASE) + pl.duration(milliseconds=(days * DAY * 1000).round().cast(pl.Int64))


def day_of(t):
    """BASE-clock datetime -> integer day since birth."""
    return (t - pl.lit(BASE)).dt.total_days().cast(pl.Int32)


def read_raw(path, cols, times=(), dates=()):
    """A raw MIMIC table: only the needed columns, ids as Int64, times parsed."""
    lf = pl.scan_csv(path, infer_schema_length=0).select(cols)        # everything as text first
    casts = []
    for c in cols:
        if c in times:
            casts.append(pl.col(c).str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False))
        elif c in dates:
            casts.append(pl.col(c).str.to_datetime("%Y-%m-%d", strict=False))
        elif c.endswith("_id") or c in ("subject_id", "icd_version", "anchor_age", "anchor_year", "itemid"):
            casts.append(pl.col(c).cast(pl.Int64, strict=False))
    return lf.with_columns(casts)


def on_clock(lf, anchors, col, keep=()):
    """subject_id, time (BASE clock) from a raw clock column."""
    return (lf.join(anchors, on="subject_id", how="inner")
              .select("subject_id", as_time(days_since_birth(pl.col(col), anchors)).alias("time"), *keep)
              .drop_nulls("time"))


def part_events(out, table, codes):
    """subject_id, time, code, value from <out>/events_parts/<table>.parquet (itemid codes)."""
    secs = (pl.col("time_of_day").str.strptime(pl.Time, "%H:%M:%S", strict=False)
              .cast(pl.Duration("us")).dt.total_seconds().fill_null(0))
    return (pl.scan_parquet(out / "events_parts" / f"{table}.parquet")
              .filter(pl.col("code").is_in([str(c) for c in codes]) & pl.col("days_since_birth").is_not_null())
              .select("subject_id", as_time(pl.col("days_since_birth") + secs / DAY).alias("time"),
                      pl.col("code").cast(pl.Int64).alias("itemid"),
                      pl.col("value").cast(pl.Float64, strict=False).alias("value")))


# ---- 2. the raw data the rules need ---------------------------------------------------------------------
def load_sources(out, rcfg, diseases):
    raw_h, raw_e = Path(rcfg["raw"]["hosp"]), Path(rcfg["raw"]["ed"])
    t0 = time.time()
    patients = read_raw(raw_h / "patients.csv.gz", ["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
                        dates=("dod",)).collect()
    anchors = patients.select("subject_id", "anchor_age", "anchor_year").lazy()
    src = {"patients": patients.select("subject_id", "gender")}

    # episodes: start day of each admission / ED visit (codes take their visit's start), visits for follow-up
    eps = (pl.scan_parquet(out / "episodes.parquet")
             .select("episode_id", "subject_id", "kind", pl.col("start_days_since_birth").alias("start_day"))
             .collect())
    src["episodes"] = eps

    # diagnosis codes: hospital (hadm_id -> H...) and ED (stay -> its admission H... if any, else E...)
    edstays = read_raw(raw_e / "edstays.csv.gz", ["subject_id", "stay_id", "hadm_id"])
    hosp_dx = (read_raw(raw_h / "diagnoses_icd.csv.gz", ["subject_id", "hadm_id", "icd_code", "icd_version"])
                 .select("subject_id", pl.concat_str(pl.lit("H"), pl.col("hadm_id").cast(pl.String)).alias("episode_id"),
                         "icd_code", "icd_version"))
    ed_dx = (read_raw(raw_e / "diagnosis.csv.gz", ["subject_id", "stay_id", "icd_code", "icd_version"])
               .join(edstays.select("stay_id", "hadm_id"), on="stay_id", how="left")
               .select("subject_id",
                       pl.when(pl.col("hadm_id").is_not_null())
                         .then(pl.concat_str(pl.lit("H"), pl.col("hadm_id").cast(pl.String)))
                         .otherwise(pl.concat_str(pl.lit("E"), pl.col("stay_id").cast(pl.String))).alias("episode_id"),
                       "icd_code", "icd_version"))
    src["dx"] = (pl.concat([hosp_dx, ed_dx]).collect()
                   .join(eps.select("episode_id", "start_day"), on="episode_id", how="inner")
                   .select("subject_id", pl.col("start_day").cast(pl.Int32).alias("day"), "icd_version",
                           pl.col("icd_code").str.replace_all(r"[.\s]", "").str.to_uppercase()))
    print(f"  diagnoses: {src['dx'].height:,} codes  ({time.time() - t0:.0f}s)", flush=True)

    # labs (collection time), from the events already converted by the trajectory step
    src["labs"] = part_events(out, "labevents", LAB_ITEMS).drop_nulls("value").collect().sort("subject_id", "time")
    print(f"  labs: {src['labs'].height:,} values  ({time.time() - t0:.0f}s)", flush=True)

    # death: date of death or in-hospital death time, the earliest
    adm = read_raw(raw_h / "admissions.csv.gz", ["subject_id", "deathtime"], times=("deathtime",))
    deaths = pl.concat([
        on_clock(patients.lazy().filter(pl.col("dod").is_not_null()), anchors, "dod"),
        on_clock(adm.filter(pl.col("deathtime").is_not_null()), anchors, "deathtime")])
    src["death_day"] = deaths.group_by("subject_id").agg(day_of(pl.col("time").min()).alias("death_day")).collect()

    # visits: admission or ED starts, and outpatient days in omr
    visits = [eps.filter(pl.col("kind").is_in(["admission", "ed_only"]))
                 .select("subject_id", pl.col("start_day").cast(pl.Int32).alias("day"))]
    if "omr" in rcfg["visits"]:
        omr = read_raw(raw_h / "omr.csv.gz", ["subject_id", "chartdate"], dates=("chartdate",))
        visits.append(on_clock(omr, anchors, "chartdate").select("subject_id", day_of(pl.col("time")).alias("day"))
                      .collect())
    src["last_visit"] = pl.concat(visits).group_by("subject_id").agg(pl.col("day").max().alias("last_visit"))
    print(f"  deaths and visits  ({time.time() - t0:.0f}s)", flush=True)

    # drugs by name (T2D onset), from the home medications (medrecon) and prescriptions
    want_drugs = {s for k in diseases for s in rcfg["diseases"][k].get("drug_sources", [])}
    names = []
    if "prescriptions" in want_drugs or "sepsis" in diseases:
        rx = read_raw(raw_h / "prescriptions.csv.gz", ["subject_id", "starttime", "stoptime", "drug", "route"],
                      times=("starttime", "stoptime"))
        rx = (rx.join(anchors, on="subject_id", how="inner")
                .select("subject_id",
                        as_time(days_since_birth(pl.col("starttime"), anchors)).alias("start"),
                        as_time(days_since_birth(pl.col("stoptime"), anchors)).alias("stop"),
                        pl.col("drug").str.to_lowercase().alias("drug"), "route")
                .drop_nulls("start").collect())
        if "sepsis" in diseases:
            is_abx = pl.any_horizontal([pl.col("drug").str.contains(a, literal=True) for a in ANTIBIOTICS])
            systemic = pl.col("route").is_null() | ~pl.col("route").str.contains(NON_SYSTEMIC_ROUTE)
            src["antibiotics"] = rx.filter(is_abx & systemic).select("subject_id", "start", "stop")
        if "prescriptions" in want_drugs:
            names.append(rx.select("subject_id", pl.col("start").alias("time"), "drug"))
        del rx
    if "medrecon" in want_drugs:
        med = read_raw(raw_e / "medrecon.csv.gz", ["subject_id", "stay_id", "charttime", "name"], times=("charttime",))
        ed_in = read_raw(raw_e / "edstays.csv.gz", ["stay_id", "intime"], times=("intime",))
        med = (med.join(ed_in, on="stay_id", how="left")
                  .with_columns(pl.coalesce("charttime", "intime").alias("charttime")))
        names.append(on_clock(med, anchors, "charttime", keep=[pl.col("name").str.to_lowercase().alias("drug")]).collect())
    src["drugs"] = pl.concat(names) if names else pl.DataFrame(schema={**EV, "drug": pl.String})
    print(f"  drugs: {src['drugs'].height:,} rows  ({time.time() - t0:.0f}s)", flush=True)

    if "sepsis" in diseases:
        micro = read_raw(raw_h / "microbiologyevents.csv.gz",
                         ["subject_id", "micro_specimen_id", "charttime", "chartdate", "spec_type_desc", "org_name"],
                         times=("charttime",), dates=("chartdate",))
        micro = (micro.filter(pl.col("spec_type_desc").str.to_uppercase().str.starts_with("BLOOD CULTURE"))
                      .with_columns(pl.coalesce("charttime", "chartdate").alias("t")))
        grown = pl.col("org_name").is_not_null() & ~pl.col("org_name").str.to_uppercase().str.contains("CANCELLED")
        src["cultures"] = (on_clock(micro, anchors, "t", keep=["micro_specimen_id", grown.alias("grown")])
                             .group_by("subject_id", "micro_specimen_id")
                             .agg(pl.col("time").min(), pl.col("grown").any())
                             .select("subject_id", "time", "grown").collect())
        src["vasopressors"] = part_events(out, "inputevents", VASOPRESSORS).select("subject_id", "time").collect()
        src["ventilation"] = part_events(out, "procedureevents", VENTILATION).select("subject_id", "time").collect()
        print(f"  sepsis sources: {src['cultures'].height:,} blood cultures  ({time.time() - t0:.0f}s)", flush=True)
    return src


# ---- 3. rule building blocks (from the MIMIC pilot, mimic_traj/cohort.py) ---------------------------------
def code_days(dx, codes):
    """subject_id, day of the diagnosis codes matching (icd_version, regex) pairs."""
    if not codes:
        return dx.select("subject_id", "day").clear()
    hit = pl.any_horizontal([(pl.col("icd_version") == v) & pl.col("icd_code").str.contains(rx) for v, rx in codes])
    return dx.filter(hit).select("subject_id", "day")


def lab(labs, item):
    return labs.filter(pl.col("itemid") == item).select("subject_id", "time", "value").sort("subject_id", "time")


def egfr(creatinine, patients):
    """CKD-EPI 2021 (race-free) from serum creatinine (mg/dL), sex and age (days since birth / 365.25)."""
    female = pl.col("gender") == "F"
    k = pl.when(female).then(0.7).otherwise(0.9)
    a = pl.when(female).then(-0.241).otherwise(-0.302)
    ratio = pl.col("value") / k
    age = day_of(pl.col("time")) / 365.25
    e = (142 * pl.min_horizontal(ratio, pl.lit(1.0)).pow(a) * pl.max_horizontal(ratio, pl.lit(1.0)).pow(-1.200)
         * pl.lit(0.9938).pow(age) * pl.when(female).then(1.012).otherwise(1.0))
    return (creatinine.join(patients, on="subject_id", how="inner").filter(pl.col("value") > 0)
                      .select("subject_id", "time", e.alias("egfr")).sort("subject_id", "time"))


def ckd_lab_onsets(egfr_df, low=60, days=90):
    """eGFR < low on consecutive values (no value >= low in between) spanning >= days: the first low value at
    least `days` after the run's first low value."""
    d = egfr_df.with_columns((pl.col("egfr") < low).alias("_low")).with_columns(
        (pl.col("_low") != pl.col("_low").shift(1)).fill_null(True).cum_sum().over("subject_id").alias("_run"))
    d = d.with_columns(pl.col("time").first().over("subject_id", "_run").alias("_run_start"))
    return d.filter(pl.col("_low") & (pl.col("time") - pl.col("_run_start") >= pl.duration(days=days))).select(
        "subject_id", "time")


def creatinine_baselines(creatinine):
    """Each creatinine value with the lowest value of the previous 48 h and 7 days (its own time left out)."""
    c = creatinine.filter((pl.col("value") > 0) & (pl.col("value") < 40)).sort("subject_id", "time")
    roll = lambda w: pl.col("value").rolling_min_by("time", window_size=w, closed="left").over("subject_id")  # noqa: E731
    return c.with_columns(roll("48h").alias("min48h"), roll("7d").alias("min7d"))


def kdigo_onsets(cb):
    rise = ((pl.col("value") - pl.col("min48h")) >= 0.3) | (pl.col("value") >= 1.5 * pl.col("min7d"))
    return cb.filter(rise.fill_null(False)).select("subject_id", "time")


def antibiotic_runs(abx):
    """Runs of consecutive calendar days with a systemic antibiotic: subject_id, r0, r1 (dates)."""
    if abx.height == 0:
        return pl.DataFrame(schema={"subject_id": pl.Int64, "r0": pl.Date, "r1": pl.Date})
    s = pl.col("start").dt.date()
    stop = pl.coalesce(pl.col("stop"), pl.col("start")).dt.date()
    stop = pl.min_horizontal(pl.max_horizontal(stop, s), s + pl.duration(days=MAX_RX_DAYS))
    days = (abx.select("subject_id", pl.date_ranges(s, stop, "1d").alias("d")).explode("d")
               .unique().sort("subject_id", "d"))
    new_run = (pl.col("d") - pl.col("d").shift(1).over("subject_id") != pl.duration(days=1)).fill_null(True)
    days = days.with_columns(new_run.cum_sum().over("subject_id").alias("_run"))
    return days.group_by("subject_id", "_run").agg(pl.col("d").min().alias("r0"), pl.col("d").max().alias("r1")).drop("_run")


def organ_dysfunction(src, cb):
    """Times of organ dysfunction for the sepsis rule."""
    labs = src["labs"]
    plt = lab(labs, PLATELETS).with_columns(
        pl.col("value").rolling_max_by("time", window_size="7d", closed="left").over("subject_id").alias("_max7d"))
    parts = [src["vasopressors"], src["ventilation"],
             lab(labs, LACTATE).filter(pl.col("value") >= 2.0),
             lab(labs, BILIRUBIN).filter(pl.col("value") >= 2.0),
             cb.filter((pl.col("value") >= 2 * pl.col("min7d")).fill_null(False)),
             plt.filter(((pl.col("value") < 100) & (pl.col("_max7d") >= 100)
                         & (pl.col("value") <= 0.5 * pl.col("_max7d"))).fill_null(False))]
    return pl.concat([p.select("subject_id", "time") for p in parts]).sort("subject_id", "time")


def ase_onsets(cultures, runs, od):
    """Blood cultures that meet the simplified Adult Sepsis Event (onset = culture time)."""
    if cultures.height == 0:
        return cultures.select("subject_id", "time")
    c = cultures.select("subject_id", "time").with_columns(pl.col("time").dt.date().alias("_dc")).with_row_index("_c")
    # antibiotics: the latest run starting by culture day + 2 is the only one that can qualify
    r = c.with_columns((pl.col("_dc") + pl.duration(days=2)).alias("_k")).sort("_k").join_asof(
        runs.sort("r0"), left_on="_k", right_on="r0", by="subject_id", strategy="backward", check_sortedness=False)
    first = pl.max_horizontal("r0", pl.col("_dc") - pl.duration(days=2))
    r = r.with_columns(((pl.col("r1") - first) >= pl.duration(days=3)).fill_null(False).alias("_abx"))
    # organ dysfunction: the first one from 48 h before the culture on, if it is within 48 h after
    o = c.with_columns((pl.col("time") - pl.duration(hours=48)).alias("_k")).sort("_k").join_asof(
        od.select("subject_id", pl.col("time").alias("_od")).sort("_od"), left_on="_k", right_on="_od",
        by="subject_id", strategy="forward", check_sortedness=False)
    o = o.with_columns((pl.col("_od") <= pl.col("time") + pl.duration(hours=48)).fill_null(False).alias("_organ"))
    ok = r.select("_c", "_abx").join(o.select("_c", "_organ"), on="_c")
    return c.join(ok, on="_c").filter(pl.col("_abx") & pl.col("_organ")).select("subject_id", "time")


def drug_days(drugs, names):
    if not names:
        return drugs.select("subject_id", day_of(pl.col("time")).alias("day")).clear()
    hit = pl.any_horizontal([pl.col("drug").str.contains(n, literal=True) for n in names])
    return drugs.filter(hit).select("subject_id", day_of(pl.col("time")).alias("day"))


def disease_events(src, key, dcfg, th):
    """onsets (subject_id, day), exclusions (subject_id, day), tests (subject_id, day, abnormal), min_tests."""
    days = lambda df: df.select("subject_id", day_of(pl.col("time")).alias("day"))  # noqa: E731
    codes = code_days(src["dx"], dcfg["codes"])
    excl = code_days(src["dx"], dcfg.get("exclude_codes", []))
    creat = lab(src["labs"], CREATININE)
    min_tests = 1
    if key == "t2d":
        a1c = lab(src["labs"], HBA1C)
        onsets = [codes, days(a1c.filter(pl.col("value") >= th["hba1c_onset"])),
                  drug_days(src["drugs"], dcfg.get("drugs", []))]
        tests = a1c.select("subject_id", "time", (pl.col("value") >= th["hba1c_abnormal"]).alias("abnormal"))
    elif key == "ckd":
        e = egfr(creat, src["patients"])
        onsets = [codes, days(ckd_lab_onsets(e, th["egfr_low"], th["egfr_days"]))]
        tests = e.select("subject_id", "time", (pl.col("egfr") < th["egfr_low"]).alias("abnormal"))
    elif key == "aki":
        onsets = [codes, days(kdigo_onsets(creatinine_baselines(creat)))]
        tests = creat.select("subject_id", "time", (pl.col("value") > th["creatinine_abnormal"]).alias("abnormal"))
        min_tests = 2                     # KDIGO needs a baseline: one value cannot show a rise
    elif key == "sepsis":
        cb = creatinine_baselines(creat)
        ase = ase_onsets(src["cultures"], antibiotic_runs(src["antibiotics"]), organ_dysfunction(src, cb))
        onsets = [codes, days(ase)]
        tests = src["cultures"].select("subject_id", "time", pl.col("grown").alias("abnormal"))
    else:
        raise ValueError(f"no rule for disease '{key}'")
    tests = tests.select("subject_id", day_of(pl.col("time")).alias("day"), "abnormal")
    return {"onsets": pl.concat([o.select("subject_id", pl.col("day").cast(pl.Int32)) for o in onsets]),
            "exclude": excl, "tests": tests, "min_tests": min_tests}


# ---- 4. labels ------------------------------------------------------------------------------------------
class DayIndex:
    """Events per patient, sorted by day: counts and last day in a window (lo, hi], for many points at once."""

    def __init__(self, ev):
        k = np.sort(ev["subject_id"].to_numpy().astype(np.int64) * 1_000_000 + ev["day"].to_numpy())
        self.k = k

    def count(self, sid, lo, hi):
        base = sid.astype(np.int64) * 1_000_000
        return np.searchsorted(self.k, base + hi, "right") - np.searchsorted(self.k, base + lo, "right")

    def last(self, sid, lo, hi):
        base = sid.astype(np.int64) * 1_000_000
        if len(self.k) == 0:
            return np.full(len(sid), -1)
        i = np.searchsorted(self.k, base + hi, "right") - 1
        ok = (i >= 0) & (self.k[np.maximum(i, 0)] > base + lo)
        return np.where(ok, self.k[np.maximum(i, 0)] - base, -1)


def label_points(pts, ev, death_day, last_visit, horizon, gap):
    """Status of every point for one disease, horizon and gap (see the module docstring)."""
    first = ev["onsets"].group_by("subject_id").agg(pl.col("day").min().alias("onset"))
    excl = ev["exclude"].group_by("subject_id").agg(pl.col("day").min().alias("excl"))
    p = (pts.select("point_id", "subject_id", pl.col("T_day").cast(pl.Int32).alias("T"))
            .join(first, on="subject_id", how="left").join(excl, on="subject_id", how="left")
            .join(death_day, on="subject_id", how="left").join(last_visit, on="subject_id", how="left")
            .with_columns(end=pl.col("T") + gap + horizon)
            .sort("point_id"))

    # tests in (T, T + g + H]
    sid, lo, hi = p["subject_id"].to_numpy(), p["T"].to_numpy(), p["end"].to_numpy()
    all_tests = DayIndex(ev["tests"])
    abnormal = DayIndex(ev["tests"].filter(pl.col("abnormal")))
    p = p.with_columns(n_tests=pl.Series(all_tests.count(sid, lo, hi)).cast(pl.Int32),
                       n_abnormal=pl.Series(abnormal.count(sid, lo, hi)).cast(pl.Int32),
                       last_test_day=pl.Series(all_tests.last(sid, lo, hi)).cast(pl.Int32))

    onset, T, end = pl.col("onset"), pl.col("T"), pl.col("end")
    tested = pl.col("n_tests") >= ev["min_tests"]
    status = (pl.when((onset <= T).fill_null(False)).then(pl.lit("prevalent"))
                .when((pl.col("excl") <= end).fill_null(False)).then(pl.lit("excluded"))
                .when((onset <= T + gap).fill_null(False)).then(pl.lit("confirmed_in_gap"))
                .when((onset <= end).fill_null(False)).then(pl.lit("positive"))
                .when((pl.col("death_day") <= end).fill_null(False)).then(pl.lit("death"))
                .when((pl.col("last_visit") < end).fill_null(True)).then(pl.lit("no_followup"))
                .when(tested & (pl.col("n_abnormal") > 0)).then(pl.lit("tested_abnormal"))
                .when(tested).then(pl.lit("tested_negative"))
                .otherwise(pl.lit("untested")))
    return p.with_columns(status=status).select(
        "point_id", "status",
        pl.col("status").replace_strict(OWN_LABEL, default=None, return_dtype=pl.Int8).alias("y"),
        pl.when(pl.col("status") == "positive").then(onset).alias("onset_day"),
        "n_tests", pl.when(pl.col("last_test_day") >= 0).then(pl.col("last_test_day")).alias("last_test_day"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory-config", default=str(HERE.parent / "bert" / "trajectory_columns.yaml"))
    ap.add_argument("--config", default=str(HERE.parent / "config" / "resilience_config.yaml"))
    ap.add_argument("--diseases", nargs="*", default=None, help="default: all diseases of the config")
    args = ap.parse_args()
    tcfg, rcfg = load_yaml(args.trajectory_config), load_yaml(args.config)
    out = Path(tcfg["paths"]["out"])
    res_dir = out / rcfg["out_dir"]
    diseases = args.diseases or list(rcfg["diseases"])
    pts = pl.read_parquet(res_dir / "points.parquet")
    t0 = time.time()

    print("reading the raw data ...", flush=True)
    src = load_sources(out, rcfg, diseases)
    labels = []
    for key in diseases:
        dcfg = rcfg["diseases"][key]
        ev = disease_events(src, key, dcfg, rcfg["thresholds"])
        for h in rcfg["horizons"]:
            lab_ = label_points(pts, ev, src["death_day"], src["last_visit"], h, dcfg["gap"])
            labels.append(lab_.with_columns(disease=pl.lit(key), horizon=pl.lit(h, pl.Int32),
                                            gap=pl.lit(dcfg["gap"], pl.Int32)))
        print(f"{key}: {ev['onsets']['subject_id'].n_unique():,} patients ever with X  ({time.time() - t0:.0f}s)",
              flush=True)

    lab_all = (pl.concat(labels)
                 .select("point_id", "disease", "horizon", "gap", "status", "y", "onset_day", "n_tests", "last_test_day")
                 .sort("disease", "horizon", "point_id"))
    lab_all.write_parquet(res_dir / "labels.parquet")

    # status counts: points and patients
    table = (lab_all.join(pts.select("point_id", "subject_id"), on="point_id")
                    .group_by("disease", "horizon", "status")
                    .agg(pl.len().alias("points"), pl.col("subject_id").n_unique().alias("patients"))
                    .with_columns(order=pl.col("status").replace_strict({s: i for i, s in enumerate(STATUSES)},
                                                                        return_dtype=pl.Int32))
                    .sort("disease", "horizon", "order").drop("order"))
    table.write_csv(res_dir / "label_counts.csv")
    with pl.Config(tbl_rows=200):
        print(table)
    print(f"written {res_dir / 'labels.parquet'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
