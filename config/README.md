# Step 1: column configuration (MIMIC-IV)

## Files

| File | Purpose |
|---|---|
| `mimic_columns.csv` | One row per column of every table: dataset, table, file, column, dtype, `use`, `locked`, role, lock reason. Edit `use` in Excel to include or exclude a column. |
| `check_config.py` | Reads the CSV, refuses it if a locked column is switched off, and returns the config for the trajectory step. |

## How to use `use` and `locked`

- `use = Y`: the column goes into the trajectory construction.
- `use = N`: the column is ignored.
- `locked = Y`: the column is a relational key (or the time/interval used to link two tables). It must stay `use = Y`. If you set it to `N`, `check_config.py` refuses to run and says which line is wrong.
- Each locked column has a `lock_reason` saying why it is needed.

## Locked columns, in short

- `subject_id`, `hadm_id`, `stay_id` in every table that links to patients, admissions or stays.
- ED `stay_id` is not ICU `stay_id`. The `table` column tells you which one applies.
- `icd_code` + `icd_version`, `itemid`, `transfers.eventtype`, and the time/interval columns used to link ED visits, ICU stays and admissions (`intime`, `outtime`, `admittime`, `dischtime`).

## Data types

`dtype` uses the names from the relational model: `int`, `float`, `string`, `datetime`, `date`. `check_config.py` maps them to Polars types (`Int64`, `Float64`, `String`, `Datetime`, `Date`).

## Check

```
python check_config.py
```

## Notes

- The inventory covers MIMIC-IV 3.1 (`hosp`, `icu`) and MIMIC-IV-ED 2.2 (`ed`), 25 tables, as listed in the relational model. Tables that exist only in MIMIC-IV 2.2 (for example `d_hcpcs`, `drgcodes`) are not included.
- Dtypes and names come from the relational model, not from reading the gzipped files. A quick check against the real headers is still recommended (see below).
- The `role` column (key, time, code, value, unit, text, static, other) is a label only. It does not change what is used.

## Verify against the real files (recommended next step)

Run once to compare the CSV with the headers of the real files, once the device shell is available again:

```
zcat mimic-iv-3.1/hosp/labevents.csv.gz | head -1
```
