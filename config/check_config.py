"""Step 1 configuration: load mimic_columns.csv and enforce locked columns.

Usage:
    python check_config.py            # validate and print a summary
    from check_config import load_config

Rules enforced:
- A column with locked=Y must have use=Y. Otherwise the script stops with an error.
- Every locked column must have a lock_reason.
- Locked columns cannot be switched off by the trajectory step: use load_config().
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("mimic_columns.csv")

# Map the config dtype to a Polars dtype name, used when reading the raw files.
POLARS_DTYPE = {
    "int": "Int64",
    "float": "Float64",
    "string": "String",
    "datetime": "Datetime",
    "date": "Date",
}


class ConfigError(ValueError):
    pass


def read_rows(path=CONFIG_PATH):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate(rows):
    errors = []
    for i, r in enumerate(rows, start=2):  # line 1 is the header
        if r["use"] not in ("Y", "N") or r["locked"] not in ("Y", "N"):
            errors.append(f"line {i}: use/locked must be Y or N")
        if r["dtype"] not in POLARS_DTYPE:
            errors.append(f"line {i}: unknown dtype {r['dtype']!r}")
        if r["locked"] == "Y":
            if r["use"] != "Y":
                errors.append(f"line {i}: {r['table']}.{r['column']} is locked but use=N")
            if not r["lock_reason"].strip():
                errors.append(f"line {i}: {r['table']}.{r['column']} is locked without a reason")
    return errors


def load_config(path=CONFIG_PATH):
    """Return {table: {"file", "columns": {name: dtype}, "used": [...], "locked": [...]}}.

    The trajectory step must read columns from "used" only. Locked columns are
    always in "used", even if someone edits use=N (the file is refused instead).
    """
    rows = read_rows(path)
    errors = validate(rows)
    if errors:
        raise ConfigError("\n".join(errors))

    tables = defaultdict(lambda: {"file": None, "columns": {}, "used": [], "locked": []})
    for r in rows:
        t = tables[r["table"]]
        t["file"] = r["file"]
        t["columns"][r["column"]] = POLARS_DTYPE[r["dtype"]]
        if r["use"] == "Y":
            t["used"].append(r["column"])
        if r["locked"] == "Y":
            t["locked"].append(r["column"])
    return dict(tables)


def main():
    try:
        tables = load_config()
    except ConfigError as e:
        print("Configuration refused:\n" + str(e))
        sys.exit(1)
    total_used = sum(len(t["used"]) for t in tables.values())
    total_all = sum(len(t["columns"]) for t in tables.values())
    print(f"{len(tables)} tables, {total_used} of {total_all} columns used")
    for name, t in tables.items():
        print(f"  {name:22s} used {len(t['used']):3d}/{len(t['columns']):3d}  locked {len(t['locked'])}")


if __name__ == "__main__":
    main()
