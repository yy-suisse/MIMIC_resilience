"""Compare the header of every MIMIC file with mimic_columns.csv.

Reads only the first line of each .csv.gz file, so it is fast.

Usage (PowerShell, from this folder):
    python check_headers.py
"""
import csv
import gzip
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "mimic_columns.csv"

# Root folder of each dataset on your disk (edit if your paths differ).
ROOTS = {
    "MIMIC-IV 3.1": Path(r"D:\MIMIC\mimic-iv-3.1"),
    "MIMIC-IV-ED 2.2": Path(r"D:\MIMIC\mimic-iv-ed-2.2"),
}


def read_header(path):
    with gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as f:
        line = f.readline()
    return [c.strip() for c in next(csv.reader([line]))]


def main():
    with open(CONFIG, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # One entry per file: the list of columns the config expects, in order.
    expected_by_file = {}
    for r in rows:
        key = (r["dataset"], r["table"], r["file"])
        expected_by_file.setdefault(key, []).append(r["column"])

    problems = 0
    for (dataset, table, rel_path), expected in expected_by_file.items():
        path = ROOTS[dataset] / rel_path
        if not path.exists():
            problems += 1
            print(f"MISSING FILE   {table}: {path}")
            continue

        actual = read_header(path)
        if actual == expected:
            print(f"OK             {table} ({len(actual)} columns)")
            continue

        problems += 1
        print(f"MISMATCH       {table}")
        only_file = [c for c in actual if c not in expected]
        only_csv = [c for c in expected if c not in actual]
        if only_file:
            print(f"  in file, not in csv: {only_file}")
        if only_csv:
            print(f"  in csv, not in file: {only_csv}")
        if not only_file and not only_csv:
            print("  same columns, different order")

    print(f"\n{len(expected_by_file)} files checked, {problems} problem(s).")


if __name__ == "__main__":
    main()
