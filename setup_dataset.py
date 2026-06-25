#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12,<3.14"
# ///
"""
Set up working files from a CLDF dataset.

Copies variables.csv and codes.csv from the CLDF directory into the project
root, creates a gold/ directory, and extracts gold-standard codings for each
source document referenced in the CLDF data.csv.

Output format per gold file:
    [
        {"id": 2, "code": "0"},
        {"id": 3, "code": null},
        ...
    ]

Usage:
    python setup_dataset.py                          # default CLDF dir
    python setup_dataset.py --cldf-dir path/to/cldf
    python setup_dataset.py --list                   # list all source keys
"""

import argparse
import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

CLDF_DIR = Path("./dataset/cldf")


def strip_pages(source_key: str) -> str:
    """'buck1952[39-41]' → 'buck1952'"""
    return re.sub(r"\[.*?\]", "", source_key).strip()


def parse_sources(source_field: str) -> list[str]:
    if not source_field:
        return []
    return [strip_pages(s) for s in source_field.split(";") if s.strip()]


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_code_names(path: Path) -> dict[str, str]:
    """Returns map of Code_ID → Name (the short code value)."""
    codes = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            codes[row["ID"]] = row["Name"]
    return codes


def build_source_index(data_rows: list[dict]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = defaultdict(list)
    for row in data_rows:
        for src in parse_sources(row["Source"]):
            if src:
                index[src].append(row)
    return index


def codings_for_source(
    rows: list[dict],
    all_variables: list[dict],
    code_names: dict[str, str],
) -> list[dict]:
    by_param: dict[str, str | None] = {}
    for row in rows:
        param_id = row["Parameter_ID"]
        if row["Code_ID"]:
            by_param[param_id] = code_names.get(row["Code_ID"], row["Code_ID"])
        elif row["Value"] not in ("", None):
            by_param[param_id] = row["Value"]
        else:
            by_param[param_id] = None

    return [
        {"id": int(var["ID"]), "code": by_param.get(var["ID"])}
        for var in all_variables
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up working files from a CLDF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cldf-dir", type=Path, default=CLDF_DIR,
        help=f"Path to CLDF directory (default: {CLDF_DIR})")
    parser.add_argument(
        "-o", "--output", default="gold",
        help="Output directory for gold files (default: gold/)")
    parser.add_argument(
        "--list", action="store_true",
        help="List all source keys and exit")
    args = parser.parse_args()

    cldf = args.cldf_dir
    for name in ("variables.csv", "codes.csv", "data.csv"):
        if not (cldf / name).exists():
            sys.exit(f"Error: {cldf / name} not found")

    # Copy variables.csv and codes.csv to project root
    if not args.list:
        for name in ("variables.csv", "codes.csv"):
            src = cldf / name
            dst = Path(name)
            shutil.copy2(src, dst)
            print(f"Copied {src} → {dst}")

    all_variables = load_csv(cldf / "variables.csv")
    code_names = load_code_names(cldf / "codes.csv")
    data_rows = load_csv(cldf / "data.csv")
    source_index = build_source_index(data_rows)

    if args.list:
        for key in sorted(source_index):
            n_params = len({r["Parameter_ID"] for r in source_index[key]})
            n_societies = len({r["Language_ID"] for r in source_index[key]})
            print(f"{key:40s}  {n_params:3d} params  {n_societies:3d} societies")
        print(f"\nTotal: {len(source_index)} sources", file=sys.stderr)
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in sorted(source_index):
        rows = source_index[key]
        codings = codings_for_source(rows, all_variables, code_names)
        out_path = output_dir / f"{key}.json"
        out_path.write_text(json.dumps(codings, indent=2), encoding="utf-8")
        n_coded = sum(1 for c in codings if c["code"] is not None)
        print(f"{key:40s}  {n_coded:3d}/{len(all_variables)} coded → {out_path}")

    print(f"\nDone. {len(source_index)} gold file(s) written to {output_dir}/",
          file=sys.stderr)


if __name__ == "__main__":
    main()
