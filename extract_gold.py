#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12,<3.14"
# ///
"""
Extract 'gold' codings from the DPLACE Pulotu CLDF dataset.

For each source document referenced in data.csv, writes a JSON file containing
all variables with their coded values (null if not coded for that source).

Output format:
    [
        {"id": 2, "code": "0"},
        {"id": 3, "code": null},
        ...
    ]

Usage:
    python extract_gold.py                        # all sources → gold/
    python extract_gold.py -s buck1952            # one source
    python extract_gold.py -o my_gold_dir         # custom output dir
    python extract_gold.py --list                 # list all source keys
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CLDF_DIR = Path("./dplace-dataset-pulotu/cldf")
VARIABLES_CSV = CLDF_DIR / "variables.csv"
CODES_CSV = CLDF_DIR / "codes.csv"
DATA_CSV = CLDF_DIR / "data.csv"


def strip_pages(source_key: str) -> str:
    """'buck1952[39-41]' → 'buck1952'"""
    return re.sub(r"\[.*?\]", "", source_key).strip()


def parse_sources(source_field: str) -> list[str]:
    """Split semicolon-separated source field into base keys."""
    if not source_field:
        return []
    return [strip_pages(s) for s in source_field.split(";") if s.strip()]


def load_variables(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_codes(path: Path) -> dict[str, str]:
    """Returns map of Code_ID → Name (the short code value)."""
    codes = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            codes[row["ID"]] = row["Name"]
    return codes


def load_data(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_source_index(data_rows: list[dict]) -> dict[str, list[dict]]:
    """Group data rows by base source key."""
    index: dict[str, list[dict]] = defaultdict(list)
    for row in data_rows:
        for src in parse_sources(row["Source"]):
            if src:
                index[src].append(row)
    return index


def codings_for_source(
    source_key: str,
    rows: list[dict],
    all_variables: list[dict],
    code_names: dict[str, str],
) -> list[dict]:
    """Build a full coding list for a source — all variables, null if absent."""
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
        description="Extract gold codings from DPLACE Pulotu CLDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-s", "--source", metavar="KEY",
                        help="Extract only this source key (e.g. buck1952)")
    parser.add_argument("-o", "--output", default="gold",
                        help="Output directory (default: gold/)")
    parser.add_argument("--list", action="store_true",
                        help="List all source keys and exit")
    args = parser.parse_args()

    all_variables = load_variables(VARIABLES_CSV)
    code_names = load_codes(CODES_CSV)
    data_rows = load_data(DATA_CSV)

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

    sources_to_process = (
        [args.source] if args.source else sorted(source_index)
    )

    if args.source and args.source not in source_index:
        sys.exit(f"Source key '{args.source}' not found. Use --list to see all keys.")

    for key in sources_to_process:
        rows = source_index[key]
        codings = codings_for_source(key, rows, all_variables, code_names)
        out_path = output_dir / f"{key}.json"
        out_path.write_text(json.dumps(codings, indent=2), encoding="utf-8")
        n_coded = sum(1 for c in codings if c["code"] is not None)
        print(f"{key:40s}  {n_coded:3d}/{len(all_variables)} coded → {out_path}")

    print(f"\nDone. {len(sources_to_process)} file(s) written to {output_dir}/",
          file=sys.stderr)


if __name__ == "__main__":
    main()
