import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pycldf

DATASET_DIR = Path("./dataset")


def strip_pages(source_key: str) -> str:
    return re.sub(r"\[.*?\]", "", source_key).strip()


def find_metadata(dataset_dir: Path) -> Path:
    matches = list(dataset_dir.rglob("*-metadata.json"))
    if not matches:
        sys.exit(f"Error: no CLDF *-metadata.json found under {dataset_dir}")
    cldf_matches = [m for m in matches if m.parent.name == "cldf"]
    return cldf_matches[0] if cldf_matches else matches[0]


def build_source_index(ds: pycldf.Dataset) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = defaultdict(list)
    for row in ds["ValueTable"]:
        for src in row["Source"]:
            key = strip_pages(src)
            if key:
                index[key].append(row)
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
            by_param[param_id] = str(row["Value"])
        else:
            by_param[param_id] = None

    return [
        {"id": var["ID"], "code": by_param.get(var["ID"])}
        for var in all_variables
    ]


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "setup",
        description="Set up working files from a CLDF dataset",
        help="Set up working files from a CLDF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", type=Path, default=DATASET_DIR,
        help=f"Path to dataset root (default: {DATASET_DIR})")
    p.add_argument(
        "-o", "--output", default="gold",
        help="Output directory for gold files (default: gold/)")
    p.add_argument(
        "--list", action="store_true",
        help="List all source keys and exit")
    p.set_defaults(func=_run)


def _run(args) -> None:
    metadata_path = find_metadata(args.dataset)
    ds = pycldf.Dataset.from_metadata(metadata_path)

    if not args.list:
        for table, dst_name in (
            (ds["ParameterTable"], "parameters.csv"),
            (ds["CodeTable"], "codes.csv"),
        ):
            src = ds.directory / table.url.string
            dst = Path(dst_name)
            shutil.copy2(src, dst)
            print(f"Copied {src} → {dst}")

    all_variables = list(ds["ParameterTable"])
    code_names = {row["ID"]: row["Name"] for row in ds["CodeTable"]}
    source_index = build_source_index(ds)

    if args.list:
        for key in sorted(source_index):
            rows = source_index[key]
            n_params = len({r["Parameter_ID"] for r in rows})
            n_societies = len({r["Language_ID"] for r in rows})
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
