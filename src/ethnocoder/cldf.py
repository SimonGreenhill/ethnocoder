import json
import re
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pycldf

TABLE_FILES = {
    "parameters": ("ParameterTable", "parameters.csv"),
    "codes": ("CodeTable", "codes.csv"),
}


def strip_pages(source_key: str) -> str:
    return re.sub(r"\[.*?\]", "", source_key).strip()


def find_metadata(dataset_dir: Path) -> Path:
    matches = list(dataset_dir.rglob("*-metadata.json"))
    if not matches:
        sys.exit(f"Error: no CLDF *-metadata.json found under {dataset_dir}")
    cldf_matches = [m for m in matches if m.parent.name == "cldf"]
    return cldf_matches[0] if cldf_matches else matches[0]


def load_dataset(dataset_dir: Path) -> pycldf.Dataset:
    return pycldf.Dataset.from_metadata(find_metadata(dataset_dir))


def build_source_index(value_rows: Iterable[dict]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = defaultdict(list)
    for row in value_rows:
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

    return [{"id": var["ID"], "code": by_param.get(var["ID"])} for var in all_variables]


def copy_table(ds: pycldf.Dataset, which: str, dest_dir: Path) -> Path:
    table_name, filename = TABLE_FILES[which]
    table = ds[table_name]
    src = Path(ds.directory) / cast(Any, table).url.string
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / filename
    shutil.copy2(src, dst)
    return dst


def write_gold(ds: pycldf.Dataset, out_dir: Path) -> int:
    all_variables = list(cast(Iterable[dict], ds["ParameterTable"]))
    code_names = {row["ID"]: row["Name"] for row in cast(Iterable[dict], ds["CodeTable"])}
    source_index = build_source_index(cast(Iterable[dict], ds["ValueTable"]))

    gold_dir = out_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    for key in sorted(source_index):
        codings = codings_for_source(source_index[key], all_variables, code_names)
        out_path = gold_dir / f"{key}.json"
        out_path.write_text(json.dumps(codings, indent=2), encoding="utf-8")
        n_coded = sum(1 for c in codings if c["code"] is not None)
        print(f"{key:40s}  {n_coded:3d}/{len(all_variables)} coded → {out_path}")
    return len(source_index)


def list_sources(ds: pycldf.Dataset) -> None:
    source_index = build_source_index(cast(Iterable[dict], ds["ValueTable"]))
    for key in sorted(source_index):
        rows = source_index[key]
        n_params = len({r["Parameter_ID"] for r in rows})
        n_societies = len({r["Language_ID"] for r in rows})
        print(f"{key:40s}  {n_params:3d} params  {n_societies:3d} societies")
    print(f"\nTotal: {len(source_index)} sources", file=sys.stderr)
