import json
import re
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast
from tqdm import tqdm
from rich import print

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
                index[f"{row['Language_ID']}_{key}"].append(row)
    return index


def codings_for_source(
    rows: list[dict],
    all_variables: list[dict],
    code_names: dict[str, str],
) -> list[dict]:
    """
    Get the codings for a specific source.
    Converts CLDF values to a list of dicts, for outputting as a gold-standard file.

    Args:
        rows: The list of value rows for a specific source.
        all_variables: The list of all parameter rows from the dataset.
        code_names: A mapping from code IDs to their values. i.e. from "GB023-1" to "1".

    Returns:
        A list of dictionaries, each containing the parameter ID and its corresponding code for the source.
    """

    # Initialise the dictionary to store codings, Comment, Source, Source_comment, Coders
    by_param: dict[
        str, # param ID
        list[
            str | None, # coding value
            str | None, # comment
            str | None, # source
            str | None, # source comment
            str | None, # coders
        ]
    ] = {}

    # Iterate over each row for this source and populate the codings.
    # Not sure why we build by_param first and the return list of dicts later?
    # Is it more efficient somehow?
    # Maybe because we build retlist from all_variables.
    for row in rows:

        # The parameter ID for this row.
        param_id = row["Parameter_ID"]

        # If there is a code ID, use the corresponding value.
        if row["Code_ID"]:
            by_param[param_id] = [code_names.get(row["Code_ID"], row["Code_ID"]), None, None, None, None]

        # Otherwise get the value from the row itself, if it exists.
        elif row["Value"] not in ("", None):
            by_param[param_id] = [str(row["Value"]), None, None, None, None]

        # If there's no value and no idea, yet there is a row entry, give a "?"
        elif row["Value"] is None:
            by_param[param_id] = ["?", None, None, None, None]

        # If there's nothing at all, don't code this variable.
        else:
            by_param[param_id] = [None, None, None, None, None]

        # Get comment, source, source comment, and coders if they exist in the row.
        by_param[param_id][1] = row.get("Comment", None)
        by_param[param_id][2] = row.get("Source", None)
        by_param[param_id][3] = row.get("Source_comment", None)
        by_param[param_id][4] = row.get("Coders", None)

        # Join Source and Coders by semicolon if necessary.
        if by_param[param_id][2] is not None:
            by_param[param_id][2] = "; ".join(by_param[param_id][2])
        if by_param[param_id][4] is not None:
            by_param[param_id][4] = "; ".join(by_param[param_id][4])

    # Build the return list of dicts.
    retlist = [
        {
            "id": var["ID"], 
            "code": by_param.get(var["ID"], [None, None, None, None, None])[0],
            "comment": by_param.get(var["ID"], [None, None, None, None, None])[1],
            "source": by_param.get(var["ID"], [None, None, None, None, None])[2],
            "source_comment": by_param.get(var["ID"], [None, None, None, None, None])[3],
            "coders": by_param.get(var["ID"], [None, None, None, None, None])[4],
        } 
        for var in all_variables
    ]

    # Return a list of dicts.
    return retlist


def copy_table(ds: pycldf.Dataset, which: str, dest_dir: Path) -> Path:
    table_name, filename = TABLE_FILES[which]
    table = ds[table_name]
    src = Path(ds.directory) / cast(Any, table).url.string
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / filename
    shutil.copy2(src, dst)
    return dst


def write_gold(ds: pycldf.Dataset, out_dir: Path, sources: list[str] | None = None) -> int:
    """
    Write gold-standard coding files for the given dataset.
    Either the entire dataset or just the specified sources will be written.

    Args:
        ds: The CLDF dataset to process.
        out_dir: The directory where gold-standard files will be written.
        sources: Optional list of source keys to include. If None, all sources are included.

    Returns:
        The number of gold-standard files written.
    """

    # Extract all variables and code names from the dataset.
    all_variables = list(cast(Iterable[dict], ds["ParameterTable"]))
    code_names = {row["ID"]: row["Name"] for row in cast(Iterable[dict], ds["CodeTable"])}

    # Inform the user we are building the source index, because it can take a little while
    print("[blue]Building source index...[/blue]")
    source_index = build_source_index(cast(Iterable[dict], ds["ValueTable"]))

    # Set up the output directory
    gold_dir = out_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    # Initialise results count
    n_written = 0

    # Loop at each source
    for key in tqdm(sorted(source_index), desc="Writing gold files"):

        # If we're not writing this source, continue.
        if sources and not any(src in key for src in sources):
            continue

        # Get all codings for this source.
        # If "code" is None, skip it.
        codings = [
            c for c in codings_for_source(source_index[key], all_variables, code_names)
            if c["code"] is not None
        ]

        # Define the path for this gold file.
        out_path = gold_dir / f"{key}.json"

        # Write this gold file
        out_path.write_text(json.dumps(codings, indent=2), encoding="utf-8")

        # Print a summary of what was written for this source.
        print(f"  [blue]{key:40s}  {len(codings):3d}/{len(all_variables)} coded → {out_path}[/blue]")

        # Increment the count of written files.
        n_written += 1

    # Return the total number of gold files written.
    return n_written


def list_sources(ds: pycldf.Dataset) -> None:
    source_index = build_source_index(cast(Iterable[dict], ds["ValueTable"]))
    for key in sorted(source_index):
        rows = source_index[key]
        n_params = len({r["Parameter_ID"] for r in rows})
        n_societies = len({r["Language_ID"] for r in rows})
        print(f"{key:40s}  {n_params:3d} params  {n_societies:3d} societies")
    print(f"\nTotal: {len(source_index)} sources", file=sys.stderr)
