import json
import re
import shutil
import sys
import copy
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

# Formerly multistate variables must be binarised when writing gold files.
# This object denotes how multistate variables ought to be converted to binary variables.
# Of course, if the binary variables already exist for the specified source and language,
# they take precedence over the corresponding multistate variable.
GRAMBANK_MULTISTATE_MAPPINGS = {
    "GB024": { # GB024: What is the order of numeral and noun in the NP? 1=Num-N, 2=N-Num, 3=both
        "binaries": ["GB024a", "GB024b"],
        "1": {"GB024a": "1", "GB024b": "0"}, # GB024a: Is the order of the numeral and noun Num-N?
        "2": {"GB024a": "0", "GB024b": "1"}, # GB024b: Is the order of the numeral and noun N-Num?
        "3": {"GB024a": "1", "GB024b": "1"}, # both
        "?": {"GB024a": "?", "GB024b": "?"}, # unknown
    },
    "GB025": { # GB025: What is the order of adnominal demonstrative and noun? 1=Dem-N, 2=N-Dem, 3=both
        "binaries": ["GB025a", "GB025b"],
        "1": {"GB025a": "1", "GB025b": "0"}, # GB025a: Is the order of adnominal demonstrative and noun Dem-N?
        "2": {"GB025a": "0", "GB025b": "1"}, # GB025b: Is the order of adnominal demonstrative and noun N-Dem?
        "3": {"GB025a": "1", "GB025b": "1"}, # both
        "?": {"GB025a": "?", "GB025b": "?"}, # unknown
    },
    "GB065": { # GB065: What is the pragmatically unmarked order of adnominal possessor noun and possessed noun? 1=Possessor-Possessed, 2=Possessed-Possessor, 3=both
        "binaries": ["GB065a", "GB065b"],
        "1": {"GB065a": "1", "GB065b": "0"}, # GB065a: Is the pragmatically unmarked order of adnominal possessor noun and possessed noun PSR-PSD?
        "2": {"GB065a": "0", "GB065b": "1"}, # GB065b: Is the pragmatically unmarked order of adnominal possessor noun and possessed noun PSD-PSR?
        "3": {"GB065a": "1", "GB065b": "1"}, # both
        "?": {"GB065a": "?", "GB065b": "?"}, # unknown
    }, 
    "GB130": { # GB130: What is the pragmatically unmarked order of S and V in intransitive clauses? 1=SV, 2=VS, 3=both
        "binaries": ["GB130a", "GB130b"],
        "1": {"GB130a": "1", "GB130b": "0"}, # GB130a: Is the pragmatically unmarked order of S and V in intransitive clauses S-V?
        "2": {"GB130a": "0", "GB130b": "1"}, # GB130b: Is the pragmatically unmarked order of S and V in intransitive clauses V-S?
        "3": {"GB130a": "1", "GB130b": "1"}, # both
        "?": {"GB130a": "?", "GB130b": "?"}, # unknown
    },
    "GB193": { # GB193: What is the order of adnominal property word and noun?: 1=ANM-N, 2=N-ANM, 3=both, 0=they cannot be used attributively
        "binaries": ["GB193a", "GB193b"],
        "0": {"GB193a": "0", "GB193b": "0"}, # They cannot be used attributively
        "1": {"GB193a": "1", "GB193b": "0"}, # GB193a: Is the order of the adnominal property word (ANM) and noun ANM-N?
        "2": {"GB193a": "0", "GB193b": "1"}, # GB193b: Is the order of the adnominal property word (ANM) and noun N-ANM?
        "3": {"GB193a": "1", "GB193b": "1"}, # both
        "?": {"GB193a": "?", "GB193b": "?"}, # unknown
    },
    "GB203": { # GB203: What is the order of the adnominal collective universal quantifier ('all') and the noun?: 1=UQ-N, 2=N-UQ, 3=both, 0=no UQ,
        "binaries": ["GB203a", "GB203b"],
        "0": {"GB203a": "0", "GB203b": "0"}, # No UQ
        "1": {"GB203a": "1", "GB203b": "0"}, # GB203a: Is the order of the adnominal collective universal quantifier (UQ) and noun UQ-N?
        "2": {"GB203a": "0", "GB203b": "1"}, # GB203b: Is the order of the adnominal collective universal quantifier (UQ) and noun N-UQ?
        "3": {"GB203a": "1", "GB203b": "1"}, # both
        "?": {"GB203a": "?", "GB203b": "?"}, # unknown
    },
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

    # Convert multistate variables to their corresponding binary variables,
    # if the binary variables don't already exist.
    # Then remove the multistate variables.
    for multistate_id in GRAMBANK_MULTISTATE_MAPPINGS.keys():

        # Figure out if the corresponding binary variables exist for this multistate variable.
        binaries = GRAMBANK_MULTISTATE_MAPPINGS[multistate_id]["binaries"] # we want it to error if it doesn't exist
        for binary_id in binaries:
            if binary_id not in by_param and multistate_id in by_param:

                # We need to add this binary variable.
                # Get the value from the multistate.
                # We want it to error if it doesn't exist.
                ls_multistate = copy.deepcopy(by_param[multistate_id])
                value_binary = GRAMBANK_MULTISTATE_MAPPINGS[multistate_id][str(ls_multistate[0])][binary_id]

                # Convert None's to empty strings for concatenation purposes.
                if ls_multistate[1] is None:
                    ls_multistate[1] = "[BINARISED FROM MULTISTATE]"
                else:
                    ls_multistate[1] += " [BINARISED FROM MULTISTATE]"
                if ls_multistate[4] is None:
                    ls_multistate[4] = "[BINARISED FROM MULTISTATE]"
                else:
                    ls_multistate[4] += " [BINARISED FROM MULTISTATE]"

                # Take the binary value, and fill in the other fields from the multistate value.
                by_param[binary_id] = [
                    value_binary, # code
                    ls_multistate[1], # comment
                    ls_multistate[2], # source
                    ls_multistate[3], # source_comment
                    ls_multistate[4], # coders
                ]

        # Remove any entries in by_param where "ID" is equal to one of the keys of GRAMBANK_MULTISTATE_MAPPINGS.
        if multistate_id in by_param:
            del by_param[multistate_id]

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
    print("[blue]Building source index (this can take a while)...[/blue]")
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

        # Total variables is all those in all_variables, minus the IDs in the keys of GRAMBANK_MULTISTATE_MAPPINGS.
        total_variables = len([v for v in all_variables if v["ID"] not in GRAMBANK_MULTISTATE_MAPPINGS])

        # Print a summary of what was written for this source.
        print(f"  [blue]{key:40s}  [green]{len(codings):3d}/{total_variables} coded → [spring_green2]{out_path}[/spring_green2][/green][/blue]")

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
