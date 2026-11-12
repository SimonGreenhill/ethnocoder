import json
from datetime import datetime
import sys
import os
from rich import print
from pathlib import Path
import pandas as pd
import csv

OLLAMA_URL = "http://127.0.0.1:11434"
GRAMBANK_CLDF_DIR = Path(os.path.join("dataset", "grambank-v3.0rc1", "cldf"))
CONTEXT_BUDGET_DEFAULT = 262144  # quarter of a million

def get_timestamp() -> str:
    """
    Get the current timestamp as a string in the format YYYYMMDD_HHMMSS.

    Returns:
        The current timestamp as a string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def strip_fences(text: str) -> str:
    """
    Strip code fences (```) from the start and end of the text, if present.

    Args:
        text: The text to strip code fences from.

    Returns:
        The text with code fences removed.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
    if text.endswith("```"):
        text = text[:text.rindex("```")]
    return text.strip()


def parse_codings(text: str) -> list[dict]:
    """
    Parse LLM response text (fences already stripped) into a list of coding dicts.

    Handles: double-brace bug, raw_response wrapper.

    Args:
        text: The LLM response text with code fences already stripped.

    Returns:
        A list of coding dictionaries parsed from the response.
    """

    # Double-brace bug
    if text.startswith("{{"):
        text = text[1:]

    # Raw response wrapper handling
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:

        # Warn the user
        print("[yellow]Warning: Failed to parse JSON from LLM response.[/yellow]", file=sys.stderr)

        raw = [{"message": "Invalid JSON returned, see raw text response for details."}]

    # If the raw response is a dict with a "raw_response" key but no "codings" key, parse the raw response.
    if isinstance(raw, dict) and "raw_response" in raw and "codings" not in raw:
        raw = json.loads(strip_fences(raw["raw_response"]))

    # Return the parsed codings, either as a list or from the "codings" key if present.
    return raw if isinstance(raw, list) else raw.get("codings", [])

def convert_model_output_to_tsv(
        fpath_model_output_json: str,
        fpath_output_tsv: str,
)->None:
    """
    Convert model output from JSON to TSV format.

    Args:
        fpath_model_output_json: Path to the JSON file containing model output.
        fpath_output_tsv: Path to the output TSV file.

    Returns:
        None
    """

    # Get JSON file
    with open(fpath_model_output_json, 'r', encoding='utf-8') as f:
        model_output_data = json.load(f)

    # Convert JSON to DataFrame
    df_model_output = pd.DataFrame(model_output_data['codings'])

    # Fix up data e.g. newlines with \n
    def escape_tsv_value(value):
        if isinstance(value, str):
            # Escape control characters, quotes, and backslashes; keep Unicode readable.
            return json.dumps(value, ensure_ascii=False)[1:-1]
        return value

    # Escape all weird values so it outputs properly
    for column in df_model_output.columns:
        df_model_output[column] = df_model_output[column].map(escape_tsv_value)

    # Output TSV
    df_model_output.to_csv(fpath_output_tsv, sep="\t", index=False)

    # Report success
    print(f"[green]Successfully converted model output to TSV: {fpath_output_tsv}[/green]")


def load_variables(path: Path) -> list[dict]:
    """
    Load variables data from a CSV file.

    Args:
        path (Path): The path to the CSV file containing variables data.

    Returns:
        list[dict]: A list of dictionaries representing the variables.
    """

    print(f"[blue]Loading variables from [bright_cyan]{path}[/bright_cyan][/blue]")

    # Open the CSV file and read its contents into a list of dictionaries.
    with open(path, encoding="utf-8") as f:
        ls_variables = list(csv.DictReader(f))

    # Process: if there are binarised variables, remove the corresponding multistate variable.
    # Binarised variables are those whose value for "ID" ends with a lower-case alphabetic character, NOT a number.
    binarised_ids = {var["ID"] for var in ls_variables if var["ID"][-1].isalpha() and var["ID"][-1].islower()}

    # Specify the variables to omit: first remove the final character from each binarised ID to get the corresponding multistate ID
    multistate_ids_to_omit = {var_id[:-1] for var_id in binarised_ids}

    # Filter out the multistate variables that have corresponding binarised versions
    ls_variables = [var for var in ls_variables if var["ID"] not in multistate_ids_to_omit]

    # Inform the user
    print(f"  [yellow]Removed multistate variables with corresponding binarised versions: {', '.join(multistate_ids_to_omit)}[/yellow]")

    # Return processed variables
    return ls_variables

def load_codes(
        path: Path,
        bool_use_question_mark: bool = True,
    ) -> dict[str, list[dict]]:
    """
    Load codes data from a CSV file and organize them by variable ID.

    Args:
        path (Path): The path to the CSV file containing codes data.
        bool_use_question_mark (bool): Whether to include the "?" code as valid, to mean "no evidence found". Defaults to True.

    Returns:
        dict[str, list[dict]]: A dictionary mapping variable IDs to lists of code dictionaries.
    """

    # Initialise the dictionary to store codes by variable ID
    codes_by_var: dict[str, list[dict]] = {}

    # Open the CSV file and read its contents
    with open(path, encoding="utf-8") as f:

        # Iterate over each row in the CSV and organize codes by variable ID
        for row in csv.DictReader(f):

            # If row["Parameter_ID"] is not already a key in the dictionary, add it with an empty list
            # and then append the current row.
            codes_by_var.setdefault(row["Parameter_ID"], []).append(row)

    # If bool_use_question_mark is True, add the "?" code to each variable's list of codes if it's not already present.
    if bool_use_question_mark:
        for var_id, codes in codes_by_var.items():
            if not any(code["Name"] == "?" for code in codes):
                codes.append({"ID": f"{var_id}-?", "Parameter_ID": var_id, "Name": "?", "Description": "No evidence found"})

    return codes_by_var