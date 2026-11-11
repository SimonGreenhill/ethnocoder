import json
from datetime import datetime
import sys
import os
from rich import print
from pathlib import Path
import pandas as pd

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