# Convert JSON files to TSV and back.

import json
from rich import print
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def json_to_tsv(
        json_file: Path,
        tsv_file: Path
    )->None:
    """
    Convert a JSON file to a TSV file.

    Input entries to use are id, code, confidence, justification, quote, pages.
    Output headers are Parameter_ID, Value, Confidence, Justification, Quote, Pages.

    Args:
        json_file (Path): Path to the input JSON file.
        tsv_file (Path): Path to the output TSV file.
    """

    # Read the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Write the TSV file
    with open(tsv_file, 'w', encoding='utf-8') as f:
        # Write the header
        f.write("Parameter_ID\tValue\tConfidence\tJustification\tQuote\tPages\n")
        # Write the data
        for entry in tqdm(data['codings'], desc="Converting JSON to TSV"):
            f.write(f"{entry.get('id','')}\t{entry.get('code','')}\t{entry.get('confidence','')}\t{entry.get('justification','')}\t{entry.get('quote','')}\t{entry.get('pages','')}\n")

    print(f"[green]Converted {json_file} to {tsv_file}[/green]")

def tsv_to_json(
        tsv_file: Path,
        json_file: Path
)->None:
    """
    Convert a TSV file to a gold-standard JSON file.
    The TSV file must at least have Parameter_ID and Value columns.
    The JSON file will map these to the dicts in a list, each of which has keys "id" and "code" respectively.

    Args:
        tsv_file (Path): Path to the input TSV file.
        json_file (Path): Path to the output JSON file.
    """
    
    # Read the TSV file with pandas
    df = pd.read_csv(tsv_file, sep='\t')

    # Initialise list for JSON output
    data = []

    # Get each row and convert to JSON format
    for _, row in df.iterrows():
        data.append({
            "id": row.get("Parameter_ID", ""),
            "code": row.get("Value", "")
        })

    # Write the JSON file
    # Because it's in "gold" format, it doesn't have the "codings" wrapper around the list
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"[green]Converted {tsv_file} to {json_file}[/green]")


if __name__ == "__main__":

    # Create argparser for input JSON and output TSV files
    import argparse
    parser = argparse.ArgumentParser(description="Convert JSON to TSV")

    # Argument: are we running JSON-to-TSV or TSV-to-JSON?
    parser.add_argument("-m", "--mode", type=str, choices=["json_to_tsv", "tsv_to_json"], default="json_to_tsv", help="Conversion mode: json_to_tsv or tsv_to_json")

    parser.add_argument("-j", "--json_file", type=str, help="Path to the input JSON file")
    parser.add_argument("-t", "--tsv_file", type=str, help="Path to the output TSV file")
    args = parser.parse_args()

    if args.mode == "json_to_tsv":
        json_to_tsv(json_file = Path(args.json_file), tsv_file = Path(args.tsv_file))
    else:
        tsv_to_json(tsv_file = Path(args.tsv_file), json_file = Path(args.json_file))