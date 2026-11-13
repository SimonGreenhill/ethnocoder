# PREPARE
# 1. Take an input directory.
# 2. It must have subdirectories, each containing a single document (either PDF or plaintext) and a JSON gold-coding file.
# 3. Get the glottocode from the JSON gold-coding file: should be the first 8 characters, before the first underscore.
# 4. Get the canonical name of that language, using pycldf and GRAMBANK_CLDF_DIR.
# 5. If everything looks good, create a dedicated output batch directory for this run: output/<DATE>_<BATCH_NUMBER>.

# RUN
# 1. Use ethnocoder.models.code.Coder to code each document specifically for the language given by the canonical name and glottocode.
# 2. Place each model output coding in a dedicated output directory, along with a copy of the gold coding.
# 3. Create the summary TSV for each document, using methods from check_single.py.
# 4. Create the overall summary TSV aggregating the results from all documents, using methods from collate_results.py.

import os
import argparse
import sys
from rich import print
from collections.abc import Iterable
from typing import cast
from pathlib import Path

from ethnocoder.datasets import cldf
from ethnocoder.common.utils import get_timestamp, GRAMBANK_CLDF_DIR, CONTEXT_BUDGET_DEFAULT
from ethnocoder.models.code import Coder
from ethnocoder.results.check_single import main as check_single_main
from ethnocoder.results.collate_results import collate_results
from ethnocoder.results.get_results_stats import summarise_tsv_results, plot_accuracy_per_stats

DIR_OUT = "output"

def prepare(
        input_directory: str,
        output_batch_directory: str | None = None,
) -> tuple[list[dict] | None, str | None]:
    """
    Prepare the input data for batch processing.

    Args:
        input_directory (str): Path to the input directory containing subdirectories with documents and gold-coding JSON files.
        output_batch_directory (str | None): Path to the output batch directory for this run. If None, a new directory will be created.
    Returns:
        list[dict]: A list of dictionaries containing information about each subdirectory, including glottocode and canonical language name.
                    Each dictionary has the keys: "subdir_path", "fname_document", "fname_gold", "glottocode", and "language_name".
                    If any subdirectory does not meet the requirements, we error and return None.
        "output_batch_directory": str | None, Path to the output batch directory for this run.
    """

    # Load the CLDF dataset.
    print(f"[blue]Loading CLDF dataset from {GRAMBANK_CLDF_DIR}...[/blue]")
    ds = cldf.load_dataset(GRAMBANK_CLDF_DIR)

    # Check it has a LanguageTable
    if "LanguageTable" in ds:
        language_names = {row["ID"]: row["Name"] for row in cast(Iterable[dict], ds["LanguageTable"])}
        print(f"[green]Found LanguageTable in CLDF dataset.[/green]")
    else:
        print(f"[red]LanguageTable not found in CLDF dataset.[/red]")
        return None, None
    
    # Initialise return list
    retlist = []

    # Check that ALL subdirectories of the input directory have the following format:
    # each containing a single document (either PDF or plaintext) and a JSON gold-coding file.
    print(f"[blue]Checking subdirectories in input directory: {input_directory}...[/blue]")
    for subdir in os.listdir(input_directory):

        # Get the subdirectory
        subdir_path = os.path.join(input_directory, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # Get the files
        files = os.listdir(subdir_path)

        # There should be exactly 2.
        if len(files) != 2:
            print(f"  [red]Subdirectory {subdir_path} does not contain exactly two files.[/red]")
            return None, None

        # There should be a JSON gold-coding file and a document file (PDF or plaintext)
        if not any(f.endswith(".json") for f in files):
            print(f"  [red]Subdirectory {subdir_path} does not contain a JSON gold-coding file.[/red]")
            return None, None
        if not any(f.endswith(".pdf") or f.endswith(".txt") for f in files):
            print(f"  [red]Subdirectory {subdir_path} does not contain a document file (PDF or plaintext).[/red]")
            return None, None

        # Get the glottocode from the JSON gold-coding file: should be the first 8 characters, before the first underscore.
        json_file = next((f for f in files if f.endswith(".json")), None)
        if json_file:
            glottocode = json_file.split("_")[0]

            # Should be EXACTLY 8 characters long
            if len(glottocode) != 8:
                print(f"  [red]Glottocode for subdirectory {subdir_path} is not exactly 8 characters long: {glottocode}[/red]")
                return None, None

            # print(f"  [green]Glottocode for subdirectory {subdir_path}: {glottocode}[/green]")
        else:
            print(f"  [red]No JSON gold-coding file found in subdirectory {subdir_path}[/red]")
            return None, None

        # Get the canonical language name for this glottocode
        # from ds["LanguageTable"] get the ID which is this glottocode, then get the Name column.
        if glottocode:
            language_name = language_names.get(glottocode)
            if not language_name:
                print(f"  [red]No canonical language name found for glottocode {glottocode} in subdirectory {subdir_path}[/red]")
                return None, None
        else:
            print(f"  [red]No glottocode found for subdirectory {subdir_path}[/red]")
            return None, None

        # Initialise lgcode as None
        lgcode = None

        # Use pycldf dataset, sources object.
        # The entry.id of each entry in ds.sources is the bibtex key, which here should be subdir.
        # We can look for a matching entry and extract the lgcode if available.
        bibtex_key = subdir
        if bibtex_key in ds.sources:
            entry = ds.sources[bibtex_key]
            if entry.get("lgcode", None):
                lgcode = entry.get('lgcode')
            else:
                print(f"  [yellow]No lgcode found in entry for bibtex key {bibtex_key}[/yellow]")
                print(entry)
        else:
            print(f"  [yellow]No entry found in ds.sources for bibtex key {bibtex_key}[/yellow]")

        # Append info to retlist
        retlist.append({
            "subdir_path": subdir_path,
            "fname_document": next((f for f in files if f.endswith(".pdf") or f.endswith(".txt")), None),
            "fname_gold": json_file,
            "glottocode": glottocode,
            "language_name": language_name,
            "lgcode": lgcode,
        })

    # If everything looks good, create a dedicated output batch directory for this run: output/<DATE>_<BATCH_NUMBER>.
    # Get timestamp
    timestamp = get_timestamp()

    # Get batch_id: the exact name of the final subdirectory in the input directory
    batch_id = os.path.basename(os.path.normpath(input_directory))

    # Build the output batch directory path and create the directory if it doesn't exist.
    if output_batch_directory is None:
        output_batch_directory = f"output/{timestamp}_{batch_id}"
    os.makedirs(output_batch_directory, exist_ok=True)

    print(f"[green]Preparation complete. Output batch directory: {output_batch_directory}[/green]")

    return retlist, output_batch_directory

def run_batch(
        input_directory: str,
        output_batch_directory: str,
        retlist: list[dict],
        args: argparse.Namespace,
) -> None:
    """
    Run the batch processing on the prepared input data.
    For each entry in retlist:
    - Create a Coder instance, passing in the model arguments (model name, api key etc), dataset arguments (parameters and codes),
    the language name and the document path.
    - Run coder.run() to process the document with the provided codes and parameters.

    Args:
        input_directory (str): Path to the input directory containing subdirectories with documents and gold-coding JSON files.
        output_batch_directory (str): Path to the output batch directory where results should be saved.
        retlist (list[dict]): List of dictionaries containing prepared data for the batch run. See prepared() docstring for details.
        args (argparse.Namespace): Command-line arguments parsed by argparse. Contains model arguments, dataset arguments, and other options for the Coder instance.
    """

    # Remove args.input_directory and args.prepare_only as they are no longer needed
    del args.input_directory
    del args.prepare_only

    # For each entry in retlist, create a Coder instance and run it
    for entry in retlist:

        print(f"[blue]Processing document for language {entry['language_name']} ({entry['glottocode']})...[/blue]")

        # Get the glottocode for convenience
        glottocode = entry["glottocode"]

        # The output directory must be added to the args namespace as out_dir,
        # so use our batch output and a subdirectory with the glottocode name.
        args.out_dir = os.path.join(output_batch_directory, glottocode)

        # The language name should be args.language
        args.language = entry["language_name"]
        args.lgcode = entry["lgcode"]

        # Specify either the pdf or text document path for the Coder instance
        if "pdf" in entry["fname_document"].lower():
            args.pdf = os.path.join(entry["subdir_path"], entry["fname_document"])
        elif "txt" in entry["fname_document"].lower():
            args.text = os.path.join(entry["subdir_path"], entry["fname_document"])
        else:
            raise ValueError(f"Unsupported document type for {entry['fname_document']}")

        # Other arg prep
        args.print_prompt = False
        args.dump = False
        args.doc_size = False
        args.bool_debug_batches = False

        # Create a Coder instance and run it
        coder = Coder(
            args = args,
        )

        # Run it.
        coder.run()

        # Inform the user that this particular document has been processed
        print(f"  [green]Processed document for language {args.language} ({glottocode}).[/green]")

    # Inform the user that the batch processing is complete
    print("[green]Batch processing complete.[/green]")


def check_per_output(
        output_batch_directory,
        retlist,
):
    """
    Step through each subdirectory, find the appropriate JSON model output file,
    run check_single_main using this and the corresponding gold-coding JSON file.
    """

    # 1. Step through each subdirectory in the output batch directory
    for subdir in os.listdir(output_batch_directory):

        # Get this subdir's retlist entry.
        retlist_entry = None
        for entry in retlist:
            if entry["glottocode"] == subdir:
                retlist_entry = entry
                break

        if retlist_entry is None:
            if subdir != "summary":
                print(f"[yellow]No retlist entry found for subdir {subdir}[/yellow]")
            continue

        subdir_path = os.path.join(output_batch_directory, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # 2. Find the appropriate JSON model output file
        json_files = [f for f in os.listdir(subdir_path) if f.endswith(".json")]
        if not json_files:
            print(f"[yellow]No JSON output file found in {subdir_path}[/yellow]")
            continue

        # Find the numbers at the end of the JSON filename.
        # The format is: <fname_document>_<number>.json
        dict_file_by_number = {}
        for f in json_files:

            # If it doesn't start with the expected document name, skip it
            if not f.startswith(retlist_entry["fname_document"].replace(".pdf","").replace(".txt","") + "_"):
                continue

            # Get the number, save to a dict with the number as the key and filename as the value
            number = int(f.split("_")[-1].split(".")[0])
            dict_file_by_number[number] = f

        # Get the highest number to identify the latest JSON output file
        if not dict_file_by_number:
            print(f"[yellow]No valid JSON output file found in {subdir_path}[/yellow]")
            continue
        latest_number = max(dict_file_by_number.keys())
        latest_json_file = dict_file_by_number[latest_number]

        # Now build the path
        fpath_model_output = os.path.join(subdir_path, latest_json_file)

        # The gold file comes from the entry in retlist whose glottocode is just this subdir
        fpath_gold = None
        for entry in retlist:
            if entry["glottocode"] == subdir:
                fpath_gold = os.path.join(entry["subdir_path"], entry["fname_gold"])
                break
        if fpath_gold is None:
            print(f"[yellow]No gold file found for glottocode {subdir}[/yellow]")
            continue
        
        # 4. Run check_single_main
        check_single_main(
            fpath_gold = Path(fpath_gold),
            fpath_model_output = Path(fpath_model_output),
            bool_output_to_terminal = False,
        )


def summarise_results(output_batch_directory, retlist, args):

    # Check per output: creates results_all.tsv in each model results subdirectory.
    check_per_output(
        output_batch_directory = output_batch_directory,
        retlist = retlist,
    )

    # Create the summary output directory path.
    summary_output_directory = os.path.join(output_batch_directory, "summary")
    os.makedirs(summary_output_directory, exist_ok=True)

    # Collate results from each subdirectory into a single summary file
    collate_results(
        main_directory = output_batch_directory,
        target_directory = summary_output_directory,
    )

    # Finally, create the full summary TSV using get_results_stats_main
    summarise_tsv_results(
        directory = summary_output_directory,
        bool_output_to_terminal = False,
    )

    # Optionally, plot accuracy per stats if needed
    plot_accuracy_per_stats(
        directory = output_batch_directory,
    )

def main(
        args: argparse.Namespace,
):

    # Prepare: check data is consistent and ready to be run in batch.
    retlist, output_batch_directory = prepare(
        input_directory = args.input_directory,
        output_batch_directory = args.output_batch_directory,
    )

    # Sanity check
    if retlist is None or output_batch_directory is None:
        print("[red]Failed to prepare input data.[/red]")
        sys.exit(1)

    # Run if required.
    if retlist and not args.prepare_only:

        # Prepare kwargs for running models
        kwargs = {
            key: value
            for key, value in vars(args).items()
            if key not in {"input_directory", "prepare_only"}
        }

        # Call the function
        run_batch(
            input_directory = args.input_directory,
            output_batch_directory = output_batch_directory,
            retlist = retlist,
            args = args,
        )

    # Summarise the results after running the batch
    if output_batch_directory is not None:
        summarise_results(
            output_batch_directory = output_batch_directory,
            retlist = retlist,
            args = args,
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare and run batch processing for ethnocoder.")

    # Directories
    parser.add_argument("-i", "--input_directory", type=str, help="Path to the input directory containing subdirectories with documents and gold-coding JSON files.")

    # Output batch directory
    parser.add_argument("-o", "--output-batch-directory", default=None, help="Path to the output batch directory for this run. If None, a new directory will be created.")

    # NB DO NOT use -p as it clashes with -p in code.py, which this script passes args to.
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare the input data without running the batch.")

    # Params for coder
    # model: the LLM to use for coding the document
    parser.add_argument("-m", "--model", default=None, help="LiteLLM model string (required)")

    # variables: CSV file containing the variables to code
    parser.add_argument("--variables", default="parameters.csv", help="Variables CSV (default: parameters.csv)")

    # codes: CSV file containing the codes for the variables
    parser.add_argument("--codes", default="codes.csv", help="Codes CSV (default: codes.csv)")

    # ids: Comma-separated list of variable IDs to code
    parser.add_argument("--ids", help="Comma-separated list of variable IDs to code (e.g. 2,3,5)")

    # api-base: Override API base URL for the LLM
    parser.add_argument("--api-base", default=None, help="Override API base URL")

    # max-chars: Truncate document text to this many characters
    parser.add_argument("--max-chars", type=int, default=None, help="Truncate document text to this many characters")

    # context-budget: Token budget per request
    parser.add_argument(
            "--context-budget",
            type=int,
            default=CONTEXT_BUDGET_DEFAULT,
            help="Token budget per request. When set, variables are coded in batches that fit "
            "this budget (reusing a cached document prefix). Omit for a single request with all variables.",
        )
    
        # response-reserve-per-var: Estimated output tokens reserved per variable when packing batches
    parser.add_argument(
            "--response-reserve-per-var",
            type=int,
            default=500,
            help="Estimated output tokens reserved per variable when packing batches (default: 500)",
        )

    # bool-review: Perform a review of the coding results where necessary
    parser.add_argument("--bool-review", action="store_true", help="Perform a review of the coding results where necessary")
    
        # bool-thinking: Enable thinking mode for the LLM, if allowed by the model
    parser.add_argument("--bool-thinking", action="store_true", help="Enable thinking mode for the LLM, if allowed by the model")

    # temperature: Set the temperature for the LLM
    # Default is currently 0.4 because we want more deterministic rather than more creative.
    parser.add_argument("--temperature", type=float, default=0.4, help="Set the temperature for the LLM")
    
    # top-p: Set the top-p (nucleus sampling) for the LLM
    # Default is 0.95, recommended by SAIA for GLM 5.3 Flash.
    parser.add_argument("--top-p", type=float, default=0.95, help="Set the top-p (nucleus sampling) for the LLM")

    args = parser.parse_args()

    main(args=args)