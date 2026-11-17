# PREPARE
# 1. Take an input directory.
# 2. It must have subdirectories, each containing a single document (either PDF or plaintext) and a JSON gold-coding file.
# 3. Get the glottocode from the JSON gold-coding file: should be the first 8 characters, before the first underscore.
# 4. Get the canonical name of that language, using pycldf and GRAMBANK_CLDF_DIR.
# 5. If everything looks good, create a dedicated output batch directory for this run: output/<DATE>_<BATCH_NUMBER>.

# RUN
# 1. Use ethnocoder.models.code.Coder to code per language, per model.
# 2. Place each model output coding in a dedicated output directory, along with a copy of the gold coding.
# 3. Create the summary TSV for each document, using methods from check_single.py.
# 4. Create the overall summary TSV aggregating the results from all documents, using methods from collate_results.py.

import os
import sys
import shutil
import argparse
import copy
import json
from rich import print
from pathlib import Path

from ethnocoder.common.utils import (
    get_timestamp, parse_csv_list, is_glottocode,
    CONTEXT_BUDGET_DEFAULT,
    FPATH_MODELS_METADATA,
)
from ethnocoder.models.code import Coder
from ethnocoder.datasets import cldf
from ethnocoder.common.utils import GRAMBANK_CLDF_DIR
from ethnocoder.plotting.plot_trials_summary import (
    collect_results, plot_results, FPATH_MAJORITY,
)
from ethnocoder.analysis.calculate_stats_for_batch import calculate_stats
from ethnocoder.plotting.plot_accuracy_time_reasoning import plot_accuracy_per_stats

# Find most recent batch input directory
BATCHES_DIR = Path(__file__).resolve().parents[3] / "docs" / "batches"

DIR_OUT = os.path.join("output", "batches")

LANGUAGE_NAMES_SPECIAL = {
    "simlish_explicit": "Simlish",
    "simlish": "Simlish",
    "mock_dutch": "Mock Dutch",
    "isla1278_124144": "Kalinago, aka Island Carib"
}

MODELS = (
    # "tiny",
    "small",
    "medium",
    "large",
)

def get_language_name_and_lgcode(
    language_identifier: str,
    json_file: str,
) -> tuple[str, str | None]:
    """
    Get the canonical language name and lgcode for a given language identifier and JSON file.

    Args:
        language_identifier (str): The identifier for the language (e.g., subdirectory name).
        json_file (str): The name of the JSON gold-coding file.

    Returns:
        tuple[str, str | None]: A tuple containing the canonical language name and the lgcode (if available).
    """
    
    ds = cldf.load_dataset(GRAMBANK_CLDF_DIR)

    if "LanguageTable" not in ds:
        raise ValueError("LanguageTable not found in CLDF dataset")

    if language_identifier in LANGUAGE_NAMES_SPECIAL:
        language_name = LANGUAGE_NAMES_SPECIAL[language_identifier]
    else:
        glottocode = json_file.split("_")[0]
        language_name = next(
            (row["Name"] for row in ds["LanguageTable"]
             if row["ID"] == glottocode),
            None,
        )
        if not language_name:
            raise ValueError(f"No language found for glottocode {glottocode}")

    lgcode = None
    glottocode, separator, source_id = Path(json_file).stem.partition("_")

    if separator and source_id and is_glottocode(glottocode):
        if source_id in ds.sources:
            lgcode = ds.sources[source_id].get("lgcode")
        else:
            print(f"[yellow]No CLDF source found for {source_id}[/yellow]")

    return language_name, lgcode

def prepare(
        input_directory: str,
        models: list[str],
        output_batch_directory: str | None = None,
        languages: list[str] | None = None,
) -> tuple[list[dict] | None, str | None]:
    """
    Prepare the input data for batch processing.

    Args:
        input_directory (str): Path to the input directory containing subdirectories with documents and gold-coding JSON files.
        output_batch_directory (str | None): Path to the output batch directory for this run. If None, a new directory will be created.
        languages (list[str] | None): Input folder names or glottocodes to include.
    Returns:
        list[dict]: A list of dictionaries containing information about each subdirectory, including glottocode and canonical language name.
                    Each dictionary has the keys: "subdir_path", "fname_document", "fname_gold", "glottocode", and "language_name".
                    If any subdirectory does not meet the requirements, we error and return None.
        "output_batch_directory": str | None, Path to the output batch directory for this run.
    """
    
    # Initialise return list
    retlist = []
    matched_languages = set()

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

        if languages is not None:
            identifiers = {subdir}
            identifiers.update(
                filename.split("_", 1)[0]
                for filename in files
                if filename.endswith(".json")
            )
            matches = identifiers.intersection(languages)
            if not matches:
                continue
            if matched_languages.intersection(matches):
                raise ValueError(f"Language identifier matches multiple inputs: {sorted(matches)}")
            matched_languages.update(matches)

        # There should be exactly 2.
        if len(files) != 2:
            print(f"  [red]Subdirectory {subdir_path} does not contain exactly two files.[/red]")
            print(f"    [red]Subdirectory {subdir_path} contains the following files: {files}[/red]")
            return None, None

        # There should be a JSON gold-coding file and a document file (PDF or plaintext)
        if not any(f.endswith(".json") for f in files):
            print(f"  [red]Subdirectory {subdir_path} does not contain a JSON gold-coding file.[/red]")
            return None, None
        if not any(f.endswith(".pdf") or f.endswith(".txt") for f in files):
            print(f"  [red]Subdirectory {subdir_path} does not contain a document file (PDF or plaintext).[/red]")
            return None, None

        json_file = next(f for f in files if f.endswith(".json"))
        document_file = next(
            f for f in files if f.endswith(".pdf") or f.endswith(".txt")
        )

        # Get language name and lgcode
        language_name, lgcode = get_language_name_and_lgcode(subdir, json_file)

        retlist.append({
            "subdir_path": subdir_path,
            "fname_document": document_file,
            "fname_gold": json_file,
            "glottocode": subdir,  # Temporary: run_batch still expects this key.
            "folder_name": subdir,
            "language_name": language_name,  # Resolved from special names or CLDF
            "lgcode": lgcode,
        })

    if languages is not None:
        missing = set(languages) - matched_languages
        if missing:
            raise ValueError(f"Language inputs not found: {sorted(missing)}")

    # CHeck for existing output batch directory
    if output_batch_directory is not None:
        output_root = Path(output_batch_directory)
        if not output_root.is_dir():
            raise FileNotFoundError(output_root)

        used_folders = set()
        for entry in retlist:
            glottocode = entry["fname_gold"].split("_", 1)[0]
            candidates = {entry["folder_name"]}
            if is_glottocode(glottocode):
                candidates.add(glottocode)

            matches = [name for name in candidates if (output_root / name).is_dir()]
            if len(matches) != 1:
                raise ValueError(f"Expected one output folder for {entry['fname_gold']}: {matches}")

            folder = matches[0]
            if folder in used_folders:
                raise ValueError(f"Multiple inputs map to output folder {folder}")
            used_folders.add(folder)

            source_gold = Path(entry["subdir_path"]) / entry["fname_gold"]
            existing_gold = output_root / folder / "gold.json"
            with source_gold.open(encoding="utf-8") as f:
                source_data = json.load(f)
            with existing_gold.open(encoding="utf-8") as f:
                existing_data = json.load(f)
            if source_data != existing_data:
                raise ValueError(f"Gold files differ: {source_gold} and {existing_gold}")

            entry["folder_name"] = folder

        print(f"[green]Preparation complete. Output batch directory: [spring_green2]{output_batch_directory}[/spring_green2][/green]")

        return retlist, output_batch_directory

    # If everything looks good, create a dedicated output batch directory for this run: output/<DATE>_<BATCH_NUMBER>.
    # Get timestamp
    timestamp = get_timestamp()

    output_batch_directory = os.path.join(DIR_OUT, timestamp)
    os.makedirs(output_batch_directory, exist_ok=False)

    # Output metadata for the batch
    batch_id = os.path.basename(os.path.normpath(input_directory))

    # Create output directories.
    for entry in retlist:
        language_directory = os.path.join(
            output_batch_directory,
            entry["folder_name"],
        )
        os.makedirs(language_directory, exist_ok=True)

        # Copy the gold JSON file to the language directory as gold.json
        shutil.copy2(
            os.path.join(entry["subdir_path"], entry["fname_gold"]),
            os.path.join(language_directory, "gold.json"),
        )

        # Create output directories for each model within the language directory
        for model in models:
            os.makedirs(
                os.path.join(language_directory, model),
                exist_ok=True,
            )
    
    # Top-level metadata file
    # Add more stuff here as needed.
    metadata = {
        "batch_id": batch_id,
        "created_time": timestamp,
    }

    # Add language labels to the metadata if available
    metadata["language_labels"] = {}
    for entry in retlist:

        # Glottocodes should be retrievable from the gold file.
        glottocode = Path(entry["fname_gold"]).stem.split("_", 1)[0]
        metadata["language_labels"][entry["folder_name"]] = {
            "glottocode": glottocode if is_glottocode(glottocode) else None,
            "language_name": entry["language_name"],
        }

    # Dump metadata to a JSON file, at the top of this batch's output directory
    metadata_path = os.path.join(output_batch_directory, f"{batch_id}_meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[green]Preparation complete. Output batch directory: [spring_green2]{output_batch_directory}[/spring_green2][/green]")

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

    # Get models
    with open(FPATH_MODELS_METADATA, encoding="utf-8") as f:
        model_configs = json.load(f)

    model_ids = {}
    for model_id, config in model_configs.items():
        nickname = config.get("nickname")
        if nickname in args.models:
            if nickname in model_ids:
                raise ValueError(f"Duplicate model nickname: {nickname}")
            model_ids[nickname] = {
                "model_id": model_id,
                "api_base": config.get("api_base", None),
            }

    missing = set(args.models) - model_ids.keys()
    if missing:
        raise ValueError(f"Unknown model nicknames: {sorted(missing)}")

    # Check for existing output directories for each model and raise an error if any are not empty.
    if output_batch_directory is not None:
        for entry in retlist:
            for model in args.models:
                model_dir = Path(output_batch_directory) / entry["folder_name"] / model
                if model_dir.exists() and any(model_dir.iterdir()):
                    raise ValueError(f"Model output directory is not empty: {model_dir}")

    # For each entry in retlist, create a Coder instance and run it
    i = 0
    n_total = len(retlist) * len(args.models)
    for model in args.models:

        print(f"[blue][bold]***** RUNNING MODEL [grey54]{model}[/grey54] *****[/bold][/blue]")

        for entry in retlist:

            run_args = copy.deepcopy(args)
            run_args.language = entry["language_name"]
            run_args.lgcode = entry["lgcode"]
            run_args.model = model_ids[model]["model_id"]
            run_args.api_base = model_ids[model]["api_base"]
            run_args.out_dir = os.path.join(
                output_batch_directory, entry["folder_name"], model
            )

            document = os.path.join(entry["subdir_path"], entry["fname_document"])
            run_args.pdf = document if document.lower().endswith(".pdf") else None
            run_args.text = document if document.lower().endswith(".txt") else None

            run_args.print_prompt = False
            run_args.dump = False
            run_args.doc_size = False
            run_args.bool_debug_batches = False

            Coder(args=run_args).run()

            i += 1
            print(f"[green][bold]{i}/{n_total}[/bold]: Processed document {entry['fname_document']} for language {entry['language_name']} ({entry['glottocode']}) with model {model}[/green]")

    # Inform the user that the batch processing is complete
    print("[green]Batch processing complete.[/green]")


def summarise_results(output_batch_directory, retlist, args):
    root = Path(output_batch_directory)
    languages = [entry["folder_name"] for entry in retlist]

    models = sorted({
        path.name
        for language in languages
        for path in (root / language).iterdir()
        if path.is_dir()
    })

    calculate_stats(root, bool_output_to_terminal=False)
    plot_accuracy_per_stats(str(root))

    model_accuracies, majority_accuracies, languages, model_times = collect_results(
        root=root,
        gold_filename=Path("gold.json"),
        majority_filepath=Path(FPATH_MAJORITY),
        languages=languages,
        models=models,
    )
    plot_results(
        model_accuracies,
        majority_accuracies,
        output_path=root / "summary" / f"trials_summary_{get_timestamp()}.png",
        languages=languages,
        models=models,
        model_times=model_times,
        root=root,
    )

def main(
        args: argparse.Namespace,
):

    # Prepare: check data is consistent and ready to be run in batch.
    retlist, output_batch_directory = prepare(
        input_directory = args.input_directory,
        models = args.models,
        output_batch_directory = args.output_batch_directory,
        languages = args.languages,
    )

    # Sanity check
    if retlist is None or output_batch_directory is None:
        print("[red]Failed to prepare input data.[/red]")
        sys.exit(1)

    # Run if required.
    if retlist and not args.summarise_only and not args.prepare_only:

        # Call the function
        run_batch(
            input_directory = args.input_directory,
            output_batch_directory = output_batch_directory,
            retlist = retlist,
            args = args,
        )

    # Summarise the results after running the batch
    if retlist and not args.prepare_only and not args.skip_summary and output_batch_directory is not None:
        summarise_results(
            output_batch_directory = output_batch_directory,
            retlist = retlist,
            args = args,
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare and run batch processing for ethnocoder.")

    # Directories
    parser.add_argument(
        "-i", 
        "--input_directory", 
        type=str, 
        help="Path to the input directory containing subdirectories with documents and gold-coding JSON files.",
    )

    # # Output batch directory
    parser.add_argument("-o", "--output-batch-directory", default=None, help="Path to the output batch directory for this run. If None, a new directory will be created.")
    
    parser.add_argument(
        "--models",
        type=parse_csv_list,
        default=list(MODELS),
        help="Comma-separated models (default: tiny,small,medium,large).",
    )

    parser.add_argument(
        "--languages",
        type=parse_csv_list,
        default=None,
        help="Comma-separated input folder names or glottocodes; defaults to all languages.",
    )

    mode = parser.add_mutually_exclusive_group()
    # NB DO NOT use -p as it clashes with -p in code.py, which this script passes args to.
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--summarise-only", action="store_true")
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Run the batch without summarising the results.",
    )

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

    if args.summarise_only and not args.output_batch_directory:
        parser.error("--summarise-only requires -o")
    if args.summarise_only and args.skip_summary:
        parser.error("--summarise-only cannot be combined with --skip-summary")

    # Find input
    if args.input_directory is None:
        if not BATCHES_DIR.is_dir():
            parser.error(f"Batch directory not found: {BATCHES_DIR}")

        batches = [
            path for path in BATCHES_DIR.iterdir()
            if path.is_dir() and path.name.isdecimal()
        ]
        if not batches:
            parser.error(f"No integer-named batch directories found in {BATCHES_DIR}")

        args.input_directory = str(max(batches, key=lambda path: int(path.name)))

    main(args=args)