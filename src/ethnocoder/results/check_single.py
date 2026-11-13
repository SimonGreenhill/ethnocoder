# 1. Argparse: gold file -g and model output file -m.
# 2. Check both for duplicates.
# 3. Check results file for missing or superfluous entries.
# 4. Loop at gold file entries:
#    a. Find the relevant entry in the results file.
#    b. Store the pair <gold, result> in a results dict.
# 5. Report: absolute number of pairs, proportion of pairs.
# Left-pad the report lines with spacing (total 3) so that the values all line up with each other.

import argparse
import json
import os
from pathlib import Path
import sys
from rich import print
import pandas as pd

from ethnocoder.datasets.cldf import GRAMBANK_MULTISTATE_MAPPINGS
from ethnocoder.common.utils import GRAMBANK_CLDF_DIR

def main(
        fpath_gold: Path,
        fpath_model_output: Path,
        bool_output_to_terminal: bool = True,
) -> None:
    """
    Compare a single model output file against the gold file.
    """

    # Open both files into json objects.
    with open(fpath_gold, "r") as f:
        gold_data = json.load(f)
    with open(fpath_model_output, "r") as f:
        model_output_data = json.load(f)

    # Gold is a list of dictionaries, with "id" and "code", and various other comments.
    gold_ids = [entry["id"] for entry in gold_data]

    # Remove 
    codings_with_ids = [entry for entry in model_output_data["codings"] if "id" in entry]

    # Results list is in "codings"
    model_output_ids = [entry["id"] for entry in codings_with_ids]

    gold_duplicates = [id_ for id_ in gold_ids if gold_ids.count(id_) > 1]
    model_output_duplicates = [id_ for id_ in model_output_ids if model_output_ids.count(id_) > 1]

    if gold_duplicates:
        print(f"[red]Gold file has duplicates:[/red] {gold_duplicates}")
    if model_output_duplicates:
        print(f"[red]Model output file has duplicates:[/red] {model_output_duplicates}")
    if len(gold_duplicates) == 0 and len(model_output_duplicates) == 0:
        print("[green]No duplicates found in either file.[/green]")

    # 3. Check model output file for missing or superfluous entries.
    missing_in_results = [id_ for id_ in gold_ids if id_ not in model_output_ids]
    superfluous_in_results = [id_ for id_ in model_output_ids if id_ not in gold_ids]

    if missing_in_results:
        print(f"[magenta]Missing in model output file: {missing_in_results}[/magenta]")
    if superfluous_in_results:
        print(f"[magenta]Superfluous in model output file: {superfluous_in_results}[/magenta]")
    if not missing_in_results and not superfluous_in_results:
        print("[green]No missing or superfluous entries in model output file.[/green]")

    # 4. Loop at gold file entries and find relevant entry in model output file.
    results_dict = {}
    stats_dict = {
        "correct": 0, # Value matches exactly, including question marks
        "incorrect": 0, # Value does not match, including where gold has a question mark and model output does not
        "hedged": 0, # Gold value is determinate but model output has a question mark
    }
    for gold_entry in gold_data:
        gold_id = gold_entry["id"]

        # Get the location of this ID in the list, then retrieve the corresponding entry from the model output file.
        model_output_index = model_output_ids.index(gold_id) if gold_id in model_output_ids else None

        if model_output_index is None:
            # print(f"  [magenta]ID {gold_id} from gold file not found in model output file.[/magenta]")
            continue

        # Get this entry from the model output file using the index.
        model_output_entry = codings_with_ids[model_output_index]

        # Check the ID matches
        if model_output_entry["id"] != gold_id:
            print(f"  [red]ID mismatch for gold ID {gold_id} in model output file.[/red]")
            sys.exit(1)
        
        # Get a unique identifier for this pair of values
        value_gold = gold_entry["code"]

        # debug
        try:
            value_model_output = model_output_entry["code"]
        except KeyError:
            print(f"  [red]ID {gold_id} from model output file missing 'code' field.[/red]")
            value_model_output = 99 # Use 99 to indicate no "code" field was returned for this ID.

        # String them together with an underscore to create a unique identifier
        unique_identifier = f"{value_gold}_{value_model_output}"

        if unique_identifier not in results_dict:
            results_dict[unique_identifier] = 0

        # Store this unique identifier in the results dictionary
        results_dict[unique_identifier] += 1

        # Update stats_dict
        if value_gold == value_model_output:
            stats_dict["correct"] += 1
        elif value_model_output == "?":
            stats_dict["hedged"] += 1
        else:
            stats_dict["incorrect"] += 1

    # Sort results_dict as follows:
    # both 0
    # both 1
    # both 2
    # both 3
    # both ?
    # gold 0, result 1 (and 2, and 3)
    # gold 1, result 0 (and 2, and 3)
    # gold 2, result 0 (and 1, and 3)
    # gold 3, result 0 (and 1, and 2)
    # gold ?, result 0 (and 1, and 2, and 3)
    # gold 0 (and 1, and 2, and 3), result ?def sort_key(item):
    def sort_key(item):
        gold, model = item[0].split("_")
        return (
            gold != model,  # matching cases first, including ?_?
            model == "?" and gold != "?",  # other model-? cases last
            gold,
            model,
        )

    # Sort the results dictionary based on the custom sort key
    results_dict = dict(sorted(results_dict.items(), key=sort_key))

    # Report results
    if bool_output_to_terminal:
        for unique_identifier, count in results_dict.items():
            gold_value, model_value = unique_identifier.split("_")

            # Print different colours depending on correctness
            if gold_value == model_value:
                print(f"[green]Gold: {gold_value} | Model: {model_value}: {count:>3}[/green]")
            elif model_value == "?":
                print(f"[yellow]Gold: {gold_value} | Model: {model_value}: {count:>3}[/yellow]")
            else:
                print(f"[red]Gold: {gold_value} | Model: {model_value}: {count:>3}[/red]")

    # Add logged data to stats dict: total time and total reasoning tokens consumed.
    # Get the directory from fpath_model_output.
    subdir = os.path.dirname(fpath_model_output)

    # Get all log filepaths.
    log_files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(".log")]

    # Initialise totals for this subdirectory.
    total_elapsed = 0
    total_elapsed_2 = 0
    total_reasoning_tokens = 0

    # Process each log file for this subdirectory.
    for log_file in log_files:
        with open(log_file, "r") as f:
            d = json.load(f)
            total_elapsed += float(d.get("elapsed", 0))
            total_elapsed_2 += float(d.get("elapsed_2", 0))

            # 3. Sum the reasoning tokens for this log entry.
            total_reasoning_tokens += int(d.get("completion_tokens_details", {}).get("reasoning_tokens", 0))

    # 3c. Store the totals for this subdirectory in the results dict.
    results_dict["resources"] = {
        "elapsed_1": total_elapsed,
        "elapsed_2": total_elapsed_2,
        "total_time": total_elapsed + total_elapsed_2,
        "total_reasoning_tokens": total_reasoning_tokens,
    }

    # Report stats
    if bool_output_to_terminal:
        total = sum(stats_dict.values())
        print(f"\n[blue]Totals: ({total} values)[/blue]")
        for key, value in stats_dict.items():

            # Print colours
            if key == "correct":
                print(f"  [green]{key.capitalize()}:   {value:>3} ({value/total:.1%})[/green]")
            elif key == "incorrect":
                print(f"  [red]{key.capitalize()}: {value:>3} ({value/total:.1%})[/red]")
            elif key == "hedged":
                print(f"  [yellow]{key.capitalize()}:    {value:>3} ({value/total:.1%})[/yellow]")
            else:
                print(f"  [blue]{key.capitalize()}: {value:>3} ({value/total:.1%})[/blue]")
            continue

    # Call dedicated output function
    output_results(
        results_dict = results_dict,
        stats_dict = stats_dict,
        gold_data = gold_data,
        model_output_data = model_output_data,
        fpath_model_output = fpath_model_output,
    )

def output_results(
        results_dict: dict,
        stats_dict: dict,
        gold_data: dict,
        model_output_data: dict,
        fpath_model_output: Path,
)->None:
    """
    Output the results of the comparison between gold data and model output data.
    JSON file: Statistics overview of results (i.e. what was printed to the terminal in main())
    TSV file: Variable-by-variable comparison of gold and model output values, with model comments and quotes.

    Args:
        results_dict (dict): Dictionary containing the comparison results between gold and model output.
        stats_dict (dict): Dictionary containing statistics of the comparison results.
        gold_data (dict): Dictionary containing the gold standard data.
        model_output_data (dict): Dictionary containing the model output data.
        fpath_model_output (Path): Path to the model output file. Determines the directory for these output files.
    """

    # Get the output directory.
    out_dir = fpath_model_output.parent

    # Ensure the output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build output dict
    output_dict = results_dict

    output_dict["totals"] = stats_dict

    # Include percentages in totals
    total = sum(stats_dict.values())
    output_dict["totals"]["correct_pct"] = round(stats_dict["correct"] * 100 / total, 1) if total > 0 else 0
    output_dict["totals"]["incorrect_pct"] = round(stats_dict["incorrect"] * 100 / total, 1) if total > 0 else 0
    output_dict["totals"]["hedged_pct"] = round(stats_dict["hedged"] * 100 / total, 1) if total > 0 else 0

    # Write the statistics to a JSON file
    out_dict_fpath = out_dir / "stats.json"
    with open(out_dict_fpath, "w") as f:
        json.dump(output_dict, f, indent=2)

    # Inform the user.
    print(f"\n[green]Statistics written to {out_dict_fpath}[/green]")

    # TSV file
    # Load everything from feature_grouping_for_analysis.csv.
    df_feature_grouping = pd.read_csv("feature_grouping_for_analysis.csv")

    # Convert feature grouping dataframe columns to strings.
    for col in df_feature_grouping.columns:
        if col == "Feature_ID":
            continue
        df_feature_grouping[col] = df_feature_grouping[col].astype(str)

    # Check all the variables are in df_feature_grouping.
    # Load the parameters csv from GRAMBANK_CLDF_DIR.
    fpath_parameters = GRAMBANK_CLDF_DIR / "parameters.csv"
    df_parameters = pd.read_csv(fpath_parameters)

    # Remove multistate variables from df_parameters and df_feature_grouping.
    df_parameters = df_parameters[~df_parameters["ID"].isin(GRAMBANK_MULTISTATE_MAPPINGS)]
    df_feature_grouping = df_feature_grouping[~df_feature_grouping["Feature_ID"].isin(GRAMBANK_MULTISTATE_MAPPINGS)]

    # Check that all entries in the ID column of df_parameters are present in df_feature_grouping.
    if not df_parameters["ID"].isin(df_feature_grouping["Feature_ID"]).all():

        # Print in red the missing IDs.
        missing_ids = df_parameters.loc[~df_parameters["ID"].isin(df_feature_grouping["Feature_ID"]), "ID"]
        print(f"[red]Missing IDs in feature grouping: {', '.join(missing_ids)}[/red]")
        sys.exit(1)

    # Add gold and model output columns to TSV.
    # Create dataframes from gold_data and model_output_data.
    df_gold = pd.DataFrame(gold_data)
    df_model_output = pd.DataFrame(model_output_data['codings'])

    # Merge df_gold into df_feature_grouping; call df_gold's "code" column "gold"
    df_gold = df_gold.rename(columns={"code": "gold"})
    df_gold = df_gold.rename(columns={"comment": "human_comment"})
    df_gold = df_gold.rename(columns={"source": "human_source"})
    df_gold = df_gold.rename(columns={"source_comment": "human_source_comment"})
    df_gold = df_gold.rename(columns={"coders": "human_coders"})
    df_feature_grouping = df_feature_grouping.merge(df_gold, how="left", left_on="Feature_ID", right_on="id")

    # Merge df_model_output into df_feature_grouping; call df_model_output's "code" column "model_output"
    df_model_output = df_model_output.rename(columns={"code": "model_output"})
    df_feature_grouping = df_feature_grouping.merge(df_model_output, how="left", left_on="Feature_ID", right_on="id")

    # Drop the redundant 'id' columns resulting from the merges
    df_feature_grouping = df_feature_grouping.drop(columns=["id_x", "id_y"], errors="ignore")

    # Extra analysis columns go here.
    # Add a "verdict" column based on comparison of gold and model_output
    df_feature_grouping["verdict"] = df_feature_grouping.apply(
        lambda row: "correct" if row["gold"] == row["model_output"] else "incorrect",
        axis=1
    )

    # If there's "incorrect" and model_output column is "?" change "incorrect" to "hedged".
    df_feature_grouping["verdict"] = df_feature_grouping.apply(
        lambda row: "hedged" if row["verdict"] == "incorrect" and row["model_output"] == "?" else row["verdict"],
        axis=1
    )

    # If the gold value is None or pd.nan, change the verdict to "missing_gold"
    df_feature_grouping["verdict"] = df_feature_grouping.apply(
        lambda row: "missing_gold" if row["gold"] is None or pd.isna(row["gold"]) else row["verdict"],
        axis=1
    )

    # If the model_output value is None, change the verdict to "missing_model"
    df_feature_grouping["verdict"] = df_feature_grouping.apply(
        lambda row: "missing_model" if row["model_output"] is None or pd.isna(row["model_output"]) else row["verdict"],
        axis=1
    )

    cols = [
                "Feature_ID", 
                "gold", 
                "model_output", 
                "verdict", 
                "confidence", 
                "justification", 
                "quote", 
                "pages", 
        ]
    cols2 = ['human_comment', 'human_source', 'human_source_comment', 'human_coders']
    cols3 = [
                "Main_domain", 
                "Finer_grouping",
            ]

    # There may not be cols2, so only include them if they exist in the DataFrame
    cols2_existing = [col for col in cols2 if col in df_feature_grouping.columns]

    # Reorder columns:
    # Feature_ID, gold, model_output, verdict, confidence, justification, quote, pages, Main_domain, Finer_grouping
    df_feature_grouping = df_feature_grouping[
        cols + cols2_existing + cols3
    ]

    # Sort by: 
    # 1. verdict (correct, incorrect, hedged)
    # 2. confidence (high, medium, low, absent)
    # 3. gold value (0, 1, 2, 3, ?)
    # 4. model_output value (0, 1, 2, 3, ?)
    # 5. Feature_ID
    df_feature_grouping["verdict"] = pd.Categorical(
        df_feature_grouping["verdict"],
        categories=["correct", "incorrect", "hedged", "missing_gold", "missing_model"],
        ordered=True
    )
    df_feature_grouping["confidence"] = pd.Categorical(
        df_feature_grouping["confidence"],
        categories=["high", "medium", "low", "absent"],
        ordered=True
    )
    df_feature_grouping["gold"] = pd.Categorical(
        df_feature_grouping["gold"],
        categories=["0", "1", "2", "3", "?"],
        ordered=True
    )
    df_feature_grouping["model_output"] = pd.Categorical(
        df_feature_grouping["model_output"],
        categories=["0", "1", "2", "3", "?"],
        ordered=True
    )
    df_feature_grouping = df_feature_grouping.sort_values(
        by=["verdict", "confidence", "gold", "model_output", "Feature_ID"]
    )

    # Inform the user we are writing the TSV files:
    print(f"[blue]Writing TSVs to {out_dir}...[/blue]")

    # Output to TSV in out_dir.
    fpath_output = out_dir / "results_all.tsv"
    df_feature_grouping.to_csv(fpath_output, sep="\t", index=False)

    # Inform the user.
    print(f"  [green]TSV written for gold and model output at [spring_green2]{fpath_output.name}[/spring_green2][/green]")

    # If there are any correct, output them to a separate TSV file.
    df_correct = df_feature_grouping[df_feature_grouping["verdict"] == "correct"]
    if not df_correct.empty:
        fpath_correct = out_dir / "results_correct.tsv"
        df_correct.to_csv(fpath_correct, sep="\t", index=False)
        print(f"  [green]→ TSV written for correct entries at [spring_green2]{fpath_correct.name}[/spring_green2][/green]")
    else:
        print(f"  [yellow]→ No correct entries found.[/yellow]")

    # If there are any incorrect, output them to a separate TSV file.
    df_incorrect = df_feature_grouping[df_feature_grouping["verdict"] == "incorrect"]
    if not df_incorrect.empty:
        fpath_incorrect = out_dir / "results_incorrect.tsv"
        df_incorrect.to_csv(fpath_incorrect, sep="\t", index=False)
        print(f"  [green]→ TSV written for incorrect entries at [spring_green2]{fpath_incorrect.name}[/spring_green2][/green]")
    else:
        print(f"  [yellow]→ No incorrect entries found.[/yellow]")

    # If there are any hedged, output them to a separate TSV file.
    df_hedged = df_feature_grouping[df_feature_grouping["verdict"] == "hedged"]
    if not df_hedged.empty:
        fpath_hedged = out_dir / "results_hedged.tsv"
        df_hedged.to_csv(fpath_hedged, sep="\t", index=False)
        print(f"  [green]→ TSV written for hedged entries at [spring_green2]{fpath_hedged.name}[/spring_green2][/green]")
    else:
        print(f"  [yellow]→ No hedged entries found.[/yellow]")

    # If there are any missing gold entries, output them to a separate TSV file.
    df_missing_gold = df_feature_grouping[df_feature_grouping["verdict"] == "missing_gold"]
    if not df_missing_gold.empty:
        fpath_missing_gold = out_dir / "results_missing_gold.tsv"
        df_missing_gold.to_csv(fpath_missing_gold, sep="\t", index=False)
        print(f"  [green]→ TSV written for missing gold entries at [spring_green2]{fpath_missing_gold.name}[/spring_green2][/green]")
    else:
        print(f"  [yellow]→ No missing gold entries found.[/yellow]")

    # If there are any missing model entries, output them to a separate TSV file.
    df_missing_model = df_feature_grouping[df_feature_grouping["verdict"] == "missing_model"]
    if not df_missing_model.empty:
        fpath_missing_model = out_dir / "results_missing_model.tsv"
        df_missing_model.to_csv(fpath_missing_model, sep="\t", index=False)
        print(f"  [green]→ TSV written for missing model entries at [spring_green2]{fpath_missing_model.name}[/spring_green2][/green]")
    else:
        print(f"  [yellow]→ No missing model entries found.[/yellow]")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--fpath_gold", required=True, help="Path to the gold file")
    parser.add_argument("-m", "--fpath_model_output", required=True, help="Path to the model output file")
    args = parser.parse_args()

    main(
        fpath_gold=Path(args.fpath_gold),
        fpath_model_output=Path(args.fpath_model_output),
    )