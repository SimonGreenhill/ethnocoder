# From a list of results TSVs, get per-variable statistics on correct, incorrect, hedged etc.

import os
import pandas as pd
import argparse
from tqdm import tqdm
from rich import print

from ethnocoder.utils import get_timestamp

def main(directory: str) -> None:
    """
    Load results TSVs from the specified directory and compute per-variable statistics.
    Output the results to a summary TSV file in the same directory.
    Also print to the console.

    Args:
        directory (str): Directory containing the results TSV files.
    """

    # From the directory, get all TSV files with format "results_all*.tsv"
    tsv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("results_all") and f.endswith(".tsv")]
    if not tsv_files:
        print(f"[red]No results TSV files found in {directory}[/red]")
        return

    # Load each TSV file into a DataFrame and concatenate them
    all_results = []
    for tsv_file in tqdm(tsv_files, desc="Loading TSV files"):
        df = pd.read_csv(tsv_file, sep="\t")
        all_results.append(df)

    # If something went wrong and no data was loaded, exit
    if not all_results:
        print(f"[red]No data found in TSV files in {directory}[/red]")
        return

    # Concatenate all loaded DataFrames into a single DataFrame
    results_df = pd.concat(all_results, ignore_index=True)

    # Fix up some values because "verdict" current combines things we want to split.
    # E.g. it's listed as "incorrect" if the model coding value is empty, we want this to be "model_missing".
    # It's list as either "incorrect" or "hedged" if the gold value is empty, we want this to be "gold_missing".
    results_df.loc[(results_df["verdict"] == "incorrect") & (results_df["model_output"].isna()), "verdict"] = "model_missing"
    results_df.loc[(results_df["verdict"].isin(["incorrect", "hedged"])) & (results_df["gold"].isna()), "verdict"] = "gold_missing"

    # Compute per-variable statistics.
    # We want percentages of correct, incorrect, and hedged responses per variable.
    summary = results_df.groupby("Feature_ID").agg(
        correct=pd.NamedAgg(column="verdict", aggfunc=lambda x: sum(v == "correct" for v in x)),
        incorrect=pd.NamedAgg(column="verdict", aggfunc=lambda x: sum(v == "incorrect" for v in x)),
        hedged=pd.NamedAgg(column="verdict", aggfunc=lambda x: sum(v == "hedged" for v in x)),
        model_missing=pd.NamedAgg(column="verdict", aggfunc=lambda x: sum(v == "model_missing" for v in x)),
        gold_missing=pd.NamedAgg(column="verdict", aggfunc=lambda x: sum(v == "gold_missing" for v in x)),
        total=pd.NamedAgg(column="verdict", aggfunc="count")
    ).reset_index()

    # Summarise the percentages. Truncate all to 1 decimal place. 
    summary["pct_correct"] = (summary["correct"] / summary["total"] * 100).round(1)
    summary["pct_incorrect"] = (summary["incorrect"] / summary["total"] * 100).round(1)
    summary["pct_hedged"] = (summary["hedged"] / summary["total"] * 100).round(1)
    summary["pct_model_missing"] = (summary["model_missing"] / summary["total"] * 100).round(1)
    summary["pct_gold_missing"] = (summary["gold_missing"] / summary["total"] * 100).round(1)

    # Sort by:
    # - pct_correct in descending order
    # - pct_incorrect in ascending order
    # - gold_missing in ascending order
    summary = summary.sort_values(by=["pct_correct", "pct_incorrect", "pct_gold_missing"], ascending=[False, True, True])

    # Output the summary to a TSV file and print it to the console
    summary_file = os.path.join(directory, f"summary_results_{get_timestamp()}.tsv")
    summary.to_csv(summary_file, sep="\t", index=False)
    print(f"[green]Summary written to {summary_file}[/green]")
    print(summary)


if __name__ == "__main__":

    # Command-line arguments

    # -d --directory of results TSVs
    parser = argparse.ArgumentParser(description="Get per-variable statistics from results TSVs")
    parser.add_argument("-d", "--directory", type=str, required=False, default=os.path.join("sandbox","summary_testing"), help="Directory of results TSVs")
    args = parser.parse_args()

    main(
        directory=args.directory,
    )

