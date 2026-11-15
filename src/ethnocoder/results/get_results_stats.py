# From a list of results TSVs, get per-variable statistics on correct, incorrect, hedged etc.

import os
import pandas as pd
import argparse
from tqdm import tqdm
from rich import print

from ethnocoder.common.utils import get_timestamp
from ethnocoder.plotting.plot_accuracy_time_reasoning import plot_accuracy_per_stats

def summarise_tsv_results(
        directory: str,
        bool_output_to_terminal: bool = True,
        tsv_files: list[str] | None = None,
    ) -> str:
    """
    Load results TSVs from the specified directory and compute per-variable statistics.
    Output the results to a summary TSV file in the same directory.
    Also print to the console.

    Args:
        directory (str): Directory containing the results TSV files.
        bool_output_to_terminal (bool): Whether to print the summary to the terminal. Defaults to True.
        tsv_files (list[str] | None): List of specific TSV files to process. If None, all TSV files matching "results_all*.tsv" in the directory will be used. Defaults to None.
    Returns:
        str: Path to the summary TSV file generated. Returns an empty string if no summary was created.
    """

    # From the directory, get all TSV files with format "results_all*.tsv"
    if tsv_files is None:
        tsv_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.startswith("results_all") and f.endswith(".tsv")
        ]
    if not tsv_files:
        print(f"[red]No results TSV files found in {directory}[/red]")
        return ""

    # Load each TSV file into a DataFrame and concatenate them
    all_results = []
    for tsv_file in tqdm(tsv_files, desc="Loading TSV files"):
        df = pd.read_csv(tsv_file, sep="\t")
        all_results.append(df)

    # If something went wrong and no data was loaded, exit
    if not all_results:
        print(f"[red]No data found in TSV files in {directory}[/red]")
        return ""

    # Concatenate all loaded DataFrames into a single DataFrame
    results_df = pd.concat(all_results, ignore_index=True)

    # Gold missing takes precedence when both values are missing.
    results_df.loc[results_df["gold"].isna(), "verdict"] = "missing_gold"
    results_df.loc[
        results_df["gold"].notna() & results_df["model_output"].isna(),
        "verdict",
    ] = "missing_model"

    summary = results_df.groupby("Feature_ID").agg(
        correct=pd.NamedAgg(column="verdict", aggfunc=lambda x: sum(v == "correct" for v in x)),
        incorrect=pd.NamedAgg(column="verdict", aggfunc=lambda x: sum(v == "incorrect" for v in x)),
        hedged=pd.NamedAgg(column="verdict", aggfunc=lambda x: sum(v == "hedged" for v in x)),
        model_missing=pd.NamedAgg(column="verdict", aggfunc=lambda x: sum(v == "missing_model" for v in x)),
        gold_missing=pd.NamedAgg(column="verdict", aggfunc=lambda x: sum(v == "missing_gold" for v in x)),
        total=pd.NamedAgg(column="verdict", aggfunc="count")
    ).reset_index()

    # Exclude missing gold from the four outcome percentages.
    eligible_total = (summary["total"] - summary["gold_missing"]).replace(0, float("nan"))
    for category in ("correct", "incorrect", "hedged", "model_missing"):
        summary[f"pct_{category}"] = (
            summary[category] / eligible_total * 100
        ).round(1).fillna(0)

    # Gold missing is reported as a percentage of all rows.
    summary["pct_gold_missing"] = (
        summary["gold_missing"] / summary["total"] * 100
    ).round(1)

    # Sort by:
    # - pct_correct in descending order
    # - pct_incorrect in ascending order
    # - gold_missing in ascending order
    summary = summary.sort_values(by=["pct_correct", "pct_incorrect", "pct_gold_missing"], ascending=[False, True, True])

    # Output the summary to a TSV file and print it to the console
    summary_file = os.path.join(directory, f"summary_results_{get_timestamp()}.tsv")
    summary.to_csv(summary_file, sep="\t", index=False)
    print(f"[green]Summary written to [spring_green2]{summary_file}[/spring_green2][/green]")

    # Output summary if requested
    if bool_output_to_terminal:
        print(summary)

    return summary_file


if __name__ == "__main__":

    # Command-line arguments

    # -d --directory of results TSVs
    parser = argparse.ArgumentParser(description="Get per-variable statistics from results TSVs")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Directory of results TSVs")

    # Argument to call summarise_tsv_results
    parser.add_argument("--summarise", action="store_true", help="Call summarise_tsv_results")

    # Argument to plot accuracy per stats, instead of summarising TSV results
    parser.add_argument("--plot", action="store_true", help="Call plot_accuracy_per_stats")

    args = parser.parse_args()

    if args.summarise:
        summarise_tsv_results(
            directory=args.directory,
            bool_output_to_terminal=True,
        )
    else:
        print("[yellow]Skipping summarise.[/yellow]")

    if args.plot:
        plot_accuracy_per_stats(
            directory=args.directory,
        )
    else:
        print("[yellow]Skipping plot.[/yellow]")

