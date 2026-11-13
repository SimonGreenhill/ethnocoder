# From a list of results TSVs, get per-variable statistics on correct, incorrect, hedged etc.

import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
from rich import print
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from ethnocoder.common.utils import get_timestamp

def summarise_tsv_results(
        directory: str,
        bool_output_to_terminal: bool = True,
    ) -> None:
    """
    Load results TSVs from the specified directory and compute per-variable statistics.
    Output the results to a summary TSV file in the same directory.
    Also print to the console.

    Args:
        directory (str): Directory containing the results TSV files.
        bool_output_to_terminal (bool): Whether to print the summary to the terminal. Defaults to True.
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
    print(f"[green]Summary written to [spring_green2]{summary_file}[/spring_green2][/green]")

    # Output summary if requested
    if bool_output_to_terminal:
        print(summary)

def plot_accuracy_per_stats(
        directory: str,
)->None:
    """
    1. Pull stats.json and EVERY .log file per subdirectory in the main directory.
    2. Sum the values of all d["elapsed"] and d["elapsed_2"] entries in all .log files for that subdirectory.
    3. Sum the values of all d["completion_tokens_details"]["reasoning_tokens"]. *NB I didn't store reasoning tokens for review passes*.
    4. Plot accuracy on the x-axis (scale 0 to 100), then twin y-axes on the same plot: total elapsed (left) and total reasoning tokens (right).
    5. Output all plots to `summary` subdirectory within the main directory.
    
    Args:
        directory (str): The main directory containing subdirectories with stats.json and .log files.
    """

    # 1. Pull stats.json and EVERY .log file per subdirectory in the main directory.
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    # Initialise results dict. Keys are subdirectory names, values are dicts with totals.
    results = {}

    # Loop through each subdirectory and process its stats and log files.
    for subdir in subdirs:

        # Get the stats filepath.
        stats_file = os.path.join(subdir, "stats.json")

        # CHANGED: stats["resources"] now holds this data.

        # # Get all log filepaths.
        # log_files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(".log")]

        # Skip any that don't have 
        if not os.path.exists(stats_file):
            if os.path.basename(subdir) != "summary":
                print(f"[yellow]Warning: stats.json not found in {subdir}[/yellow]")
            continue
        with open(stats_file, "r") as f:
            dict_stats = json.load(f)

        # # 2. Sum the values of all d["elapsed"] and d["elapsed_2"] entries in all .log files for that subdirectory.
        # # Initialise totals for this subdirectory.
        # total_elapsed = 0
        # total_elapsed_2 = 0
        # total_reasoning_tokens = 0

        # # Process each log file for this subdirectory.
        # for log_file in log_files:
        #     with open(log_file, "r") as f:
        #         d = json.load(f)
        #         total_elapsed += float(d.get("elapsed", 0))
        #         total_elapsed_2 += float(d.get("elapsed_2", 0))

        #         # 3. Sum the reasoning tokens for this log entry.
        #         total_reasoning_tokens += int(d.get("completion_tokens_details", {}).get("reasoning_tokens", 0))

        totals = dict_stats.get("totals", {})

        # 3c. Store the totals for this subdirectory in the results dict.
        results[os.path.basename(subdir)] = {
            "elapsed_1": dict_stats["resources"]["elapsed_1"],
            "elapsed_2": dict_stats["resources"]["elapsed_2"],
            "total_time": dict_stats["resources"]["total_time"],
            "total_reasoning_tokens": dict_stats["resources"]["total_reasoning_tokens"],
            "accuracy": dict_stats.get("totals", {}).get("correct_pct", None),
            "size": totals["correct"] + totals["incorrect"] + totals["hedged"],
        }

    # Remove any entries with None accuracy.
    results = {k: v for k, v in results.items() if v["accuracy"] is not None}

    # 4. Plot accuracy on the x-axis (scale 0 to 100), 
    # then twin y-axes on the same plot: total elapsed (left) and total reasoning tokens (right).
    accuracies = [v["accuracy"] for v in results.values()]
    total_elapsed = [v["total_time"] for v in results.values()]
    total_reasoning_tokens = [v["total_reasoning_tokens"] for v in results.values()]

    # Create plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.set_xlabel("Accuracy")
    ax1.set_ylabel("Time (s)")
    ax2.set_ylabel("Reasoning (1000x tokens)")

    # Plot each subdirectory's data points with unique colors and markers.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    subdirectory_handles = []

    # Scale sizes.
    sizes = [v["size"] for v in results.values()]
    max_size = max(sizes) if sizes else 1
    min_size = min(sizes) if sizes else 1

    # Plot in a loop so we get the elapsed and the reasoning in the same colour, different shapes.
    for i, (basename, values) in enumerate(sorted(results.items())):
        color = colors[i % len(colors)]
        accuracy = values["accuracy"]

        # Get marker size
        if min_size == max_size:
            markersize = 9  # All counts equal: use the midpoint.
        else:
            markersize = 6 + 6 * (values["size"] - min_size) / (max_size - min_size)

        ax1.plot(
            accuracy, values["total_time"],
            marker="o", linestyle="None", color=color, markersize=markersize,
            markeredgecolor="black", markeredgewidth=0.6,
        )
        ax2.plot(
            accuracy, values["total_reasoning_tokens"],
            marker="^", linestyle="None", color=color, markersize=markersize,
            markeredgecolor="black", markeredgewidth=0.6,
        )
        # ax1.annotate(basename, (accuracy, values["total_time"]), color=color)

        subdirectory_handles.append(
            Line2D([], [], color=color, marker="s", linestyle="None", label=f"{basename} ({values['size']})")
        )

    ax1.set_xlim(0, 100)

    subdirectory_legend = ax1.legend(
        handles=subdirectory_handles, title="Glottocode", loc="upper left"
    )
    ax1.add_artist(subdirectory_legend)

    metric_handles = [
        Line2D([], [], color="black", marker="o", linestyle="None", label="Time"),
        Line2D([], [], color="black", marker="^", linestyle="None", label="Reasoning"),
    ]
    ax2.legend(handles=metric_handles, title="Metric", loc="lower left")

    # Make the total time axis go from 0 to 1.1* the max
    ax1.set_ylim(0, 1.1 * max(total_elapsed))

    # Make the total reasoning tokens axis go from 0 to 1.1* the max
    ax2.set_ylim(0, 1.1 * max(total_reasoning_tokens))

    # Scale the total reasoning tokens by 1000 for better visualization
    ax2.set_yticks([tick for tick in ax2.get_yticks() if tick % 1000 == 0])
    ax2.set_yticklabels([str(int(tick / 1000)) for tick in ax2.get_yticks()])

    # Set x-ticks to have the percentage sign
    ax1.set_xticks([tick for tick in ax1.get_xticks() if tick % 10 == 0])
    ax1.set_xticklabels([f"{int(tick)}%" for tick in ax1.get_xticks()])

    # 5. Output to the summary subdirectory
    summary_dir = os.path.join(directory, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    output_path = os.path.join(summary_dir, f"accuracy_time_reasoning_{get_timestamp()}.png")
    plt.savefig(output_path)
    print(f"[green]Saved plot to [spring_green2]{output_path}[/spring_green2][/green]")


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

