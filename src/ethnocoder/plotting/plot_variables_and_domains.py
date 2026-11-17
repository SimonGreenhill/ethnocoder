# Written by Terra, not checked!

import argparse
import json
from pathlib import Path
from rich import print

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from ethnocoder.common.utils import get_timestamp, parse_csv_list


CATEGORIES = {
    "pct_correct": ("Correct", "green"),
    "pct_incorrect": ("Incorrect", "red"),
    "pct_hedged": ("Hedged", "gold"),
    "pct_model_missing": ("Model missing", "blue"),
}


def draw_stacked_percentages(
    axis: Axes,
    data: pd.DataFrame,
    label_column: str,
    title: str,
    title_fontsize: int = 20,
    highlights: dict[str, str] | None = None,
) -> None:
    """Draw stacked percentage bars on an existing axis."""

    # Font sizes
    axis.set_title(label=title, fontsize=title_fontsize)
    axis.set_xlabel(xlabel="Percentage", fontsize=18)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    axis.tick_params(axis="x", labelsize=16)
    axis.tick_params(axis="y", labelsize=16)

    # dealing with empty data
    if data.empty:
        axis.set_xlim(left=0, right=100)
        axis.set_yticks(ticks=[])
        axis.text(x=0.5, y=0.5, s="No variables", ha="center",
                  transform=axis.transAxes)
        return

    data = data.sort_values(by="pct_correct", ascending=False, na_position="last")
    positions = range(len(data))
    labels = [
        f"{label} (no gold)" if missing else str(label)
        for label, missing in zip(
            data[label_column],
            data[list(CATEGORIES)].isna().all(axis=1),
        )
    ]
    left = pd.Series(data=0.0, index=data.index)

    for column, (label, color) in CATEGORIES.items():
        values = data[column].fillna(value=0)
        axis.barh(y=positions, width=values, left=left, label=label, color=color)
        left = left + values

    axis.set_yticks(ticks=list(positions), labels=labels)
    if highlights:
        for tick, feature_id in zip(axis.get_yticklabels(), data[label_column]):
            if feature_id in highlights:
                color = highlights[feature_id]
                tick.set_fontweight("bold")
                tick.set_color("black")
                tick.set_bbox(dict(boxstyle="round", facecolor=color, edgecolor=color, alpha=0.4))
    axis.invert_yaxis()
    axis.set_xlim(left=0, right=max(100, left.max()))


def plot_stacked_percentages(
    data: pd.DataFrame,
    label_column: str,
    title: str,
    output_path: Path,
) -> None:
    """Save a single stacked-bar plot."""
    if data.empty:
        raise ValueError(f"No data to plot: {title}")

    fig, axis = plt.subplots(figsize=(12, max(4, len(data) * 0.25 + 2)))
    draw_stacked_percentages(
        axis=axis, data=data, label_column=label_column, title=title
    )
    fig.legend(
        handles=[Patch(facecolor=color, label=label) for label, color in CATEGORIES.values()],
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        borderaxespad=0,
        fontsize=22,
        ncol=len(CATEGORIES),
    )
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname=output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[green]Saved plot: [spring_green2]{output_path}[/spring_green2][/green]")


def plot_variables_by_domain(
    data: pd.DataFrame,
    domains: list[str],
    models: list[str],
    output_path: Path,
    bool_highlights: bool = False,
) -> None:
    """Save variable bars in four columns, with two domains stacked in the last."""
    if data.empty:
        raise ValueError("No variables to plot")

    expected = {"clause", "nominal domain", "verbal domain", "pronoun", "numeral"}
    if set(domains) != expected:
        raise ValueError(f"Expected domains {expected}, found: {domains}")

    highlights = {}
    if bool_highlights:
        for filename, color in (("top_20.json", "green"), ("bottom_20.json", "red")):
            path = Path("sandbox/intercoder_reliability") / filename
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    entries = json.load(f)
                if not isinstance(entries, dict):
                    raise ValueError(f"Expected a JSON object in {path}")
                overlap = highlights.keys() & entries.keys()
                if overlap:
                    raise ValueError(f"Variables appear in both highlight files: {sorted(overlap)}")
                highlights.update(dict.fromkeys(entries, color))

    counts = data.groupby("Main_domain").size()
    pronoun_count = counts.get("pronoun", 0)
    numeral_count = counts.get("numeral", 0)
    largest_group = max(counts.max(), pronoun_count + numeral_count)

    fig = plt.figure(figsize=(25, max(7, largest_group * 0.25 + 4)))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=4,
        height_ratios=(pronoun_count + 4, numeral_count + 4),
    )

    panels = [
        (fig.add_subplot(grid[:, 0]), "clause"),
        (fig.add_subplot(grid[:, 1]), "nominal domain"),
        (fig.add_subplot(grid[:, 2]), "verbal domain"),
        (fig.add_subplot(grid[0, 3]), "pronoun"),
        (fig.add_subplot(grid[1, 3]), "numeral"),
    ]
    for axis, domain in panels:
        draw_stacked_percentages(
            axis=axis,
            data=data[data["Main_domain"] == domain],
            label_column="Feature_ID",
            title=domain,
            title_fontsize=30,
            highlights=highlights,
        )

    fig.suptitle(f"Variables — models: {', '.join(models)}")
    fig.legend(
        handles=[Patch(facecolor=color, label=label) for label, color in CATEGORIES.values()],
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        borderaxespad=0,
        fontsize=22,
        ncol=len(CATEGORIES),
    )
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname=output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[green]Saved plot: [spring_green2]{output_path}[/spring_green2][/green]")


def plot_variables_and_domains(
    summary_file: Path,
    models: list[str],
    grouping_file: Path = Path("feature_grouping_for_analysis.csv"),
    output_dir: Path | None = None,
    bool_highlights: bool = False,
) -> None:
    """Plot variable percentages and unweighted domain averages."""
    if not models:
        raise ValueError("At least one model is required for the plot titles")

    summary = pd.read_csv(summary_file, sep="\t")
    grouping = pd.read_csv(
        grouping_file,
        usecols=["Feature_ID", "Main_domain", "Finer_grouping"],
    )
    summary = summary.merge(
        grouping, on="Feature_ID", how="left", validate="many_to_one"
    )

    missing_groups = summary[
        summary[["Main_domain", "Finer_grouping"]].isna().any(axis=1)
    ]
    if not missing_groups.empty:
        raise ValueError(
            f"Missing domain assignments: {missing_groups['Feature_ID'].tolist()}"
        )

    # The summary stores zero percentages when no gold is evaluable.
    # Treat those as undefined so they do not affect domain averages.
    no_gold = summary["total"] == summary["gold_missing"]
    summary.loc[no_gold, list(CATEGORIES)] = float("nan")

    main_domains = (
        summary.groupby("Main_domain", as_index=False)[list(CATEGORIES)].mean()
    )
    finer_domains = (
        summary.groupby("Finer_grouping", as_index=False)[list(CATEGORIES)].mean()
    )

    timestamp = get_timestamp()

    if output_dir is None:
        output_dir = summary_file.parent
        output_path_variables_by_domain = output_dir / f"variables_{timestamp}.png"
        output_path_main = output_dir / f"main_domains_{timestamp}.png"
        output_path_finer = output_dir / f"finer_domains_{timestamp}.png"
    else:
        # Don't add the timestamp if the output directory was specified
        output_path_variables_by_domain = output_dir / f"variables.png"
        output_path_main = output_dir / f"main_domains.png"
        output_path_finer = output_dir / f"finer_domains.png"

    # For all three filepaths, add "_highlights" if bool_highlights is True
    if bool_highlights:
        output_path_variables_by_domain = output_path_variables_by_domain.with_name(
            output_path_variables_by_domain.stem + "_highlights.png"
        )
        output_path_main = output_path_main.with_name(
            output_path_main.stem + "_highlights.png"
        )
        output_path_finer = output_path_finer.with_name(
            output_path_finer.stem + "_highlights.png"
        )

    model_label = ", ".join(models)

    plot_variables_by_domain(
        data=summary,
        domains=sorted(grouping["Main_domain"].dropna().unique().tolist()),
        models=models,
        output_path=output_path_variables_by_domain,
        bool_highlights=bool_highlights,
    )

    for data, label, name, filename in (
        (main_domains, "Main_domain", "Main domains", "main_domains"),
        (finer_domains, "Finer_grouping", "Finer domains", "finer_domains"),
    ):
        plot_stacked_percentages(
            data=data,
            label_column=label,
            title=f"{name} — models: {model_label}",
            output_path=output_path_main if filename == "main_domains" else output_path_finer,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot variable and domain statistics.")
    parser.add_argument("summary_file", type=Path)

    # Output directory for the plots (defaults to the summary file's parent directory).
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the plots (default: summary file's parent directory).",
    )

    parser.add_argument("--models", type=parse_csv_list, required=True)
    parser.add_argument(
        "--grouping-file",
        type=Path,
        default=Path("feature_grouping_for_analysis.csv"),
    )

    # Bool to plot highlights on the variables-by-domain plot.
    parser.add_argument(
        "--bool-highlights",
        action="store_true",
        help="Whether to plot highlights on the variables-by-domain plot.",
    )

    args = parser.parse_args()
    plot_variables_and_domains(
        summary_file = args.summary_file, 
        models = args.models,
        grouping_file = args.grouping_file,
        output_dir = args.output_dir,
        bool_highlights = args.bool_highlights,
    )