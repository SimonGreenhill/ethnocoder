import argparse
import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
from rich import print

from ethnocoder.common.utils import get_timestamp


def plot_accuracy_per_stats(directory: str) -> None:
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)

    results = {}
    for language_dir in sorted(root.iterdir()):
        if not language_dir.is_dir() or language_dir.name in {"summary", "metasummary"}:
            continue

        # Support both older flat batches and language/model batches.
        flat_stats = language_dir / "stats.json"
        if flat_stats.is_file():
            stats_files = [(language_dir.name, flat_stats)]
        else:
            model_dirs = sorted(path for path in language_dir.iterdir() if path.is_dir())
            if not model_dirs:
                raise FileNotFoundError(f"No stats.json or model directories in {language_dir}")
            stats_files = [
                (f"{language_dir.name}/{model_dir.name}", model_dir / "stats.json")
                for model_dir in model_dirs
            ]

        for label, stats_file in stats_files:
            with stats_file.open(encoding="utf-8") as file:
                stats = json.load(file)

            totals = stats["totals"]
            resources = stats["resources"]
            results[label] = {
                "accuracy": totals["correct_pct"],
                "time": resources["total_time"],
                "reasoning": resources["total_reasoning_tokens"],
                "size": sum(
                    totals[key]
                    for key in ("correct", "incorrect", "hedged", "model_missing")
                ),
            }

    if not results:
        raise ValueError(f"No results found in {root}")

    fig, time_axis = plt.subplots()
    reasoning_axis = time_axis.twinx()
    time_axis.set_xlabel("Accuracy")
    time_axis.set_ylabel("Time (s)")
    reasoning_axis.set_ylabel("Reasoning (1000x tokens)")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sizes = [values["size"] for values in results.values()]
    min_size, max_size = min(sizes), max(sizes)
    language_handles = []

    for index, (label, values) in enumerate(sorted(results.items())):
        color = colors[index % len(colors)]
        size = values["size"]
        marker_size = (
            9 if min_size == max_size
            else 6 + 6 * (size - min_size) / (max_size - min_size)
        )

        time_axis.plot(
            values["accuracy"], values["time"],
            marker="o", linestyle="None", color=color,
            markersize=marker_size, markeredgecolor="black",
        )
        reasoning_axis.plot(
            values["accuracy"], values["reasoning"],
            marker="^", linestyle="None", color=color,
            markersize=marker_size, markeredgecolor="black",
        )
        language_handles.append(
            Line2D([], [], color=color, marker="s", linestyle="None",
                   label=f"{label} ({size})")
        )

    time_axis.set_xlim(0, 100)
    time_axis.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    reasoning_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000:g}")
    )
    time_axis.set_ylim(0, max(1, 1.1 * max(v["time"] for v in results.values())))
    reasoning_axis.set_ylim(0, max(1, 1.1 * max(v["reasoning"] for v in results.values())))

    language_legend = time_axis.legend(
        handles=language_handles, title="Glottocode / model", loc="upper left"
    )
    time_axis.add_artist(language_legend)
    reasoning_axis.legend(
        handles=[
            Line2D([], [], color="black", marker="o", linestyle="None", label="Time"),
            Line2D([], [], color="black", marker="^", linestyle="None", label="Reasoning"),
        ],
        title="Metric",
        loc="lower left",
    )

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_path = summary_dir / f"accuracy_time_reasoning_{get_timestamp()}.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[green]Saved plot to {output_path}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot batch accuracy, time and reasoning.")
    parser.add_argument("directory", help="Batch output directory")
    args = parser.parse_args()
    plot_accuracy_per_stats(args.directory)