"""
Plot model accuracy and language-specific baselines for a per-model per-language batch.

Expected directory layout:

    experiment_directory/
        simlish_explicit/
            gold_<language>.json
            tiny/
                .../stats.json
            small/
                .../stats.json
            medium/
                .../stats.json
            large/
                .../stats.json
        simlish/
            ...
        mock_dutch/
            ...
        island_carib/
            ...

Each model directory may contain nested directories, but it must contain
exactly one stats.json file. The script reads accuracy from
stats.json["totals"]["correct_pct"].

Gold and majority JSON files should contain either:
  * a list of {"id": ..., "code": ...} records;
  * an object with a "codings" list of those records; or
  * an object mapping IDs to codes.

For each language, the typicality baseline is the percentage of gold IDs whose
gold code matches the code in majority.json. Gold IDs missing from that file
count as incorrect, and are reported as a warning.

The plot is saved as trials_summary.png in the input directory by
default, then shown. Use --output to choose a different output path.
"""

import os
import math
import argparse
import json
from pathlib import Path
from rich import print

import matplotlib.pyplot as plt

from ethnocoder.common.utils import get_timestamp, parse_csv_list

FPATH_MAJORITY = "majority.json"

MODEL_SCATTERS = {
    "tiny": {
        "shape": "P",
        "color": "black"
    },
    "small": {
        "shape": "h",
        "color": "red"
    },
    "medium": {
        "shape": "o",
        "color": "#009E73"
    },
    "large": {
        "shape": "s",
        "color": "#E69F00"
    },
    "frontier": {
        "shape": "^",
        "color": "#0072B2"
    }
}

# MODELS = (
#     # "tiny",
#     "small",
#     "medium",
#     "large",
# )

CHANCE_BASELINE = 33.3

def load_codings(path: Path) -> dict[str, object]:
    """
    Load coding records from a JSON file and return them as an ID-to-code map.

    Accepted JSON shapes are:
      * A list of records, each with "id" and "code" fields.
      * An object with a "codings" field containing such a list.
      * An object mapping IDs directly to codes.

    Duplicate IDs and malformed records raise ValueError rather than being
    silently discarded, since either could change the calculated baseline.
    """
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Model-output files in this project may wrap their records in "codings".
    if isinstance(data, dict) and "codings" in data:
        data = data["codings"]

    # A direct mapping is convenient for majority.json files that store
    # {"feature-id": "most-common-code"} rather than a list of records.
    if isinstance(data, dict):
        records = [{"id": record_id, "code": code} for record_id, code in data.items()]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(
            f"{path} must contain a list of coding records or an ID-to-code object."
        )

    codings: dict[str, object] = {}

    for index, record in enumerate(records):
        if not isinstance(record, dict) or "id" not in record or "code" not in record:
            raise ValueError(
                f"{path}: record {index} must be an object with 'id' and 'code' fields."
            )

        # IDs are strings in the project's gold files; stringifying here also
        # makes numeric IDs in a mapping compare consistently with those files.
        record_id = str(record["id"])

        if record_id in codings:
            raise ValueError(f"{path}: duplicate coding ID {record_id!r}.")

        codings[record_id] = record["code"]

    if not codings:
        raise ValueError(f"{path} contains no coding records.")

    return codings


def find_stats_file(model_directory: Path) -> Path:
    """
    Find the single stats.json file belonging to a model.

    Searching recursively allows result files to live a few levels beneath
    the model directory. Requiring exactly one file avoids choosing an
    arbitrary trial when the directory contains multiple results.
    """

    if not model_directory.is_dir():
        return None
        # raise FileNotFoundError(f"Model directory does not exist: {model_directory}")

    stats_files = sorted(model_directory.rglob("stats.json"))

    if len(stats_files) != 1:
        raise ValueError(
            f"Expected exactly one stats.json under {model_directory}, "
            f"found {len(stats_files)}."
        )

    return stats_files[0]


def load_accuracy(stats_path: Path) -> float:
    """
    Read and validate a model's accuracy percentage from stats.json.

    The expected field is stats["totals"]["correct_pct"], matching the
    project's existing statistics output.
    """
    with stats_path.open("r", encoding="utf-8") as file:
        stats = json.load(file)

    try:
        accuracy = float(stats["totals"]["correct_pct"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{stats_path} must contain a numeric "
            "'totals.correct_pct' value."
        ) from error

    if not 0 <= accuracy <= 100:
        raise ValueError(
            f"{stats_path}: correct_pct must be between 0 and 100, got {accuracy}."
        )

    return accuracy

def load_model_stats(stats_path: Path) -> tuple[float, float]:
    """
    Load a model's accuracy and elapsed time from stats.json.

    Returns a tuple of (accuracy, elapsed_time).
    """
    with stats_path.open("r", encoding="utf-8") as file:
        stats = json.load(file)

    try:
        accuracy = float(stats["totals"]["correct_pct"])
        elapsed = float(stats["resources"]["total_time"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{stats_path} must contain numeric 'totals.correct_pct' and 'resources.total_time' values."
        ) from error

    if not 0 <= accuracy <= 100:
        raise ValueError(
            f"{stats_path}: correct_pct must be between 0 and 100, got {accuracy}."
        )

    if elapsed < 0:
        raise ValueError(
            f"{stats_path}: elapsed time must be non-negative, got {elapsed}."
        )

    return accuracy, elapsed

def calculate_majority_accuracy(
    gold: dict[str, object],
    majority: dict[str, object],
) -> tuple[float, list[str]]:
    """
    Compare gold codes with majority values for IDs present in both files.

    For ties, choose the first value in sorted order. An empty
    most_common_value list counts as incorrect. Return the accuracy
    percentage and the gold IDs absent from majority.
    """
    common_ids = gold.keys() & majority.keys()
    missing_ids = sorted(gold.keys() - majority.keys())

    if not common_ids:
        return 0.0, missing_ids

    correct = 0

    for id_ in sorted(common_ids):
        entry = majority[id_]

        if not isinstance(entry, dict) or "most_common_value" not in entry:
            raise ValueError(f"{id_}: missing majority 'most_common_value'.")

        values = entry["most_common_value"]
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"{id_}: 'most_common_value' must be a list of strings.")

        if not values:
            print(f"Warning: {id_} has no majority value; counting as incorrect.")
            continue

        # Sorting resolves ties deterministically: "0" before "1".
        if gold[id_] == sorted(values)[0]:
            correct += 1

    return 100 * correct / len(common_ids), missing_ids


def collect_results(
    root: Path,
    gold_filename: Path,
    majority_filepath: Path,
    languages: list[str],
    models: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, float], list[str]]: 
    """
    Load accuracy for every language/model and calculate each typicality baseline.
    
    majority_filepath is absolute.

    gold_filename is resolved relative to each language
    directory.
    """
    model_accuracies: dict[str, dict[str, float]] = {}
    majority_accuracies: dict[str, float] = {}

    # If languages is None, default to all subdirectories in root, except summary and metasummary.
    if languages is None:
        languages = [
            d.name
            for d in root.iterdir()
            if d.is_dir() and d.name not in {"summary", "metasummary"}
        ]

    # Initialise model total time count
    model_times = {model: 0.0 for model in models}

    # Loop at languages
    for language in languages:
        language_directory = root / language

        if not language_directory.is_dir():
            raise FileNotFoundError(
                f"Language directory does not exist: {language_directory}"
            )

        # The gold and majority files are language-specific, so each baseline
        # is evaluated against the gold records in that same language folder.
        gold_path = language_directory / gold_filename
        majority_path = majority_filepath
        use_majority = True

        if not gold_path.is_file():
            raise FileNotFoundError(f"Gold file does not exist: {gold_path}")
        if not majority_path.is_file():
            print(f"[yellow]Majority file does not exist: {majority_path}[/yellow]")
            use_majority = False

        gold = load_codings(gold_path)
        if use_majority:
            majority = load_codings(majority_path)
            baseline, missing_ids = calculate_majority_accuracy(gold, majority)
            majority_accuracies[language] = baseline
        else:
            print(f"[yellow]Using 0.0 for majority accuracy for language {language}[/yellow]")
            majority_accuracies[language] = 0.0
            missing_ids = []

        if missing_ids:
            print(
                f"Warning: {language}: majority.json is missing "
                f"{len(missing_ids)} gold ID(s); they are excluded from the count."
            )

        # Each expected model contributes one point to this language's group.
        model_accuracies[language] = {}
        for model in models:
            model_directory = language_directory / model
            stats_path = find_stats_file(model_directory)

            if stats_path is None:
                print(f"[yellow]Stats file not found for model {model} in language {language}[/yellow]")
                continue

            # Get model accuracies and times
            accuracy, elapsed = load_model_stats(stats_path)
            model_accuracies[language][model] = accuracy
            model_times[model] += elapsed

    return model_accuracies, majority_accuracies, languages, model_times


def plot_results(
    model_accuracies: dict[str, dict[str, float]],
    majority_accuracies: dict[str, float],
    output_path: Path,
    languages: list[str],
    models: list[str],
    model_times: dict[str, float],
    bool_averages: bool = False,
    root: Path | None = None,
) -> None:
    """
    Draw grouped model-accuracy points and chance/typicality baselines.

    Each model is horizontally offset within each language group. The chance
    baseline spans the full plot; typicality baselines are dashed segments
    centered on their respective language groups.
    """

    # quick fix: if the languages are JUST mock_dutch and island_carib,
    # put mock_dutch first if the languages are JUST mock_dutch and island_carib.
    # Otherwise order x-axis groups by majority_accuracies, lowest to highest.
    if set(languages) == {"mock_dutch", "island_carib"}:
        languages.sort(key=lambda lang: lang != "mock_dutch")
    else:
        languages.sort(key=lambda lang: majority_accuracies[lang])

    
    # Ensure that the majority_accuracies dictionary is also ordered according to the sorted languages.
    majority_accuracies = {lang: majority_accuracies[lang] for lang in languages}
    # Ensure that the model_accuracies dictionary is also ordered according to the sorted languages.
    model_accuracies = {lang: model_accuracies[lang] for lang in languages}


    figure, axis = plt.subplots(figsize=(11, 6))

    group_positions = {language: index for index, language in enumerate(languages)}

    # Offset model points slightly so all four values remain visible within
    # each language group instead of overlapping at the same x coordinate.
    model_spacing = 0.12
    model_offsets = {
        model: (index - (len(models) - 1) / 2) * model_spacing
        for index, model in enumerate(models)
    }

    # Use distinct colors and marker shapes to make the model series legible
    # in both the plot and its legend.
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D")

    for model_index, model in enumerate(models):

        x_values = [
            group_positions[language] + model_offsets[model]
            for language in languages
        ]
        # y_values = [
        #     model_accuracies[language][model]
        #     for language in languages
        # ]
        y_values = [
            model_accuracies[language].get(model, float("nan"))
            for language in languages
        ]

        # Get total time for this model across all languages.
        total_seconds = model_times[model]
        minutes, seconds = divmod(round(total_seconds), 60)
        hours, minutes = divmod(minutes, 60)
        label = f"{hours}h {minutes:02d}m {seconds:02d}s"

        axis.scatter(
            x_values,
            y_values,
            color=MODEL_SCATTERS[model]["color"] if model in MODEL_SCATTERS else colors[model_index % len(colors)],
            marker=MODEL_SCATTERS[model]["shape"] if model in MODEL_SCATTERS else markers[model_index % len(markers)],
            s=65,
            label=f"{model} ({label})",
            zorder=3,
        )

    # Chance is the same for every language, so it is shown as one full-width
    # dashed line.
    axis.axhline(
        CHANCE_BASELINE,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Chance ({CHANCE_BASELINE:.1f}%)",
        zorder=1,
    )

    # Majority accuracy varies by language; use one dashed segment per group.
    for index, language in enumerate(languages):
        axis.hlines(
            y=majority_accuracies[language],
            xmin=index - 0.32,
            xmax=index + 0.32,
            color="black",
            linestyle="--",
            linewidth=1.5,
            # Label only the first segment to keep a single legend entry.
            label="Typicality baseline" if index == 0 else "_nolegend_",
            zorder=2,
        )

    # Averages lines
    if bool_averages:
        for model_index, model in enumerate(models):
            
            # Get the average accuracy
            avg_accuracies = [model_accuracies[language].get(model, float("nan")) for language in languages]

            # Remove nan
            avg_accuracies = [acc for acc in avg_accuracies if not math.isnan(acc)]

            # Average over remaining values
            avg_accuracy = sum(avg_accuracies) / len(avg_accuracies) if avg_accuracies else float("nan")

            # Plot accuracy as a horizontal line across the plot, in the model's colour.
            axis.axhline(
                y=avg_accuracy,
                color=MODEL_SCATTERS[model]["color"] if model in MODEL_SCATTERS else colors[model_index % len(colors)],
                linestyle=":",
                linewidth=1.5,
                label="_nolegend",#f"{model} average" if model_index == 0 else "_nolegend_",
                zorder=2,
            )

    # Xtick labels should be the actual language names, from language_labels where possible.
    axis.set_xticks(range(len(languages)))

    # If root is not None, pull the file ending _meta.json from the root
    # and look for the entry "language_labels" to map language codes to their metadata.
    language_labels = {}
    if root is not None:

        # Search for exactly one file ending with _meta.json in the root directory.
        meta_files = list(root.glob("*_meta.json"))
        if len(meta_files) == 1:
            with open(meta_files[0], "r", encoding="utf-8") as f:
                meta_data = json.load(f)
                if "language_labels" in meta_data:
                    language_labels = meta_data["language_labels"]

    # Default to existing <lang> identifier if not found.
    language_labels = [language_labels.get(lang, lang) for lang in languages]

    # If it's a dict, get "glottocode", unless that's None, in which case use the language name.
    for i, lang in enumerate(language_labels):
        if isinstance(lang, dict) and "glottocode" in lang and lang["glottocode"] is not None:
            language_labels[i] = lang["glottocode"]
        elif isinstance(lang, dict) and "language_name" in lang and lang["language_name"] is not None:
            language_labels[i] = lang["language_name"]
        else:
            language_labels[i] = str(lang) # fallback

    # Rotate x-axis tick labels for better readability.
    if len(languages) > 3:
        axis.set_xticklabels(
            language_labels,
            rotation=45,
            ha="right"
        )
    else:
        axis.set_xticklabels(language_labels)

    axis.set_xlim(-0.5, len(languages) - 0.5)
    axis.set_ylim(0, 100)
    axis.set_xlabel("Language", fontsize=16)
    axis.set_ylabel("Accuracy (%)", fontsize=16)
    axis.set_title("Accuracy by language and model", fontsize=20)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Model / baseline", loc="lower right", fontsize=14)

    # Increase fontsize of both axes labels and tick labels.
    axis.xaxis.label.set_size(16)
    axis.yaxis.label.set_size(16)
    axis.tick_params(axis="both", which="major", labelsize=14)


    figure.tight_layout()

    # Create the output parent if a custom --output path points to a new folder.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[green]Saved plot to: [spring_green2]{output_path}[/spring_green2][/green]")

    # # Show the figure for interactive use, while still keeping the saved PNG.
    # plt.show()
    plt.close(figure)


def main() -> None:
    """
    Parse command-line arguments, load the experiment results, and plot them.
    """
    parser = argparse.ArgumentParser(
        description="Plot per-language per-model results (accuracy) with baselines per language."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        required=True,
        help="Directory containing the language subdirectories.",
    )

    # Output filepath
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output filepath for the summary plot (default: inside the experiment directory).",
    )

    parser.add_argument(
        "--languages",
        type=parse_csv_list,
        default=None,#list(LANGUAGES),
        help="Comma-separated languages (defaults to discovered subdirectories).",
    )
    parser.add_argument(
        "--models",
        type=parse_csv_list,
        default=None,
        help="Comma-separated models (defaults to discovered subdirectories).",
    )
    parser.add_argument(
        "--gold-file",
        type=Path,
        default=Path("gold.json"),
        help=(
            "Gold filename or path relative to each language directory "
            "(default: gold.json)."
        ),
    )
    parser.add_argument(
        "--majority-file",
        type=Path,
        default=FPATH_MAJORITY,
        help=(
            "Majority filename or path relative to each language directory "
            f"(default: {FPATH_MAJORITY})."
        ),
    )
    parser.add_argument(
        "--bool-averages",
        action="store_true",
        help="Whether to plot average accuracy as horizontal lines per model.",
    )
    args = parser.parse_args()

    root = args.directory.expanduser().resolve()

    if not root.is_dir():
        parser.error(f"Input directory does not exist or is not a directory: {root}")

    # Relative output paths are interpreted from the current working directory;
    # if omitted, save the plot inside the experiment directory.
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
    else:
        output_path = root / "summary" / f"trials_summary_{get_timestamp()}.png"

    try:
        languages = args.languages
        if languages is None:
            languages = sorted(
                path.name for path in root.iterdir()
                if path.is_dir() and path.name not in {"summary", "metasummary"}
            )
        if not languages:
            raise ValueError(f"No language directories found in {root}")

        models = args.models
        if models is None:
            model_names = set()
            for language in languages:
                language_dir = root / language
                if not language_dir.is_dir():
                    raise FileNotFoundError(f"Language directory not found: {language_dir}")
                model_names.update(
                    path.name for path in language_dir.iterdir() if path.is_dir()
                )
            models = sorted(model_names)
            if not models:
                raise ValueError(f"No model directories found in {root}")

        model_accuracies, majority_accuracies, languages, model_times = collect_results(
            root=root,
            gold_filename=args.gold_file,
            majority_filepath=args.majority_file,
            languages=languages,
            models=models,
        )
        plot_results(
            model_accuracies,
            majority_accuracies,
            output_path,
            languages=languages,
            models=models,
            model_times=model_times,
            bool_averages=args.bool_averages,
            root=root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()