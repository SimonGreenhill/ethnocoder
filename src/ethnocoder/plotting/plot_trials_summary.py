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

For each language, the majority baseline is the percentage of gold IDs whose
gold code matches the code in majority.json. Gold IDs missing from that file
count as incorrect, and are reported as a warning.

The plot is saved as trials_summary.png in the input directory by
default, then shown. Use --output to choose a different output path.
"""

import os
import argparse
import json
from pathlib import Path
from rich import print

import matplotlib.pyplot as plt

from ethnocoder.common.utils import get_timestamp, parse_csv_list

FPATH_MAJORITY = os.path.join("sandbox", "majority.json")

LANGUAGE_LABELS = { # quick fix, needs doing properly with the CLDF dataset
    "mock_dutch": "Mock Dutch",
    "island_carib": "Kalinago (Island Carib)",
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
        raise FileNotFoundError(f"Model directory does not exist: {model_directory}")

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
    Load accuracy for every language/model and calculate each majority baseline.
    
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
            model_accuracies[language][model] = load_accuracy(stats_path)

    return model_accuracies, majority_accuracies, languages


def plot_results(
    model_accuracies: dict[str, dict[str, float]],
    majority_accuracies: dict[str, float],
    output_path: Path,
    languages: list[str],
    models: list[str],
) -> None:
    """
    Draw grouped model-accuracy points and chance/majority baselines.

    Each model is horizontally offset within each language group. The chance
    baseline spans the full plot; majority baselines are dashed segments
    centered on their respective language groups.
    """

    # Prep: order x-axis groups by majority_accuracies, lowest to highest.
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
        y_values = [
            model_accuracies[language][model]
            for language in languages
        ]

        axis.scatter(
            x_values,
            y_values,
            color=colors[model_index % len(colors)],
            marker=markers[model_index % len(markers)],
            s=65,
            label=model,
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
            label="Majority baseline" if index == 0 else "_nolegend_",
            zorder=2,
        )

    # Xtick labels should be the actual language names, from LANGUAGE_LABELS where possible.
    axis.set_xticks(range(len(languages)))
    axis.set_xticklabels([LANGUAGE_LABELS.get(lang, lang) for lang in languages])

    axis.set_xlim(-0.5, len(languages) - 0.5)
    axis.set_ylim(0, 100)
    axis.set_xlabel("Language")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Accuracy by language and model")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Model / baseline", loc="lower right")

    # Increase fontsize of both axes labels and tick labels.
    axis.xaxis.label.set_size(16)
    axis.yaxis.label.set_size(16)
    axis.tick_params(axis="both", which="major", labelsize=14)

    # Rotate x-axis tick labels for better readability.
    axis.set_xticklabels(
        [LANGUAGE_LABELS.get(lang, lang) for lang in languages],
        rotation=45,
        ha="right"
    )

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
    args = parser.parse_args()

    root = args.directory.expanduser().resolve()

    if not root.is_dir():
        parser.error(f"Input directory does not exist or is not a directory: {root}")

    # Relative output paths are interpreted from the current working directory;
    # if omitted, save the plot inside the experiment directory.
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

        model_accuracies, majority_accuracies, languages = collect_results(
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
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()