# Written by Terra, not checked!

import argparse
from pathlib import Path
from rich import print

from ethnocoder.common.utils import parse_csv_list
from ethnocoder.results.get_results_stats import summarise_tsv_results
from ethnocoder.plotting.plot_variables_and_domains import plot_variables_and_domains


def analyse_variables(
    root: Path,
    requested_models: list[str] | None = None,
    excluded_languages: list[str] | None = None,
) -> tuple[Path, list[str]]:
    """
    Analyse variables for a batch of models and languages.

    Args:
        root: Path to the batch output directory.
        requested_models: List of models to include; defaults to all discovered models.
        excluded_languages: Language directory names to exclude from the analysis.

    Returns:
        A tuple containing the path to the summary file and the list of models used.
    """
    
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    languages = sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name not in {"summary", "metasummary"}
    )

    # Exclude incomplete languages before selecting models or reading TSVs.
    if excluded_languages:
        excluded = set(excluded_languages)
        missing = excluded - {path.name for path in languages}

        if missing:
            raise ValueError(f"Language directories not found: {sorted(missing)}")

        languages = [path for path in languages if path.name not in excluded]
        print(f"[yellow]Excluding languages: {', '.join(excluded_languages)}[/yellow]")

    if not languages:
        raise ValueError(f"No language directories found in {root}")

    available_models = sorted({
        model.name
        for language in languages
        for model in language.iterdir()
        if model.is_dir()
    })
    models = requested_models if requested_models is not None else available_models
    if not models:
        raise ValueError(f"No model directories found in {root}")

    files = []
    for language in languages:
        for model in models:
            result = language / model / "results_all.tsv"
            if not result.is_file():
                raise FileNotFoundError(f"Missing evaluation TSV: {result}")
            files.append(str(result))

    summary_dir = root / "summary"
    summary_dir.mkdir(exist_ok=True)
    summary_file = summarise_tsv_results(
        directory=str(summary_dir),
        bool_output_to_terminal=False,
        tsv_files=files,
    )
    if not summary_file:
        raise ValueError("No summary was produced")

    print(f"[green]Combined models: {', '.join(models)}[/green]")
    print(f"[green]Summary file: [spring_green2]{summary_file}[/spring_green2][/green]")
    return Path(summary_file), models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse batch variables.")
    parser.add_argument("directory", type=Path, help="Batch output directory")
    parser.add_argument(
        "--models",
        type=parse_csv_list,
        default=None,
        help="Comma-separated models; defaults to all discovered models.",
    )
    parser.add_argument(
        "--exclude-languages",
        type=parse_csv_list,
        default=None,
        help="Comma-separated language directory names to exclude.",
    )
    args = parser.parse_args()

    summary_file, models = analyse_variables(
        root=args.directory,
        requested_models=args.models,
        excluded_languages=args.exclude_languages,
    )
    plot_variables_and_domains(summary_file=summary_file, models=models)