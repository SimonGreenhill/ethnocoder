import argparse
import shutil
from pathlib import Path

from ethnocoder.common.utils import parse_csv_list
from ethnocoder.results.check_single import main as check_single
from ethnocoder.results.get_results_stats import summarise_tsv_results


def calculate_stats(
        root: Path, 
        delete_existing_tsvs: bool = False,
        bool_output_to_terminal: bool = True,
        excluded_languages: list[str] | None = None,
    ) -> None:
    """
    Calculate statistics for a batch of language directories.

    Args:
        root (Path): The root directory containing language subdirectories.
        delete_existing_tsvs (bool): Whether to delete existing TSV files before processing.
        bool_output_to_terminal (bool): Whether to print output to the terminal.
        excluded_languages: Language directory names to exclude from processing and summaries.
    """

    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    # Avoid allowing the deletion option on broad directories.
    if root in {Path("/"), Path.home().resolve(), Path.cwd().resolve()}:
        raise ValueError(f"Choose a batch directory, not {root}")

    language_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name not in {"summary", "metasummary"}
    )

    # Validate exclusions before processing or deleting any files.
    if excluded_languages:
        excluded = set(excluded_languages)
        missing = excluded - {path.name for path in language_dirs}

        if missing:
            raise ValueError(f"Language directories not found: {sorted(missing)}")

        # The deletion option affects the entire batch, including excluded languages.
        if delete_existing_tsvs:
            raise ValueError(
                "--delete-existing-tsvs cannot be used with --exclude-languages"
            )

        language_dirs = [path for path in language_dirs if path.name not in excluded]
        print(f"Excluding languages: {', '.join(excluded_languages)}")

    if not language_dirs:
        raise ValueError(f"No language directories found in {root}")

    jobs = []
    for language_dir in language_dirs:
        gold = language_dir / "gold.json"
        if not gold.is_file():
            raise FileNotFoundError(f"Gold file not found: {gold}")

        model_dirs = sorted(path for path in language_dir.iterdir() if path.is_dir())
        if not model_dirs:
            raise ValueError(f"No model directories found in {language_dir}")

        for model_dir in model_dirs:
            outputs = []
            for path in model_dir.glob("*.json"):
                prefix, separator, number = path.stem.rpartition("_")
                if separator and number.isdecimal():
                    outputs.append((prefix, int(number), path))

            if not outputs:
                raise FileNotFoundError(f"No numbered model output in {model_dir}")
            if len({prefix for prefix, _, _ in outputs}) != 1:
                raise ValueError(f"Multiple numbered output series in {model_dir}")

            latest = max(outputs, key=lambda item: item[1])[2]
            jobs.append((language_dir.name, model_dir.name, gold, latest))

    models = {model for _, model, _, _ in jobs}
    summaries = {}
    for language, model, _, _ in jobs:
        summary_dir = root / "summary"
        if len(models) > 1:
            summary_dir = summary_dir / model
        summaries.setdefault(summary_dir, set()).add(f"results_all_{language}.tsv")

    # Keep the existing stale-input check for full-batch summaries.
    if not delete_existing_tsvs and not excluded_languages:
        for summary_dir, expected in summaries.items():
            if summary_dir.is_dir():
                stale = [
                    path for path in summary_dir.glob("results_all*.tsv")
                    if path.name not in expected
                ]
                if stale:
                    raise ValueError(
                        f"Unexpected summary inputs in {summary_dir}: {stale}. "
                        "Remove them or use --delete-existing-tsvs."
                    )

    if delete_existing_tsvs:
        for path in sorted(root.rglob("*.tsv")):
            print(f"Deleting {path}")
            path.unlink()

    for language, model, gold, output in jobs:
        print(f"Evaluating {language} / {model}: {output}")
        check_single(gold, output, bool_output_to_terminal=bool_output_to_terminal)

    for language, model, _, output in jobs:
        summary_dir = root / "summary"
        if len(models) > 1:
            summary_dir = summary_dir / model
        summary_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            output.parent / "results_all.tsv",
            summary_dir / f"results_all_{language}.tsv",
        )

    # Pass only this run's TSVs so old files cannot enter a partial summary.
    for summary_dir, filenames in summaries.items():
        summarise_tsv_results(
            directory=str(summary_dir),
            bool_output_to_terminal=bool_output_to_terminal,
            tsv_files=[str(summary_dir / name) for name in sorted(filenames)],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalculate batch statistics.")
    parser.add_argument("directory", type=Path, help="Batch output directory")
    parser.add_argument(
        "--delete-existing-tsvs",
        action="store_true",
        help="Delete all existing TSV files under the batch directory first.",
    )
    parser.add_argument(
        "--exclude-languages",
        type=parse_csv_list,
        help="Comma-separated language directory names to exclude.",
    )
    args = parser.parse_args()
    calculate_stats(
        root=args.directory,
        delete_existing_tsvs=args.delete_existing_tsvs,
        bool_output_to_terminal=True,
        excluded_languages=args.exclude_languages,
    )