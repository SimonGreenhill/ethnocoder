"""Create per-variable TSVs from per-language results in a summary directory.

Run with a summary directory and comma-separated variable IDs, for example:

    python -m ethnocoder.analysis.create_variable_summary summary --variables GB001,GB002

Each output is written to ``summary/variables/<variable>.tsv`` and contains
the matching rows from ``results_all_<language>.tsv`` files.
"""

import argparse
from pathlib import Path
from rich import print
import pandas as pd

from ethnocoder.common.utils import parse_csv_list


def create_variable_summaries(summary_dir: Path, variables: list[str]) -> None:
    """Combine per-language result rows into one TSV per variable.

    The language is taken from each input filename and inserted after the
    confidence column. Output rows follow the verdict, confidence, gold,
    model output, and feature ID sort order used by check_single. A warning
    is printed when a variable is absent from an input file; an output is
    still written even if the variable is absent from every file.

    Args:
        summary_dir: Directory containing ``results_all_<language>.tsv`` files.
        variables: Feature IDs to extract and use as output filenames.

    Raises:
        NotADirectoryError: If the summary directory does not exist.
        FileNotFoundError: If there are no per-language result files.
        ValueError: If variable IDs or input filenames are invalid, or an
            input file has no ``Feature_ID`` column.
    """
    # Validate the input directory and IDs before creating output files.
    if not summary_dir.is_dir():
        raise NotADirectoryError(summary_dir)
    if not variables:
        raise ValueError("Provide at least one variable")
    if any(variable in {"", ".", ".."} or "/" in variable or "\\" in variable for variable in variables):
        raise ValueError("Variable IDs must be valid file names")

    # Read only per-language results directly inside the supplied directory.
    files = sorted(path for path in summary_dir.glob("results_all_*.tsv") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No per-language results found in {summary_dir}")

    # Collect one filtered frame per input file for every requested variable.
    rows = {variable: [] for variable in variables}
    for path in files:
        # Preserve the complete language suffix, including any underscores.
        language = path.stem.removeprefix("results_all_")
        if not language:
            raise ValueError(f"Missing language in file name: {path}")
        # Keep values as text, including empty cells, when copying them to outputs.
        data = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if "Feature_ID" not in data.columns:
            raise ValueError(f"Missing Feature_ID column in {path}")
        data.insert(data.columns.get_loc("confidence") + 1, "language", language)
        for variable in variables:
            matching = data[data["Feature_ID"] == variable]
            if matching.empty:
                print(f"Warning: {variable} not found in {path}")
            # Empty frames retain the input columns for header-only outputs.
            rows[variable].append(matching)

    # Write the resulting files beneath the supplied summary directory.
    output_dir = summary_dir / "variables"
    output_dir.mkdir(exist_ok=True)
    # These category orders match check_single rather than alphabetical order.
    sort_order = {
        "verdict": ["correct", "incorrect", "hedged", "missing_gold", "missing_model"],
        "confidence": ["high", "medium", "low", "absent"],
        "gold": ["0", "1", "2", "3", "?"],
        "model_output": ["0", "1", "2", "3", "?"],
    }
    for variable, frames in rows.items():
        output_path = output_dir / f"{variable}.tsv"
        combined = pd.concat(frames, ignore_index=True)
        # Map known values to sort ranks; unknown and empty values sort last.
        # The original cell values remain unchanged in the output.
        combined = combined.sort_values(
            by=["verdict", "confidence", "gold", "model_output", "Feature_ID"],
            key=lambda column: column.map(
                {value: rank for rank, value in enumerate(sort_order[column.name])}
            ) if column.name in sort_order else column,
        )
        combined.to_csv(output_path, sep="\t", index=False)
        print(f"[green]Saved [spring_green2]{output_path}[/spring_green2][/green]")


if __name__ == "__main__":
    # Reuse the project's comma-separated list parser for variable IDs.
    parser = argparse.ArgumentParser(description="Create per-variable summaries.")
    parser.add_argument("summary_directory", type=Path, help="Summary directory")
    parser.add_argument(
        "--variables",
        type=parse_csv_list,
        required=True,
        help="Comma-separated variable IDs",
    )
    args = parser.parse_args()
    create_variable_summaries(args.summary_directory, args.variables)