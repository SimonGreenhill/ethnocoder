#!/usr/bin/env python3
"""
Compare gold and coded JSON files for cultural trait codings.

Usage:
    python evaluate.py cain1971                      # auto-resolve gold/cain1971.json and cain1971.json
    python evaluate.py coded.json gold.json          # explicit paths
    python evaluate.py cain1971 --gold-dir my_gold   # custom gold directory
    python evaluate.py cain1971 --coded-dir llama3.2 # coded file in a subdirectory
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()

CONF_STYLE = {
    "high":   "green",
    "medium": "yellow",
    "low":    "red",
    "absent": "red",
}

VARIABLES_CSV = Path("./variables.csv")
GOLD_DIR = Path("./gold")

def strip_fences(text: str) -> str:
    """Strip markdown code fences that models add despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
    if text.endswith("```"):
        text = text[:text.rindex("```")]
    return text.strip()


def normalize_code(value) -> str:
    """Normalize a code value to a canonical string for comparison."""
    if value is None:
        return ""
    s = str(value).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, OverflowError):
        return s


def load_codings(path: Path) -> list[dict]:
    """Load a codings JSON file, handling format variants."""
    text = strip_fences(path.read_text(encoding="utf-8"))
    if text.startswith("{{"):
        text = text[1:]
    raw = json.loads(text)
    if isinstance(raw, dict) and "raw_response" in raw and "codings" not in raw:
        raw = json.loads(strip_fences(raw["raw_response"]))
    if isinstance(raw, list):
        return raw
    return raw.get("codings", [])


def load_codings_as_dict(path: Path) -> dict[str, str]:
    """Load codings and return {id: normalized_code}, skipping nulls."""
    return {
        str(r.get("id") or r.get("variable")): normalize_code(r.get("code"))
        for r in load_codings(path)
        if r.get("code") is not None
    }


def eval_codings(gold_path: Path, coded_path: Path, variables_path: Path) -> None:
    gold_raw = load_codings(gold_path)
    coded_list = load_codings(coded_path)

    # Load variable names for display
    var_names: dict[str, str] = {}
    with open(variables_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            var_names[row["ID"]] = row["Name"]

    # Index gold by id; skip nulls
    gold: dict[str, str] = {
        str(r["id"]): normalize_code(r["code"])
        for r in gold_raw
        if r.get("code") is not None
    }

    # Index coded by id (handle both "id" and older "variable" key)
    coded: dict[str, str] = {}
    coded_confidence: dict[str, str] = {}
    coded_justification: dict[str, str] = {}
    for r in coded_list:
        vid = str(r.get("id") or r.get("variable"))
        coded[vid] = normalize_code(r.get("code"))
        if r.get("confidence"):
            coded_confidence[vid] = r["confidence"]
        if r.get("justification"):
            coded_justification[vid] = r["justification"]

    matches = 0
    mismatches = 0
    missing = 0

    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column("ID", width=6)
    table.add_column("Variable", width=35)
    table.add_column("Gold", width=10)
    table.add_column("Coded", width=10)
    table.add_column("Conf", width=8)
    table.add_column("Match", width=5)

    for var_id, gold_code in sorted(gold.items(), key=lambda x: int(x[0])):
        name = var_names.get(var_id, "?")[:35]
        confidence = coded_confidence.get(var_id, "")
        justification = coded_justification.get(var_id, "")
        conf_style = CONF_STYLE.get(confidence, "")
        conf_text = Text(confidence, style=conf_style) if confidence else Text("")

        if var_id not in coded:
            coded_code = "(missing)"
            mark = Text("?", style="yellow")
            missing += 1
        else:
            coded_code = coded[var_id]
            if gold_code == coded_code:
                mark = Text("✓", style="green")
                matches += 1
            else:
                mark = Text("✗", style="red")
                mismatches += 1

        table.add_row(var_id, name, gold_code, coded_code, conf_text, mark)
        if justification:
            table.add_row("", Text(justification, style="dim"), "", "", "", "")

    console.print(table)

    total = matches + mismatches + missing
    pct = 100 * matches / (matches + mismatches) if (matches + mismatches) > 0 else 0
    style = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
    console.print()
    console.print(f"Correct:  [{style}]{matches}/{matches + mismatches} ({pct:.1f}%)[/{style}]")
    if missing:
        console.print(f"[yellow]Missing:  {missing} variables present in gold but absent from coded output[/yellow]")
    console.print(f"[dim]Skipped:  {len(gold_raw) - total} variables with null gold code[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare gold and coded JSON files for cultural trait codings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("doc", help="Coded JSON file")
    parser.add_argument(
        "--variables",
        default=str(VARIABLES_CSV),
        help=f"Variables CSV for names (default: {VARIABLES_CSV})",
    )

    args = parser.parse_args()

    coded = Path(args.doc)
    gold = GOLD_DIR / coded.name
    for p in (coded, gold):
        if not p.exists():
            sys.exit(f"Error: {p} not found")

    eval_codings(gold, coded, Path(args.variables))


if __name__ == "__main__":
    main()
