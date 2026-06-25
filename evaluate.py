#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12,<3.14"
# ///
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
import textwrap
from pathlib import Path

# ANSI colour helpers — disabled automatically when not a TTY
def _is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _is_tty() else text

def green(t: str) -> str: return _c("32", t)
def red(t: str) -> str:   return _c("31", t)
def yellow(t: str) -> str: return _c("33", t)
def dim(t: str) -> str:   return _c("2", t)

CONF_COLOUR = {
    "high":   green,
    "medium": yellow,
    "low":    red,
    "absent": red,
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


def eval_codings(gold_path: Path, coded_path: Path, variables_path: Path) -> None:
    with open(gold_path, encoding="utf-8") as f:
        gold_raw = json.load(f)
    coded_text = strip_fences(coded_path.read_text(encoding="utf-8"))
    # Fix double-brace bug from older prefill code
    if coded_text.startswith("{{"):
        coded_text = coded_text[1:]
    coded_raw = json.loads(coded_text)
    # Old log-entry format: {"timestamp":..., "raw_response": "<json string>"}
    if isinstance(coded_raw, dict) and "raw_response" in coded_raw and "codings" not in coded_raw:
        coded_raw = json.loads(strip_fences(coded_raw["raw_response"]))

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
    codings_list = coded_raw if isinstance(coded_raw, list) else coded_raw.get("codings", [])
    coded: dict[str, str] = {}
    coded_confidence: dict[str, str] = {}
    coded_justification: dict[str, str] = {}
    for r in codings_list:
        vid = str(r.get("id") or r.get("variable"))
        coded[vid] = normalize_code(r.get("code"))
        if r.get("confidence"):
            coded_confidence[vid] = r["confidence"]
        if r.get("justification"):
            coded_justification[vid] = r["justification"]

    matches = 0
    mismatches = 0
    missing = 0  # in gold but not in coded output

    col_id = 6
    col_name = 35
    col_gold = 10
    col_coded = 10
    col_conf = 8
    wrap_width = 100

    header = (
        f"{'ID':<{col_id}}  {'Variable':<{col_name}}  {'Gold':<{col_gold}}  {'Coded':<{col_coded}}  {'Conf':<{col_conf}}  Match"
    )
    print(dim(header))
    print(dim("-" * len(header)))

    indent = " " * (col_id + 2)

    for var_id, gold_code in sorted(gold.items(), key=lambda x: int(x[0])):
        name = var_names.get(var_id, "?")[:col_name]
        confidence = coded_confidence.get(var_id, "")
        justification = coded_justification.get(var_id, "")
        conf_coloured = CONF_COLOUR.get(confidence, str)(confidence) if confidence else ""

        if var_id not in coded:
            coded_code = "(missing)"
            mark = yellow("?")
            missing += 1
        else:
            coded_code = coded[var_id]
            if gold_code == coded_code:
                mark = green("✓")
                matches += 1
            else:
                mark = red("✗")
                mismatches += 1

        # Pad confidence to col_conf using raw string length (colour codes add invisible chars)
        conf_pad = conf_coloured + " " * max(0, col_conf - len(confidence))
        print(
            f"{var_id:<{col_id}}  {name:<{col_name}}  {gold_code:<{col_gold}}  {coded_code:<{col_coded}}  {conf_pad}  {mark}"
        )
        if justification:
            for line in textwrap.wrap(justification, width=wrap_width - len(indent)):
                print(dim(indent + line))

    total = matches + mismatches + missing
    pct = 100 * matches / (matches + mismatches) if (matches + mismatches) > 0 else 0
    colour = green if pct >= 80 else yellow if pct >= 50 else red
    print()
    print(f"Correct:  {colour(f'{matches}/{matches + mismatches} ({pct:.1f}%)')}")
    if missing:
        print(yellow(f"Missing:  {missing} variables present in gold but absent from coded output"))
    print(dim(f"Skipped:  {len(gold_raw) - total} variables with null gold code"))


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
