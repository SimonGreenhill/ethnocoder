import argparse
import csv
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ethnocoder.utils import strip_fences, parse_codings

console = Console()

CONF_STYLE = {
    "high":   "green",
    "medium": "yellow",
    "low":    "red",
    "absent": "red",
}

PARAMETERS_CSV = Path("./parameters.csv")
GOLD_DIR = Path("./gold")


def normalize_code(value) -> str:
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
    return parse_codings(strip_fences(path.read_text(encoding="utf-8")))


def load_codings_as_dict(path: Path) -> dict[str, str]:
    return {
        str(r.get("id") or r.get("variable")): normalize_code(r.get("code"))
        for r in load_codings(path)
        if r.get("code") is not None
    }


def eval_codings(gold_path: Path, coded_path: Path, variables_path: Path) -> None:
    gold_raw = load_codings(gold_path)
    coded_list = load_codings(coded_path)

    var_names: dict[str, str] = {}
    with open(variables_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            var_names[row["ID"]] = row["Name"]

    gold: dict[str, str] = {
        str(r["id"]): normalize_code(r["code"])
        for r in gold_raw
        if r.get("code") is not None
    }

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

    matches = mismatches = missing = 0

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


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "evaluate",
        description="Compare gold and coded JSON files for cultural trait codings",
        help="Compare gold and coded JSON files for cultural trait codings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("doc", help="Coded JSON file")
    p.add_argument(
        "--variables",
        default=str(PARAMETERS_CSV),
        help=f"Variables CSV for names (default: {PARAMETERS_CSV})",
    )
    p.set_defaults(func=_run)


def _run(args) -> None:
    coded = Path(args.doc)
    gold = GOLD_DIR / coded.name
    for p in (coded, gold):
        if not p.exists():
            sys.exit(f"Error: {p} not found")
    eval_codings(gold, coded, Path(args.variables))
