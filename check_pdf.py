#!/usr/bin/env python3
"""
Print per-document statistics from the docs/ directory.

Usage:
    python check_pdf.py
    python check_pdf.py --docs-dir docs --gold-dir gold
"""

import argparse
import json
from pathlib import Path

import pymupdf
from rich.console import Console
from rich.table import Table

console = Console()


def pdf_stats(pdf_path: Path) -> dict:
    doc = pymupdf.open(str(pdf_path))
    pages = len(doc)
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    chars = len(text)
    return {"pages": pages, "chars": chars}


def gold_stats(gold_path: Path) -> dict:
    o = json.loads(gold_path.read_text())
    if "codings" in o:
        o = o["codings"]
    coded = sum(1 for x in o if x.get("code") is not None)
    return {"coded": coded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--gold-dir", type=Path, default=Path("gold"))
    args = parser.parse_args()

    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("Source", min_width=30)
    table.add_column("Pages", justify="right")
    table.add_column("Chars", justify="right")
    table.add_column("Coded", justify="right")

    rows = []
    for gold_file in sorted(args.gold_dir.glob("*.json")):
        stem = gold_file.stem
        pdf_path = args.docs_dir / f"{stem}.pdf"
        if not pdf_path.exists():
            continue

        ps = pdf_stats(pdf_path)
        gs = gold_stats(gold_file)
        rows.append({"stem": stem, **ps, **gs})
        table.add_row(stem, str(ps["pages"]), f"{ps['chars']:,d}", str(gs["coded"]))

    if rows:
        n = len(rows)
        avg = lambda k: sum(r[k] for r in rows) / n
        mn = lambda k: min(r[k] for r in rows)
        mx = lambda k: max(r[k] for r in rows)
        table.add_section()
        table.add_row("[dim]mean[/dim]", f"[dim]{avg('pages'):.0f}[/dim]", f"[dim]{avg('chars'):,.0f}[/dim]", f"[dim]{avg('coded'):.1f}[/dim]")
        table.add_row("[dim]min[/dim]", f"[dim]{mn('pages')}[/dim]", f"[dim]{mn('chars'):,d}[/dim]", f"[dim]{mn('coded')}[/dim]")
        table.add_row("[dim]max[/dim]", f"[dim]{mx('pages')}[/dim]", f"[dim]{mx('chars'):,d}[/dim]", f"[dim]{mx('coded')}[/dim]")

    console.print(table)
    if rows:
        console.print(f"\n{n} documents")


if __name__ == "__main__":
    main()
