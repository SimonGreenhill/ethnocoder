#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "pymupdf",
# ]
# ///
"""
Print per-document statistics from the docs/ directory.

Usage:
    python check_pdf.py
    python check_pdf.py --docs-dir docs --gold-dir gold
"""

import json
from pathlib import Path

import pymupdf


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
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--gold-dir", type=Path, default=Path("gold"))
    args = parser.parse_args()

    header = f"{'source':<30s} {'pages':>5s} {'chars':>10s} {'coded':>5s}"
    print(header)
    print("-" * len(header))

    rows = []
    for gold_file in sorted(args.gold_dir.glob("*.json")):
        stem = gold_file.stem
        pdf_path = args.docs_dir / f"{stem}.pdf"
        if not pdf_path.exists():
            continue

        ps = pdf_stats(pdf_path)
        gs = gold_stats(gold_file)
        rows.append({"stem": stem, **ps, **gs})
        print(f"{stem:<30s} {ps['pages']:5d} {ps['chars']:10,d} {gs['coded']:5d}")

    print(f"\n{'':30s} {'pages':>5s} {'chars':>10s} {'coded':>5s}")
    if rows:
        n = len(rows)
        avg = lambda k: sum(r[k] for r in rows) / n
        print(f"{'mean':<30s} {avg('pages'):5.0f} {avg('chars'):10,.0f} {avg('coded'):5.1f}")
        mn = lambda k: min(r[k] for r in rows)
        mx = lambda k: max(r[k] for r in rows)
        print(f"{'min':<30s} {mn('pages'):5d} {mn('chars'):10,d} {mn('coded'):5d}")
        print(f"{'max':<30s} {mx('pages'):5d} {mx('chars'):10,d} {mx('coded'):5d}")
        print(f"\n{n} documents")


if __name__ == "__main__":
    main()
