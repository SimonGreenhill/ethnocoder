#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = []
# ///
"""
Run code_traits.py on all PDFs in docs/ under a size limit.

Usage:
    python run_batch.py
    python run_batch.py --max-mb 5
    python run_batch.py --model claude-opus-4-6 --max-mb 2 --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

DOCS_DIR = Path("docs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run code_traits.py on PDFs")
    parser.add_argument("model", help=f"Model to use")
    parser.add_argument("--max-mb", type=float, default=1.0, help="Max file size in MB (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without running it")
    parser.add_argument("--force", action="store_true", help="Re-run even if output already exists")
    args = parser.parse_args()

    max_bytes = args.max_mb * 1_000_000
    out_dir = Path(args.model.split("/")[-1])

    pdfs = sorted(DOCS_DIR.glob("*.pdf"), key=lambda p: p.stat().st_size)
    eligible = [p for p in pdfs if p.stat().st_size <= max_bytes]
    skipped_size = len(pdfs) - len(eligible)

    already_done = {p.stem for p in out_dir.glob("*.json")} if out_dir.exists() else set()
    todo = [p for p in eligible if args.force or p.stem not in already_done]
    skipped_done = len(eligible) - len(todo)

    print(f"PDFs in {DOCS_DIR}/: {len(pdfs)} total, {len(eligible)} under {args.max_mb}MB")
    print(f"Already coded: {skipped_done}  |  To run: {len(todo)}  |  Over size limit: {skipped_size}")
    print(f"Output dir: {out_dir}/")
    print()

    if not todo:
        print("Nothing to do.")
        return

    ok = 0
    failed = []
    for i, pdf in enumerate(todo, 1):
        size_mb = pdf.stat().st_size / 1_000_000
        print(f"[{i}/{len(todo)}] {pdf.name} ({size_mb:.2f}MB)", flush=True)
        if args.dry_run:
            print(f"  → would run: uv run code_traits.py code --model {args.model} {pdf}")
            continue
        result = subprocess.run(
            ["uv", "run", "code_traits.py", "code", "--model", args.model, str(pdf)],
        )
        if result.returncode == 0:
            ok += 1
        else:
            print(f"  FAILED (exit {result.returncode})", file=sys.stderr)
            failed.append(pdf.name)

    if not args.dry_run:
        print()
        print(f"Done: {ok}/{len(todo)} succeeded")
        if failed:
            print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
