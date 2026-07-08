import json
import sys
from pathlib import Path

from ethnocoder.code import code_pdf, load_codes, load_variables, model_dirname

DOCS_DIR = Path("docs")
PARAMETERS_CSV = Path("parameters.csv")
CODES_CSV = Path("codes.csv")


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "batch",
        description="Run ethnocoder code on all PDFs in docs/ under a size limit",
    )
    p.add_argument("model", help="LiteLLM model string")
    p.add_argument("--max-mb", type=float, default=1.0, help="Max file size in MB (default: 1.0)")
    p.add_argument("--dry-run", action="store_true", help="Print what would run without running it")
    p.add_argument("--force", action="store_true", help="Re-run even if output already exists")
    p.set_defaults(func=_run)


def _run(args) -> None:
    max_bytes = args.max_mb * 1_000_000
    out_dir = Path(model_dirname(args.model))

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

    variables = load_variables(PARAMETERS_CSV)
    codes_by_var = load_codes(CODES_CSV)

    ok = 0
    failed = []
    for i, pdf in enumerate(todo, 1):
        size_mb = pdf.stat().st_size / 1_000_000
        print(f"[{i}/{len(todo)}] {pdf.name} ({size_mb:.2f}MB)", flush=True)
        if args.dry_run:
            print(f"  → would run: ethnocoder code --model {args.model} {pdf}")
            continue
        try:
            codings = code_pdf(pdf, variables, codes_by_var, model=args.model)
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{pdf.stem}.json"
            out_path.write_text(json.dumps({"codings": codings}, indent=2), encoding="utf-8")
            print(f"  JSON saved → {out_path}", file=sys.stderr)
            ok += 1
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failed.append(pdf.name)

    if not args.dry_run:
        print()
        print(f"Done: {ok}/{len(todo)} succeeded")
        if failed:
            print(f"Failed: {', '.join(failed)}")
            raise SystemExit(1)
