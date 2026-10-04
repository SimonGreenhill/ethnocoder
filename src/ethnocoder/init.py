import argparse
from importlib.resources import files
from pathlib import Path

from ethnocoder import cldf


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "init",
        description="Initialise a working directory from a CLDF dataset",
        help="Initialise a working directory from a CLDF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("datasetdir", type=Path, help="Path to CLDF dataset root")
    p.add_argument("workingdir", type=Path, help="Working directory to create/populate")
    p.set_defaults(func=_run)


def _run(args) -> None:
    workingdir: Path = args.workingdir
    workingdir.mkdir(parents=True, exist_ok=True)

    ds = cldf.load_dataset(args.datasetdir)
    for which in ("parameters", "codes"):
        dst = cldf.copy_table(ds, which, workingdir)
        print(f"Copied {which} → {dst}")
    n = cldf.write_gold(ds, workingdir)
    print(f"Wrote {n} gold file(s) to {workingdir / 'gold'}/")

    docs_dir = workingdir / "docs"
    docs_dir.mkdir(exist_ok=True)
    print(f"Created {docs_dir}/ (place PDF source documents here)")

    prompt_dst = workingdir / "PROMPT.md"
    if prompt_dst.exists():
        print(f"{prompt_dst} already exists — leaving it unchanged")
    else:
        content = files("ethnocoder.data").joinpath("PROMPT.md").read_text(encoding="utf-8")
        prompt_dst.write_text(content, encoding="utf-8")
        print(f"Created {prompt_dst}")
