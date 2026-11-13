import argparse
from pathlib import Path
from rich import print

from ethnocoder.datasets import cldf
from ethnocoder.common.utils import GRAMBANK_CLDF_DIR

def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "create-gold",
        description="Create gold files from a CLDF dataset",
        help="Create gold files from a CLDF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasetdir", default=GRAMBANK_CLDF_DIR, type=Path, required=False, help="Path to CLDF dataset root")
    p.add_argument("--workingdir", default=Path("."), type=Path, required=False, help="Working directory to create/populate")

    # Argument for specifying a comma-separated list of sources.
    # Only these sources will be used to generate gold files.
    p.add_argument(
        "--sources",
        type=str,
        help="Comma-separated list of sources to use for generating gold files",
    )

    p.set_defaults(func=_run)


def _run(args) -> None:
    workingdir: Path = args.workingdir
    workingdir.mkdir(parents=True, exist_ok=True)

    ds = cldf.load_dataset(args.datasetdir)

    # Create gold files using the specified sources, if provided.
    sources = None
    if args.sources:
        sources = [s.strip() for s in args.sources.split(",")]
    n = cldf.write_gold(ds, workingdir, sources=sources)

    print(f"[green]Wrote {n} gold file(s) to [spring_green2]{workingdir / 'gold'}[/spring_green2][/green]")