import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from ethnocoder import cldf


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "dataset",
        description="Inspect or extract files from a CLDF dataset",
        help="Inspect or extract files from a CLDF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("datasetdir", type=Path, help="Path to CLDF dataset root")
    p.add_argument("--list", action="store_true", help="List source keys and stats")
    p.add_argument(
        "--extract", nargs="+", choices=["gold", "parameters", "codes"],
        metavar="{gold,parameters,codes}",
        help="Extract one or more outputs")
    p.add_argument(
        "-o", "--output", type=Path, default=Path("."),
        help="Output directory (default: .)")
    p.set_defaults(func=_run)


def _run(args) -> None:
    ds = cldf.load_dataset(args.datasetdir)

    if args.list:
        cldf.list_sources(ds)

    if args.extract:
        for which in args.extract:
            if which == "gold":
                n = cldf.write_gold(ds, args.output)
                print(f"Wrote {n} gold file(s) to {args.output / 'gold'}/")
            else:
                dst = cldf.copy_table(ds, which, args.output)
                print(f"Copied {which} → {dst}")

    if not args.list and not args.extract:
        n_params = len(list(cast(Iterable[dict], ds["ParameterTable"])))
        n_codes = len(list(cast(Iterable[dict], ds["CodeTable"])))
        n_sources = len(cldf.build_source_index(cast(Iterable[dict], ds["ValueTable"])))
        print(f"{n_params} parameters, {n_codes} codes, {n_sources} sources")
