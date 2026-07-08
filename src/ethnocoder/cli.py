import argparse

from ethnocoder import batch, check, code, evaluate, init, setup, summarise


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ethnocoder",
        description="Automated coding of cultural trait variables from PDF source documents",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    code.add_subparser(subparsers)
    batch.add_subparser(subparsers)
    evaluate.add_subparser(subparsers)
    summarise.add_subparser(subparsers)
    setup.add_subparser(subparsers)
    check.add_subparser(subparsers)
    init.add_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)
