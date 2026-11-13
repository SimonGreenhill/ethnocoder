import argparse

from ethnocoder.batch import batch
from ethnocoder.datasets import dataset
from ethnocoder.datasets import create_gold, init
from ethnocoder.models import code
from ethnocoder.results import check, evaluate, summarise


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
    dataset.add_subparser(subparsers)
    check.add_subparser(subparsers)
    init.add_subparser(subparsers)
    create_gold.add_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
