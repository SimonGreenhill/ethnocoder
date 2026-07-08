import argparse
from pathlib import Path

from ethnocoder.evaluate import load_codings_as_dict

GOLD_DIR = Path("gold")


def compare(gold: dict[str, str], coded: dict[str, str]) -> tuple[int, int]:
    same, diff = 0, 0
    for vid, gold_code in gold.items():
        if vid not in coded or gold_code != coded[vid]:
            diff += 1
        else:
            same += 1
    return (same, same + diff)


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "summarise",
        description="Summarise accuracy across all documents for a model",
    )
    p.add_argument("modeldir", type=Path, help="Directory of coded JSON outputs")
    p.set_defaults(func=_run)


def _run(args) -> None:
    overall = {"same": 0, "total": 0}
    for p in sorted(args.modeldir.glob("*.json")):
        coded = load_codings_as_dict(p)
        gold = load_codings_as_dict(GOLD_DIR / p.name)
        same, total = compare(gold, coded)
        m = same / total if total else 0
        print(f"{p.stem:20s}\t{same:5d}\t{total:5d}\t{m:0.4f}")
        overall["same"] += same
        overall["total"] += total

    if overall["total"]:
        pc = (overall["same"] / overall["total"]) * 100
        print(f"\n{overall['same']} / {overall['total']} = {pc:0.2f}%")
    else:
        print("\nNo comparable variables found.")
