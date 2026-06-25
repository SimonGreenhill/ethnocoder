#!/usr/bin/env python3
# coding=utf-8
"""Summarise accuracy across all documents for a model."""

from pathlib import Path

from evaluate import load_codings_as_dict

GOLD_DIR = Path("gold")


def compare(gold: dict[str, str], coded: dict[str, str]) -> tuple[int, int]:
    same, diff = 0, 0
    for vid, gold_code in gold.items():
        if vid not in coded or gold_code != coded[vid]:
            diff += 1
        else:
            same += 1
    return (same, same + diff)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare coded JSON outputs against gold standard')
    parser.add_argument("modeldir", help='modeldir', type=Path)
    args = parser.parse_args()

    overall = {'same': 0, 'total': 0}
    for p in sorted(args.modeldir.glob("*.json")):
        coded = load_codings_as_dict(p)
        gold = load_codings_as_dict(GOLD_DIR / p.name)
        same, total = compare(gold, coded)
        m = same / total if total else 0
        print(f"{p.stem:20s}\t{same:5d}\t{total:5d}\t{m:0.4f}")
        overall['same'] += same
        overall['total'] += total

    if overall['total']:
        pc = (overall['same'] / overall['total']) * 100
        print(f"\n{overall['same']} / {overall['total']} = {pc:0.2f}%")
    else:
        print("\nNo comparable variables found.")
