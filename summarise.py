#!/usr/bin/env python3
# coding=utf-8
"""..."""

import json
from pathlib import Path

GOLD_DIR = Path("gold")

def get(f):
    try:
        o = json.loads(f.read_text())
    except Exception as e:
        print(f"ERROR -- {f}")
        print(f.read_text())
        raise
    
    if 'codings' in o:
        return convert(o['codings'])
    return convert(o)


# converts list of dicts to dict
def convert(alist):
    def _(x):
        if 'id' in x:
            return int(x["id"])
        elif 'variable_id' in x:
            return int(x["variable_id"])
        raise ValueError("?")
    
    return {_(l): l['code'] for l in alist}


def compare(gold, code):
    same, diff = 0, 0
    for i in gold:
        if gold[i] is None:
            continue
        elif i not in code or str(gold[i]) != str(code[i]):
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
    for p in args.modeldir.glob("*.json"):
        coded = get(p)
        gold = get(GOLD_DIR / p.name)
        same, total = compare(gold, coded)
        m = same / total
        print(f"{p.stem:20s}\t{same:5d}\t{total:5d}\t{m:0.4f}")
        overall['same'] += same
        overall['total'] += total

    pc = (overall['same'] / overall['total']) * 100
    print(f"\n{overall['same']} / {overall['total']} = {pc:0.2f}%")