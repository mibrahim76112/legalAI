#!/usr/bin/env python3
"""Fixed stratified subsample of the v2 dev set for the BASE-model baseline.

Why a subsample: zero-shot generation is slow because untuned models are
verbose -- an earlier full-dev zero-shot run hit a 3h wall at 86%. Four base
models over full dev is not affordable.

Why it is still a fair comparison: the adapter models are rescored on THESE
EXACT ROWS offline from archived predictions, so base and adapter are compared
on identical examples. The subsample numbers therefore differ from the
full-dev headline figures and must not be mixed with them.
"""
import json, random, argparse
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260919


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nli", type=int, default=250)
    ap.add_argument("--n-cuad", type=int, default=250)
    ap.add_argument("--n-note", type=int, default=100)
    ap.add_argument("--out", default="v2/baseline_subset.jsonl")
    args = ap.parse_args()

    dev = [json.loads(l) for l in open("v2/combined/dev.jsonl", encoding="utf-8")]
    rng = random.Random(SEED)
    keep = []

    # ContractNLI: equal per verdict so the rare class is represented
    by = defaultdict(list)
    for i, r in enumerate(dev):
        if r["task"] == "contractnli":
            by[r["verdict"]].append(i)
    per = args.n_nli // len(by)
    for k in sorted(by):
        rng.shuffle(by[k]); keep += by[k][:per]

    # CUAD: half present / half absent, spread across categories
    pres, absn = defaultdict(list), defaultdict(list)
    for i, r in enumerate(dev):
        if r["task"] == "cuad":
            (pres if r["present"] else absn)[r["category"]].append(i)
    half = args.n_cuad // 2
    for pool in (pres, absn):
        cats = sorted(pool)
        k = 0
        got = 0
        while got < half and any(pool[c] for c in cats):
            c = cats[k % len(cats)]
            if pool[c]:
                rng.shuffle(pool[c]); keep.append(pool[c].pop()); got += 1
            k += 1

    notes = [i for i, r in enumerate(dev) if r["task"] == "risknote"]
    rng.shuffle(notes); keep += notes[:args.n_note]

    keep = sorted(set(keep))
    rows = [dict(dev[i], _dev_index=i) for i in keep]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"subset: {len(rows)} rows  {dict(Counter(r['task'] for r in rows))}")
    print(f"  NLI verdicts : {dict(Counter(r['verdict'] for r in rows if r['task']=='contractnli'))}")
    print(f"  CUAD present : {dict(Counter(r['present'] for r in rows if r['task']=='cuad'))}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
