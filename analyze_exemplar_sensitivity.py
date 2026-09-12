#!/usr/bin/env python3
"""
W2 analysis: how much of the 3-shot macro-F1 is exemplar choice?

Combines the median-length triple used for the headline ranking ("base") with
the four random triples, giving 5 observations per model. Reports the spread and
- the question that actually matters - whether the RANKING ORDER changes when
the triple changes.
"""
import json, glob, argparse, statistics as st
from pathlib import Path
from aggregate import prf, UNPARSED

BASE_DIR = "/scratch/ibi761/legalai/results"
EXVAR_DIR = "/scratch/ibi761/legalai/results_exvar"
# gaps the published 3-shot ranking rests on
PUBLISHED_GAPS = {"Qwen3-14B vs Qwen3-8B": 0.016, "Qwen3-8B vs phi-4": 0.003}


def load(fp):
    rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
    rows.sort(key=lambda r: r["example_id"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=BASE_DIR)
    ap.add_argument("--exvar-dir", default=EXVAR_DIR)
    args = ap.parse_args()

    scores = {}   # model -> {tag: macro_f1}
    # headline triple: 3-shot, spec prompt, no tag
    for fp in glob.glob(str(Path(args.base_dir) / "raw__*__threeshot__spec.jsonl")):
        rows = load(fp)
        m = rows[0]["model"]
        gold = [r["gold_verdict"] for r in rows]
        pred = [r["parsed_verdict"] or UNPARSED for r in rows]
        scores.setdefault(m, {})["base"] = prf(pred, gold)["macro_f1"]
    # pre-rename runs had no system tag in the filename
    for fp in glob.glob(str(Path(args.base_dir) / "raw__*__threeshot.jsonl")):
        rows = load(fp)
        m = rows[0]["model"]
        if "base" in scores.get(m, {}):
            continue
        gold = [r["gold_verdict"] for r in rows]
        pred = [r["parsed_verdict"] or UNPARSED for r in rows]
        scores.setdefault(m, {})["base"] = prf(pred, gold)["macro_f1"]

    for fp in glob.glob(str(Path(args.exvar_dir) / "raw__*.jsonl")):
        rows = load(fp)
        m, tag = rows[0]["model"], rows[0].get("fewshot_tag", "?")
        gold = [r["gold_verdict"] for r in rows]
        pred = [r["parsed_verdict"] or UNPARSED for r in rows]
        scores.setdefault(m, {})[tag] = prf(pred, gold)["macro_f1"]

    if not scores:
        raise SystemExit("no results found")

    tags = ["base", "t1", "t2", "t3", "t4"]
    print(f"{'model':28s}" + "".join(f"{t:>8}" for t in tags)
          + f"{'mean':>8}{'sd':>7}{'range':>8}")
    spreads = {}
    for m in sorted(scores):
        v = [scores[m].get(t) for t in tags]
        got = [x for x in v if x is not None]
        cells = "".join(f"{x:8.3f}" if x is not None else f"{'-':>8}" for x in v)
        rng = max(got) - min(got)
        spreads[m] = rng
        sd = st.stdev(got) if len(got) > 1 else 0.0
        print(f"{m:28s}{cells}{st.mean(got):8.3f}{sd:7.3f}{rng:8.3f}")

    print("\n--- ranking order per triple ---")
    orders = {}
    for t in tags:
        present = {m: s[t] for m, s in scores.items() if t in s}
        if len(present) < 2:
            continue
        o = sorted(present, key=lambda m: -present[m])
        orders[t] = o
        print(f"  {t:5s} " + " > ".join(f"{m.split('/')[-1]}({present[m]:.3f})" for m in o))

    uniq = {tuple(o) for o in orders.values()}
    print(f"\n  distinct orders across {len(orders)} triples: {len(uniq)}")
    if len(uniq) == 1:
        print("  -> ORDER IS STABLE under exemplar choice")
    else:
        print("  -> ORDER CHANGES with the exemplar triple; the published")
        print("     3-shot order is not a property of the models alone")

    print("\n--- exemplar spread vs the gaps the ranking rests on ---")
    worst = max(spreads.values())
    print(f"  largest per-model spread from exemplar choice : {worst:.3f}")
    for name, gap in PUBLISHED_GAPS.items():
        flag = "SWAMPED" if worst > gap else "ok"
        print(f"  published gap {name:26s} {gap:.3f}   -> {flag}")
    print("\n  If the spread exceeds a gap, that gap cannot be attributed to")
    print("  model capability: a different arbitrary triple could reverse it.")


if __name__ == "__main__":
    main()
