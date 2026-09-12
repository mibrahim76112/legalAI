#!/usr/bin/env python3
"""
Build the frozen screen inputs, reusing 02_build_sft_data.py's own row loader
so the eval set is row-for-row identical to sft_data/baseline/valid.jsonl.

That identity is the point: these base-model numbers become the "base" column
of the final fine-tuned-vs-base table, so both must score the same rows under
the same filtering (strip + MIN_PREMISE_CHARS) and the same nda-N ids.

Outputs (data/):
  eval_dev.jsonl   - rows we score
  fewshot.json     - 3 exemplars, one per class, TRAIN only, shared by all models
  hypotheses.json  - nda-N -> policy text
"""
import json, argparse, importlib.util, statistics
from pathlib import Path
from collections import Counter

CLASSES = ["Entailment", "Contradiction", "NotMentioned"]


def load_builder(path="02_build_sft_data.py"):
    """Import the SFT builder despite its numeric filename."""
    spec = importlib.util.spec_from_file_location("sft_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-split", default="dev")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    b = load_builder()
    rows, dropped = b.load_rows()          # same strip + MIN_PREMISE_CHARS as SFT
    hyp2id = b.hypothesis_ids()            # hypothesis text -> nda-N
    print(f"loaded {len(rows)} rows (dropped {dropped} under "
          f"{b.MIN_PREMISE_CHARS} chars) - same filter as SFT")

    train = [r for r in rows if r["subset"] == "train"]
    ev    = [r for r in rows if r["subset"] == args.eval_split]

    outd = Path(args.out); outd.mkdir(exist_ok=True)

    # ---- eval set ------------------------------------------------------
    # chunk_id is b.chunk_id(premise), the same key the SFT files carry, so
    # per-clause error analysis can join base and fine-tuned predictions.
    with open(outd / "eval_dev.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(ev):
            f.write(json.dumps({
                "example_id":    f"{args.eval_split}-{i:05d}",
                "chunk_id":      b.chunk_id(r["premise"]),
                "hypothesis_id": hyp2id[r["hypothesis"]],
                "premise":       r["premise"],
                "hypothesis":    r["hypothesis"],
                "gold_verdict":  r["label"],
            }, ensure_ascii=False) + "\n")

    # ---- 3-shot exemplars ----------------------------------------------
    # One per class, TRAIN only, median-length within its class: deterministic
    # and not cherry-picked. Premises that also occur in the eval split are
    # excluded so no exemplar can echo a scored row.
    ev_prem = {r["premise"] for r in ev}
    fewshot = []
    for cls in CLASSES:
        pool = [r for r in train if r["label"] == cls and r["premise"] not in ev_prem]
        pool.sort(key=lambda r: (len(r["premise"]), r["premise"]))
        pick = pool[len(pool) // 2]
        fewshot.append({
            "verdict":       cls,
            "premise":       pick["premise"],
            "hypothesis":    pick["hypothesis"],
            "hypothesis_id": hyp2id[pick["hypothesis"]],
            "n_chars":       len(pick["premise"]),
        })
    json.dump(fewshot, open(outd / "fewshot.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    json.dump({hyp2id[h]: h for h in hyp2id},
              open(outd / "hypotheses.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- report ---------------------------------------------------------
    c, n = Counter(r["label"] for r in ev), len(ev)
    print(f"\neval split : {args.eval_split}  n={n}")
    for cls in CLASSES:
        print(f"  {cls:14s} {c[cls]:5d}  {100*c[cls]/n:5.1f}%")
    print(f"hypotheses : {len(set(hyp2id.values()))}")
    print(f"premise chars: median={statistics.median(len(r['premise']) for r in ev):.0f} "
          f"max={max(len(r['premise']) for r in ev)}")
    print("\n3-shot exemplars (train only):")
    for e in fewshot:
        print(f"  {e['verdict']:14s} {e['hypothesis_id']:8s} {e['n_chars']:4d} chars")


if __name__ == "__main__":
    main()
