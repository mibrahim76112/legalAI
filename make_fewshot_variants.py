#!/usr/bin/env python3
"""
Emit alternative balanced 3-shot triples to measure exemplar sensitivity (W2).

The screen ranked models using ONE triple (median-length per class, in
data/fewshot.json). That is fair between models but says nothing about how much
the *level* of each score depends on which triple was drawn - and the 3-shot
condition moves Qwen3-8B by +0.090, so the assumption that it is insensitive was
never safe.

These variants are RANDOM draws under fixed seeds, not hand-picked: the quantity
we want is the variance over the natural pool of eligible exemplars, which is
what a different arbitrary choice would have cost us.

Constraints held identical to data/fewshot.json:
  - one exemplar per class (balanced)
  - TRAIN split only
  - no premise that also occurs in the eval split
"""
import json, random, argparse, importlib.util
from pathlib import Path

CLASSES = ["Entailment", "Contradiction", "NotMentioned"]


def load_builder(path="02_build_sft_data.py"):
    spec = importlib.util.spec_from_file_location("sft_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--eval-split", default="dev")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    b = load_builder()
    rows, _ = b.load_rows()
    hyp2id = b.hypothesis_ids()

    train = [r for r in rows if r["subset"] == "train"]
    ev_prem = {r["premise"] for r in rows if r["subset"] == args.eval_split}

    pools = {c: [r for r in train
                 if r["label"] == c and r["premise"] not in ev_prem]
             for c in CLASSES}
    print("eligible train exemplars per class:")
    for c in CLASSES:
        print(f"  {c:14s} {len(pools[c])}")

    outd = Path(args.out)
    for seed in args.seeds:
        rng = random.Random(seed)
        triple = []
        for c in CLASSES:
            pick = rng.choice(pools[c])
            triple.append({
                "verdict": c,
                "premise": pick["premise"],
                "hypothesis": pick["hypothesis"],
                "hypothesis_id": hyp2id[pick["hypothesis"]],
                "n_chars": len(pick["premise"]),
            })
        p = outd / f"fewshot_t{seed}.json"
        json.dump(triple, open(p, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        chars = "/".join(str(e["n_chars"]) for e in triple)
        hids = ",".join(e["hypothesis_id"] for e in triple)
        print(f"  seed {seed} -> {p.name}  chars={chars:18s} hyps={hids}")


if __name__ == "__main__":
    main()
