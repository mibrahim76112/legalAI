#!/usr/bin/env python3
"""
Build Arm B training data: reasoning-first targets from filtered STaR rationales.

Verified-only: every row is one where the teacher reached the gold verdict
BLIND and then passed all four grounding filters. No conditioned/rationalised
rows -- those are a separate arm.

dev/test are copied unchanged from Arm A. Teacher output never enters them.
"""
import json, argparse, statistics as st
from pathlib import Path
from collections import Counter
from reasoning_filters import derive_length_bounds, apply_all

SYSTEM = (
    "You are a junior legal assistant reviewing a contract. Given the CONTRACT "
    "and the QUESTION, determine whether the contract supports the stated position "
    "(Entailment), conflicts with it (Contradiction), or does not address it "
    "(NotMentioned). First explain your reasoning: name the governing clause, quote "
    "it verbatim, and apply it to the position. Then give the verdict and the exact "
    "sentence(s) that justify it, or an empty evidence list if none apply. Respond "
    "with JSON only, in the form "
    '{"reasoning": "...", "verdict": "...", "evidence": [...]}.'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="doc_sft_armb")
    ap.add_argument("--f2-strict", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open("reasoning/pass1_raw.jsonl", encoding="utf-8")]
    armA = {}
    contracts = {}
    for l in open("doc_sft/train.jsonl", encoding="utf-8"):
        r = json.loads(l)
        armA[(r["doc_id"], r["hypothesis_id"])] = r
        u = [m["content"] for m in r["messages"] if m["role"] == "user"][0]
        contracts[r["doc_id"]] = u.split("CONTRACT: ", 1)[1].rsplit("\n\nQUESTION: ", 1)[0]

    lo, hi = derive_length_bounds(rows)
    kept, rej, reasons, ev = apply_all(rows, contracts, lo, hi, f2_strict=args.f2_strict)
    print(f"filters: kept {len(kept)}/{len(rows)} = {len(kept)/len(rows):.1%} "
          f"(length bounds {lo}..{hi})")

    outd = Path(args.out); outd.mkdir(exist_ok=True)
    built = 0
    with open(outd / "train.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            src = armA.get((r["doc_id"], r["hypothesis_id"]))
            if src is None:
                continue
            user = [m["content"] for m in src["messages"] if m["role"] == "user"][0]
            # reasoning FIRST: dict insertion order is preserved by json.dumps,
            # so the verdict is generated after the analysis, not before it.
            target = json.dumps({"reasoning": " ".join(r["reasoning"].split()),
                                 "verdict": src["verdict"],
                                 "evidence": src["evidence"]}, ensure_ascii=False)
            assert target.index('"reasoning"') < target.index('"verdict"'), "key order broken"
            f.write(json.dumps({
                "doc_id": src["doc_id"], "file_name": src["file_name"],
                "hypothesis_id": src["hypothesis_id"], "split": "train",
                "verdict": src["verdict"], "evidence": src["evidence"],
                "teacher": r["teacher"], "pass": r["pass"], "conditioned": False,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user},
                             {"role": "assistant", "content": target}],
            }, ensure_ascii=False) + "\n")
            built += 1

    # dev/test unchanged from Arm A, but with the Arm B system prompt so the
    # prompt the model sees at eval matches what it was trained under.
    for src_name, dst in (("valid", "valid"), ("test", "test")):
        n = 0
        with open(outd / f"{dst}.jsonl", "w", encoding="utf-8") as f:
            for l in open(f"doc_sft/{src_name}.jsonl", encoding="utf-8"):
                r = json.loads(l)
                r["messages"] = [{"role": "system", "content": SYSTEM}] + \
                                [m for m in r["messages"] if m["role"] != "system"]
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
        print(f"  {dst}.jsonl {n} rows (gold targets unchanged, Arm B system prompt)")

    c = Counter(json.loads(l)["verdict"] for l in open(outd / "train.jsonl", encoding="utf-8"))
    print(f"  train.jsonl {built} rows  {dict(c)}")
    json.dump({"kept": built, "of": len(rows), "yield": built/len(rows),
               "length_bounds": [lo, hi], "f2_strict": args.f2_strict,
               "class_counts": dict(c), "conditioned_rows": 0,
               "reject_reasons": dict(reasons), "evidence_modes": dict(ev)},
              open(outd / "_manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
