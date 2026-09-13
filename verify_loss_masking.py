#!/usr/bin/env python3
"""
Prove prompt tokens are masked BEFORE the first fine-tune, not after.

Unmasked prompt loss is silent: nothing errors, the loss curve looks plausible,
and the model spends its capacity reconstructing contract text instead of
learning the task. This prints the actual label tensor and checks the boundary
token by token.
"""
import json, argparse
from doc_sft_seq import build_example, detect_think_kwarg, IGNORE_INDEX
from doc_harness import build_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-file", default="models_doc.txt")
    ap.add_argument("--data", default="doc_sft/train.jsonl")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--show", type=int, default=1)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    models = [l.strip() for l in open(args.models_file) if l.strip()]

    for m in models:
        try:
            tok = AutoTokenizer.from_pretrained(m)
        except Exception as e:
            print(f"=== {m} : UNAVAILABLE ({type(e).__name__}) ===\n")
            continue
        kw, _ = detect_think_kwarg(tok, build_messages("x", "y"))
        print("=" * 78)
        print(f"{m}   (think kwarg: {kw})")
        print("=" * 78)

        ex = build_example(tok, rows[0]["messages"], kw, args.max_seq_len)
        n_p, n_t = ex["n_prompt"], ex["n_target"]
        labels, iid = ex["labels"], ex["input_ids"]

        # --- the checks -------------------------------------------------
        all_masked = all(l == IGNORE_INDEX for l in labels[:n_p])
        none_masked_after = all(l != IGNORE_INDEX for l in labels[n_p:])
        aligned = labels[n_p:] == iid[n_p:]
        frac = n_t / len(iid)

        print(f"  total tokens        : {len(iid)}")
        print(f"  prompt tokens       : {n_p}  (all == -100 : {all_masked})")
        print(f"  target tokens       : {n_t}  (none == -100 : {none_masked_after})")
        print(f"  labels==input after boundary : {aligned}")
        print(f"  fraction of loss on the TARGET : {frac:.2%}"
              f"   (unmasked would be 100% of {len(iid)} tokens)")

        print(f"\n  --- label tensor around the boundary (index {n_p}) ---")
        lo, hi = max(0, n_p - 4), min(len(iid), n_p + 6)
        for i in range(lo, hi):
            mark = "  <-- TARGET STARTS" if i == n_p else ""
            piece = tok.decode([iid[i]])
            print(f"    [{i:5d}] input={iid[i]:>7}  label={labels[i]:>7}  "
                  f"{piece!r:>22}{mark}")

        print(f"\n  --- first 6 labels (must all be -100) ---")
        print(f"    {labels[:6]}")
        print(f"  --- last 6 labels (must be real token ids) ---")
        print(f"    {labels[-6:]}")
        print(f"  --- decoded target from labels ---")
        real = [l for l in labels if l != IGNORE_INDEX]
        print(f"    {tok.decode(real)[:150]!r}")

        # D1: the masked prefix must BE the inference prompt, byte for byte.
        infer_prompt = tok.apply_chat_template(
            [x for x in rows[0]["messages"] if x["role"] != "assistant"],
            tokenize=False, add_generation_prompt=True, **({kw: False} if kw else {}))
        built_prompt = tok.decode(iid[:n_p])
        prefix_ok = built_prompt.strip().endswith(infer_prompt.strip()[-60:])
        print(f"  train prompt ENDS WITH the inference prompt tail : {prefix_ok}")
        print(f"    inference tail: {infer_prompt[-46:]!r}")
        print(f"    train    tail: {built_prompt[-46:]!r}")

        ok = all_masked and none_masked_after and aligned and prefix_ok
        print(f"\n  >>> MASKING + TRAIN/INFER ALIGNMENT {'CORRECT' if ok else 'BROKEN'}\n")

        # aggregate over a sample so the ratio is not a one-off
        s = rows[:200]
        tot_p = tot_t = 0
        for r in s:
            e = build_example(tok, r["messages"], kw, args.max_seq_len)
            if e:
                tot_p += e["n_prompt"]; tot_t += e["n_target"]
        print(f"  over {len(s)} examples: prompt={tot_p} target={tot_t} "
              f"-> target is {tot_t/(tot_p+tot_t):.2%} of tokens")
        print(f"  Without masking, ~{tot_p/(tot_p+tot_t):.1%} of the loss would be "
              f"contract reconstruction.\n")


if __name__ == "__main__":
    main()
