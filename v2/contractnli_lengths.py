#!/usr/bin/env python3
"""ContractNLI dev prompt lengths, in the SAME file order the report uses."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))

BUDGET = 16384 - 1024
ORDER = ["<2K", "2-4K", "4-8K", ">8K"]


def bucket(n):
    return "<2K" if n < 2000 else "2-4K" if n < 4000 else "4-8K" if n < 8000 else ">8K"


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    rows = [json.loads(l) for l in open("v2/combined/dev.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("task") == "contractnli"]
    lens = []
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        e = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                    enable_thinking=False)
        if hasattr(e, "input_ids"):
            e = e["input_ids"]
        if e and isinstance(e[0], list):
            e = e[0]
        lens.append(len(list(e)))
    sl = sorted(lens)
    out = {"n": len(lens), "budget": BUDGET,
           "p50": sl[len(sl)//2], "p99": sl[int(.99*len(sl))], "max": sl[-1],
           "min": sl[0],
           "n_over_budget": sum(1 for x in lens if x > BUDGET),
           "lengths": lens,
           "per_row_bucket": [bucket(x) for x in lens],
           "bucket_order": ORDER}
    json.dump(out, open("v2/_contractnli_lengths.json", "w"))
    print(f"n={out['n']} p50={out['p50']} p99={out['p99']} max={out['max']} "
          f"over_budget={out['n_over_budget']}")
    from collections import Counter
    print("buckets:", dict(Counter(out["per_row_bucket"])))


if __name__ == "__main__":
    main()
