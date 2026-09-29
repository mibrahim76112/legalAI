#!/usr/bin/env python3
"""Merge chunked quantization runs back into one full-dev prediction file.

Chunks are contiguous slices of v2/combined/dev.jsonl in order, so
concatenating chunk 0..3 reproduces the full dev row order exactly. That
matters: the bf16 reference (raw__llama_3task.jsonl) is in that same order and
alignment is by position, since the rows carry no unique key.
"""
import json, argparse, subprocess, sys
from pathlib import Path

B = Path("/scratch/ibi761/legalai/v2_baseline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)      # nf4 | fp8
    ap.add_argument("--out-tag", required=True)     # full_llama_nf4
    ap.add_argument("--chunks", type=int, default=4)
    args = ap.parse_args()

    parts = []
    for i in range(args.chunks):
        fp = B / f"raw__chunk_{args.prefix}_{i}.jsonl"
        if not fp.exists():
            print(f"MISSING {fp.name}")
            return 1
        parts.append([l for l in open(fp, encoding="utf-8")])
        print(f"  chunk {i}: {len(parts[-1])} rows")
    merged = [l for p in parts for l in p]
    dev_n = sum(1 for _ in open("v2/combined/dev.jsonl", encoding="utf-8"))
    assert len(merged) == dev_n, f"got {len(merged)}, dev has {dev_n}"
    out = B / f"raw__{args.out_tag}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(merged)
    print(f"  merged {len(merged)} rows -> {out.name}")
    subprocess.run([sys.executable, "v2/eval_multitask.py", "--base", "Qwen/Qwen3-4B",
                    "--rescore", str(out), "--tag", args.out_tag,
                    "--out-dir", str(B)], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "v2/evidence_eval.py", "--raw", str(out),
                    "--tag", f"quant_{args.out_tag}", "--out-dir", "v2/evidence_eval"],
                   check=True, stdout=subprocess.DEVNULL)
    print(f"  scored -> metrics__{args.out_tag}.json + evidence__quant_{args.out_tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
