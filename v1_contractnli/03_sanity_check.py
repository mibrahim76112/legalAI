"""Sanity-check the formatted mlx-lm files under sft_data/.

Checks schema, label leakage, target JSON validity, and token length.

Usage:
    python 03_sanity_check.py [--tokenizer mlx-community/Qwen2.5-7B-Instruct-4bit]
                              [--sample 400] [--show 5]
"""

import argparse
import glob
import json
import os

VERDICTS = {"Entailment", "Contradiction", "NotMentioned"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def check_schema(rows, path):
    sample = rows[0]
    if "prompt" in sample and "completion" in sample:
        kind = "CompletionsDataset (prompt/completion)"
    elif "messages" in sample:
        kind = "ChatDataset (messages)"
    elif "text" in sample:
        kind = "TextDataset (text)"
    else:
        raise SystemExit(f"{path}: no key mlx-lm recognises: {sorted(sample)}")
    bad = [i for i, r in enumerate(rows) if sorted(r) != sorted(sample)]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} rows with inconsistent keys, first at {bad[0]}")
    return kind


def check_content(rows, path):
    leaks, malformed = [], []
    for i, r in enumerate(rows):
        msgs = r["messages"]
        # system turn names all three verdicts by design
        user_text = " ".join(m["content"] for m in msgs if m["role"] == "user")
        target = [m for m in msgs if m["role"] == "assistant"]
        if len(target) != 1:
            malformed.append((i, "expected exactly one assistant turn"))
            continue
        try:
            obj = json.loads(target[0]["content"])
        except json.JSONDecodeError as e:
            malformed.append((i, f"assistant turn is not JSON: {e}"))
            continue
        if obj.get("verdict") not in VERDICTS:
            malformed.append((i, f"bad verdict {obj.get('verdict')!r}"))
        if obj.get("verdict") in user_text:
            leaks.append(i)
    return leaks, malformed


def token_stats(rows, tok, sample_n):
    lens = []
    for r in rows[:sample_n]:
        lens.append(len(tok.apply_chat_template(r["messages"])))
    lens.sort()
    n = len(lens)
    return {
        "sampled": n,
        "min": lens[0],
        "p50": lens[n // 2],
        "p95": lens[int(0.95 * n) - 1],
        "max": lens[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sft_data")
    ap.add_argument("--tokenizer", default=None,
                    help="HF/mlx model id; omit to skip token-length stats")
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    paths = sorted(glob.glob(os.path.join(args.root, "*", "*.jsonl")))
    for path in paths:
        rows = load(path)
        kind = check_schema(rows, path)
        leaks, malformed = check_content(rows, path)
        line = f"{path:44} {len(rows):5} rows  {kind}"
        if tok:
            s = token_stats(rows, tok, args.sample)
            line += (f"  tokens[n={s['sampled']}] min={s['min']} p50={s['p50']} "
                     f"p95={s['p95']} max={s['max']}")
        print(line)
        if leaks:
            print(f"    LABEL LEAK in {len(leaks)} rows, first at index {leaks[0]}")
        for i, why in malformed[:3]:
            print(f"    MALFORMED row {i}: {why}")

    if args.show and paths:
        rows = load(paths[0])
        print(f"\n--- first {args.show} formatted examples from {paths[0]} ---")
        for r in rows[: args.show]:
            for m in r["messages"]:
                content = m["content"]
                if len(content) > 400:
                    content = content[:400] + " ...[truncated for display]"
                print(f"  [{m['role']}] {content}")
            print()


if __name__ == "__main__":
    main()
