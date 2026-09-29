"""Evaluate a base or LoRA-tuned model on a formatted ContractNLI split.

Reports macro F1, per-class metrics and schema-adherence, and writes
per-example predictions for later error slicing.
"""

import argparse
import collections
import json
import random
import re
import time

VERDICTS = ["Entailment", "Contradiction", "NotMentioned"]
UNPARSEABLE = "UNPARSEABLE"


def load_rows(path, limit=None, seed=0):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def gold_of(row):
    target = [m for m in row["messages"] if m["role"] == "assistant"][0]["content"]
    return json.loads(target)["verdict"]


def few_shot_turns(path, k, seed=0):
    """k demonstrations from the training file, one class at a time."""
    rows = load_rows(path)
    by_label = collections.defaultdict(list)
    for r in rows:
        by_label[gold_of(r)].append(r)
    rng = random.Random(seed)
    picked = []
    for i in range(k):
        label = VERDICTS[i % len(VERDICTS)]
        picked.append(rng.choice(by_label[label]))
    turns = []
    for r in picked:
        turns += [m for m in r["messages"] if m["role"] != "system"]
    return turns


def build_prompt(row, tokenizer, shots):
    msgs = [m for m in row["messages"] if m["role"] == "system"]
    msgs += shots
    msgs += [m for m in row["messages"] if m["role"] == "user"]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True)


def parse(text):
    """Lenient parse: real JSON first, then a bare verdict mention."""
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and obj.get("verdict") in VERDICTS:
            return obj["verdict"], True
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r"\b(NotMentioned|Entailment|Contradiction)\b", text)
    return (m.group(1) if m else UNPARSEABLE), False


def score(golds, preds):
    labels = VERDICTS + [UNPARSEABLE]
    conf = {g: collections.Counter() for g in VERDICTS}
    for g, p in zip(golds, preds):
        conf[g][p] += 1
    per_class = {}
    for c in VERDICTS:
        tp = conf[c][c]
        fp = sum(conf[g][c] for g in VERDICTS if g != c)
        fn = sum(conf[c][p] for p in labels if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(conf[c].values())}
    acc = sum(conf[c][c] for c in VERDICTS) / max(1, len(golds))
    macro_f1 = sum(per_class[c]["f1"] for c in VERDICTS) / len(VERDICTS)
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per_class,
            "confusion": {g: dict(conf[g]) for g in VERDICTS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft_data/baseline/test.jsonl")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--few-shot-data", default="sft_data/baseline/train.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--completion-batch-size", type=int, default=16)
    ap.add_argument("--prefill-batch-size", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from mlx_lm import batch_generate, load

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    rows = load_rows(args.data, args.limit)
    shots = few_shot_turns(args.few_shot_data, args.few_shot) if args.few_shot else []

    prompts = [build_prompt(r, tokenizer, shots) for r in rows]
    golds = [gold_of(r) for r in rows]

    t0 = time.time()
    resp = batch_generate(
        model, tokenizer, prompts,
        max_tokens=args.max_tokens,
        verbose=True,
        completion_batch_size=args.completion_batch_size,
        prefill_batch_size=args.prefill_batch_size,
    )
    elapsed = time.time() - t0

    preds, well_formed = [], 0
    for text in resp.texts:
        p, ok = parse(text)
        preds.append(p)
        well_formed += ok

    metrics = score(golds, preds)
    metrics["schema_adherence"] = well_formed / max(1, len(rows))
    metrics["unparseable"] = preds.count(UNPARSEABLE) / max(1, len(rows))
    metrics["n"] = len(rows)
    metrics["seconds"] = round(elapsed, 1)
    metrics["config"] = {
        "data": args.data, "model": args.model, "adapter_path": args.adapter_path,
        "few_shot": args.few_shot, "tag": args.tag,
    }

    tag = args.tag or (args.adapter_path or f"base-{args.few_shot}shot")
    print(f"\n=== {tag} on {args.data} (n={len(rows)}, {elapsed:.0f}s) ===")
    print(f"accuracy        {metrics['accuracy']:.3f}")
    print(f"macro F1        {metrics['macro_f1']:.3f}")
    print(f"schema adherence{metrics['schema_adherence']:8.3f}   unparseable {metrics['unparseable']:.3f}")
    print(f"{'class':16}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>7}")
    for c in VERDICTS:
        m = metrics["per_class"][c]
        print(f"{c:16}{m['precision']:7.3f}{m['recall']:7.3f}{m['f1']:7.3f}{m['support']:7d}")
    print("confusion (gold -> predicted):")
    for g in VERDICTS:
        print(f"  {g:16}{metrics['confusion'][g]}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        pred_path = args.out.replace(".json", ".preds.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for r, g, p, text in zip(rows, golds, preds, resp.texts):
                f.write(json.dumps({"chunk_id": r.get("chunk_id"), "gold": g,
                                    "pred": p, "raw": text}, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out} and {pred_path}")


if __name__ == "__main__":
    main()
