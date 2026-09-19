#!/usr/bin/env python3
"""
Turn the measured token counts into the report + max_seq_len recommendation.

The part that matters most is the truncation-safety check: a document that gets
cut is only acceptable if NO gold evidence span falls in the removed region.
If it does, the training target cites text the model cannot see, which teaches
it to hallucinate citations. That is corruption, not a smaller example.
"""
import json, csv, argparse
from pathlib import Path
from collections import defaultdict

THRESHOLDS = [4096, 8192, 16384, 32768]
CLASSES = ["Entailment", "Contradiction", "NotMentioned"]


def pct(v, q):
    v = sorted(v)
    return v[min(int(q * len(v)), len(v) - 1)] if v else 0


def load_csv(p):
    return list(csv.DictReader(open(p)))


def evidence_spans(doc, span_idx):
    out = []
    for i in span_idx:
        if 0 <= i < len(doc["spans"]):
            out.append(tuple(doc["spans"][i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="CONTEXT_BUDGET.md")
    args = ap.parse_args()

    docr = load_csv(Path(args.data) / "doc_token_counts.csv")
    seqr = load_csv(Path(args.data) / "seq_token_counts.csv")
    for r in docr:
        r["n_tokens"] = int(r["n_tokens"]); r["n_chars"] = int(r["n_chars"])
    for r in seqr:
        for k in ("input_tokens", "output_tokens", "total_tokens", "n_evidence_spans"):
            r[k] = int(r[k])

    models = sorted({r["model"] for r in docr})
    L = []
    A = L.append

    A("# Context-length budget for document-level SFT\n")
    A("Task moves from clause-level to document-level: input is a full NDA plus one")
    A("policy question, output is `{\"verdict\": ..., \"evidence\": [...]}`.\n")
    A("All counts are **measured end-to-end** on the real training sequences")
    A("(chat template + system prompt + full document + question, and the")
    A("serialized target JSON). Nothing is estimated by adding part counts, because")
    A("subword boundaries make token counts non-additive.\n")
    A(f"Corpus: **607 documents** (train 423 / dev 61 / test 123), "
      f"**{len(seqr)//len(models)} (document, hypothesis) pairs**.\n")

    # ---------- Part 1 ----------
    A("\n## Part 1 — Document length distribution\n")
    A("Tokens for `document.text` alone, all 607 documents.\n")
    A("| tokenizer | min | p50 | p90 | p95 | p99 | max |")
    A("|---|---|---|---|---|---|---|")
    for m in models:
        v = [r["n_tokens"] for r in docr if r["model"] == m]
        A(f"| {m.split('/')[-1]} | {min(v)} | {pct(v,.50)} | {pct(v,.90)} | "
          f"{pct(v,.95)} | {pct(v,.99)} | {max(v)} |")

    A("\nDocuments exceeding each threshold (of 607):\n")
    A("| tokenizer | >4096 | >8192 | >16384 | >32768 |")
    A("|---|---|---|---|---|")
    for m in models:
        v = [r["n_tokens"] for r in docr if r["model"] == m]
        A(f"| {m.split('/')[-1]} | " +
          " | ".join(str(sum(1 for x in v if x > t)) for t in THRESHOLDS) + " |")

    over = defaultdict(list)
    for r in docr:
        if r["n_tokens"] > 16384:
            over[r["model"]].append(r)
    A("\n### Documents over 16384 tokens\n")
    if not any(over.values()):
        A("None, under any of the four tokenizers.\n")
    else:
        A("| tokenizer | doc_id | file_name | split | chars | tokens |")
        A("|---|---|---|---|---|---|")
        for m in models:
            for r in sorted(over[m], key=lambda x: -x["n_tokens"]):
                A(f"| {m.split('/')[-1]} | {r['doc_id']} | `{r['file_name']}` | "
                  f"{r['split']} | {r['n_chars']} | {r['n_tokens']} |")

    # ---------- Part 2 ----------
    A("\n## Part 2 — Full sequence budget (measured)\n")
    A("### Output (target JSON) tokens, by verdict\n")
    A("| tokenizer | verdict | n | p50 | p95 | p99 | max |")
    A("|---|---|---|---|---|---|---|")
    for m in models:
        for c in CLASSES:
            v = [r["output_tokens"] for r in seqr
                 if r["model"] == m and r["verdict"] == c]
            if v:
                A(f"| {m.split('/')[-1]} | {c} | {len(v)} | {pct(v,.50)} | "
                  f"{pct(v,.95)} | {pct(v,.99)} | {max(v)} |")

    A("\n### Total sequence length (input + output)\n")
    A("| tokenizer | p50 | p90 | p95 | p99 | max |")
    A("|---|---|---|---|---|---|")
    for m in models:
        v = [r["total_tokens"] for r in seqr if r["model"] == m]
        A(f"| {m.split('/')[-1]} | {pct(v,.50)} | {pct(v,.90)} | {pct(v,.95)} | "
          f"{pct(v,.99)} | {max(v)} |")

    A("\nSequences exceeding each threshold:\n")
    A("| tokenizer | total pairs | >4096 | >8192 | >16384 | >32768 |")
    A("|---|---|---|---|---|---|")
    for m in models:
        v = [r["total_tokens"] for r in seqr if r["model"] == m]
        A(f"| {m.split('/')[-1]} | {len(v)} | " +
          " | ".join(f"{sum(1 for x in v if x > t)} "
                     f"({100*sum(1 for x in v if x > t)/len(v):.1f}%)"
                     for t in THRESHOLDS) + " |")

    # ---------- Part 3 ----------
    A("\n## Part 3 — Tokenizer efficiency\n")
    means = {m: sum(r["n_tokens"] for r in docr if r["model"] == m) /
                sum(1 for r in docr if r["model"] == m) for m in models}
    best = min(means, key=means.get)
    A(f"Mean tokens per document. Baseline = **{best.split('/')[-1]}** (most efficient).\n")
    A("| tokenizer | mean tokens/doc | vs best | chars/token |")
    A("|---|---|---|---|")
    for m in sorted(models, key=lambda x: means[x]):
        ch = sum(r["n_chars"] for r in docr if r["model"] == m) / \
             sum(r["n_tokens"] for r in docr if r["model"] == m)
        d = 100 * (means[m] - means[best]) / means[best]
        A(f"| {m.split('/')[-1]} | {means[m]:.0f} | "
          f"{'baseline' if m==best else f'+{d:.1f}%'} | {ch:.2f} |")
    json.dump({"means": means, "best": best}, open("data/_eff.json", "w"))

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"-> {args.out}  ({len(L)} lines)")
    print(f"most efficient: {best} ({means[best]:.0f} tok/doc)")
    for m in sorted(models, key=lambda x: means[x]):
        print(f"  {m:30s} {means[m]:8.0f}  {100*(means[m]-means[best])/means[best]:+6.1f}%")


if __name__ == "__main__":
    main()
