#!/usr/bin/env python3
"""Emit a Markdown file of risk notes for human review.

Stratified across categories so the sample is not 30 notes about one clause
type, and drawn with the recorded seed so the same 30 come back on a rerun.
Each entry shows the source clause next to the note, because a note cannot be
judged without the text it claims to describe.
"""
import json, random, argparse, textwrap
from collections import defaultdict, Counter
from pathlib import Path

SEED = 20260919


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="v2/risk_notes/train.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="v2/RISK_NOTE_SAMPLES.md")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.notes, encoding="utf-8")]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    rng = random.Random(SEED)
    for v in by_cat.values():
        rng.shuffle(v)

    # round-robin over categories until n reached -> even coverage
    picked, cats = [], sorted(by_cat)
    i = 0
    while len(picked) < args.n and any(by_cat[c] for c in cats):
        c = cats[i % len(cats)]
        if by_cat[c]:
            picked.append(by_cat[c].pop())
        i += 1

    L = []; A = L.append
    A("# Risk-note samples for review\n")
    A(f"{len(picked)} notes drawn from `{args.notes}` "
      f"({len(rows)} kept notes total), stratified across "
      f"{len({p['category'] for p in picked})} categories, seed {SEED}.\n")
    A("These are **model-generated** (teacher Qwen3-32B). Nothing here is human")
    A("legal advice and nothing has been checked by a lawyer.\n")
    A("For each note, the question is narrow: **does the note describe the")
    A("clause accurately, without inventing anything the clause does not say?**")
    A("Style is secondary. Mark each one and count the rejects at the end.\n")
    A("---\n")
    for i, r in enumerate(picked, 1):
        A(f"### {i}. {r['category']}\n")
        A(f"*Contract:* `{r['contract'][:90]}`\n")
        A("**Clause:**\n")
        A("```")
        A(textwrap.fill(r["clause"][:1200], 88))
        A("```\n")
        A(f"**Note:** {r['note']}\n")
        A(f"<sub>{r['n_words']} words, {r['n_sentences']} sentence(s)</sub>\n")
        A("- [ ] accurate   - [ ] inaccurate   - [ ] invents facts\n")
    A("---\n")
    A("## Tally\n")
    A("accurate: ___ / %d    inaccurate: ___    invents facts: ___\n" % len(picked))
    A("If more than about 5 of 30 are inaccurate or invent facts, the synthetic")
    A("task should not go into the training mix as it stands.\n")

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"-> {args.out}  ({len(picked)} notes, "
          f"{len({p['category'] for p in picked})} categories)")
    print("category spread:", dict(Counter(p["category"] for p in picked)))


if __name__ == "__main__":
    main()
