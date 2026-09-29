#!/usr/bin/env python3
"""Near-duplicate detection across splits — the check a hash cannot do.

A SHA-256 over normalized text only catches byte-identical documents. Two
copies of the same NDA differing by a date, a party name, or a single clause
hash completely differently, so hashing reports them as unrelated. That is the
common case in contract corpora, where the same template is filed repeatedly
with the counterparty swapped.

Method: MinHash over word shingles, the standard corpus-dedup approach.
  - 5-word shingles, so local edits perturb only the shingles they touch
  - 256 permutations; estimated Jaccard = fraction of matching signature slots
  - exact Jaccard recomputed for every candidate pair above the screen, so no
    reported number is an estimate

Reported at several thresholds because "duplicate" is a judgement call:
  >= 0.95  effectively the same document
  >= 0.80  same template, minor edits
  >= 0.60  strong family resemblance (common in NDAs; not necessarily leakage)
"""
import json, sys, re, argparse
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from text_norm import norm                                     # noqa: E402

SPL = ["train", "dev", "test"]
WORD = re.compile(r"[a-z0-9]+")


def shingles(text, k=5):
    w = WORD.findall(norm(text))
    return {hash(" ".join(w[i:i + k])) for i in range(max(len(w) - k + 1, 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=256)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--screen", type=float, default=0.35)
    ap.add_argument("--out", default="v2/_near_dup.json")
    args = ap.parse_args()
    import numpy as np

    def body(r):
        u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
        b = u.split("\n\nQUESTION:")[0].split("\n\nCLAUSE CATEGORY:")[0]
        return b[len("CONTRACT: "):] if b.startswith("CONTRACT: ") else b

    docs = {}
    for sp in SPL:
        for r in (json.loads(l) for l in
                  open(f"v2/combined/{sp}.jsonl", encoding="utf-8")):
            key = ("nli", r["doc_id"]) if r["task"] == "contractnli" \
                else ("cuad", r.get("contract"))
            if r["task"] == "risknote":
                continue                      # clause-level, not a document
            if key not in docs:
                docs[key] = (sp, body(r))
    keys = sorted(docs, key=str)
    print(f"[data] {len(keys)} distinct contracts "
          f"({sum(1 for k in keys if k[0]=='nli')} NDA, "
          f"{sum(1 for k in keys if k[0]=='cuad')} commercial)", flush=True)

    sets = [shingles(docs[k][1], args.k) for k in keys]
    rng = np.random.default_rng(20260919)
    A = rng.integers(1, 2**61 - 1, args.perms, dtype=np.int64)
    B = rng.integers(0, 2**61 - 1, args.perms, dtype=np.int64)
    M = 2**61 - 1
    sig = np.empty((len(keys), args.perms), dtype=np.int64)
    for i, s in enumerate(sets):
        if not s:
            sig[i] = 0
            continue
        v = np.fromiter((x & 0x7FFFFFFFFFFFFFFF for x in s), dtype=np.int64,
                        count=len(s))
        sig[i] = ((np.outer(v, A) + B) % M).min(axis=0)
    print(f"[minhash] {args.perms} permutations, {args.k}-word shingles", flush=True)

    est = (sig[:, None, :] == sig[None, :, :]).mean(axis=2)
    np.fill_diagonal(est, 0.0)
    cand = [(i, j) for i in range(len(keys)) for j in range(i + 1, len(keys))
            if est[i, j] >= args.screen]
    print(f"[screen] {len(cand)} candidate pairs at estimated Jaccard >= {args.screen}",
          flush=True)

    exact = []
    for i, j in cand:
        a, b = sets[i], sets[j]
        u = len(a | b)
        exact.append((len(a & b) / u if u else 0.0, i, j))
    exact.sort(reverse=True)

    print("\nCROSS-SPLIT near-duplicates (the ones that matter):\n")
    print(f"  {'Jaccard':>8s}  {'split A':>6s} {'split B':>7s}  documents")
    hits = {0.95: 0, 0.80: 0, 0.60: 0}
    shown = 0
    for jac, i, j in exact:
        sa, sb = docs[keys[i]][0], docs[keys[j]][0]
        if sa == sb:
            continue
        for t in hits:
            if jac >= t:
                hits[t] += 1
        if shown < 12 and jac >= 0.60:
            print(f"  {jac:8.3f}  {sa:>6s} {sb:>7s}  {str(keys[i])[:34]} | {str(keys[j])[:34]}")
            shown += 1
    print(f"\n  cross-split pairs at Jaccard >= 0.95: {hits[0.95]}")
    print(f"  cross-split pairs at Jaccard >= 0.80: {hits[0.80]}")
    print(f"  cross-split pairs at Jaccard >= 0.60: {hits[0.60]}")

    within = sum(1 for jac, i, j in exact
                 if docs[keys[i]][0] == docs[keys[j]][0] and jac >= 0.80)
    print(f"\n  within-split pairs at Jaccard >= 0.80: {within} (harmless, but "
          f"inflates effective sample size)")

    json.dump({"n_docs": len(keys), "perms": args.perms, "shingle_k": args.k,
               "screen": args.screen, "cross_split": hits, "within_split_080": within,
               "top": [{"jaccard": round(j, 4), "a": list(keys[x]),
                        "split_a": docs[keys[x]][0], "b": list(keys[y]),
                        "split_b": docs[keys[y]][0]}
                       for j, x, y in exact[:40]]},
              open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
