#!/usr/bin/env python3
"""Split integrity: duplicates within splits, and leakage across them.

Four checks, each reported with a number rather than a claim:
  1. contract-level overlap across train/dev/test (the split unit)
  2. exact duplicate ROWS inside each split
  3. near-duplicate CONTRACTS by normalized-text hash, including across the
     two corpora, since ContractNLI and CUAD were assembled independently
  4. identical evidence spans appearing in more than one split
"""
import json, hashlib, sys
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from text_norm import norm                                     # noqa: E402

SPL = ["train", "dev", "test"]


def h(s):
    return hashlib.sha256(norm(s).encode()).hexdigest()


def load(sp):
    return [json.loads(l) for l in
            open(f"v2/combined/{sp}.jsonl", encoding="utf-8")]


def docid(r):
    if r["task"] == "contractnli":
        return ("nli", r.get("doc_id"))
    return ("cuad", r.get("contract"))


def main():
    data = {sp: load(sp) for sp in SPL}

    print("1. ROW COUNTS\n")
    print(f"  {'split':6s} {'total':>7s} {'ContractNLI':>12s} {'CUAD':>7s} {'RiskNote':>9s}")
    for sp in SPL:
        c = Counter(r["task"] for r in data[sp])
        print(f"  {sp:6s} {len(data[sp]):7d} {c['contractnli']:12d} "
              f"{c['cuad']:7d} {c['risknote']:9d}")
    tot = sum(len(v) for v in data.values())
    print(f"  {'TOTAL':6s} {tot:7d}")

    print("\n2. DOCUMENT COUNTS (the split unit)\n")
    docs = {sp: set(docid(r) for r in data[sp]) for sp in SPL}
    print(f"  {'split':6s} {'NDAs':>6s} {'commercial':>11s} {'total docs':>11s}")
    for sp in SPL:
        n = sum(1 for d in docs[sp] if d[0] == "nli")
        c = sum(1 for d in docs[sp] if d[0] == "cuad")
        print(f"  {sp:6s} {n:6d} {c:11d} {len(docs[sp]):11d}")

    print("\n3. CONTRACT-LEVEL LEAKAGE (same document in two splits)\n")
    bad = 0
    for a in range(3):
        for b in range(a + 1, 3):
            ov = docs[SPL[a]] & docs[SPL[b]]
            bad += len(ov)
            print(f"  {SPL[a]:5s} vs {SPL[b]:5s}: {len(ov)} shared documents")
    print(f"  -> {'PASS' if bad == 0 else 'FAIL'}")

    print("\n4. EXACT DUPLICATE ROWS WITHIN A SPLIT\n")
    for sp in SPL:
        keys = [h(json.dumps(r["messages"], sort_keys=True)) for r in data[sp]]
        d = len(keys) - len(set(keys))
        print(f"  {sp:6s} {d} duplicate rows of {len(keys)}")

    print("\n5. NEAR-DUPLICATE CONTRACT TEXT (normalized hash, across everything)\n")
    txt = {}
    for sp in SPL:
        for r in data[sp]:
            u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
            body = u.split("\n\nQUESTION:")[0].split("\n\nCLAUSE CATEGORY:")[0]
            if body.startswith("CONTRACT: "):
                body = body[len("CONTRACT: "):]
            txt.setdefault(docid(r), (sp, body))
    byhash = defaultdict(list)
    for k, (sp, body) in txt.items():
        byhash[h(body)].append((sp, k))
    dupes = {k: v for k, v in byhash.items() if len(v) > 1}
    print(f"  distinct contracts: {len(txt)}   identical-text groups: {len(dupes)}")
    cross = 0
    for k, v in dupes.items():
        if len({s for s, _ in v}) > 1:
            cross += 1
            print(f"    CROSS-SPLIT: {v}")
    print(f"  -> cross-split identical contracts: {cross} "
          f"({'PASS' if cross == 0 else 'FAIL'})")

    print("\n6. EVIDENCE SPANS SHARED ACROSS SPLITS\n")
    spans = {sp: set(h(e) for r in data[sp] for e in (r.get("evidence") or []))
             for sp in SPL}
    for sp in SPL:
        print(f"  {sp:6s} {len(spans[sp])} distinct gold spans")
    for a in range(3):
        for b in range(a + 1, 3):
            ov = spans[SPL[a]] & spans[SPL[b]]
            print(f"  {SPL[a]:5s} vs {SPL[b]:5s}: {len(ov)} shared spans")
    print("  (boilerplate spans CAN legitimately repeat; only document overlap "
          "is true leakage)")


if __name__ == "__main__":
    main()
