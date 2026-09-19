#!/usr/bin/env python3
"""
Part 4: is truncating at a candidate max_seq_len SAFE?

A truncated document is only acceptable if every gold evidence span survives the
cut. If a span falls in the removed region, the training target cites text the
model cannot see - which teaches it to fabricate citations. That is a corrupted
label, not a shorter example, and it is worse than dropping the row.

Method: tokenize each at-risk document WITH offset mappings, convert the token
budget into a character cutoff, and compare against the real gold span offsets.
"""
import json, csv, argparse
from pathlib import Path
from collections import defaultdict


def load_splits():
    from huggingface_hub import hf_hub_download
    out = {}
    for s in ("train", "dev", "test"):
        p = hf_hub_download(repo_id="trismik/ContractNLI",
                            filename=f"{s}.json", repo_type="dataset")
        out[s] = json.load(open(p, encoding="utf-8"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seq-len", type=int, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--models-file", default="models.txt")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    seqr = [r for r in csv.DictReader(open(Path(args.data) / "seq_token_counts.csv"))]
    docr = [r for r in csv.DictReader(open(Path(args.data) / "doc_token_counts.csv"))]
    for r in seqr:
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            r[k] = int(r[k])
    doctok = {(r["model"], r["doc_id"]): int(r["n_tokens"]) for r in docr}

    splits = load_splits()
    docs = {}
    for s in ("train", "dev", "test"):
        for d in splits[s]["documents"]:
            docs[str(d["id"])] = (s, d)

    models = [l.strip() for l in open(args.models_file) if l.strip()]
    L = args.max_seq_len
    print(f"=== truncation safety at max_seq_len = {L} ===\n")

    summary = {}
    for m in models:
        at_risk = [r for r in seqr if r["model"] == m and r["total_tokens"] > L]
        short = m.split("/")[-1]
        if not at_risk:
            print(f"{short:26s} 0 sequences exceed {L} - nothing truncated")
            summary[m] = {"pairs": 0, "docs": 0, "evidence_lost": 0, "lost_docs": []}
            continue

        tok = AutoTokenizer.from_pretrained(m)
        offs_cache = {}
        lost_pairs, lost_docs = 0, set()
        risk_docs = {r["doc_id"] for r in at_risk}

        for r in at_risk:
            did = r["doc_id"]
            s, d = docs[did]
            if did not in offs_cache:
                enc = tok(d["text"], add_special_tokens=False,
                          return_offsets_mapping=True)
                offs_cache[did] = enc["offset_mapping"]
            offs = offs_cache[did]

            # tokens the template/system/question consume, measured not assumed
            overhead = r["input_tokens"] - doctok[(m, did)]
            doc_budget = L - overhead - r["output_tokens"]
            if doc_budget <= 0:
                lost_pairs += 1; lost_docs.add(did); continue
            if doc_budget >= len(offs):
                continue
            char_cut = offs[doc_budget - 1][1]

            ann = d["annotation_sets"][0]["annotations"][r["hypothesis_id"]]
            gold = [d["spans"][i] for i in ann.get("spans", [])
                    if 0 <= i < len(d["spans"])]
            if any(end > char_cut for _, end in gold):
                lost_pairs += 1
                lost_docs.add(did)

        print(f"{short:26s} {len(at_risk):5d} pairs over budget "
              f"({len(risk_docs)} docs) | evidence LOST in {lost_pairs} pairs "
              f"({len(lost_docs)} docs)")
        for did in sorted(lost_docs, key=int):
            s, d = docs[did]
            print(f"    doc {did:>4s} [{s:5s}] {d['file_name']}  "
                  f"{doctok[(m,did)]} tok")
        summary[m] = {"pairs": len(at_risk), "docs": len(risk_docs),
                      "evidence_lost": lost_pairs,
                      "lost_docs": sorted(lost_docs, key=int)}

    json.dump(summary, open(Path(args.data) / f"_trunc_{L}.json", "w"), indent=2)
    print(f"\n-> {args.data}/_trunc_{L}.json")


if __name__ == "__main__":
    main()
