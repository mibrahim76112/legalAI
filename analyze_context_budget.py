#!/usr/bin/env python3
"""
Context-length budget for document-level SFT + tokenizer efficiency comparison.

CPU only, login-node safe: all 607 documents tokenize in ~2s per tokenizer.
Every number here is MEASURED end-to-end, not estimated from part counts -
subword boundaries make token counts non-additive, and a budget built on
addition would be wrong by an unknown margin exactly where it matters.

  python analyze_context_budget.py

Outputs:
  CONTEXT_BUDGET.md            report
  data/doc_token_counts.csv    per document x tokenizer
  data/seq_token_counts.csv    per (document, hypothesis) x tokenizer
"""
import json, csv, argparse, statistics as st
from pathlib import Path
from collections import defaultdict

SYSTEM = (
    "You are a junior legal assistant reviewing a contract. Given the CONTRACT "
    "and the QUESTION, determine whether the contract supports the stated position "
    "(Entailment), conflicts with it (Contradiction), or does not address it "
    "(NotMentioned). Return the exact sentence(s) from the contract that justify "
    "your answer, or 'no related clause' if none apply. Respond with JSON only, "
    "in the form {\"verdict\": \"...\", \"evidence\": [...]}."
)
USER_TMPL = "CONTRACT: {text}\n\nQUESTION: {hypothesis}"
CLASSES = ["Entailment", "Contradiction", "NotMentioned"]
THRESHOLDS = [4096, 8192, 16384, 32768]


def ids(enc):
    """transformers>=5 returns BatchEncoding from tokenize=True."""
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):
        enc = enc[0]
    return list(enc)


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0
    i = min(int(q * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[i]


def load_splits(hf_home):
    from huggingface_hub import hf_hub_download
    out = {}
    for s in ("train", "dev", "test"):
        p = hf_hub_download(repo_id="trismik/ContractNLI", filename=f"{s}.json",
                            repo_type="dataset")
        out[s] = json.load(open(p, encoding="utf-8"))
    return out


def evidence_texts(doc, span_idx):
    """Resolve annotation span INDICES through doc['spans'] to the real text.

    doc['spans'] is a list of [start, end] char offsets. The annotation carries
    indices into that list. Deliberately NOT min..max of the range: the
    annotated spans are often discontiguous, and spanning the gap would invent
    evidence the annotator did not mark.
    """
    out = []
    for i in span_idx:
        if 0 <= i < len(doc["spans"]):
            a, b = doc["spans"][i]
            out.append(doc["text"][a:b].strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-file", default="models.txt")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    splits = load_splits(None)
    labels = splits["dev"]["labels"]          # hypothesis text per nda-N
    models = [l.strip() for l in open(args.models_file) if l.strip()]

    docs = []      # (split, doc)
    for s in ("train", "dev", "test"):
        for d in splits[s]["documents"]:
            docs.append((s, d))
    print(f"documents: {len(docs)}  (train/dev/test = "
          f"{len(splits['train']['documents'])}/{len(splits['dev']['documents'])}"
          f"/{len(splits['test']['documents'])})")

    doc_rows, seq_rows = [], []
    for m in models:
        tok = AutoTokenizer.from_pretrained(m)
        short = m.split("/")[-1]
        print(f"\n=== {m} ===", flush=True)

        # ---- Part 1: document token counts -----------------------------
        for split, d in docs:
            n = len(tok(d["text"], add_special_tokens=False)["input_ids"])
            doc_rows.append({"model": m, "split": split, "doc_id": d["id"],
                             "file_name": d["file_name"], "n_chars": len(d["text"]),
                             "n_tokens": n})

        # ---- Part 2: full training sequences ---------------------------
        n_pairs = 0
        for split, d in docs:
            ann = d["annotation_sets"][0]["annotations"]
            for hid, a in ann.items():
                hyp = labels[hid]["hypothesis"]
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user",
                         "content": USER_TMPL.format(text=d["text"], hypothesis=hyp)}]
                inp = len(ids(tok.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True)))
                ev = evidence_texts(d, a.get("spans", []))
                target = json.dumps({"verdict": a["choice"], "evidence": ev},
                                    ensure_ascii=False)
                out = len(tok(target, add_special_tokens=False)["input_ids"])
                seq_rows.append({"model": m, "split": split, "doc_id": d["id"],
                                 "file_name": d["file_name"], "hypothesis_id": hid,
                                 "verdict": a["choice"], "n_evidence_spans": len(ev),
                                 "input_tokens": inp, "output_tokens": out,
                                 "total_tokens": inp + out})
                n_pairs += 1
        print(f"  pairs measured: {n_pairs}")

    outd = Path(args.out); outd.mkdir(exist_ok=True)
    with open(outd / "doc_token_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(doc_rows[0])); w.writeheader(); w.writerows(doc_rows)
    with open(outd / "seq_token_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seq_rows[0])); w.writeheader(); w.writerows(seq_rows)
    print(f"\n-> {outd}/doc_token_counts.csv  ({len(doc_rows)} rows)")
    print(f"-> {outd}/seq_token_counts.csv  ({len(seq_rows)} rows)")


if __name__ == "__main__":
    main()
