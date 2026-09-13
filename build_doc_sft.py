#!/usr/bin/env python3
"""
PHASE 0 - document-level SFT data from official trismik/ContractNLI.

One example per (document, hypothesis) pair, 17 per document, official splits
used as-is (no re-splitting).

  input  = system prompt + full contract text + one policy question
  target = {"verdict": ..., "evidence": [<resolved gold span texts>]}

Evidence is resolved through doc["spans"] using only the ANNOTATED span indices,
never the min..max range: annotated spans are frequently discontiguous and
spanning the gap would invent evidence the annotator did not mark.

The policy question is the `hypothesis` field from the official labels dict
VERBATIM - rewording it into a question would add uncontrolled variance against
a fixed gold phrasing.

Outputs doc_sft/{train,valid,test}.jsonl
"""
import json, argparse, hashlib
from pathlib import Path
from collections import Counter, defaultdict

# Says "empty evidence list" so the prompt and the target agree. Training on []
# while the prompt asks for a sentinel string teaches the model to disobey.
SYSTEM = (
    "You are a junior legal assistant reviewing a contract. Given the CONTRACT "
    "and the QUESTION, determine whether the contract supports the stated position "
    "(Entailment), conflicts with it (Contradiction), or does not address it "
    "(NotMentioned). Return the exact sentence(s) from the contract that justify "
    "your answer, or an empty evidence list if none apply. Respond with JSON only, "
    "in the form {\"verdict\": \"...\", \"evidence\": [...]}."
)
USER_TMPL = "CONTRACT: {text}\n\nQUESTION: {hypothesis}"

SPLIT_FILE = {"train": "train", "dev": "valid", "test": "test"}
# label-corrupting pairs: gold evidence falls beyond an 8192-token truncation
DROP = {("622", "nda-10"), ("622", "nda-19"), ("162", "nda-19")}


def norm(s):
    return " ".join(s.lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="doc_sft")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    splits = {}
    for s in ("train", "dev", "test"):
        p = hf_hub_download(repo_id="trismik/ContractNLI", filename=f"{s}.json",
                            repo_type="dataset")
        splits[s] = json.load(open(p, encoding="utf-8"))
    labels = splits["dev"]["labels"]

    outd = Path(args.out); outd.mkdir(exist_ok=True)
    stats, bad_evidence, dropped, whitespace_spans = {}, [], [], []

    for split in ("train", "dev", "test"):
        rows = []
        for d in splits[split]["documents"]:
            did = str(d["id"])
            text = d["text"]
            ann = d["annotation_sets"][0]["annotations"]
            for hid, a in ann.items():
                if (did, hid) in DROP:
                    dropped.append((split, did, hid, a["choice"]))
                    continue
                ev, ws_dropped = [], 0
                for i in a.get("spans", []):
                    if 0 <= i < len(d["spans"]):
                        s0, s1 = d["spans"][i]
                        span = text[s0:s1].strip()
                        # ContractNLI carries exactly one whitespace-only
                        # annotated span (dev doc 582 nda-19, a single " ").
                        # Keeping it would put "" in the training target and
                        # make that example un-scoreable: an empty string can
                        # never be "covered", so no model could ever earn a TP
                        # on it. Dropping it is the only reading consistent
                        # with "the exact sentence(s) that justify your answer".
                        if not span:
                            ws_dropped += 1
                            continue
                        ev.append(span)
                if ws_dropped:
                    whitespace_spans.append({"split": split, "doc_id": did,
                                             "hypothesis_id": hid,
                                             "dropped": ws_dropped,
                                             "kept": len(ev)})
                user = USER_TMPL.format(text=text, hypothesis=labels[hid]["hypothesis"])
                target = json.dumps({"verdict": a["choice"], "evidence": ev},
                                    ensure_ascii=False)

                # VERIFY: every evidence string must occur verbatim in the input
                for e in ev:
                    if norm(e) not in norm(user):
                        bad_evidence.append({"split": split, "doc_id": did,
                                             "hypothesis_id": hid,
                                             "evidence": e[:120]})
                rows.append({
                    "doc_id": did, "file_name": d["file_name"],
                    "hypothesis_id": hid, "split": split,
                    "verdict": a["choice"], "evidence": ev,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": target},
                    ],
                })
        fn = outd / f"{SPLIT_FILE[split]}.jsonl"
        with open(fn, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = Counter(r["verdict"] for r in rows)
        stats[split] = (len(rows), dict(c))
        print(f"{str(fn):26s} {len(rows):6d} rows  {dict(c)}")

    print(f"\ndropped (label-corrupting): {len(dropped)}")
    for s, did, hid, ch in dropped:
        print(f"  {s:6s} doc {did:>4s} {hid:8s} {ch}")

    print(f"\nwhitespace-only gold spans dropped: {len(whitespace_spans)}")
    for w in whitespace_spans:
        print(f"  {w['split']:6s} doc {w['doc_id']:>4s} {w['hypothesis_id']:8s} "
              f"dropped {w['dropped']}, kept {w['kept']}")

    print(f"\nEVIDENCE VERBATIM CHECK: {len(bad_evidence)} failures")
    if bad_evidence:
        for b in bad_evidence[:20]:
            print(f"  {b['split']:6s} doc {b['doc_id']:>4s} {b['hypothesis_id']:8s} "
                  f"{b['evidence']!r}")
        json.dump(bad_evidence, open(outd / "_bad_evidence.json", "w"), indent=2)
    else:
        print("  all evidence strings occur verbatim in their input")

    json.dump({"stats": {k: v[1] for k, v in stats.items()},
               "counts": {k: v[0] for k, v in stats.items()},
               "dropped": dropped, "n_bad_evidence": len(bad_evidence),
               "whitespace_spans_dropped": whitespace_spans,
               "system_prompt": SYSTEM},
              open(outd / "_manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
