#!/usr/bin/env python3
"""
Oracle-evidence condition: does perfect retrieval fix verdicts?

LEAK CONTROL. Gold evidence is EMPTY for every NotMentioned row, so supplying
"here are the relevant excerpts" would announce the label. This run is therefore
restricted to GOLD-BEARING rows (Entailment + Contradiction only), where every
row has excerpts and their presence carries no information.

That makes this an upper bound on ENTAILMENT-vs-CONTRADICTION discrimination
given perfect retrieval -- NOT a three-way oracle. The comparison is against the
same row subset from the ordinary run, so it is paired.
"""
import json, argparse
from pathlib import Path
from collections import Counter

SYS = ("You are a junior legal assistant reviewing a contract. You are given the "
       "CONTRACT, the QUESTION, and the EXCERPTS that a senior reviewer has "
       "already identified as the governing language. Using those excerpts, "
       "determine whether the contract supports the stated position (Entailment), "
       "conflicts with it (Contradiction), or does not address it (NotMentioned). "
       'Respond with JSON only, in the form {"verdict": "...", "evidence": [...]}.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="doc_sft/valid.jsonl")
    ap.add_argument("--out", default="doc_sft_oracle")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    kept = Counter(); skipped = 0
    with open(out / "valid.jsonl", "w", encoding="utf-8") as f:
        for l in open(args.src, encoding="utf-8"):
            r = json.loads(l)
            if not r["evidence"]:          # NotMentioned -> would leak
                skipped += 1
                continue
            u = [m["content"] for m in r["messages"] if m["role"] == "user"][0]
            body = u.split("CONTRACT: ", 1)[1]
            text, q = body.rsplit("\n\nQUESTION: ", 1)
            exc = "\n".join(f"- {e}" for e in r["evidence"])
            r["messages"] = [
                {"role": "system", "content": SYS},
                {"role": "user",
                 "content": f"CONTRACT: {text}\n\nEXCERPTS:\n{exc}\n\nQUESTION: {q}"},
                {"role": "assistant", "content": json.dumps(
                    {"verdict": r["verdict"], "evidence": r["evidence"]},
                    ensure_ascii=False)}]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept[r["verdict"]] += 1
    for s in ("train", "test"):
        (out / f"{s}.jsonl").write_text(Path(f"doc_sft/{s}.jsonl").read_text(encoding="utf-8"),
                                        encoding="utf-8")
    print(f"oracle eval rows: {sum(kept.values())}   {dict(kept)}")
    print(f"skipped (NotMentioned, would leak): {skipped}")


if __name__ == "__main__":
    main()
