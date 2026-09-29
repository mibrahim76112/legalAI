#!/usr/bin/env python3
"""
Two-task combined dataset: ContractNLI (Task 1) + CUAD (Task 2).

Task 3 (risk notes) is included when --risk-notes is passed. It is a
clause-level task by design, which makes the three tasks chain: Task 2 locates
the clause in the contract, Task 3 explains what that clause does. Feeding the
whole contract for a note about one clause would also push Task 3 into the same
truncation problem Task 2 already has, for no benefit.

max_seq_len = 16384, chosen from a MEASUREMENT, not convenience:
  cap     CUAD rows truncated   evidence DESTROYED
  8192            1428                826  (38.3% of present rows)
  16384            755                375  (17.4%)
  32768            341                 71  ( 3.3%)
8192 would train 826 rows on targets citing text the model cannot see --
the same corruption that caused 3 ContractNLI rows to be dropped in v1, at
275x the scale. 32768 halves the loss again but a 14B LoRA at 32k does not
fit one H100 (v1 peaked at 49GB for 14B at 8192).

Rows whose evidence does not survive truncation are DROPPED, not truncated.
"""
import os, sys, json, random, argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from text_norm import norm                      # noqa: E402

SEED = 20260919
MAX_SEQ = 16384


RN_SYSTEM = (
    "You are a contract review assistant. Given a clause and its category, "
    "write EXACTLY ONE sentence for a business reader explaining what the "
    "clause does in practice and what brings it into effect."
)
RN_USER = "CLAUSE CATEGORY: {cat}\n\nCLAUSE: {clause}"


def risk_note_row(r):
    """Risk-note records are raw generations, not SFT rows -- build messages."""
    return {
        "task": "risknote",
        "contract": r["contract"], "category": r["category"],
        "split": r["split"], "evidence": [],
        "messages": [
            {"role": "system", "content": RN_SYSTEM},
            {"role": "user", "content": RN_USER.format(cat=r["category"],
                                                       clause=r["clause"])},
            {"role": "assistant", "content": r["note"]},
        ],
    }


def survives(tok, r, cap):
    """Does this row's evidence survive prompt-left truncation at `cap`?"""
    msgs = [m for m in r["messages"] if m["role"] != "assistant"]
    tgt = [m for m in r["messages"] if m["role"] == "assistant"][0]["content"]
    pid = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    if hasattr(pid, "input_ids"):
        pid = pid["input_ids"]
    tid = tok(tgt, add_special_tokens=False)["input_ids"]
    if len(pid) + len(tid) <= cap:
        return True
    keep = cap - len(tid)
    if keep <= 0:
        return False
    ev = r.get("evidence") or []
    if not ev:
        return True                      # nothing to lose
    hay = norm(tok.decode(pid[-keep:]))
    return all(norm(e) in hay for e in ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="v2/combined")
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ)
    ap.add_argument("--cuad-train", default="",
                    help="override the CUAD TRAIN file, e.g. the windowed build "
                         "at v2/cuad_windowed/train.jsonl. dev/test always come "
                         "from v2/cuad/ so evaluation stays comparable.")
    ap.add_argument("--risk-notes", default="",
                    help="directory of generated risk notes; omit for the "
                         "two-task ablation")
    ap.add_argument("--cuad-share", type=float, default=0.45,
                    help="target CUAD share of the TRAIN mix")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    SPLIT = {"train": ("train", "train"), "dev": ("valid", "dev"),
             "test": ("test", "test")}
    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    stats = {}
    dropped_rows = []

    for split, (nli_f, cuad_f) in SPLIT.items():
        nli = [json.loads(l) for l in
               open(f"v1_contractnli/doc_sft/{nli_f}.jsonl", encoding="utf-8")]
        cuad_src = (args.cuad_train if (split == "train" and args.cuad_train)
                    else f"v2/cuad/{cuad_f}.jsonl")
        cuad = [json.loads(l) for l in open(cuad_src, encoding="utf-8")]
        if split == "train":
            print(f"  CUAD train source: {cuad_src} ({len(cuad)} rows)")
        for r in nli:
            r["task"] = "contractnli"
        for r in cuad:
            r["task"] = "cuad"

        # The truncation-survival filter applies to TRAIN ONLY.
        # Dropping an evidence-truncated row from dev/test would curate the
        # evaluation set toward shorter contracts and report an optimistic
        # number. A model that cannot handle a long contract must be scored on
        # it, not excused from it. On train the same row is poison, because its
        # target cites text the model cannot see.
        dropped = Counter()
        if split == "train":
            keep_c, keep_n = [], []
            for r in cuad:
                if survives(tok, r, args.max_seq_len):
                    keep_c.append(r)
                else:
                    dropped["cuad_evidence_truncated"] += 1
                    dropped_rows.append({"task": "cuad", "contract": r.get("contract"),
                                         "category": r.get("category")})
            for r in nli:
                if survives(tok, r, args.max_seq_len):
                    keep_n.append(r)
                else:
                    dropped["contractnli_evidence_truncated"] += 1
                    dropped_rows.append({"task": "contractnli",
                                         "doc_id": r.get("doc_id"),
                                         "hypothesis_id": r.get("hypothesis_id")})
        else:
            keep_c, keep_n = cuad, nli

        keep_r = []
        if args.risk_notes:
            rn_path = Path(args.risk_notes) / f"{split}.jsonl"
            if rn_path.exists():
                for r in (json.loads(l) for l in open(rn_path, encoding="utf-8")):
                    row = risk_note_row(r)
                    if split != "train" or survives(tok, row, args.max_seq_len):
                        keep_r.append(row)
                    else:
                        dropped["risknote_truncated"] += 1

        # Mixture: subsample CUAD so it is ~45% of train and ContractNLI does not
        # dominate. Dev/test keep everything -- a mixture ratio is a training
        # choice, not an evaluation one.
        if split == "train":
            want = int(args.cuad_share / (1 - args.cuad_share) * len(keep_n))
            if len(keep_c) > want:
                rng.shuffle(keep_c); keep_c = keep_c[:want]
        rows = keep_n + keep_c + keep_r
        rng.shuffle(rows)
        with open(outd / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = {"total": len(rows),
                        "contractnli": len(keep_n), "cuad": len(keep_c),
                        "risknote": len(keep_r),
                        "cuad_share": len(keep_c) / len(rows),
                        "dropped": dict(dropped)}
        print(f"{split:5s} total={len(rows):6d}  nli={len(keep_n):5d} "
              f"cuad={len(keep_c):5d} ({len(keep_c)/len(rows):.1%})  "
              f"risknote={len(keep_r):5d}  dropped={dict(dropped)}")

    json.dump(dropped_rows, open(outd / "_dropped_train_rows.json", "w"), indent=2)
    json.dump({"seed": SEED, "max_seq_len": args.max_seq_len,
               "cuad_share_target": args.cuad_share,
               "tasks": (["contractnli", "cuad", "risknote"]
                         if args.risk_notes else ["contractnli", "cuad"]),
               "risk_notes_source": args.risk_notes or None,
               "cuad_train_source": args.cuad_train or "v2/cuad/train.jsonl",
               "truncation_filter": "TRAIN ONLY; dev/test keep every row",
               "splits": stats},
              open(outd / "_manifest.json", "w"), indent=2)
    print(f"\n-> {outd}/")


if __name__ == "__main__":
    main()
