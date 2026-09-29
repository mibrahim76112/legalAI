#!/usr/bin/env python3
"""What would a WINDOWED CUAD training set look like?

Direct-truncation training drops 375 rows whose gold evidence falls outside the
16,384-token window, and every surviving truncated row keeps the contract TAIL.
Windowed training would instead emit one row per (contract, category, window),
with the evidence inside the window by construction.

This measures the yield and the shape change before anything is rebuilt:
  - how many of the 375 dropped rows come back
  - how many training rows the windowed build produces
  - the present/absent balance, which windowing distorts: a present clause
    lives in ONE window, so the other windows of that contract become absent
    rows for that category, and naive windowing inflates the negative class.
"""
import sys, json, argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_norm import norm                                     # noqa: E402
from reasoning_filters import _grounded                        # noqa: E402
from cuad_windowed import (windows_for, contract_text,          # noqa: E402
                           SYSTEM, USER_TMPL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", default="v2/cuad/train.jsonl")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", default="v2/_windowed_train_yield.json")
    args = ap.parse_args()
    budget = args.max_model_len - args.max_new_tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    bycon = defaultdict(list)
    for r in rows:
        bycon[r["contract"]].append(r)
    cats = sorted({r["category"] for r in rows})
    probe = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": USER_TMPL.format(text="",
                                                          category=max(cats, key=len))}]
    e = tok.apply_chat_template(probe, tokenize=True, add_generation_prompt=True,
                                enable_thinking=False)
    if hasattr(e, "input_ids"):
        e = e["input_ids"]
    if e and isinstance(e[0], list):
        e = e[0]
    overhead = len(list(e))

    c = Counter()
    win_hist = Counter()
    for cname, rs in bycon.items():
        t = contract_text(rs[0])
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        wins = windows_for(t, enc["offset_mapping"], len(enc["input_ids"]),
                           budget, overhead)
        win_hist[len(wins)] += 1
        wnorm = [norm(t[w["char_start"]:w["char_end"]]) for w in wins]
        for r in rs:
            ev = r.get("evidence") or []
            if not ev:
                # absent row -> one row per window, all absent
                c["absent_rows_out"] += len(wins)
                c["absent_rows_in"] += 1
                continue
            c["present_rows_in"] += 1
            placed = False
            for wi, hay in enumerate(wnorm):
                allin = all(_grounded(g, hay) for g in ev)
                anyin = any(_grounded(g, hay) for g in ev)
                if allin:
                    c["present_rows_out"] += 1
                    placed = True
                elif anyin:
                    c["windows_with_PARTIAL_evidence"] += 1
                else:
                    c["absent_rows_out_from_present_contract"] += 1
            if not placed:
                c["present_rows_UNPLACEABLE"] += 1

    tot_out = (c["present_rows_out"] + c["absent_rows_out"]
               + c["absent_rows_out_from_present_contract"])
    print(f"budget {budget}, {len(bycon)} train contracts\n")
    print("windows per contract:", dict(sorted(win_hist.items())))
    print(f"\ncurrent direct-truncation training set:")
    print(f"  rows                          6246  (2155 present / 4091 absent)")
    print(f"  dropped, evidence truncated    375  (all present rows)")
    print(f"  usable                        5871  (1780 present / 4091 absent)")
    print(f"\nwindowed training set would be:")
    print(f"  present rows (all spans in one window) {c['present_rows_out']:6d}")
    print(f"  present rows UNPLACEABLE               {c['present_rows_UNPLACEABLE']:6d}")
    print(f"  windows holding only SOME spans        {c['windows_with_PARTIAL_evidence']:6d}  <- ambiguous, needs a rule")
    print(f"  absent rows from absent contracts      {c['absent_rows_out']:6d}")
    print(f"  absent rows from present contracts     {c['absent_rows_out_from_present_contract']:6d}  <- NEW negatives windowing creates")
    print(f"  total rows                             {tot_out:6d}  ({tot_out/5871:.2f}x current)")
    pres = c["present_rows_out"]
    print(f"  present rate                           {pres/tot_out:6.1%}  (current 30.3%)")
    print(f"\nrecovered vs current: {pres - 1780:+d} present rows")
    json.dump(dict(c) | {"windows_per_contract": {str(k): v for k, v in win_hist.items()},
                         "total_rows_out": tot_out},
              open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
