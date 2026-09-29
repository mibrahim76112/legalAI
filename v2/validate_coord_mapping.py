#!/usr/bin/env python3
"""Validate window -> original-contract coordinate mapping.

Uses GOLD evidence spans as the payload: they are real substrings of real
contracts, so a correct mapping must round-trip every one of them. Each mapped
span is checked by slicing the ORIGINAL contract at the computed offsets and
comparing the normalized result to the span.

Reports separately for spans landing:
  - near the START of a window (first 10% of its tokens)
  - near the END of a window (last 10%)
  - inside the OVERLAP region shared by two windows
because those are the three places an off-by-one would show up.
"""
import sys, json, argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_norm import norm                                     # noqa: E402
from reasoning_filters import _grounded                        # noqa: E402
from cuad_windowed import (norm_with_map, locate, windows_for,  # noqa: E402
                           contract_text, SYSTEM, USER_TMPL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", default="v2/combined/dev.jsonl")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", default="v2/_coord_validation.json")
    args = ap.parse_args()
    budget = args.max_model_len - args.max_new_tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    rows = [r for r in rows if r.get("task") == "cuad"]
    bycon = defaultdict(list)
    for r in rows:
        bycon[r["contract"]].append(r)
    cats = sorted({r["category"] for r in rows})
    longest = max(cats, key=len)
    probe = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": USER_TMPL.format(text="", category=longest)}]
    e = tok.apply_chat_template(probe, tokenize=True, add_generation_prompt=True,
                                enable_thinking=False)
    if hasattr(e, "input_ids"):
        e = e["input_ids"]
    if e and isinstance(e[0], list):
        e = e[0]
    overhead = len(list(e))

    res = Counter()
    pos = defaultdict(Counter)
    failures = []
    visible_b = Counter()
    for cname, rs in bycon.items():
        t = contract_text(rs[0])
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        offs, n_tok = enc["offset_mapping"], len(enc["input_ids"])
        wins = windows_for(t, offs, n_tok, budget, overhead)
        wnorm = {}
        for w in wins:
            wt = t[w["char_start"]:w["char_end"]]
            wnorm[w["window"]] = (wt,) + norm_with_map(wt)
        ov_lo = {}
        for a, b in zip(wins, wins[1:]):
            ov_lo[b["window"]] = (b["char_start"], a["char_end"])

        for r in rs:
            for g in (r.get("evidence") or []):
                res["gold_spans"] += 1
                hit = False
                for w in wins:
                    wt, wn, wi = wnorm[w["window"]]
                    if not _grounded(g, wn):
                        continue
                    hit = True
                    loc = locate(g, wt, wn, wi)
                    if loc is None:
                        res["located_FAIL"] += 1
                        failures.append({"contract": cname, "why": "no_locate",
                                         "span": g[:120]})
                        continue
                    ls, le = loc
                    cs = w["char_start"] + ls
                    ce = w["char_start"] + le
                    # THE round-trip: slice the ORIGINAL contract at the mapped
                    # offsets and require it to match the span.
                    if norm(t[cs:ce]) == norm(g) or \
                       norm(t[cs:ce]) == norm(g).strip(".,;:!?’'\"-— "):
                        res["roundtrip_OK"] += 1
                    else:
                        res["roundtrip_FAIL"] += 1
                        failures.append({"contract": cname, "why": "roundtrip",
                                         "span": g[:120],
                                         "got": t[cs:ce][:120]})
                    span_tok_frac = (ls / max(len(wt), 1))
                    if span_tok_frac <= 0.10:
                        pos["start_of_window"][
                            "ok" if res["roundtrip_FAIL"] == 0 else "seen"] += 1
                        pos["start_of_window"]["n"] += 1
                    if (le / max(len(wt), 1)) >= 0.90:
                        pos["end_of_window"]["n"] += 1
                    if w["window"] in ov_lo:
                        lo, hi = ov_lo[w["window"]]
                        if cs >= lo and ce <= hi:
                            pos["in_overlap"]["n"] += 1
                    break
                if hit:
                    visible_b["visible"] += 1
                else:
                    visible_b["INVISIBLE"] += 1
                    failures.append({"contract": cname, "why": "not_in_any_window",
                                     "span": g[:120]})

    print("coordinate mapping validation (gold spans as payload)")
    print(f"  gold spans tested      : {res['gold_spans']}")
    print(f"  round-trip OK          : {res['roundtrip_OK']}")
    print(f"  round-trip FAIL        : {res['roundtrip_FAIL']}")
    print(f"  could not locate       : {res['located_FAIL']}")
    print(f"\nposition coverage of tested spans:")
    for k in ("start_of_window", "end_of_window", "in_overlap"):
        print(f"  {k:18s} {pos[k]['n']}")
    print(f"\nB evidence visibility (span appears in SOME window):")
    print(f"  visible   : {visible_b['visible']}")
    print(f"  INVISIBLE : {visible_b['INVISIBLE']}")
    tot = visible_b['visible'] + visible_b['INVISIBLE']
    print(f"  visibility: {visible_b['visible']/tot:.4%}" if tot else "")
    json.dump({"counts": dict(res), "position_coverage": {k: dict(v) for k, v in pos.items()},
               "b_visibility": dict(visible_b),
               "failures_sample": failures[:20]},
              open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
