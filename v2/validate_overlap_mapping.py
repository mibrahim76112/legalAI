#!/usr/bin/env python3
"""Targeted test of the one case gold spans did not exercise: evidence that
falls INSIDE the region shared by two consecutive windows.

No gold CUAD span happened to land in an overlap, so that branch of the
coordinate mapping was untested by validate_coord_mapping.py. Rather than
report the branch as unverified, this drives it directly with real contract
text sampled from the overlap region itself.

The test is stronger than the gold-span one: a span in an overlap is visible
from TWO windows at different local offsets, and BOTH must map to the SAME
original-contract offsets. That is exactly the off-by-one that window
arithmetic gets wrong.
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
    ap.add_argument("--per-overlap", type=int, default=12)
    ap.add_argument("--out", default="v2/_overlap_validation.json")
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
    probe = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": USER_TMPL.format(
                 text="", category=max(cats, key=len))}]
    e = tok.apply_chat_template(probe, tokenize=True, add_generation_prompt=True,
                                enable_thinking=False)
    if hasattr(e, "input_ids"):
        e = e["input_ids"]
    if e and isinstance(e[0], list):
        e = e[0]
    overhead = len(list(e))

    res = Counter()
    fails = []
    tested_overlaps = 0
    for cname, rs in bycon.items():
        t = contract_text(rs[0])
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        wins = windows_for(t, enc["offset_mapping"], len(enc["input_ids"]),
                           budget, overhead)
        if len(wins) < 2:
            continue
        wn = {}
        for w in wins:
            wt = t[w["char_start"]:w["char_end"]]
            wn[w["window"]] = (wt,) + norm_with_map(wt)
        for a, b in zip(wins, wins[1:]):
            lo, hi = b["char_start"], a["char_end"]      # shared region
            if hi - lo < 200:
                continue
            tested_overlaps += 1
            res["overlap_regions"] += 1
            # sample real substrings spread across the shared region
            step = max(1, (hi - lo) // (args.per_overlap + 1))
            for k in range(1, args.per_overlap + 1):
                s = lo + k * step
                seg = t[s:s + 160]
                # trim to whole words so the span looks like quoted text
                seg = seg[seg.find(" ") + 1: seg.rfind(" ")] if " " in seg else seg
                if len(norm(seg)) < 40:
                    continue
                res["spans_tested"] += 1
                got = {}
                for w in (a, b):
                    wt, wnorm, widx = wn[w["window"]]
                    if not _grounded(seg, wnorm):
                        res["not_grounded_in_window"] += 1
                        fails.append({"contract": cname, "why": "not_grounded",
                                      "window": w["window"], "span": seg[:90]})
                        continue
                    loc = locate(seg, wt, wnorm, widx)
                    if loc is None:
                        res["locate_FAIL"] += 1
                        fails.append({"contract": cname, "why": "no_locate",
                                      "window": w["window"], "span": seg[:90]})
                        continue
                    ls, le = loc
                    got[w["window"]] = (w["char_start"] + ls, w["char_start"] + le)
                if len(got) == 2:
                    (s1, e1), (s2, e2) = got[a["window"]], got[b["window"]]
                    if (s1, e1) == (s2, e2):
                        res["both_windows_AGREE"] += 1
                    else:
                        res["both_windows_DISAGREE"] += 1
                        fails.append({"contract": cname, "why": "disagree",
                                      "a": [s1, e1], "b": [s2, e2],
                                      "span": seg[:90]})
                    # and the mapped offsets must still round-trip
                    if norm(t[s1:e1]) == norm(seg):
                        res["roundtrip_OK"] += 1
                    else:
                        res["roundtrip_FAIL"] += 1
                        fails.append({"contract": cname, "why": "roundtrip",
                                      "span": seg[:90], "got": t[s1:e1][:90]})

    print("OVERLAP-REGION coordinate mapping test")
    print(f"  multi-window contracts with a usable overlap : {tested_overlaps}")
    for k in ("spans_tested", "both_windows_AGREE", "both_windows_DISAGREE",
              "roundtrip_OK", "roundtrip_FAIL", "locate_FAIL",
              "not_grounded_in_window"):
        print(f"  {k:30s} {res[k]}")
    json.dump({"counts": dict(res), "failures": fails[:20]},
              open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
