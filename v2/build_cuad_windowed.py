#!/usr/bin/env python3
"""Windowed CUAD TRAINING data. Train split only; dev/test stay whole.

Why this exists
---------------
Direct-truncation training drops 375 rows whose gold evidence falls outside the
16,384-token window, and every surviving truncated row keeps the contract TAIL
because truncation is left-side. Windowed inference then feeds the model heads
and middles of long contracts -- 23.8% of calls in experiment B -- a prompt
shape it never saw in training.

Windowing the TRAINING data fixes both: evidence is inside the window by
construction, and train/inference prompt geometry match.

The labelling rule
------------------
A window is labelled by WHAT IS IN IT, not by what the contract contains:

  present = true  + ONLY the gold spans grounded in THIS window
  present = false + []          if no gold span is in this window

Trimming the target to the visible spans is what makes every target
supportable. It also means:
  - the 156 rows whose spans straddle a boundary are no longer unplaceable;
    their spans are split across the windows that hold them
  - ZERO rows are dropped for truncation, because truncation cannot orphan a
    span any more

This matches inference exactly: experiment B asks each window "is the clause in
THIS window" and ORs grounded votes over windows. Per-window labelling is the
training-side mirror of that question.

Cost: windowing creates new negatives (windows of a contract that HAS the
clause, but not in this window). That is a correct label and is what teaches
scanning, but it lowers the present rate. Reported, not hidden.

Dev and test are NOT windowed. They stay whole so evaluation remains
comparable to runs A and B; windowing is applied at inference by
v2/cuad_windowed.py.
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

SEED = 20260919


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--src", default="v2/cuad/train.jsonl")
    ap.add_argument("--out", default="v2/cuad_windowed/train.jsonl")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--content-tokens", type=int, default=14000)
    ap.add_argument("--stride", type=int, default=12000)
    args = ap.parse_args()
    budget = args.max_model_len - args.max_new_tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)

    rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
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

    out, c = [], Counter()
    win_hist = Counter()
    spans_in, spans_out = 0, 0
    for cname, rs in bycon.items():
        t = contract_text(rs[0])
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        wins = windows_for(t, enc["offset_mapping"], len(enc["input_ids"]),
                           budget, overhead, args.content_tokens, args.stride)
        win_hist[len(wins)] += 1
        wtext = [t[w["char_start"]:w["char_end"]] for w in wins]
        wnorm = [norm(x) for x in wtext]
        for r in rs:
            ev = r.get("evidence") or []
            spans_in += len(ev)
            for wi, w in enumerate(wins):
                vis = [g for g in ev if _grounded(g, wnorm[wi])]
                spans_out += len(vis)
                present = bool(vis)
                c["present" if present else "absent"] += 1
                if ev and not present:
                    c["negative_from_present_contract"] += 1
                if ev and vis and len(vis) < len(ev):
                    c["window_with_PARTIAL_spans"] += 1
                out.append({
                    "contract": cname, "category": r["category"], "split": "train",
                    "present": present, "evidence": vis,
                    "window": wi, "n_windows": len(wins),
                    "windowed": bool(w["windowed"]),
                    "win_char_start": w["char_start"], "win_char_end": w["char_end"],
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER_TMPL.format(
                            text=wtext[wi], category=r["category"])},
                        {"role": "assistant", "content": json.dumps(
                            {"present": present, "evidence": vis},
                            ensure_ascii=False)},
                    ]})

    # Every target must be supportable by its own prompt. Assert it.
    bad = 0
    for r in out:
        if not r["evidence"]:
            continue
        hay = norm([m for m in r["messages"] if m["role"] == "user"][0]["content"])
        if not all(_grounded(g, hay) for g in r["evidence"]):
            bad += 1
    assert bad == 0, f"{bad} windowed rows cite text outside their own prompt"

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(out)
    print(f"windows per contract: {dict(sorted(win_hist.items()))}")
    print(f"\nwindowed CUAD train rows : {n}")
    print(f"  present                : {c['present']} ({c['present']/n:.1%})")
    print(f"  absent                 : {c['absent']} ({c['absent']/n:.1%})")
    print(f"  of absent, from a contract that HAS the clause elsewhere: "
          f"{c['negative_from_present_contract']}")
    print(f"  windows holding only SOME of a row's spans (target trimmed): "
          f"{c['window_with_PARTIAL_spans']}")
    print(f"\ngold spans in source {spans_in}, placed into windows {spans_out} "
          f"({spans_out/spans_in:.2f}x -- >1 because overlap duplicates some)")
    print(f"rows dropped for truncation: 0  (asserted: {bad} unsupportable targets)")
    json.dump({"seed": SEED, "src": args.src, "budget": budget,
               "content_tokens": args.content_tokens, "stride": args.stride,
               "rows": n, "counts": dict(c),
               "spans_in": spans_in, "spans_placed": spans_out,
               "windows_per_contract": {str(k): v for k, v in win_hist.items()},
               "rows_dropped": 0,
               "labelling_rule": "per-window; target trimmed to spans grounded "
                                 "in that window; mirrors the OR-over-grounded-"
                                 "votes aggregation used at inference"},
              open(outp.parent / "_manifest.json", "w"), indent=2)
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
