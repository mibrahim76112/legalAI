#!/usr/bin/env python3
"""A's evidence visibility, at SPAN level, reconstructed exactly.

A left-truncated each prompt to the last `budget` tokens of the rendered chat
template, so the visible contract is a SUFFIX. This reproduces that cut token
for token and asks, for every gold span, whether it survived.

Reported at span level so it is comparable to B's span-level visibility.
Row-level counts are reported too, since a row fails if ANY of its spans is cut.
"""
import sys, json, argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_norm import norm                                     # noqa: E402
from reasoning_filters import _grounded                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", default="v2/combined/dev.jsonl")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", default="v2/_a_visibility.json")
    args = ap.parse_args()
    budget = args.max_model_len - args.max_new_tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    rows = [r for r in rows if r.get("task") == "cuad"]
    c = Counter()
    invisible_keys, by_cat = [], defaultdict(Counter)
    for r in rows:
        ev = r.get("evidence") or []
        if not ev:
            continue
        msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        enc = tok.apply_chat_template(msgs, tokenize=True,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
        if hasattr(enc, "input_ids"):
            enc = enc["input_ids"]
        if enc and isinstance(enc[0], list):
            enc = enc[0]
        enc = list(enc)
        visible_text = tok.decode(enc[-budget:]) if len(enc) > budget else None
        hay = norm(visible_text) if visible_text is not None else None
        c["rows_with_evidence"] += 1
        row_ok = True
        for g in ev:
            c["spans_total"] += 1
            ok = True if hay is None else _grounded(g, hay)
            if ok:
                c["spans_visible"] += 1
                by_cat[r["category"]]["visible"] += 1
            else:
                c["spans_invisible"] += 1
                by_cat[r["category"]]["invisible"] += 1
                row_ok = False
        if row_ok:
            c["rows_fully_visible"] += 1
        else:
            c["rows_with_invisible_span"] += 1
            invisible_keys.append([r["contract"], r["category"]])

    print(f"A visibility at budget {budget} (suffix kept by left-truncation)")
    for k in ("rows_with_evidence", "rows_fully_visible", "rows_with_invisible_span",
              "spans_total", "spans_visible", "spans_invisible"):
        print(f"  {k:28s} {c[k]}")
    print(f"  span visibility             {c['spans_visible']/c['spans_total']:.4%}")
    json.dump({**dict(c),
               "budget": budget,
               "span_visibility": c["spans_visible"] / c["spans_total"],
               "invisible_keys": invisible_keys,
               "by_category": {k: dict(v) for k, v in by_cat.items()}},
              open(args.out, "w"), indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
