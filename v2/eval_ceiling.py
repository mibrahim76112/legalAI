#!/usr/bin/env python3
"""How much gold evidence is invisible at EVAL time?

At training, an evidence-truncated row is poison and was dropped. At eval,
nothing is dropped -- a long contract is a legitimate hard case. But if the
gold span falls outside the prompt budget, the model is being asked to quote
text it cannot see. Those rows put a CEILING on achievable evidence recall
that is not a model failure, and the ceiling has to be reported next to the
score or the score is misread.

Budget here is max_model_len - max_new_tokens = 15360, matching eval exactly,
not the 16384 used at training.
"""
import sys, json, argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from text_norm import norm                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v2/combined/dev.jsonl")
    ap.add_argument("--budget", type=int, default=15360)
    ap.add_argument("--out", default="v2/_eval_ceiling.json")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    stat = defaultdict(Counter)
    lost_by_cat = Counter()
    ev_by_cat = Counter()
    for r in rows:
        t = r["task"]
        ev = r.get("evidence") or []
        stat[t]["rows"] += 1
        if ev:
            stat[t]["rows_with_evidence"] += 1
            if t == "cuad":
                ev_by_cat[r["category"]] += 1
        msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        enc = tok.apply_chat_template(msgs, tokenize=True,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
        if hasattr(enc, "input_ids"):
            enc = enc["input_ids"]
        if enc and isinstance(enc[0], list):
            enc = enc[0]
        enc = list(enc)
        if len(enc) <= args.budget:
            continue
        stat[t]["truncated"] += 1
        if not ev:
            continue
        hay = norm(tok.decode(enc[-args.budget:]))
        if not all(norm(e) in hay for e in ev):
            stat[t]["evidence_invisible"] += 1
            if t == "cuad":
                lost_by_cat[r["category"]] += 1

    print(f"prompt budget {args.budget} tokens ({args.data})\n")
    print(f"{'task':14s} {'rows':>6s} {'w/ evid':>8s} {'trunc':>6s} "
          f"{'EVID INVISIBLE':>15s} {'% of w/ evid':>13s}")
    for t, c in sorted(stat.items()):
        we = c["rows_with_evidence"]
        print(f"{t:14s} {c['rows']:6d} {we:8d} {c['truncated']:6d} "
              f"{c['evidence_invisible']:15d} "
              f"{(c['evidence_invisible']/we if we else 0):12.1%}")
    if lost_by_cat:
        print(f"\nCUAD per-category recall ceiling:")
        print(f"{'category':34s} {'w/ evid':>8s} {'invisible':>10s} {'CEILING':>9s}")
        for cat in sorted(ev_by_cat, key=lambda c: lost_by_cat[c]/max(ev_by_cat[c],1),
                          reverse=True):
            n, l = ev_by_cat[cat], lost_by_cat[cat]
            print(f"{cat:34s} {n:8d} {l:10d} {1-l/max(n,1):8.1%}")
    json.dump({"budget": args.budget, "data": args.data,
               "by_task": {t: dict(c) for t, c in stat.items()},
               "cuad_evidence_rows_by_category": dict(ev_by_cat),
               "cuad_invisible_by_category": dict(lost_by_cat)},
              open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
