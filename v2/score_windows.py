#!/usr/bin/env python3
"""Score any windows__*.jsonl with the fixed OR-over-grounded-votes rule.

Lets two windowed runs from DIFFERENT checkpoints be compared directly, which
compare_windowed.py cannot do (it compares windowed vs truncated for one
checkpoint).
"""
import sys, json, argparse
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from text_norm import norm                                     # noqa: E402
from reasoning_filters import _grounded                        # noqa: E402


def f1(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    return (2*p*r/(p+r) if p+r else 0.0), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--lengths",
                    default="/scratch/ibi761/legalai/v2_results/cost__q4_3task_win.json")
    args = ap.parse_args()
    clen = json.load(open(args.lengths))["contract_tokens"]

    def bucket(n):
        return "<16K" if n < 16000 else "16-32K" if n < 32000 else ">32K"

    res = {}
    for wf, lab in zip(args.windows, args.labels):
        agg = {}
        ung = 0
        for w in (json.loads(l) for l in open(wf, encoding="utf-8")):
            k = (w["contract"], w["category"])
            a = agg.setdefault(k, {"gold": w["gold_present"],
                                   "gold_ev": w["gold_evidence"],
                                   "pred": False, "ev": [], "votes": 0})
            ung += w["n_ungrounded"]
            if w["vote_present"]:
                a["votes"] += 1
                a["pred"] = True
                a["ev"].extend(s["text"] for s in w["grounded_spans"])
        res[lab] = (agg, ung)

    keys = sorted(set.intersection(*[set(a) for a, _ in res.values()]))
    print(f"comparing {len(keys)} (contract, category) pairs\n")

    def score(agg, sub):
        c = Counter(); ok = no = 0
        for k in sub:
            x = agg[k]
            g, p = x["gold"], x["pred"]
            c["tp" if (g and p) else "fp" if (p and not g) else
              "fn" if (g and not p) else "tn"] += 1
            if g and x["gold_ev"]:
                hay = norm(" ".join(x["ev"]))
                (ok := ok + 1) if all(_grounded(e, hay) for e in x["gold_ev"]) else (no := no + 1)
        f, p_, r_ = f1(c["tp"], c["fp"], c["fn"])
        return {"f1": f, "p": p_, "r": r_, "evr": ok/(ok+no) if ok+no else 0.0,
                "n": sum(c.values()), "evn": ok+no, **c}

    print(f"{'run':26s} {'presF1':>7s} {'presP':>6s} {'presR':>6s} {'evRec':>6s} "
          f"{'TP':>4s} {'FP':>4s} {'FN':>4s} {'ungrounded':>11s}")
    for lab, (agg, ung) in res.items():
        s = score(agg, keys)
        print(f"{lab:26s} {s['f1']:7.4f} {s['p']:6.3f} {s['r']:6.3f} {s['evr']:6.3f} "
              f"{s['tp']:4d} {s['fp']:4d} {s['fn']:4d} {ung:11d}")
    print(f"\nby contract length:")
    for b in ("<16K", "16-32K", ">32K"):
        sub = [k for k in keys if bucket(clen[k[0]]) == b]
        if not sub:
            continue
        print(f"  {b:8s} n={len(sub):4d} contracts={len({k[0] for k in sub}):3d}", end="")
        for lab, (agg, _) in res.items():
            s = score(agg, sub)
            print(f"   {lab}: F1 {s['f1']:.3f} R {s['r']:.3f} evR {s['evr']:.3f}", end="")
        print()


if __name__ == "__main__":
    main()
