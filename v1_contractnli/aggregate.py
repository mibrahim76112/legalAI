#!/usr/bin/env python3
"""
Merge the per-cell raw outputs into the screen's results table.

Ranking metric   : macro-F1, 3-shot. Macro because Contradiction is ~9% of dev
                   and accuracy would just track the two majority classes.
Diagnostics      : per-class P/R/F1, JSON validity, parse-failure rate,
                   mean output tokens. Reported, never used to rank.
Noise            : bootstrap CI per cell + PAIRED bootstrap between models,
                   because with 89 Contradiction examples in dev a small
                   macro-F1 gap can easily be sampling noise.
"""
import json, glob, csv, argparse, random
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["Entailment", "Contradiction", "NotMentioned"]
UNPARSED = "__unparsed__"   # never equals a gold label, so it scores as wrong


def prf(pred, gold):
    """Per-class precision/recall/F1 + macro-F1."""
    out, f1s = {}, []
    for c in CLASSES:
        tp = sum(1 for p, g in zip(pred, gold) if p == c and g == c)
        fp = sum(1 for p, g in zip(pred, gold) if p == c and g != c)
        fn = sum(1 for p, g in zip(pred, gold) if p != c and g == c)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        out[c] = {"precision": P, "recall": R, "f1": F, "support": tp + fn}
        f1s.append(F)
    out["macro_f1"] = sum(f1s) / len(f1s)
    out["accuracy"] = sum(1 for p, g in zip(pred, gold) if p == g) / len(gold)
    return out


def boot_ci(pred, gold, n=2000, seed=0):
    rng = random.Random(seed)
    N = len(gold)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        vals.append(prf([pred[i] for i in idx], [gold[i] for i in idx])["macro_f1"])
    vals.sort()
    return vals[int(.025 * n)], vals[int(.975 * n)]


def paired_boot(a_pred, b_pred, gold, n=2000, seed=0):
    """P(model A macro-F1 > model B) on the SAME resampled examples.

    Paired because both models saw identical rows; pairing removes the
    example-difficulty variance that dominates an unpaired comparison.
    """
    rng = random.Random(seed)
    N = len(gold)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        g = [gold[i] for i in idx]
        diffs.append(prf([a_pred[i] for i in idx], g)["macro_f1"]
                     - prf([b_pred[i] for i in idx], g)["macro_f1"])
    diffs.sort()
    return {
        "mean_diff": sum(diffs) / n,
        "ci_low": diffs[int(.025 * n)],
        "ci_high": diffs[int(.975 * n)],
        "p_a_better": sum(1 for d in diffs if d > 0) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/scratch/ibi761/legalai/results")
    ap.add_argument("--out", default="results")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    cells = {}
    for fp in sorted(glob.glob(str(Path(args.dir) / "raw__*.jsonl"))):
        rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
        if not rows:
            continue
        # system prompt is part of the identity: spec and sft runs are
        # different measurements and must never merge into one cell.
        key = (rows[0]["model"], rows[0]["condition"],
               rows[0].get("system_prompt", "spec"))
        rows.sort(key=lambda r: r["example_id"])       # align across models
        cells[key] = rows

    if not cells:
        raise SystemExit(f"no raw__*.jsonl found in {args.dir}")

    outd = Path(args.out); outd.mkdir(exist_ok=True)
    table = []
    for (model, cond, system), rows in sorted(cells.items()):
        gold = [r["gold_verdict"] for r in rows]
        pred = [r["parsed_verdict"] or UNPARSED for r in rows]
        m = prf(pred, gold)
        lo, hi = boot_ci(pred, gold, args.bootstrap)
        n = len(rows)
        rec = {
            "model": model, "condition": cond, "system": system, "n": n,
            "macro_f1": m["macro_f1"], "macro_f1_lo": lo, "macro_f1_hi": hi,
            "accuracy": m["accuracy"],
            "json_valid_rate": sum(r["json_valid"] for r in rows) / n,
            "parse_fail_rate": sum(1 for r in rows if r["parsed_verdict"] is None) / n,
            "truncated_rate": sum(1 for r in rows if r.get("finish_reason") == "length") / n,
            "mean_output_tokens": sum(r["n_output_tokens"] for r in rows) / n,
        }
        for c in CLASSES:
            for k in ("precision", "recall", "f1"):
                rec[f"{c}_{k}"] = m[c][k]
            rec[f"{c}_support"] = m[c]["support"]
        table.append(rec)

    with open(outd / "results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    json.dump(table, open(outd / "results_table.json", "w"), indent=2)

    # ---- console report -------------------------------------------------
    print(f"{'model':28s}{'cond':11s}{'macroF1':>9}{'  95% CI':>18}"
          f"{'acc':>7}{'JSON':>7}{'fail':>7}{'tok':>7}")
    for r in sorted(table, key=lambda x: (-x["macro_f1"], x["model"])):
        print(f"{r['model']:28s}{r['condition']:11s}{r['macro_f1']:9.3f}"
              f"  [{r['macro_f1_lo']:.3f},{r['macro_f1_hi']:.3f}]"
              f"{r['accuracy']:7.3f}{r['json_valid_rate']:7.1%}"
              f"{r['parse_fail_rate']:7.1%}{r['mean_output_tokens']:7.1f}")

    print("\nPer-class (recall is the commercially important column):")
    print(f"{'model':28s}{'cond':11s}" + "".join(f"{c[:6]+' P/R/F1':>24}" for c in CLASSES))
    for r in sorted(table, key=lambda x: (x["model"], x["condition"])):
        s = f"{r['model']:28s}{r['condition']:11s}"
        for c in CLASSES:
            s += f"{r[c+'_precision']:7.3f}{r[c+'_recall']:8.3f}{r[c+'_f1']:9.3f}"
        print(s)

    # ---- ranking + is the gap real? -------------------------------------
    systems = {k[2] for k in cells}
    if len(systems) > 1:
        print(f"\nNOTE: {len(systems)} system prompts present {sorted(systems)}; "
              f"ranking is computed per prompt.")
    three = {m: rows for (m, c, sy), rows in cells.items() if c == "threeshot"}
    if len(three) >= 2:
        gold = [r["gold_verdict"] for r in next(iter(three.values()))]
        preds = {m: [r["parsed_verdict"] or UNPARSED for r in rows]
                 for m, rows in three.items()}
        order = sorted(preds, key=lambda m: -prf(preds[m], gold)["macro_f1"])
        print("\nRANKING (macro-F1, 3-shot):")
        for i, m in enumerate(order, 1):
            print(f"  {i}. {m:30s} {prf(preds[m], gold)['macro_f1']:.3f}")

        print("\nPaired bootstrap, adjacent pairs (is the gap real?):")
        cmps = []
        for a, b in zip(order, order[1:]):
            d = paired_boot(preds[a], preds[b], gold, args.bootstrap)
            verdict = ("separated" if d["ci_low"] > 0 else "WITHIN NOISE")
            print(f"  {a} > {b}: d={d['mean_diff']:+.3f} "
                  f"[{d['ci_low']:+.3f},{d['ci_high']:+.3f}] "
                  f"P(better)={d['p_a_better']:.2f}  -> {verdict}")
            cmps.append({"a": a, "b": b, **d, "verdict": verdict})
        json.dump({"ranking": order, "comparisons": cmps},
                  open(outd / "ranking.json", "w"), indent=2)

    print(f"\n-> {outd}/results_table.csv, results_table.json, ranking.json")


if __name__ == "__main__":
    main()
