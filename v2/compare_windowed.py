#!/usr/bin/env python3
"""Side-by-side: truncated vs windowed CUAD inference, same checkpoint.

Answers one question: how much of the truncation loss is recoverable at
INFERENCE alone, with no retraining. That decides whether CUAD gets rebuilt
around windows or whether the current training data stands.
"""
import sys, json, argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from text_norm import norm                                    # noqa: E402
from reasoning_filters import _grounded                       # noqa: E402

FOCUS = ["Non-Transferable License", "Ip Ownership Assignment",
         "Exclusivity", "Non-Compete"]


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def bucket(n):
    return "<16K" if n < 16000 else "16-32K" if n < 32000 else ">32K"


def covered(gold, pred):
    """Every gold span present in the concatenated prediction -- the same
    strict rule the truncated run is scored with."""
    if not gold:
        return None
    hay = norm(" ".join(pred))
    return all(_grounded(g, hay) for g in gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, help="windows__TAG.jsonl")
    ap.add_argument("--truncated", required=True, help="raw__TAG.jsonl")
    ap.add_argument("--meta", required=True, help="windowmeta__TAG.json")
    ap.add_argument("--ceiling", default="v2/_eval_ceiling.json")
    ap.add_argument("--out", default="v2/WINDOWED_CUAD.md")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    clen = meta["contract_tokens"]
    wrows = [json.loads(l) for l in open(args.windows, encoding="utf-8")]

    # ---- windowed aggregation: OR over windows, grounded votes only ----
    agg = {}
    for w in wrows:
        k = (w["contract"], w["category"])
        a = agg.setdefault(k, {"gold": w["present_gold"],
                               "gold_evidence": w["gold_evidence"],
                               "pred": False, "evidence": [],
                               "n_windows": 0, "n_votes": 0,
                               "n_ungrounded_claims": 0})
        a["n_windows"] += 1
        if w["pred_present"] and not w["grounded_evidence"]:
            a["n_ungrounded_claims"] += 1
        if w["vote_present"]:
            a["n_votes"] += 1
            a["pred"] = True
            a["evidence"].extend(w["grounded_evidence"])

    # ---- truncated rows, restricted to the same (contract, category) pairs ----
    trows = [json.loads(l) for l in open(args.truncated, encoding="utf-8")]
    trows = [r for r in trows if r.get("task") == "cuad"]
    from doc_harness import strip_wrappers, _find_json

    def tparse(raw):
        t = strip_wrappers(raw)
        if not t:
            return None, []
        d = _find_json(t)
        if not isinstance(d, dict) or "present" not in d:
            return None, []
        ev = d.get("evidence") or []
        return bool(d["present"]), [str(x) for x in ev] if isinstance(ev, list) else []

    trunc = {}
    for r in trows:
        k = (r["contract"], r["category"])
        if k not in agg:
            continue                       # subset run -- compare like with like
        p, ev = tparse(r["raw"])
        trunc[k] = {"gold": bool(r["present"]), "gold_evidence": r["gold_evidence"],
                    "pred": bool(p), "evidence": ev, "unparseable": p is None}

    keys = sorted(set(agg) & set(trunc))
    print(f"comparing {len(keys)} (contract, category) pairs")

    def score(d, subset=None):
        c = Counter(); ev_tp = ev_fn = 0
        for k in (subset if subset is not None else keys):
            x = d[k]
            g, p = x["gold"], x["pred"]
            c["tp" if (g and p) else "fp" if (p and not g) else
              "fn" if (g and not p) else "tn"] += 1
            if g and x["gold_evidence"]:
                if covered(x["gold_evidence"], x["evidence"]):
                    ev_tp += 1
                else:
                    ev_fn += 1
        f, p_, r_ = f1(c["tp"], c["fp"], c["fn"])
        return {"n": sum(c.values()), "f1": f, "precision": p_, "recall": r_,
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
                "evidence_recall": ev_tp / (ev_tp + ev_fn) if ev_tp + ev_fn else 0.0,
                "evidence_n": ev_tp + ev_fn}

    L = []; A = L.append
    A("# Windowed vs truncated CUAD inference\n")
    A("Same checkpoints, no retraining. An **inference-time** change only.\n")
    A(f"Windows of {meta['content_tokens']:,} content tokens, stride "
      f"{meta['stride']:,} ({meta['content_tokens']-meta['stride']:,} overlap), "
      f"covering each contract in full.\n")
    A("**Aggregation rule, fixed in advance and not tuned.** A window's "
      "`present: true` counts only if the span it emitted occurs verbatim in "
      "that window. A window claiming presence with an ungrounded span is "
      "hallucinating, not detecting, and that is precisely the false-positive "
      "class that OR-ing over N windows would amplify. Contract-level verdict "
      "is present if **any** window casts a grounded vote.\n")
    A(f"Cost: {meta['n_calls']:,} generation calls against "
      f"{meta['n_calls_truncated_equivalent']:,} truncated "
      f"({meta['n_calls']/meta['n_calls_truncated_equivalent']:.2f}x), "
      f"{meta['wall_seconds']/60:.1f} min wall clock, "
      f"{meta['sec_per_call']:.3f}s per call.\n")
    A(f"Windows per contract: {meta['windows_per_contract']}\n")

    tw, tt = score(agg), score(trunc)
    A("\n## 1-2. Present-class detection and evidence recall\n")
    A("| strategy | F1 | precision | recall | TP | FP | FN | evidence recall |")
    A("|---|---|---|---|---|---|---|---|")
    for nm, s in (("truncated", tt), ("windowed", tw)):
        A(f"| {nm} | {s['f1']:.3f} | {s['precision']:.3f} | {s['recall']:.3f} | "
          f"{s['tp']} | {s['fp']} | {s['fn']} | {s['evidence_recall']:.3f} "
          f"(n={s['evidence_n']}) |")
    A(f"\nDelta: F1 {tw['f1']-tt['f1']:+.3f}, precision "
      f"{tw['precision']-tt['precision']:+.3f}, recall {tw['recall']-tt['recall']:+.3f}, "
      f"evidence recall {tw['evidence_recall']-tt['evidence_recall']:+.3f}.\n")

    A("\n## 3. Stratified by contract length\n")
    A("The window only binds on long contracts, so a pooled number hides the "
      "effect. This is where it should appear or not at all.\n")
    A("| bucket | contracts | pairs | trunc F1 | wind F1 | trunc ev-recall | wind ev-recall |")
    A("|---|---|---|---|---|---|---|")
    for b in ("<16K", "16-32K", ">32K"):
        sub = [k for k in keys if bucket(clen[k[0]]) == b]
        if not sub:
            continue
        a_, t_ = score(agg, sub), score(trunc, sub)
        ncon = len({k[0] for k in sub})
        A(f"| {b} | {ncon} | {len(sub)} | {t_['f1']:.3f} | {a_['f1']:.3f} | "
          f"{t_['evidence_recall']:.3f} | {a_['evidence_recall']:.3f} |")

    A("\n## 4. Achievable ceiling\n")
    inv_w = sum(1 for k in keys
                if agg[k]["gold"] and agg[k]["gold_evidence"]
                and not any(_grounded(g, norm(" ".join(agg[k]["evidence"])))
                            for g in agg[k]["gold_evidence"])
                and agg[k]["n_votes"] == 0)
    try:
        ceil = json.load(open(args.ceiling))
        bt = ceil["by_task"]["cuad"]
        A(f"- **Truncated**: {bt['evidence_invisible']} of "
          f"{bt['rows_with_evidence']} evidence-bearing dev rows have their gold "
          f"span outside the kept window, a ceiling of "
          f"{1-bt['evidence_invisible']/bt['rows_with_evidence']:.1%}.")
    except Exception:
        A("- Truncated ceiling: see `_eval_ceiling.json`.")
    A("- **Windowed**: every token of every contract appears in some window, so "
      "the ceiling is **100%** by construction. Whether the measured recall "
      "approaches it is a separate question answered by the table above.\n")

    A("\n## 5. The double-penalised categories\n")
    A("These lost the most training rows to the 16,384 cut AND have the lowest "
      "truncated eval ceilings, because both effects track contract length.\n")
    A("| category | pairs | trunc F1 | wind F1 | trunc recall | wind recall | "
      "trunc ev-rec | wind ev-rec |")
    A("|---|---|---|---|---|---|---|---|")
    for cat in FOCUS:
        sub = [k for k in keys if k[1] == cat]
        if not sub:
            continue
        a_, t_ = score(agg, sub), score(trunc, sub)
        A(f"| {cat} | {len(sub)} | {t_['f1']:.3f} | {a_['f1']:.3f} | "
          f"{t_['recall']:.3f} | {a_['recall']:.3f} | "
          f"{t_['evidence_recall']:.3f} | {a_['evidence_recall']:.3f} |")

    A("\n## What the aggregation rule rejected\n")
    ung = sum(a["n_ungrounded_claims"] for a in agg.values())
    multi = Counter(a["n_votes"] for a in agg.values())
    A(f"- {ung:,} window-level `present: true` claims were discarded because the "
      f"span they cited was not in the window they were shown. Without the "
      f"groundedness requirement each of these would have become a "
      f"contract-level false positive.")
    A(f"- Votes per pair: {dict(sorted(multi.items()))}. Pairs with more than one "
      f"vote are clauses found in overlapping windows, which is expected.\n")

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    json.dump({"truncated": tt, "windowed": tw,
               "ungrounded_window_claims": ung,
               "votes_per_pair": {str(k): v for k, v in multi.items()}},
              open(str(Path(args.out).with_suffix(".json")), "w"), indent=2)
    print("\n".join(L[:40]))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
