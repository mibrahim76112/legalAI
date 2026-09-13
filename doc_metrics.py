#!/usr/bin/env python3
"""Metrics for the document-level screen. Every number recomputable from the
raw JSONL with no GPU time."""
import json, random
from collections import Counter
from doc_harness import CLASSES, evidence_case, jaccard, norm

UNPARSED = "__unparsed__"


def verdict_metrics(pred, gold):
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


def evidence_metrics(rows):
    """Strict ContractEval-style F1/F2 on example-level TP/FN/FP, plus a
    partial-credit variant and Jaccard. TN (gold empty, pred empty) is excluded
    from F1/F2 by construction - it is correct abstention, not a retrieval hit."""
    c = Counter()
    fracs, jacs = [], []
    false_abstain = 0
    for r in rows:
        gold, pred = r["gold_evidence"], r["parsed_evidence"]
        case, frac = evidence_case(gold, pred)
        c[case] += 1
        if gold:
            fracs.append(frac)
            if not pred:
                false_abstain += 1
        j = jaccard(gold, pred)
        if j is not None:
            jacs.append(j)
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * P * R / (P + R) if P + R else 0.0
    f2 = 5 * tp / (5 * tp + 4 * fn + fp) if (5 * tp + 4 * fn + fp) else 0.0
    n_gold = tp + fn
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": P, "recall": R, "f1_strict": f1, "f2_strict": f2,
            "partial_credit": sum(fracs) / len(fracs) if fracs else 0.0,
            "jaccard": sum(jacs) / len(jacs) if jacs else 0.0,
            "false_abstention_rate": false_abstain / n_gold if n_gold else 0.0}


def joint_metrics(rows):
    """Verdict correct AND all gold spans covered.

    Reported two ways because the brief's definition is vacuously true when
    gold evidence is empty:
      lenient - literal reading; empty gold counts as 'all covered'
      strict  - additionally requires the model NOT to invent evidence when
                gold is empty (i.e. no evidence false positive)
    """
    lenient = strict = 0
    for r in rows:
        ok_v = (r["parsed_verdict"] == r["gold_verdict"])
        case, frac = evidence_case(r["gold_evidence"], r["parsed_evidence"])
        covered = (frac == 1.0)
        if ok_v and covered:
            lenient += 1
            if not (not r["gold_evidence"] and r["parsed_evidence"]):
                strict += 1
    n = len(rows)
    return {"joint_lenient": lenient / n, "joint_strict": strict / n}


def format_metrics(rows):
    n = len(rows)
    js = Counter(r["json_state"] for r in rows)
    return {"n": n,
            "not_json": js.get("not_json", 0) / n,
            "json_wrong_schema": js.get("json_wrong_schema", 0) / n,
            "schema_valid": sum(1 for r in rows if r["schema_valid"]) / n,
            "parse_fail": sum(1 for r in rows if r["parsed_verdict"] is None) / n,
            "truncated": sum(1 for r in rows if r.get("finish_reason") == "length") / n,
            "parse_modes": dict(Counter(r["parse_mode"] for r in rows))}


def cell_metrics(rows):
    gold = [r["gold_verdict"] for r in rows]
    pred = [r["parsed_verdict"] or UNPARSED for r in rows]
    v = verdict_metrics(pred, gold)
    out = {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"]}
    for c in CLASSES:
        for k in ("precision", "recall", "f1", "support"):
            out[f"{c}_{k}"] = v[c][k]
    out.update({f"ev_{k}": val for k, val in evidence_metrics(rows).items()})
    out.update(joint_metrics(rows))
    out.update({f"fmt_{k}": val for k, val in format_metrics(rows).items()
                if k != "parse_modes"})
    tk = sorted(r["n_output_tokens"] for r in rows)
    out["mean_output_tokens"] = sum(tk) / len(tk)
    out["p99_output_tokens"] = tk[int(.99 * len(tk))]
    return out


def paired_boot(a_pred, b_pred, gold, n=2000, seed=0):
    rng = random.Random(seed)
    N = len(gold)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        g = [gold[i] for i in idx]
        diffs.append(verdict_metrics([a_pred[i] for i in idx], g)["macro_f1"]
                     - verdict_metrics([b_pred[i] for i in idx], g)["macro_f1"])
    diffs.sort()
    return {"mean_diff": sum(diffs) / n, "ci_low": diffs[int(.025 * n)],
            "ci_high": diffs[int(.975 * n)],
            "p_a_better": sum(1 for d in diffs if d > 0) / n}


def boot_ci(pred, gold, n=2000, seed=0):
    rng = random.Random(seed)
    N = len(gold)
    v = sorted(verdict_metrics(*zip(*[(pred[i], gold[i])
               for i in [rng.randrange(N) for _ in range(N)]]))["macro_f1"]
               for _ in range(n))
    return v[int(.025 * n)], v[int(.975 * n)]
