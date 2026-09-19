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


# Gold spans of <=4 normalized chars are section numbers ("2.1", "4.3", "8.2").
# They are trivially substring-matchable: any output containing "2.1" anywhere
# covers them. They are real annotations so they are NOT filtered from the
# headline metric -- that would invent a convention. Instead every evidence
# metric is reported twice, with and without them, so the inflation is a
# measured number rather than a caveat.
TRIVIAL_SPAN_CHARS = 4


def _strip_trivial(gold):
    return [g for g in gold if len(norm(g)) > TRIVIAL_SPAN_CHARS]


def evidence_metrics(rows, exclude_trivial=False):
    """Strict ContractEval-style F1/F2 on example-level TP/FN/FP, plus a
    partial-credit variant and Jaccard. TN (gold empty, pred empty) is excluded
    from F1/F2 by construction - it is correct abstention, not a retrieval hit."""
    c = Counter()
    fracs, jacs = [], []
    false_abstain = 0
    n_excluded = 0
    for r in rows:
        gold, pred = r["gold_evidence"], r["parsed_evidence"]
        if exclude_trivial and gold:
            gold = _strip_trivial(gold)
            if not gold:
                # every gold span was trivial; the example has no meaningful
                # retrieval target left. Drop it rather than reclassify it as
                # gold-empty, which would wrongly turn a TP/FN into an FP/TN.
                n_excluded += 1
                continue
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
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "n_excluded": n_excluded,
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
    out.update({f"evNT_{k}": val for k, val in
                evidence_metrics(rows, exclude_trivial=True).items()})
    # SIGNED, and it genuinely goes both ways: a trivial span inflates when the
    # model happens to emit "2.1" anywhere, and deflates when the model quotes
    # the clause text but omits its section number, losing the whole example to
    # FN under strict all-spans-covered matching.
    out["ev_f1_trivial_delta"] = out["ev_f1_strict"] - out["evNT_f1_strict"]
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


# ---- soft evidence metrics ------------------------------------------------
# Strict coverage (every gold span a contiguous substring) scores a near-miss
# identically to retrieving unrelated text. The FN overlap analysis showed
# median token overlap of 0.55-0.65 with p90 ~0.97, so a large share of "failures"
# are boundary errors, not retrieval errors. ROUGE-L and token-overlap P/R/F1
# quantify that; they are REPORTED ALONGSIDE strict, never instead of it, since
# strict is what compares to ContractEval.
def _lcs(a, b):
    """Length of the longest common subsequence of two token lists."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(gold_spans, pred_spans, beta=1.0):
    """ROUGE-L P/R/F over concatenated, normalized evidence."""
    g = norm(" ".join(gold_spans)).split()
    p = norm(" ".join(pred_spans)).split()
    if not g and not p:
        return None
    if not g or not p:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    l = _lcs(p, g)
    P = l / len(p)
    R = l / len(g)
    if P + R == 0:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    b2 = beta * beta
    return {"p": P, "r": R, "f": (1 + b2) * P * R / (b2 * P + R)}


def token_overlap(gold_spans, pred_spans):
    """Unordered token-set P/R/F1 -- ROUGE-L's order-insensitive companion."""
    g = set(norm(" ".join(gold_spans)).split())
    p = set(norm(" ".join(pred_spans)).split())
    if not g and not p:
        return None
    if not g or not p:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    i = len(g & p)
    P, R = i / len(p), i / len(g)
    return {"p": P, "r": R, "f": 2 * P * R / (P + R) if P + R else 0.0}


def soft_evidence_metrics(rows, gold_bearing_only=True):
    """Mean ROUGE-L and token-overlap over examples. Restricted by default to
    rows WITH gold evidence: on gold-empty rows the correct behaviour is to
    return nothing, which these metrics cannot express."""
    rl, to = [], []
    for r in rows:
        g, p = r["gold_evidence"], r["parsed_evidence"]
        if gold_bearing_only and not g:
            continue
        a = rouge_l(g, p)
        b = token_overlap(g, p)
        if a:
            rl.append(a)
        if b:
            to.append(b)
    if not rl:
        return {}
    m = lambda xs, k: sum(x[k] for x in xs) / len(xs)
    return {"rougeL_p": m(rl, "p"), "rougeL_r": m(rl, "r"), "rougeL_f": m(rl, "f"),
            "tokovl_p": m(to, "p"), "tokovl_r": m(to, "r"), "tokovl_f": m(to, "f"),
            "n_soft": len(rl)}


def hallucination_metrics(rows, contracts):
    """Fabricated-citation rate: predicted evidence text that does NOT occur in
    the source contract.

    This is distinct from a wrong citation. A model can quote the WRONG clause
    (grounded but unhelpful) or INVENT text (ungrounded). For a legal reviewer
    the second is far worse: a fabricated quote is indistinguishable from a real
    one without checking the source, which is the work the tool is meant to save.

    Matching uses the same doc_harness.norm as everything else, with edge
    punctuation stripped -- a model adding a closing period to a quoted heading
    is not fabricating.
    """
    n_span = n_hall = 0
    ex_total = ex_hall = 0
    for r in rows:
        pred = r["parsed_evidence"]
        if not pred:
            continue
        hay = norm(contracts.get(r["doc_id"], ""))
        ex_total += 1
        bad = 0
        for s in pred:
            n_span += 1
            t = norm(s)
            if t and t not in hay and t.strip(".,;:!?'\"-—’ ") not in hay:
                n_hall += 1; bad += 1
        ex_hall += (bad > 0)
    return {"span_hallucination_rate": n_hall / n_span if n_span else 0.0,
            "example_hallucination_rate": ex_hall / ex_total if ex_total else 0.0,
            "n_spans": n_span, "n_hallucinated": n_hall,
            "n_examples_with_evidence": ex_total}
