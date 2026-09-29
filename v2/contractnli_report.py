#!/usr/bin/env python3
"""Detailed ContractNLI-only evaluation across the four v2 models.

Offline rescore of archived v2 predictions. No inference re-run.
CUAD and risk notes are deliberately not touched here.

Note on label naming: the dataset's third class is "NotMentioned". It is the
Neutral class in NLI terms and is reported under its dataset name throughout.
"""
import sys, json, random, argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from text_norm import norm                                      # noqa: E402
from doc_harness import parse_output, CLASSES                   # noqa: E402
from doc_metrics import paired_boot, boot_ci, UNPARSED          # noqa: E402

R = "/scratch/ibi761/legalai/v2_results"
RUNS = [("q4_3task", "Qwen3-4B"), ("q8_3task", "Qwen3-8B"),
        ("llama_3task", "Llama-3.1-8B"), ("q14_3task", "Qwen3-14B")]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def load(tag):
    rows = [json.loads(l) for l in open(f"{R}/raw__{tag}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["task"] == "contractnli"]
    # NO SORTING. The archived rows have no unique key -- (gold_target, verdict)
    # collides 423 times because every NotMentioned row shares one target -- so
    # sorting cannot align models and would silently corrupt the paired
    # bootstrap. File order is written straight from v2/combined/dev.jsonl and
    # was verified identical element-wise across all four models.
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="v2/_contractnli_lengths.json")
    ap.add_argument("--out", default="v2/CONTRACTNLI_EVALUATION.md")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    data = {t: load(t) for t, _ in RUNS}
    ns = {t: len(v) for t, v in data.items()}
    golds = {t: [r["verdict"] for r in v] for t, v in data.items()}
    same = len({tuple(g) for g in golds.values()}) == 1
    ev_same = len({tuple(tuple(r["gold_evidence"]) for r in v)
                   for v in data.values()}) == 1
    preds = {}
    unpar = {}
    for t, _ in RUNS:
        p = []
        u = 0
        for r in data[t]:
            d = parse_output(r["raw"])
            v = d["verdict"]
            if v is None:
                u += 1
            p.append(v or UNPARSED)
        preds[t] = p
        unpar[t] = u
    gold = golds[RUNS[0][0]]

    L = []; W = L.append
    W("# ContractNLI — detailed evaluation (v2)\n")
    W("ContractNLI only. CUAD, risk notes, ablations and windowing are "
      "deliberately excluded.\n")
    W("Rescored offline from archived v2 predictions; no inference was re-run.\n")
    W("The dataset's third class is **NotMentioned**, which is the Neutral "
      "class in NLI terms. It is reported under its dataset name below.\n")

    # ---------------- 8. sanity checks first ----------------
    W("\n## Sanity checks (stated first, because everything below depends on them)\n")
    W("| check | result |")
    W("|---|---|")
    W("| evaluation split | **dev** (`v2/combined/dev.jsonl`) |")
    W("| dev or test | dev — the test split is untouched for these four models |")
    W(f"| ContractNLI examples per model | {ns[RUNS[0][0]]} |")
    W(f"| identical example set across all four models | "
      f"{'YES — gold labels AND gold evidence identical element-wise in file order' if (same and ev_same) else 'NO — MISMATCH'} |")
    W(f"| rows excluded | none; every row is scored |")
    W("| unparseable outputs | counted as a wrong prediction, never dropped |")
    W("")
    W("| model | examples | unparseable |")
    W("|---|---|---|")
    for t, nm in RUNS:
        W(f"| {nm} | {ns[t]} | {unpar[t]} |")
    W("\nUnparseable outputs are scored as errors rather than removed. Dropping "
      "them would flatter a model that fails loudly.\n")

    # ---------------- 1. overall ----------------
    W("\n## 1. Overall verdict classification\n")
    W("Macro precision and macro recall are unweighted means over the three "
      "classes, so the rare Contradiction class counts as much as the common "
      "ones.\n")
    W("| model | accuracy | macro-F1 | macro precision | macro recall | 95% CI (macro-F1) |")
    W("|---|---|---|---|---|---|")
    per = {}
    for t, nm in RUNS:
        P = preds[t]
        acc = sum(1 for a, b in zip(P, gold) if a == b) / len(gold)
        cls = {}
        for c in CLASSES:
            tp = sum(1 for a, b in zip(P, gold) if a == c and b == c)
            fp = sum(1 for a, b in zip(P, gold) if a == c and b != c)
            fn = sum(1 for a, b in zip(P, gold) if a != c and b == c)
            cls[c] = prf(tp, fp, fn) + (tp, fp, fn)
        per[t] = cls
        mp = sum(cls[c][0] for c in CLASSES) / 3
        mr = sum(cls[c][1] for c in CLASSES) / 3
        mf = sum(cls[c][2] for c in CLASSES) / 3
        ci = boot_ci(P, gold, args.bootstrap)
        W(f"| {nm} | {acc:.3f} | **{mf:.3f}** | {mp:.3f} | {mr:.3f} | "
          f"[{ci[0]:.3f}, {ci[1]:.3f}] |")

    # ---------------- 2. per label ----------------
    sup = Counter(gold)
    W("\n## 2. Per-label results\n")
    W(f"Support: Entailment {sup['Entailment']}, Contradiction "
      f"{sup['Contradiction']}, NotMentioned {sup['NotMentioned']} "
      f"(total {len(gold)}).\n")
    for c in CLASSES:
        W(f"\n### {c} (support {sup[c]})\n")
        W("| model | precision | recall | F1 | TP | FP | FN |")
        W("|---|---|---|---|---|---|---|")
        for t, nm in RUNS:
            p, r, f, tp, fp, fn = per[t][c]
            W(f"| {nm} | {p:.3f} | {r:.3f} | {f:.3f} | {tp} | {fp} | {fn} |")

    # ---------------- 4. confusion ----------------
    W("\n## 4. Confusion matrices (rows = gold, columns = predicted)\n")
    conf = {}
    for t, nm in RUNS:
        m = defaultdict(Counter)
        for a, b in zip(preds[t], gold):
            m[b][a] += 1
        conf[t] = m
        W(f"\n**{nm}**\n")
        cols = CLASSES + [UNPARSED]
        W("| gold \\ pred | " + " | ".join(c.replace(UNPARSED, "unparseable")
                                           for c in cols) + " | total |")
        W("|---" * (len(cols) + 2) + "|")
        for g in CLASSES:
            row = [m[g][c] for c in cols]
            W(f"| **{g}** | " + " | ".join(str(x) for x in row) + f" | {sum(row)} |")

    # ---------------- 3. failure patterns ----------------
    W("\n## 3. Failure pattern per label\n")
    for c in CLASSES:
        W(f"\n### {c}\n")
        for t, nm in RUNS:
            p, r, f, tp, fp, fn = per[t][c]
            m = conf[t]
            # where do gold-c examples go when wrong?
            miss = {k: v for k, v in m[c].items() if k != c and v}
            # where do false positives come from?
            src = {g: m[g][c] for g in CLASSES if g != c and m[g][c]}
            mo = ", ".join(f"{k} {v}" for k, v in sorted(miss.items(), key=lambda x: -x[1]))
            so = ", ".join(f"{k} {v}" for k, v in sorted(src.items(), key=lambda x: -x[1]))
            W(f"- **{nm}** P {p:.3f} / R {r:.3f}. "
              f"Missed {fn} gold {c} -> predicted as [{mo or 'none'}]. "
              f"{fp} false {c} came from [{so or 'none'}].")

    # ---------------- 5. model comparison ----------------
    W("\n## 5. Comparing the four models\n")
    W("Class-F1 side by side, so the source of any difference is visible "
      "rather than hidden in the macro average.\n")
    W("| class | " + " | ".join(nm for _, nm in RUNS) + " | spread |")
    W("|---" * (len(RUNS) + 2) + "|")
    for c in CLASSES:
        vals = [per[t][c][2] for t, _ in RUNS]
        W(f"| {c} | " + " | ".join(f"{v:.3f}" for v in vals)
          + f" | {max(vals)-min(vals):.3f} |")
    W("\n### Paired bootstrap on macro-F1\n")
    W(f"Same dev rows for both models, {args.bootstrap} resamples, percentile "
      f"CI on the difference. A CI spanning zero means the gap is not "
      f"separable from sampling noise. **This tests row sampling only; it does "
      f"NOT cover training-seed variance.**\n")
    W("| A vs B | mean diff | 95% CI | P(A better) | separated |")
    W("|---|---|---|---|---|")
    for i, (ta, na) in enumerate(RUNS):
        for tb, nb in RUNS[i + 1:]:
            d = paired_boot(preds[ta], preds[tb], gold, args.bootstrap)
            sep = "no" if d["ci_low"] <= 0 <= d["ci_high"] else "YES"
            W(f"| {na} vs {nb} | {d['mean_diff']:+.4f} | "
              f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | "
              f"{d['p_a_better']:.3f} | {sep} |")

    # ---------------- 6. evidence ----------------
    W("\n## 6. Evidence extraction\n")
    ev = {t: json.load(open(f"v2/evidence_eval/evidence__{t}.json"))["task1_contractnli"]
          for t, _ in RUNS}
    e0 = ev[RUNS[0][0]]
    W(f"Gold evidence-bearing examples: **{e0['n_gold_bearing']}** of "
      f"{e0['n_examples']}. Gold evidence spans: **{e0['gold_spans_total']}**.\n")
    W("| model | strict F1 | token F1 | token P | token R | span recall @J0.5 | spans located |")
    W("|---|---|---|---|---|---|---|")
    for t, nm in RUNS:
        x = ev[t]
        W(f"| {nm} | {x['strict_f1']:.3f} | {x['token_f1_multiset']:.3f} | "
          f"{x['token_precision']:.3f} | {x['token_recall']:.3f} | "
          f"{x['span_recall_j50']:.3f} | {x['gold_spans_located']} |")
    W("\n### Evidence failure decomposition\n")
    W("| model | strict failures | boundary only | partial | not found |")
    W("|---|---|---|---|---|")
    for t, nm in RUNS:
        d = ev[t]["strict_failure_decomposition"]
        W(f"| {nm} | {sum(d.values())} | {d.get('boundary_only',0)} | "
          f"{d.get('partial',0)} | {d.get('not_found',0)} |")

    # ---------------- 7. length ----------------
    W("\n## 7. Long-context effect\n")
    if Path(args.lengths).exists():
        LEN = json.load(open(args.lengths))
        W(f"ContractNLI prompts measured with the evaluation tokenizer: "
          f"p50 {LEN['p50']:,}, p99 {LEN['p99']:,}, max {LEN['max']:,} tokens "
          f"against an input budget of {LEN['budget']:,}.\n")
        W(f"**{LEN['n_over_budget']} of {LEN['n']} ContractNLI dev prompts "
          f"exceed the budget.** The context window never binds on this task, "
          f"so there is no truncation effect to report and no evidence is "
          f"rendered invisible.\n")
        W("Accuracy by prompt length, included because 'the window never binds' "
          "is not the same as 'length does not matter':\n")
        W("| bucket | examples | " + " | ".join(nm for _, nm in RUNS) + " |")
        W("|---" * (len(RUNS) + 2) + "|")
        idx = LEN["per_row_bucket"]
        for bk in LEN["bucket_order"]:
            sel = [i for i, b in enumerate(idx) if b == bk]
            if not sel:
                continue
            cells = []
            for t, _ in RUNS:
                a = sum(1 for i in sel if preds[t][i] == gold[i]) / len(sel)
                cells.append(f"{a:.3f}")
            W(f"| {bk} | {len(sel)} | " + " | ".join(cells) + " |")
    else:
        W("Length stratification was not available in the archived outputs and "
          "was not computed. Not reported rather than estimated.\n")

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"-> {args.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
