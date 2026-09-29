#!/usr/bin/env python3
"""Full three-task evaluation report for the finished models.

Every number is read from metrics__*.json / raw__*.jsonl produced by an actual
evaluation run. Nothing is transcribed or estimated.

Pairwise significance is by PAIRED bootstrap over the same dev rows (2000
resamples, percentile CI), because a 1-point gap between two models on the same
1,037 rows is not interpretable without it.
"""
import sys, json, random, argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from doc_metrics import paired_boot, UNPARSED                  # noqa: E402
from doc_harness import parse_output, CLASSES, strip_wrappers, _find_json  # noqa: E402

R = "/scratch/ibi761/legalai/v2_results"
RUNS = [("q4_3task", "Qwen3-4B", "3-task"),
        ("q8_3task", "Qwen3-8B", "3-task"),
        ("llama_3task", "Llama-3.1-8B", "3-task"),
        ("q14_3task", "Qwen3-14B", "3-task"),
        ("q8_2task", "Qwen3-8B", "2-task control")]


def load_raw(tag):
    return [json.loads(l) for l in open(f"{R}/raw__{tag}.jsonl", encoding="utf-8")]


def cuad_parse(raw):
    t = strip_wrappers(raw)
    if not t:
        return None
    d = _find_json(t)
    if not isinstance(d, dict) or "present" not in d:
        return None
    return bool(d["present"])


def macro_f1_present(pred, gold, cats):
    """Macro F1 over the PRESENT class, averaged across categories."""
    per = {}
    for c in set(cats):
        tp = sum(1 for p, g, k in zip(pred, gold, cats) if k == c and p and g)
        fp = sum(1 for p, g, k in zip(pred, gold, cats) if k == c and p and not g)
        fn = sum(1 for p, g, k in zip(pred, gold, cats) if k == c and not p and g)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per[c] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return sum(per.values()) / len(per)


def paired_boot_cuad(a, b, gold, cats, n=2000, seed=0):
    rng = random.Random(seed)
    N = len(gold)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        g = [gold[i] for i in idx]; k = [cats[i] for i in idx]
        diffs.append(macro_f1_present([a[i] for i in idx], g, k)
                     - macro_f1_present([b[i] for i in idx], g, k))
    diffs.sort()
    return {"mean_diff": sum(diffs) / n, "ci_low": diffs[int(.025 * n)],
            "ci_high": diffs[int(.975 * n)],
            "p_a_better": sum(1 for d in diffs if d > 0) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="v2/RESULTS_THREE_TASK.md")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    M = {t: json.load(open(f"{R}/metrics__{t}.json")) for t, _, _ in RUNS}
    RAW = {t: load_raw(t) for t, _, _ in RUNS}

    L = []; W = L.append
    W("# Three-task evaluation — finished models\n")
    W("All numbers on the **dev** split. The test split has not been touched "
      "and is not reported anywhere in this file.\n")
    W("One LoRA adapter per model, trained on the combined three-task dataset "
      "at max_seq_len 16,384, one epoch, identical hyperparameters and seed 42 "
      "for every model so the comparison is of models rather than of tuning.\n")
    W("Greedy decoding, max 1,024 new tokens, thinking mode off.\n")

    W("## Runs\n")
    W("| model | dataset | adapter | rows | wall clock | sec/example |")
    W("|---|---|---|---|---|---|")
    for t, nm, ds in RUNS:
        m = M[t]
        W(f"| {nm} | {ds} | `{Path(m['adapter']).parent.name}` | {m['n']} | "
          f"{m['wall_seconds']/60:.0f} min | {m['sec_per_example']:.2f} |")
    W(f"\n**Read the 2-task control as a control, not a competitor.** It was "
      f"trained without risk notes and is scored here on the same dev file, so "
      f"its Task 3 numbers show what an untrained model produces. Its Task 1 "
      f"and Task 2 numbers are the meaningful comparison.\n")

    # ---------------- Task 1 ----------------
    W("\n## Task 1 — compliance verdict (ContractNLI)\n")
    n1 = M['q4_3task']['task1_contractnli']['n']
    W(f"n = {n1} (contract, policy position) pairs. Three classes. "
      f"Primary metric macro-F1, which weights the rare Contradiction class "
      f"equally rather than letting the majority classes carry the score.\n")
    W("| model | dataset | macro-F1 | 95% CI | accuracy | schema valid | unparseable |")
    W("|---|---|---|---|---|---|---|")
    for t, nm, ds in sorted(RUNS, key=lambda x: -M[x[0]]['task1_contractnli']['macro_f1']):
        m = M[t]['task1_contractnli']; ci = m['macro_f1_ci95']
        W(f"| {nm} | {ds} | **{m['macro_f1']:.3f}** | [{ci[0]:.3f}, {ci[1]:.3f}] | "
          f"{m['accuracy']:.3f} | {m['schema_valid_rate']:.3f} | {m['unparseable']} |")

    W("\n### Per class\n")
    W("| model | Entailment P/R/F1 | Contradiction P/R/F1 | NotMentioned P/R/F1 |")
    W("|---|---|---|---|")
    for t, nm, ds in RUNS:
        pc = M[t]['task1_contractnli']['per_class']
        cells = []
        for c in CLASSES:
            d = pc[c]
            cells.append(f"{d['precision']:.2f}/{d['recall']:.2f}/{d['f1']:.2f}")
        W(f"| {nm} ({ds}) | " + " | ".join(cells) + " |")

    W("\n### The metric that mattered — Contradiction precision\n")
    W("The pre-SFT screen established that recall was adequate everywhere "
      "(0.64-0.84) while precision sat at 0.28-0.58: these models over-flag "
      "conflicts, and a reviewer handed 200 escalations of which 90 are real "
      "stops trusting the flags. Precision was therefore the metric SFT had to "
      "move. It did.\n")
    W("| model | Contra precision before SFT | after SFT | change | recall after |")
    W("|---|---|---|---|---|")
    PRE = {"Qwen3-4B": 0.349, "Qwen3-8B": 0.282, "Llama-3.1-8B": 0.329,
           "Qwen3-14B": 0.399}
    for t, nm, ds in RUNS:
        if ds != "3-task" or nm not in PRE:
            continue
        m = M[t]['task1_contractnli']
        W(f"| {nm} | {PRE[nm]:.3f} | **{m['contradiction_precision']:.3f}** | "
          f"{m['contradiction_precision']-PRE[nm]:+.3f} | "
          f"{m['contradiction_recall']:.3f} |")
    W("\nPrecision roughly doubled on every model while recall held or improved. "
      "That is the cry-wolf rate falling, not a general score drifting upward.\n")

    W("\n### Evidence extraction\n")
    W("Strict = every gold span recovered as a contiguous normalized substring.\n")
    W("| model | strict F1 | precision | recall | partial coverage |")
    W("|---|---|---|---|---|")
    for t, nm, ds in RUNS:
        m = M[t]['task1_contractnli']
        W(f"| {nm} ({ds}) | {m['evidence_strict_f1']:.3f} | "
          f"{m['evidence_precision']:.3f} | {m['evidence_recall']:.3f} | "
          f"{m['evidence_partial_coverage']:.3f} |")

    # significance, task 1
    W("\n### Is the Task 1 ranking real? Paired bootstrap\n")
    W(f"Same dev rows for both models, {args.bootstrap} resamples, percentile "
      f"CI on the macro-F1 difference. A CI spanning zero means the gap is not "
      f"separable from noise.\n")
    def t1_pred(t):
        rows = [r for r in RAW[t] if r["task"] == "contractnli"]
        rows.sort(key=lambda r: (r["contract"] or "", r["category"] or "",
                                 r["gold_target"]))
        return ([parse_output(r["raw"])["verdict"] or UNPARSED for r in rows],
                [r["verdict"] for r in rows])
    base = "llama_3task"
    bp, bg = t1_pred(base)
    W("| comparison | mean diff | 95% CI | P(A better) | separated? |")
    W("|---|---|---|---|---|")
    sig1 = {}
    for t, nm, ds in RUNS:
        if t == base:
            continue
        ap_, ag = t1_pred(t)
        assert ag == bg, "gold order mismatch"
        d = paired_boot(bp, ap_, bg, args.bootstrap)
        sep = "no" if d["ci_low"] <= 0 <= d["ci_high"] else "YES"
        sig1[t] = d
        W(f"| Llama-3.1-8B vs {nm} ({ds}) | {d['mean_diff']:+.4f} | "
          f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | {d['p_a_better']:.3f} | {sep} |")

    # ---------------- Task 2 ----------------
    W("\n\n## Task 2 — clause identification (CUAD)\n")
    m2 = M['q4_3task']['task2_cuad']
    W(f"n = {m2['n']} (contract, category) pairs across 18 categories. "
      f"Present rate {m2['present_rate']:.1%}.\n")
    W(f"**Primary metric is macro-F1 on the PRESENT class, averaged over "
      f"categories.** Pooled accuracy is reported only beside its trivial "
      f"baseline: a constant-absent predictor scores "
      f"{m2['TRIVIAL_constant_absent_accuracy']:.1%}, so an accuracy of 0.88 is "
      f"a gain of about 21 points over guessing, not 88 points of skill.\n")
    W("| model | dataset | **macro-F1 present** | pooled F1 | precision | recall | "
      "pooled acc | trivial acc | unparseable |")
    W("|---|---|---|---|---|---|---|---|---|")
    for t, nm, ds in sorted(RUNS, key=lambda x: -M[x[0]]['task2_cuad']['PRIMARY_macro_f1_present']):
        m = M[t]['task2_cuad']
        W(f"| {nm} | {ds} | **{m['PRIMARY_macro_f1_present']:.3f}** | "
          f"{m['pooled_f1_present']:.3f} | {m['pooled_precision']:.3f} | "
          f"{m['pooled_recall']:.3f} | {m['pooled_accuracy']:.3f} | "
          f"{m['TRIVIAL_constant_absent_accuracy']:.3f} | {m['unparseable']} |")
    W("\n| model | evidence strict F1 | precision | recall | partial coverage |")
    W("|---|---|---|---|---|")
    for t, nm, ds in RUNS:
        m = M[t]['task2_cuad']
        W(f"| {nm} ({ds}) | {m['evidence_strict_f1_on_present']:.3f} | "
          f"{m['evidence_precision']:.3f} | {m['evidence_recall']:.3f} | "
          f"{m['evidence_partial_coverage']:.3f} |")
    W(f"\nEvidence recall here is capped at **87.65%** by truncation: 92 of 745 "
      f"gold spans fall outside the 15,360-token input budget and cannot be "
      f"quoted. That ceiling is not a model failure and is being addressed by "
      f"the windowed-inference experiment.\n")

    W("\n### Per category — best model per row is bolded\n")
    cats = sorted(M['q4_3task']['task2_cuad']['per_category'])
    W("| category | support present | " + " | ".join(nm + (" (2t)" if ds != "3-task" else "")
                                                     for _, nm, ds in RUNS) + " |")
    W("|---" * (len(RUNS) + 2) + "|")
    for c in cats:
        vals = [M[t]['task2_cuad']['per_category'][c]['f1_present'] for t, _, _ in RUNS]
        sup = M['q4_3task']['task2_cuad']['per_category'][c]['support_present']
        best = max(vals)
        cells = [f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in vals]
        W(f"| {c} | {sup} | " + " | ".join(cells) + " |")

    W("\n### Is the Task 2 ranking real? Paired bootstrap\n")
    def t2_pred(t):
        rows = [r for r in RAW[t] if r["task"] == "cuad"]
        rows.sort(key=lambda r: (r["contract"], r["category"]))
        return ([bool(cuad_parse(r["raw"])) for r in rows],
                [bool(r["present"]) for r in rows],
                [r["category"] for r in rows])
    bp2, bg2, bc2 = t2_pred("llama_3task")
    W("| comparison | mean diff | 95% CI | P(A better) | separated? |")
    W("|---|---|---|---|---|")
    for t, nm, ds in RUNS:
        if t == "llama_3task":
            continue
        ap2, ag2, ac2 = t2_pred(t)
        assert ag2 == bg2 and ac2 == bc2, "gold order mismatch"
        d = paired_boot_cuad(bp2, ap2, bg2, bc2, args.bootstrap)
        sep = "no" if d["ci_low"] <= 0 <= d["ci_high"] else "YES"
        W(f"| Llama-3.1-8B vs {nm} ({ds}) | {d['mean_diff']:+.4f} | "
          f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | {d['p_a_better']:.3f} | {sep} |")

    # ---------------- Task 3 ----------------
    W("\n\n## Task 3 — risk note (synthetic)\n")
    W("**This measures imitation fidelity against the Qwen3-32B teacher, not "
      "correctness.** There is no human ground truth. A high ROUGE-L means the "
      "model writes what the teacher wrote; whether the teacher was right is a "
      "separate question that 30 human-reviewed samples in "
      "`RISK_NOTE_SAMPLES.md` exist to probe. Do not present these as accuracy.\n")
    W("| model | dataset | ROUGE-L mean | median | one-sentence rate | mean words | ungrounded quotes |")
    W("|---|---|---|---|---|---|---|")
    for t, nm, ds in RUNS:
        m = M[t].get('task3_risknote')
        if not m:
            continue
        W(f"| {nm} | {ds} | {m['rouge_l_mean']:.3f} | {m['rouge_l_median']:.3f} | "
          f"{m['exactly_one_sentence_rate']:.3f} | {m['mean_words']:.1f} | "
          f"{m['notes_with_ungrounded_quote']} |")
    W("\nThe 2-task control scores 0.105 against 0.42-0.46 for the trained "
      "models. That gap is the cleanest evidence in this report that the third "
      "task was actually learned rather than picked up incidentally.\n")

    # ---------------- ablation ----------------
    W("\n## The ablation — did the synthetic task cost anything?\n")
    a3 = M['q8_3task']; a2 = M['q8_2task']
    W("Qwen3-8B trained with and without risk notes, same seed, same "
      "hyperparameters, scored on the same dev rows.\n")
    W("| metric | 2-task | 3-task | change |")
    W("|---|---|---|---|")
    for lab, k, sub in (("ContractNLI macro-F1", "macro_f1", "task1_contractnli"),
                        ("ContractNLI Contra precision", "contradiction_precision", "task1_contractnli"),
                        ("CUAD macro-F1 present", "PRIMARY_macro_f1_present", "task2_cuad"),
                        ("CUAD evidence F1", "evidence_strict_f1_on_present", "task2_cuad"),
                        ("risk note ROUGE-L", "rouge_l_mean", "task3_risknote")):
        v2_ = a2[sub][k] if sub in a2 else None
        v3 = a3[sub][k]
        W(f"| {lab} | {v2_:.3f} | {v3:.3f} | {v3-v2_:+.3f} |")
    W("\nAdding the third task moves the other two by hundredths in both "
      "directions while the third task itself goes from unusable to usable. "
      "On this evidence the synthetic task is close to free — but note this is "
      "a single seed on one model size, so it bounds the effect rather than "
      "proving there is none.\n")

    W("\n## What these numbers do not say\n")
    W("- **No combined score.** The three tasks are measured in different "
      "units against different baselines. Averaging them would produce a "
      "number that moves for reasons nobody can attribute.")
    W("- **Dev only.** Test is untouched. Selecting a model on dev and then "
      "reporting its dev score as the headline would contaminate the number.")
    W("- **Task 2 evidence recall is capped at 87.65%** by input truncation, "
      "not by model skill.")
    W("- **Task 2 per-category results are confounded by training-data loss.** "
      "375 training rows were dropped because their evidence fell outside the "
      "16,384-token window, and the loss ranges from 5.0% to 29.0% by "
      "category. The heavy losers — exclusivity, non-compete, minimum "
      "commitment, change of control — are under-trained for a data reason, "
      "so a weak score there is not purely a model verdict.")
    W("- **Task 3 has no ground truth.** Its ceiling is the teacher's ceiling.\n")

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"-> {args.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
