#!/usr/bin/env python3
"""Assemble v2/EVIDENCE_EVALUATION.md from the evidence_eval JSON outputs."""
import json
from pathlib import Path

E = "v2/evidence_eval"
RUNS = [("q4_3task", "Qwen3-4B"), ("q8_3task", "Qwen3-8B"),
        ("llama_3task", "Llama-3.1-8B"), ("q14_3task", "Qwen3-14B"),
        ("llama_3task_TEST", "Llama-3.1-8B (TEST)")]
D = {t: json.load(open(f"{E}/evidence__{t}.json")) for t, _ in RUNS}
L = []; W = L.append

W("# Evidence evaluation — v2, Tasks 1 and 2\n")
W("Rescored **offline** from archived v2 predictions. No inference was re-run, "
  "no model retrained, no v1 numbers or v1 data used. v1 was read for metric "
  "DEFINITIONS only.\n")
W("**Task 3 (synthetic risk notes) is OUT OF SCOPE and is not scored here.** "
  "Risk notes have no annotated evidence spans — the target is free prose from "
  "a teacher model — so a span-based evidence metric is undefined for them. "
  "This is stated rather than silently omitted.\n")

W("\n## 1. How the strict metric is actually computed (verified in code)\n")
W("Source: `v1_contractnli/doc_harness.py::covered_flags` / `evidence_case`, "
  "reimplemented identically in `v2/evidence_eval.py` so the diagnostics run "
  "over the same text.\n")
W("```\nhay   = norm(\" \".join(predicted_spans))\ncover = [norm(g) in hay for g in gold_spans]\n"
  "TP : gold non-empty AND all(cover)\nFN : gold non-empty AND not all(cover)\n"
  "FP : gold EMPTY     AND predicted non-empty\nTN : gold EMPTY     AND predicted empty\n"
  "F1 = 2TP / (2TP + FP + FN)\n```")
W("`norm(s) = \" \".join(s.lower().split())`. Containment is a **contiguous "
  "substring** test on normalized text.\n")
W("**Multiple / discontinuous gold spans.** All must be covered; one miss "
  "fails the whole example. The predicted spans are concatenated *before* the "
  "test, so a gold span may be satisfied by text that straddles what the model "
  "emitted as two separate spans. This is the only place where prediction "
  "structure is ignored. It is example-level, not span-level, and not "
  "token-level.\n")
W("Gold-span structure is not an edge case: **46.2% of CUAD present rows carry "
  "more than one gold span** (up to 17), and ContractNLI's 614 gold-bearing dev "
  "rows carry 1,227 spans.\n")

W("\n## 2. Correction to a previously reported number\n")
W("The CUAD evidence figure reported earlier was computed **only over "
  "gold-present rows**. Gold-absent rows never entered the confusion matrix, so "
  "a false positive was structurally impossible and precision came out as "
  "exactly 1.000. A precision of 1.000 should have been treated as a defect "
  "signal and was not.\n")
b = D["q4_3task"]["task2_cuad"]
W("| CUAD evidence, Qwen3-4B dev | precision | recall | F1 |")
W("|---|---|---|---|")
W("| previously reported (present rows only) | 1.000 | 0.391 | 0.562 |")
W(f"| **corrected (all rows, FP counted)** | **{b['strict_precision']:.3f}** | "
  f"{b['strict_recall']:.3f} | **{b['strict_f1']:.3f}** |")
W("\nAll strict figures below are the corrected full computation.\n")

W("\n## 3. Span matching rule — GREEDY ONE-TO-ONE\n")
W("Stated explicitly because the choice changes the number.\n")
W("- Jaccard is on token **sets**: `J = |set(g) & set(p)| / |set(g) | set(p)|`")
W("- all (gold, predicted) pairs scored, sorted by J descending, accepted while "
  "`J >= 0.5` and **neither** span is already matched")
W("- **one predicted span may satisfy at most one gold span**")
W("- **multiple predicted spans may NOT be combined to satisfy one gold span**")
W("- matching is one-to-one, not many-to-many; same rule for both tasks\n")
W("Rationale: a many-to-many rule lets one over-long predicted span 'locate' "
  "every gold span at once, rewarding exactly the over-extraction identified "
  "earlier in this project as the commercial failure mode. The cost is that "
  "one-to-one is **conservative** for a model that splits one clause into two "
  "predicted spans — that counts as one located gold span, not two.\n")
W("The 0.5 threshold is fixed and was not tuned.\n")

W("\n## 4. Token-level F1 — a deliberate departure from v1\n")
W("v1's `token_overlap` uses **set** intersection, discarding repetition. "
  "Standard token-level F1 (SQuAD, Rajpurkar et al. 2016) uses **multiset** "
  "intersection. Legal clauses repeat common tokens heavily, so the two differ "
  "materially. v2 uses multiset as primary and **also reports the set-based "
  "variant**, labelled, so v1 and v2 figures appearing in the same deck are "
  "comparable like with like.\n")
W("Computed over the union of spans per example, on gold-bearing rows only: on "
  "a gold-empty row the correct output is an empty list and token overlap is "
  "undefined, so including them would turn correct abstentions into free "
  "points.\n")

for task, lab in (("task1_contractnli", "Task 1 — ContractNLI"),
                  ("task2_cuad", "Task 2 — CUAD")):
    W(f"\n## {lab}\n")
    W("| model | **strict F1** | token F1 | token P | token R | span recall @J0.5 | "
      "set-based tok F1 (v1 style) |")
    W("|---|---|---|---|---|---|---|")
    for t, nm in RUNS:
        x = D[t][task]
        W(f"| {nm} | **{x['strict_f1']:.3f}** | {x['token_f1_multiset']:.3f} | "
          f"{x['token_precision']:.3f} | {x['token_recall']:.3f} | "
          f"{x['span_recall_j50']:.3f} | {x['token_f1_setbased_v1style']:.3f} |")
    W("\n| model | examples | gold-bearing | strict TP | strict FN | strict FP | "
      "strict TN | gold spans | located |")
    W("|---|---|---|---|---|---|---|---|---|")
    for t, nm in RUNS:
        x = D[t][task]
        W(f"| {nm} | {x['n_examples']} | {x['n_gold_bearing']} | {x['strict_TP']} | "
          f"{x['strict_FN']} | {x['strict_FP']} | {x['strict_TN']} | "
          f"{x['gold_spans_total']} | {x['gold_spans_located']} |")
    W("\n### Strict-failure decomposition\n")
    W("`boundary_only` = every gold span located at J>=0.5 but containment "
      "failed (right region, wrong edges). `partial` = some located. "
      "`not_found` = none located.\n")
    W("| model | strict failures | boundary only | partial | not found |")
    W("|---|---|---|---|---|")
    for t, nm in RUNS:
        d = D[t][task]["strict_failure_decomposition"]
        tot = sum(d.values())
        W(f"| {nm} | {tot} | {d.get('boundary_only',0)} | {d.get('partial',0)} | "
          f"{d.get('not_found',0)} |")

pc = D["q4_3task"]["task2_cuad"]["present_class"]
W("\n## Task 2 — Present/Absent classification\n")
W("Qwen3-4B, dev. The constant-absent baseline is printed beside accuracy "
  "every time accuracy appears.\n")
W("| metric | value |")
W("|---|---|")
W(f"| PRESENT precision | {pc['precision']:.3f} |")
W(f"| PRESENT recall | {pc['recall']:.3f} |")
W(f"| PRESENT F1 | {pc['f1']:.3f} |")
W(f"| accuracy | {pc['accuracy']:.3f} |")
W(f"| **constant-absent baseline (measured)** | **{pc['CONSTANT_ABSENT_BASELINE_measured']:.3f}** |")
W(f"| present rate | {pc['present_rate']:.3f} |")
W(f"| TP / FP / FN / TN | {pc['tp']} / {pc['fp']} / {pc['fn']} / {pc['tn']} |")
W(f"\n**The baseline is {pc['CONSTANT_ABSENT_BASELINE_measured']:.1%} on this "
  f"dev split, not 65.5%.** 65.5% corresponds to a 34.5% present rate; our dev "
  f"split is {pc['present_rate']:.1%} present. The measured value is used "
  f"throughout. (Test split measures 65.2%.)\n")

cl = D["q4_3task"]["task2_cuad"]["truncation_ceiling_overall"]
W("\n## Truncation ceiling\n")
W("16K truncation places some gold spans outside the visible window, capping "
  "achievable evidence recall through no fault of the model.\n")
W("| level | visible | total | ceiling |")
W("|---|---|---|---|")
W(f"| span-level | {cl['spans_visible']} | {cl['spans_total']} | **{cl['span_level']:.2%}** |")
W(f"| row-level | {cl['rows_with_evidence']-cl['rows_with_an_invisible_span']} | "
  f"{cl['rows_with_evidence']} | **{cl['row_level']:.2%}** |")
W("\nThe ~89.9% figure is the **row-level** ceiling. The span-level ceiling is "
  "87.65%, and span-level is the right companion to span recall.\n")

W("\n## Task 2 — per category, with n and ceiling\n")
W("Small cells must not be over-read; n is given for every one.\n")
W("| category | n | n present | gold spans | strict F1 | token F1 | span recall | "
  "span ceiling | spans invisible |")
W("|---|---|---|---|---|---|---|---|---|")
for c, v in sorted(D["q4_3task"]["task2_cuad"]["per_category"].items(),
                   key=lambda kv: -kv[1]["strict_f1"]):
    ceil = v.get("truncation_ceiling_span_level")
    W(f"| {c} | {v['n']} | {v['n_present']} | {v['gold_spans']} | "
      f"{v['strict_f1']:.3f} | {v['token_f1']:.3f} | {v['span_recall_j50']:.3f} | "
      + (f"{ceil:.1%}" if ceil is not None else "n/a")
      + f" | {v.get('spans_invisible','n/a')} |")

W("\n## Task 2 — stratified by contract token length\n")
W("The ceiling is a length effect, so the evidence metrics are broken out the "
  "same way.\n")
W("| bucket | examples | gold-bearing | gold spans | strict F1 | token F1 | span recall |")
W("|---|---|---|---|---|---|---|")
for k in ("<16K", "16-32K", ">32K"):
    v = D["q4_3task"]["task2_cuad"].get("by_contract_length", {}).get(k)
    if not v:
        continue
    W(f"| {k} | {v['n_examples']} | {v['n_gold_bearing']} | {v['gold_spans_total']} | "
      f"{v['strict_f1']:.3f} | {v['token_f1_multiset']:.3f} | {v['span_recall_j50']:.3f} |")

W("\n## Claims of comparability to published work\n")
W("**No parity claim is made, for either task.**\n")
W("- The Jaccard 0.5 threshold was supplied as CUAD's published value. The "
  "CUAD dataset paper is Hendrycks et al., *CUAD: An Expert-Annotated NLP "
  "Dataset for Legal Contract Review* (NeurIPS 2021 Datasets and Benchmarks). "
  "**I could not verify the threshold or its exact definition against the "
  "paper from this offline cluster.** Independently of that, CUAD's headline "
  "metrics are AUPR and Precision at 80% Recall over a span-extraction "
  "formulation, which is **not** what is computed here, so our numbers are not "
  "comparable to CUAD's published results regardless of the threshold.")
W("- An earlier v1 report described the strict metric as "
  "'ContractEval-comparable'. **That claim is unverified** — the exact "
  "ContractEval definition was not checked against its source in this run. It "
  "should be softened to 'strict containment, defined here' in any slide until "
  "verified.\n")
W("If a slide needs a comparability claim, the honest form is: *these are "
  "internally consistent metrics computed on our own contract-level splits; "
  "they are not directly comparable to published leaderboard numbers.*\n")

Path("v2/EVIDENCE_EVALUATION.md").write_text("\n".join(L), encoding="utf-8")
print(f"-> v2/EVIDENCE_EVALUATION.md ({len(L)} lines)")
