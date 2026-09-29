# Evidence evaluation — v2, Tasks 1 and 2

Rescored **offline** from archived v2 predictions. No inference was re-run, no model retrained, no v1 numbers or v1 data used. v1 was read for metric DEFINITIONS only.

**Task 3 (synthetic risk notes) is OUT OF SCOPE and is not scored here.** Risk notes have no annotated evidence spans — the target is free prose from a teacher model — so a span-based evidence metric is undefined for them. This is stated rather than silently omitted.


## 1. How the strict metric is actually computed (verified in code)

Source: `v1_contractnli/doc_harness.py::covered_flags` / `evidence_case`, reimplemented identically in `v2/evidence_eval.py` so the diagnostics run over the same text.

```
hay   = norm(" ".join(predicted_spans))
cover = [norm(g) in hay for g in gold_spans]
TP : gold non-empty AND all(cover)
FN : gold non-empty AND not all(cover)
FP : gold EMPTY     AND predicted non-empty
TN : gold EMPTY     AND predicted empty
F1 = 2TP / (2TP + FP + FN)
```
`norm(s) = " ".join(s.lower().split())`. Containment is a **contiguous substring** test on normalized text.

**Multiple / discontinuous gold spans.** All must be covered; one miss fails the whole example. The predicted spans are concatenated *before* the test, so a gold span may be satisfied by text that straddles what the model emitted as two separate spans. This is the only place where prediction structure is ignored. It is example-level, not span-level, and not token-level.

Gold-span structure is not an edge case: **46.2% of CUAD present rows carry more than one gold span** (up to 17), and ContractNLI's 614 gold-bearing dev rows carry 1,227 spans.


## 2. Correction to a previously reported number

The CUAD evidence figure reported earlier was computed **only over gold-present rows**. Gold-absent rows never entered the confusion matrix, so a false positive was structurally impossible and precision came out as exactly 1.000. A precision of 1.000 should have been treated as a defect signal and was not.

| CUAD evidence, Qwen3-4B dev | precision | recall | F1 |
|---|---|---|---|
| previously reported (present rows only) | 1.000 | 0.391 | 0.562 |
| **corrected (all rows, FP counted)** | **0.794** | 0.391 | **0.524** |

All strict figures below are the corrected full computation.


## 3. Span matching rule — GREEDY ONE-TO-ONE

Stated explicitly because the choice changes the number.

- Jaccard is on token **sets**: `J = |set(g) & set(p)| / |set(g) | set(p)|`
- all (gold, predicted) pairs scored, sorted by J descending, accepted while `J >= 0.5` and **neither** span is already matched
- **one predicted span may satisfy at most one gold span**
- **multiple predicted spans may NOT be combined to satisfy one gold span**
- matching is one-to-one, not many-to-many; same rule for both tasks

Rationale: a many-to-many rule lets one over-long predicted span 'locate' every gold span at once, rewarding exactly the over-extraction identified earlier in this project as the commercial failure mode. The cost is that one-to-one is **conservative** for a model that splits one clause into two predicted spans — that counts as one located gold span, not two.

The 0.5 threshold is fixed and was not tuned.


## 4. Token-level F1 — a deliberate departure from v1

v1's `token_overlap` uses **set** intersection, discarding repetition. Standard token-level F1 (SQuAD, Rajpurkar et al. 2016) uses **multiset** intersection. Legal clauses repeat common tokens heavily, so the two differ materially. v2 uses multiset as primary and **also reports the set-based variant**, labelled, so v1 and v2 figures appearing in the same deck are comparable like with like.

Computed over the union of spans per example, on gold-bearing rows only: on a gold-empty row the correct output is an empty list and token overlap is undefined, so including them would turn correct abstentions into free points.


## Task 1 — ContractNLI

| model | **strict F1** | token F1 | token P | token R | span recall @J0.5 | set-based tok F1 (v1 style) |
|---|---|---|---|---|---|---|
| Qwen3-4B | **0.731** | 0.801 | 0.852 | 0.808 | 0.723 | 0.822 |
| Qwen3-8B | **0.739** | 0.804 | 0.848 | 0.818 | 0.727 | 0.826 |
| Llama-3.1-8B | **0.747** | 0.812 | 0.856 | 0.823 | 0.736 | 0.832 |
| Qwen3-14B | **0.763** | 0.825 | 0.864 | 0.838 | 0.750 | 0.843 |
| Llama-3.1-8B (TEST) | **0.779** | 0.837 | 0.873 | 0.842 | 0.763 | 0.853 |

| model | examples | gold-bearing | strict TP | strict FN | strict FP | strict TN | gold spans | located |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 1037 | 614 | 396 | 218 | 74 | 349 | 1227 | 887 |
| Qwen3-8B | 1037 | 614 | 402 | 212 | 72 | 351 | 1227 | 892 |
| Llama-3.1-8B | 1037 | 614 | 405 | 209 | 65 | 358 | 1227 | 903 |
| Qwen3-14B | 1037 | 614 | 419 | 195 | 66 | 357 | 1227 | 920 |
| Llama-3.1-8B (TEST) | 2091 | 1188 | 832 | 356 | 116 | 787 | 2404 | 1834 |

### Strict-failure decomposition

`boundary_only` = every gold span located at J>=0.5 but containment failed (right region, wrong edges). `partial` = some located. `not_found` = none located.

| model | strict failures | boundary only | partial | not found |
|---|---|---|---|---|
| Qwen3-4B | 218 | 18 | 140 | 60 |
| Qwen3-8B | 212 | 21 | 135 | 56 |
| Llama-3.1-8B | 209 | 17 | 135 | 57 |
| Qwen3-14B | 195 | 21 | 121 | 53 |
| Llama-3.1-8B (TEST) | 356 | 20 | 227 | 109 |

## Task 2 — CUAD

| model | **strict F1** | token F1 | token P | token R | span recall @J0.5 | set-based tok F1 (v1 style) |
|---|---|---|---|---|---|---|
| Qwen3-4B | **0.524** | 0.571 | 0.628 | 0.572 | 0.412 | 0.593 |
| Qwen3-8B | **0.555** | 0.575 | 0.635 | 0.579 | 0.428 | 0.599 |
| Llama-3.1-8B | **0.583** | 0.603 | 0.666 | 0.594 | 0.447 | 0.624 |
| Qwen3-14B | **0.568** | 0.590 | 0.652 | 0.590 | 0.444 | 0.614 |
| Llama-3.1-8B (TEST) | **0.517** | 0.547 | 0.627 | 0.532 | 0.416 | 0.570 |

| model | examples | gold-bearing | strict TP | strict FN | strict FP | strict TN | gold spans | located |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 1098 | 366 | 143 | 223 | 37 | 695 | 745 | 307 |
| Qwen3-8B | 1098 | 366 | 153 | 213 | 32 | 700 | 745 | 319 |
| Llama-3.1-8B | 1098 | 366 | 162 | 204 | 28 | 704 | 745 | 333 |
| Qwen3-14B | 1098 | 366 | 157 | 209 | 30 | 702 | 745 | 331 |
| Llama-3.1-8B (TEST) | 1836 | 639 | 240 | 399 | 49 | 1148 | 1356 | 564 |

### Strict-failure decomposition

`boundary_only` = every gold span located at J>=0.5 but containment failed (right region, wrong edges). `partial` = some located. `not_found` = none located.

| model | strict failures | boundary only | partial | not found |
|---|---|---|---|---|
| Qwen3-4B | 223 | 8 | 89 | 126 |
| Qwen3-8B | 213 | 11 | 81 | 121 |
| Llama-3.1-8B | 204 | 6 | 87 | 111 |
| Qwen3-14B | 209 | 10 | 85 | 114 |
| Llama-3.1-8B (TEST) | 399 | 15 | 156 | 228 |

## Task 2 — Present/Absent classification

Qwen3-4B, dev. The constant-absent baseline is printed beside accuracy every time accuracy appears.

| metric | value |
|---|---|
| PRESENT precision | 0.879 |
| PRESENT recall | 0.732 |
| PRESENT F1 | 0.799 |
| accuracy | 0.877 |
| **constant-absent baseline (measured)** | **0.667** |
| present rate | 0.333 |
| TP / FP / FN / TN | 268 / 37 / 98 / 695 |

**The baseline is 66.7% on this dev split, not 65.5%.** 65.5% corresponds to a 34.5% present rate; our dev split is 33.3% present. The measured value is used throughout. (Test split measures 65.2%.)


## Truncation ceiling

16K truncation places some gold spans outside the visible window, capping achievable evidence recall through no fault of the model.

| level | visible | total | ceiling |
|---|---|---|---|
| span-level | 653 | 745 | **87.65%** |
| row-level | 329 | 366 | **89.89%** |

The ~89.9% figure is the **row-level** ceiling. The span-level ceiling is 87.65%, and span-level is the right companion to span recall.


## Task 2 — per category, with n and ceiling

Small cells must not be over-read; n is given for every one.

| category | n | n present | gold spans | strict F1 | token F1 | span recall | span ceiling | spans invisible |
|---|---|---|---|---|---|---|---|---|
| Renewal Term | 61 | 24 | 27 | 0.762 | 0.707 | 0.630 | 92.6% | 2 |
| Anti-Assignment | 61 | 44 | 82 | 0.732 | 0.759 | 0.585 | 89.0% | 9 |
| Notice Period To Terminate Renewal | 61 | 14 | 16 | 0.727 | 0.642 | 0.562 | 93.8% | 1 |
| Termination For Convenience | 61 | 21 | 29 | 0.632 | 0.709 | 0.586 | 100.0% | 0 |
| Non-Transferable License | 61 | 12 | 25 | 0.609 | 0.707 | 0.400 | 84.0% | 4 |
| Insurance | 61 | 20 | 73 | 0.571 | 0.581 | 0.315 | 74.0% | 19 |
| Audit Rights | 61 | 28 | 65 | 0.564 | 0.635 | 0.431 | 90.8% | 6 |
| Cap On Liability | 61 | 29 | 61 | 0.533 | 0.726 | 0.557 | 95.1% | 3 |
| License Grant | 61 | 29 | 78 | 0.500 | 0.663 | 0.449 | 89.7% | 8 |
| Exclusivity | 61 | 20 | 42 | 0.444 | 0.479 | 0.476 | 88.1% | 5 |
| Change Of Control | 61 | 16 | 30 | 0.417 | 0.549 | 0.500 | 100.0% | 0 |
| Revenue/Profit Sharing | 61 | 20 | 49 | 0.385 | 0.493 | 0.388 | 85.7% | 7 |
| Ip Ownership Assignment | 61 | 13 | 25 | 0.375 | 0.386 | 0.280 | 68.0% | 8 |
| Post-Termination Services | 61 | 21 | 42 | 0.357 | 0.314 | 0.167 | 100.0% | 0 |
| Uncapped Liability | 61 | 10 | 12 | 0.333 | 0.358 | 0.250 | 100.0% | 0 |
| Covenant Not To Sue | 61 | 10 | 14 | 0.308 | 0.333 | 0.214 | 78.6% | 3 |
| Minimum Commitment | 61 | 20 | 50 | 0.160 | 0.296 | 0.180 | 68.0% | 16 |
| Non-Compete | 61 | 15 | 25 | 0.118 | 0.269 | 0.120 | 96.0% | 1 |

## Task 2 — stratified by contract token length

The ceiling is a length effect, so the evidence metrics are broken out the same way.

| bucket | examples | gold-bearing | gold spans | strict F1 | token F1 | span recall |
|---|---|---|---|---|---|---|
| <16K | 918 | 265 | 471 | 0.575 | 0.632 | 0.503 |
| 16-32K | 144 | 82 | 221 | 0.400 | 0.426 | 0.267 |
| >32K | 36 | 19 | 53 | 0.182 | 0.360 | 0.208 |

## Claims of comparability to published work

**No parity claim is made, for either task.**

- The Jaccard 0.5 threshold was supplied as CUAD's published value. The CUAD dataset paper is Hendrycks et al., *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review* (NeurIPS 2021 Datasets and Benchmarks). **I could not verify the threshold or its exact definition against the paper from this offline cluster.** Independently of that, CUAD's headline metrics are AUPR and Precision at 80% Recall over a span-extraction formulation, which is **not** what is computed here, so our numbers are not comparable to CUAD's published results regardless of the threshold.
- An earlier v1 report described the strict metric as 'ContractEval-comparable'. **That claim is unverified** — the exact ContractEval definition was not checked against its source in this run. It should be softened to 'strict containment, defined here' in any slide until verified.

If a slide needs a comparability claim, the honest form is: *these are internally consistent metrics computed on our own contract-level splits; they are not directly comparable to published leaderboard numbers.*
