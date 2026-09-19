# Phase 0 diagnostic — dev split

Predictions: `raw____localscratch__ibi761.21892913.0__merged_Qwen_Qwen3-14B__nothink__dev__sft.jsonl`  (1037 rows)

Analysis only. **Dev split** — test is reserved and untouched.


## 0.1 Per-hypothesis breakdown

| hyp | n | gold E/C/NM | macro-F1 | acc | ev F1 | ev exact | ev partial |
|---|---|---|---|---|---|---|---|
| nda-20 | 61 | 13/12/36 | 0.572 | 0.541 | **0.448** | 0.393 | 0.810 |
| nda-18 | 61 | 11/0/50 | 0.593 | 0.934 | **0.632** | 0.885 | 0.720 |
| nda-7 | 61 | 39/14/8 | 0.686 | 0.803 | **0.711** | 0.574 | 0.751 |
| nda-16 | 61 | 22/2/37 | 0.585 | 0.869 | **0.727** | 0.803 | 0.761 |
| nda-3 | 61 | 45/0/16 | 0.585 | 0.902 | **0.730** | 0.672 | 0.706 |
| nda-10 | 61 | 29/0/32 | 0.590 | 0.885 | **0.731** | 0.770 | 0.736 |
| nda-19 | 61 | 43/0/18 | 0.570 | 0.885 | **0.737** | 0.672 | 0.767 |
| nda-1 | 61 | 19/10/32 | 0.748 | 0.820 | **0.750** | 0.803 | 0.694 |
| nda-17 | 61 | 9/8/44 | 0.793 | 0.869 | **0.765** | 0.869 | 0.765 |
| nda-4 | 61 | 50/3/8 | 0.719 | 0.902 | **0.783** | 0.672 | 0.783 |
| nda-2 | 61 | 6/44/11 | 0.554 | 0.836 | **0.791** | 0.705 | 0.786 |
| nda-5 | 61 | 54/2/5 | 0.546 | 0.934 | **0.816** | 0.705 | 0.806 |
| nda-15 | 61 | 39/0/22 | 0.594 | 0.902 | **0.838** | 0.803 | 0.829 |
| nda-13 | 61 | 47/0/14 | 0.667 | 1.000 | **0.840** | 0.787 | 0.872 |
| nda-11 | 61 | 8/0/53 | 0.613 | 0.967 | **0.857** | 0.967 | 0.750 |
| nda-12 | 61 | 40/0/21 | 0.654 | 0.984 | **0.892** | 0.869 | 0.930 |
| nda-8 | 61 | 45/0/16 | 0.652 | 0.984 | **0.930** | 0.902 | 0.908 |

## 0.3 Evidence-failure attribution

253 failures of 1037 rows (24.4%).

| category | n | % of failures | % of all rows |
|---|---|---|---|
| (a) missed — pred empty, gold non-empty | 33 | 13.0% | 3.2% |
| (b) over-extraction — pred non-empty, gold empty | 67 | 26.5% | 6.5% |
| (c) fabricated — span absent from contract | 4 | 1.6% | 0.4% |
| (d) partial — grounded, partial overlap with gold | 116 | 45.8% | 11.2% |
| (e) wrong span — grounded, no overlap | 33 | 13.0% | 3.2% |

## 0.3b How near are the near-misses? (item 8)

Gold spans recovered / required, over 186 gold-bearing failures:

| p10 | p25 | p50 | p75 | p90 | share ≥0.5 | share =0 |
|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.40 | 0.50 | 0.67 | 47.8% | 36.0% |

Strict evidence F1 (**primary, unchanged**): **0.7719**

Partial-credit variant (reported alongside, NOT a substitute): 0.7960


## 0.4 Positional check

Gold-evidence first-half share, per hypothesis (min 30.2%, max 96.0%):

| hyp | first-half share | n spans | ev F1 |
|---|---|---|---|
| nda-2 | 96.0% | 101 | 0.791 |
| nda-1 | 94.8% | 58 | 0.750 |
| nda-7 | 94.2% | 104 | 0.711 |
| nda-5 | 93.1% | 102 | 0.816 |
| nda-3 | 90.7% | 75 | 0.730 |
| nda-4 | 90.0% | 90 | 0.783 |
| nda-8 | 84.6% | 78 | 0.930 |
| nda-10 | 83.6% | 67 | 0.731 |
| nda-12 | 74.3% | 101 | 0.892 |
| nda-13 | 72.7% | 121 | 0.840 |
| nda-17 | 69.6% | 23 | 0.765 |
| nda-11 | 66.7% | 9 | 0.857 |
| nda-18 | 56.7% | 30 | 0.632 |
| nda-16 | 43.1% | 51 | 0.727 |
| nda-15 | 35.3% | 68 | 0.838 |
| nda-19 | 30.2% | 86 | 0.737 |
| nda-20 | 30.2% | 63 | 0.448 |

## 0.5 Decision rule

- worst quartile by evidence F1 (4 hypotheses): ['nda-16', 'nda-18', 'nda-20', 'nda-7']
- (c) fabricated + (a) missed = **14.6%** of all failures  (threshold ≥40% → PROCEED)
- (d) partial + (e) wrong span **within the worst quartile** = **14.2%** of all failures

### BRANCH TAKEN: **SKIP Phases 1 and 2**


---

# FINDINGS — branch taken, numbers, confidence

## Branch: SKIP Phases 1 and 2. CUAD would not fix this.

The decision rule required **(c) fabricated + (a) missed ≥ 40%** of evidence
failures for extractive supervision to be the plausible fix. Measured:

| | share of 253 failures |
|---|---|
| (c) fabricated + (a) missed | **14.6%** |
| (d) partial + (e) wrong span | **58.8%** |
| (b) over-extraction | 26.5% |

**14.6% against a 40% threshold — not close.** Fabrication alone is 4 rows
(1.6%). The fine-tuned model essentially does not invent text, so there is no
grounding deficit for CUAD to repair.

Failures are dominated by **partial overlap (45.8%)**: the model finds grounded
text in the right region and clips the span differently from the annotator. That
is a span-boundary disagreement, not a retrieval failure, and adding a second
extractive dataset with its own boundary conventions would not resolve it — it
could plausibly worsen it.

## Item 8 — how near are the near-misses

Gold spans recovered / required, over 186 gold-bearing failures:

| p10 | p25 | p50 | p75 | p90 | ≥0.5 | =0 |
|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.40 | 0.50 | 0.67 | 47.8% | 36.0% |

Nearly half of "total failures" recover at least half the required spans. The
strict metric scores a row recovering 2 of 3 gold spans identically to one
recovering none.

| | |
|---|---|
| strict evidence F1 (**primary, unchanged**) | **0.7719** |
| partial-credit variant (alongside, not a substitute) | 0.7960 |

The gap is only 0.024 in aggregate because most rows pass strictly; the
distribution above is where the strictness actually bites.

**This materially changes how the CUAD hypothesis should be read.** The premise
was "the evidence field is weak from too little extractive supervision". The
evidence field is not weak in the retrieval sense — it is grounded (1.6%
fabrication) and usually partially correct (median 0.40 of spans recovered among
failures). The deficit is boundary agreement with ContractNLI's annotators,
which more extractive data from a different corpus does not address.

## Implicated hypotheses (worst quartile by evidence F1)

| hyp | ev F1 | n | gold E/C/NM | position |
|---|---|---|---|---|
| nda-20 | **0.448** | 61 | 13/12/36 | "may retain **some** CI even after return or destruction" |
| nda-18 | 0.632 | 61 | 11/0/50 | "shall not solicit **some** of DP's representatives" |
| nda-7 | 0.711 | 61 | 39/14/8 | "may share **some** CI with **some** third-parties" |
| nda-16 | 0.727 | 61 | 22/2/37 | "shall destroy or return **some** CI upon termination" |

All four are **partial-scope positions containing "some"**. These are the same
hypotheses that dominated the Contradiction false-positive analysis earlier in
the project (nda-7 alone produced 18% of all false conflicts). A position of the
form "may retain *some* information" has no single canonical governing span, so
annotators and model disagree about extent — which is exactly the
inconsistent-annotation-precision problem LegalBench-RAG cites when excluding
whole ContractNLI categories.

nda-18 is worth singling out: **0 Contradiction examples in dev** out of 61.

## What a filtered training set would look like

Dropping those four hypotheses:

| split | before | after | change |
|---|---|---|---|
| train | 7,188 | 5,496 | −1,692 (−23.5%) |
| dev | 1,037 | 793 | −244 (−23.5%) |
| train Contradiction | 841 | 535 | **−36.4%** |
| dev Contradiction | 95 | 67 | −29.5% |

The cost is concentrated on Contradiction, which is already the scarcest and
weakest class — dropping these four removes over a third of Contradiction
training signal to fix a span-boundary problem. **I would not recommend the
filter as a training intervention.** Its value is diagnostic: it identifies
which four positions to caveat in reporting, or to exclude from an
evidence-quality headline while retaining for verdict training.

## Confidence

**High** on the branch decision. 14.6% versus a 40% threshold is not a marginal
call, and the mechanism is consistent across two independent analyses in this
project (failure taxonomy here, Contradiction false-positive concentration
earlier), both implicating the same partial-scope hypotheses.

**Moderate** on the filtered-set recommendation. It rests on one model's dev
predictions (Qwen3-14B, Arm A). The per-hypothesis pattern was stable across
models in the earlier false-positive analysis, but I have not re-run this
taxonomy across all three models — that is one cheap next step if the
recommendation is to be acted on.

## Caveats, stated

- **Dev split throughout.** Test is reserved and untouched.
- **Checkpoint criterion: final checkpoint, no selection.** `load_best_model_at_end`
  is not set and `save_total_limit=2` means the best checkpoint is usually not
  retained. Identical across all arms, so parity holds, but it is a limitation:
  eval loss rises off its minimum in most runs, so the reported models are
  past their own optimum.
- Failure taxonomy is a new measurement; `evidence_metrics` and `evidence_case`
  were imported and used unchanged.
