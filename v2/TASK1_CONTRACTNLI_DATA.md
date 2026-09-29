# Task 1 — ContractNLI dataset (existing, frozen)

Data lives at `v1_contractnli/doc_sft/{train,valid,test}.jsonl` and is **read
by path**, never copied into `v2/`. `v1_contractnli/` is frozen: it holds a
complete set of measured results that are a deliverable in their own right.

Nothing new was built for Task 1. Every figure below was **recomputed** in
this run against the files themselves, not carried over from earlier notes.

| | |
|---|---|
| source | `trismik/ContractNLI`, official splits used as-is |
| build script | `v1_contractnli/build_doc_sft.py` |
| target schema | `{"verdict": ..., "evidence": [...]}` |
| normalizer | `text_norm.norm` (lowercase, collapse whitespace, strip) |

## Row counts and class distribution

| split | rows | documents | Entailment | Contradiction | NotMentioned |
|---|---|---|---|---|---|
| train | 7,188 | 423 | 3,527 (49.1%) | 841 (11.7%) | 2,820 (39.2%) |
| dev | 1,037 | 61 | 519 (50.0%) | 95 (9.2%) | 423 (40.8%) |
| test | 2,091 | 123 | 968 (46.3%) | 220 (10.5%) | 903 (43.2%) |

Contradiction is the minority class everywhere, 9.2–11.7%. That imbalance
is the backdrop for the model-side weakness on that class.

## Split is document-level — verified

| pair | shared documents |
|---|---|
| train ∩ dev | **0** |
| train ∩ test | **0** |
| dev ∩ test | **0** |

607 distinct documents total (423/61/123). No contract contributes rows to
more than one split, so there is no document-level leakage.

## Evidence-span validation

| | |
|---|---|
| spans checked | **11,965** |
| not found verbatim in source | **0** |
| **dataset-level fabrication rate** | **0.0000%** |

Recomputed in this run with `text_norm.norm`. Every gold evidence span
occurs verbatim in its own contract.

### Two different fabrication numbers — do not conflate them

| level | value | what it is a property of |
|---|---|---|
| **dataset** | **0.0000%** | ContractNLI's *annotations* — the number for this document |
| model | 0.3–2.7% fine-tuned, 17.7–19.3% base | model *predictions* — an evaluation result |

The model-level figures belong to the v1 evaluation, not to data quality.

## Build-time exclusions

- **3 label-corrupting pairs dropped** (doc 622/nda-10, doc 622/nda-19, doc 162/nda-19) — gold evidence fell beyond
  the 8192-token cut, so the target cited text the model could not see.
- **1 whitespace-only span dropped** (dev doc 582, nda-19) — span-level, the row was retained with 5 of 6 spans.
- Evidence-verbatim failures at build time: **0**.

## The four ambiguous "some"-quantifier positions — RETAINED, not filtered

`nda-7`, `nda-16`, `nda-18`, `nda-20`. All four are partial-scope positions
whose wording contains *some*:

- **nda-7** — Receiving Party may share some Confidential Information with some third-parties (including consultants, a
- **nda-16** — Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.
- **nda-18** — Receiving Party shall not solicit some of Disclosing Party's representatives.
- **nda-20** — Receiving Party may retain some Confidential Information even after the return or destruction of Confiden

These positions have no single canonical governing span, so annotator and
model disagree about *extent* rather than about substance. They were
implicated independently by three v1 analyses: the partial-overlap failure
taxonomy, the Contradiction false-positive concentration, and the
over-extraction breakdown.

### The decision, and its measured cost

| | |
|---|---|
| filtered? | **No — retained** |
| train rows lost if removed | 7,188 → 5,496 = **−23.5%** |
| Contradiction signal lost | 841 → 535 = **−36.4%** |

**Rationale.** Removing them costs over a third of the Contradiction
training signal — already the scarcest and weakest class — to fix a
span-boundary disagreement rather than a labelling error. The decision was
to **retain them and report the ambiguity**, treating the filter as a
diagnostic that names which positions to caveat rather than as a training
intervention.

## Existing per-hypothesis breakdown

`v1_contractnli/phase0/per_hypothesis_dev.csv` — 17 rows covering n, gold
class distribution, per-class P/R/F1, macro-F1, accuracy, evidence exact
rate, evidence F1 and partial credit. Worst quartile by evidence F1:

| hypothesis | n | evidence F1 | macro-F1 |
|---|---|---|---|
| nda-20 | 61 | **0.448** | 0.572 |
| nda-18 | 61 | **0.632** | 0.593 |
| nda-7 | 61 | **0.711** | 0.686 |
| nda-16 | 61 | **0.727** | 0.585 |

(Dev split. Test was never evaluated in v1.)

