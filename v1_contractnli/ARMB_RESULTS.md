# Arm B — reasoning-first targets. Negative result.

**Pre-registered prediction: reasoning-first would improve Contradiction
precision, Arm A's worst metric. FALSIFIED.** With training data held fixed,
reasoning made Contradiction precision *worse* on all three models, and cost 3x
the output tokens.

## Method

STaR (Zelikman et al. 2022) over the TRAIN split. Teacher Qwen3-32B (ungated,
apache-2.0), blind Pass 1: the teacher never saw the gold verdict, so agreement
with gold is a genuine derivation signal rather than post-hoc rationalisation.
ContractNLI ships **no rationales** (annotations carry only `choice` and
`spans`), so there is no ground-truth reasoning — agreement is the only
correctness signal available.

Four deterministic filters, no LLM judge (per brief): quotes must appear in the
contract under the same `doc_harness.norm` the eval uses; evidence must cohere
with gold; the prose verdict must match the JSON field; length within
data-derived bounds (p1/p99, floor 20).

**Verified-only: 2,733 rows of 7,188 (38.0%). Zero conditioned/rationalised
rows.** Rows the teacher disagreed with are enriched in exactly the cases where
its judgment is unreliable, so they were held back rather than capped.

## The confound, and the control that removes it

Arm B trained on 2,733 rows against Arm A's 7,188, so the first comparison
varied **target schema AND data volume** at once. A subset-matched control —
Arm A targets on the exact same 2,733 (document, hypothesis) pairs — isolates
the schema.

### Qwen3-14B

| arm | macro-F1 | Contra P | evidence F1 | joint | out tok |
|---|---|---|---|---|---|
| A full (7,188) | 0.839 | 0.650 | 0.772 | 0.746 | 74 |
| A matched (2,733) | 0.793 | 0.608 | 0.588 | 0.583 | 79 |
| B reasoning (2,733) | 0.739 | 0.524 | 0.464 | 0.469 | 216 |
| **data-loss cost** | −0.046 | −0.042 | **−0.184** | −0.163 | |
| **reasoning effect** | **−0.054** | **−0.084** | −0.124 | −0.115 | **+137** |

Both effects are real and they split by metric: **losing 62% of the data costs
evidence F1 most**, **reasoning costs Contradiction precision most**.

## Reasoning effect, paired bootstrap (1,200 resamples, data fixed)

| model | macro-F1 | Contra precision | evidence F1 | joint |
|---|---|---|---|---|
| Qwen3-4B | −0.056 [−0.084,−0.028] | −0.112 [−0.179,−0.046] | −0.092 [−0.123,−0.063] | −0.067 [−0.092,−0.042] |
| Qwen3-8B | −0.059 [−0.085,−0.032] | −0.094 [−0.157,−0.035] | −0.093 [−0.122,−0.061] | −0.078 [−0.101,−0.052] |
| Qwen3-14B | −0.054 [−0.079,−0.026] | −0.084 [−0.148,−0.026] | −0.124 [−0.153,−0.094] | −0.114 [−0.141,−0.089] |

**All 12 comparisons separated, all negative.** No confidence interval crosses
zero. This is not noise.

## This is not an implementation failure

| | |
|---|---|
| rows emitting a reasoning field | **100%** |
| schema validity | 98.7–99.4% |
| parse failures | 0.0% |
| truncation | <1% |

The rationales are present, well-formed, and read as genuine derivations. The
mechanism works; it just does not help.

## The interesting part: reasoning helps base models and hurts fine-tuned ones

Phase 1 measured thinking mode on the same task and the same models:

| model | base | base + thinking | gain |
|---|---|---|---|
| Qwen3-14B | 0.399 | 0.576 | **+0.177** |
| Qwen3-8B | 0.282 | 0.464 | **+0.183** |
| Qwen3-4B | 0.349 | 0.444 | +0.096 |

Reasoning gave an untrained model +0.152 mean on Contradiction precision. The
same intervention, distilled into a fine-tuned model, costs −0.097 mean. SFT
already teaches the clause-level discrimination; forcing the model to articulate
it first appears to interfere rather than help.

## Secondary finding: STaR's yield costs more than its reasoning

The 62% filter rejection costs **−0.18 to −0.21 evidence F1** — larger than the
reasoning effect on that metric. If reasoning were pursued further, raising
yield matters more than improving the prompt.

## What was NOT run

Pass 2 (conditioned/rationalised rows) was deliberately not used. STaR shows
rationalisation helps in domains with unambiguously correct answers; here the
label itself is contested and teacher disagreement correlates with label
ambiguity. Whether rationalisation helps or hurts in a contested-label domain
remains open and is the more interesting follow-up.
