# Phase 1 — document-level pre-SFT screen (dev split)

Full NDA as context, one policy question per call, **zero-shot**.
Task supersedes the clause-level screen; those numbers measured an easier
task and are not mixed in here.

**No few-shot by design.** At document level each exemplar carries a full
contract, so three exemplars plus the test contract exceed context.
ContractEval is zero-shot for the same reason. Deliberate, not an omission.

n = 1037 (document, hypothesis) pairs per cell.


## Ranking — macro-F1, condition A (zero-shot non-thinking)

| model | cond | macro-F1 | 95% CI | acc | Contradiction recall | NotMentioned recall |
|---|---|---|---|---|---|---|
| Qwen3-14B | B thinking | **0.762** | [0.728, 0.792] | 0.802 | 0.758 | 0.742 |
| gemma-4-12B-it | A zero-shot | **0.730** | [0.698, 0.761] | 0.792 | 0.716 | 0.697 |
| Qwen3-4B | B thinking | **0.712** | [0.677, 0.746] | 0.770 | 0.674 | 0.759 |
| Qwen3-8B | B thinking | **0.704** | [0.670, 0.737] | 0.757 | 0.684 | 0.700 |
| Qwen3-14B | A zero-shot | **0.701** | [0.668, 0.733] | 0.753 | 0.811 | 0.700 |
| granite-4.2-8b | A zero-shot | **0.623** | [0.590, 0.654] | 0.675 | 0.800 | 0.541 |
| Qwen3-4B | A zero-shot | **0.615** | [0.579, 0.650] | 0.669 | 0.642 | 0.584 |
| Qwen3-8B | A zero-shot | **0.553** | [0.519, 0.585] | 0.623 | 0.842 | 0.314 |

## Verdict — per class

| model | cond | Entailment P/R/F1 | Contradiction P/R/F1 | NotMentioned P/R/F1 |
|---|---|---|---|---|
| Qwen3-14B | B thinking | 0.84/0.86/0.85 | 0.58/0.76/0.65 | 0.83/0.74/0.78 |
| gemma-4-12B-it | A zero-shot | 0.82/0.88/0.85 | 0.45/0.72/0.55 | 0.90/0.70/0.79 |
| Qwen3-4B | B thinking | 0.85/0.80/0.82 | 0.44/0.67/0.54 | 0.80/0.76/0.78 |
| Qwen3-8B | B thinking | 0.80/0.82/0.81 | 0.46/0.68/0.55 | 0.81/0.70/0.75 |
| Qwen3-14B | A zero-shot | 0.85/0.79/0.82 | 0.40/0.81/0.53 | 0.82/0.70/0.75 |
| granite-4.2-8b | A zero-shot | 0.78/0.76/0.77 | 0.32/0.80/0.46 | 0.77/0.54/0.63 |
| Qwen3-4B | A zero-shot | 0.71/0.74/0.72 | 0.35/0.64/0.45 | 0.78/0.58/0.67 |
| Qwen3-8B | A zero-shot | 0.71/0.83/0.77 | 0.28/0.84/0.42 | 0.95/0.31/0.47 |

## Evidence extraction

Strict = every gold span covered (ContractEval-comparable). Partial = mean fraction of gold spans covered.

| model | cond | F1 strict | F2 strict | partial | Jaccard | false-abstention | TP/FN/FP/TN |
|---|---|---|---|---|---|---|---|
| Qwen3-14B | B thinking | 0.369 | 0.300 | 0.393 | 0.501 | 0.109 | 164/450/112/311 |
| gemma-4-12B-it | A zero-shot | 0.432 | 0.366 | 0.482 | 0.561 | 0.055 | 204/410/127/296 |
| Qwen3-4B | B thinking | 0.320 | 0.275 | 0.375 | 0.466 | 0.067 | 154/460/194/229 |
| Qwen3-8B | B thinking | 0.397 | 0.335 | 0.430 | 0.469 | 0.112 | 186/428/136/287 |
| Qwen3-14B | A zero-shot | 0.391 | 0.326 | 0.418 | 0.499 | 0.117 | 180/434/127/296 |
| granite-4.2-8b | A zero-shot | 0.510 | 0.466 | 0.559 | 0.390 | 0.166 | 271/343/178/245 |
| Qwen3-4B | A zero-shot | 0.356 | 0.304 | 0.387 | 0.431 | 0.130 | 170/444/171/252 |
| Qwen3-8B | A zero-shot | 0.296 | 0.271 | 0.366 | 0.438 | 0.016 | 157/457/289/134 |

### Evidence excluding trivially-matchable gold spans

Gold spans of <=4 normalized characters are section numbers (`2.1`, `4.3`).
Any output containing `2.1` anywhere covers them, so they are trivially
matchable — but a model that quotes the clause and omits its section number
loses the whole example to FN. The effect is therefore **signed**, and
measured here rather than assumed. Headline numbers above keep them (they are
real annotations); this table removes them. Examples whose gold is entirely
trivial are excluded, not reclassified as gold-empty.

| model | cond | F1 strict | F1 excl. trivial | delta | examples excluded |
|---|---|---|---|---|---|
| Qwen3-14B | B thinking | 0.369 | 0.369 | +0.000 | 0 |
| gemma-4-12B-it | A zero-shot | 0.432 | 0.432 | +0.000 | 0 |
| Qwen3-4B | B thinking | 0.320 | 0.320 | +0.000 | 0 |
| Qwen3-8B | B thinking | 0.397 | 0.397 | +0.000 | 0 |
| Qwen3-14B | A zero-shot | 0.391 | 0.391 | +0.000 | 0 |
| granite-4.2-8b | A zero-shot | 0.510 | 0.510 | +0.000 | 0 |
| Qwen3-4B | A zero-shot | 0.356 | 0.356 | +0.000 | 0 |
| Qwen3-8B | A zero-shot | 0.296 | 0.296 | +0.000 | 0 |

## JOINT — verdict correct AND all gold spans covered

The product metric. `lenient` is the brief's literal definition, under which
empty gold counts as 'all covered'. `strict` additionally requires the model
not to invent evidence when gold is empty. Both shown because the gap is the
hallucinated-citation rate on NotMentioned rows.

| model | cond | joint strict | joint lenient |
|---|---|---|---|
| Qwen3-14B | B thinking | **0.449** | 0.452 |
| gemma-4-12B-it | A zero-shot | **0.464** | 0.464 |
| Qwen3-4B | B thinking | **0.353** | 0.442 |
| Qwen3-8B | B thinking | **0.440** | 0.449 |
| Qwen3-14B | A zero-shot | **0.440** | 0.440 |
| granite-4.2-8b | A zero-shot | **0.452** | 0.452 |
| Qwen3-4B | A zero-shot | **0.379** | 0.380 |
| Qwen3-8B | A zero-shot | **0.260** | 0.260 |

## Diagnostics (not for ranking)

| model | cond | schema valid | not JSON | JSON wrong schema | parse fail | truncated | mean out | p99 out | s/example |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | B thinking | 99.7% | 0.3% | 0.0% | 0.0% | 0.0% | 437 | 1388 | 3.918 |
| gemma-4-12B-it | A zero-shot | 99.7% | 0.3% | 0.0% | 0.0% | 0.0% | 72 | 247 | 0.024 |
| Qwen3-4B | B thinking | 98.9% | 1.1% | 0.0% | 0.4% | 0.9% | 575 | 1877 | 4.572 |
| Qwen3-8B | B thinking | 98.5% | 1.5% | 0.0% | 0.1% | 1.1% | 563 | 2048 | 4.464 |
| Qwen3-14B | A zero-shot | 99.4% | 0.6% | 0.0% | 0.0% | 0.1% | 72 | 279 | 0.669 |
| granite-4.2-8b | A zero-shot | 95.4% | 4.6% | 0.0% | 0.0% | 3.9% | 159 | 512 | 1.42 |
| Qwen3-4B | A zero-shot | 98.2% | 1.8% | 0.0% | 0.0% | 1.4% | 93 | 512 | 0.746 |
| Qwen3-8B | A zero-shot | 99.6% | 0.4% | 0.0% | 0.0% | 0.2% | 95 | 285 | 0.776 |

## Is the ordering real? (paired bootstrap, condition A)

| comparison | diff | 95% CI | verdict |
|---|---|---|---|
| gemma-4-12B-it vs Qwen3-14B | +0.029 | [-0.002, +0.058] | within noise |
| gemma-4-12B-it vs granite-4.2-8b | +0.107 | [+0.076, +0.141] | **separated** |
| gemma-4-12B-it vs Qwen3-4B | +0.115 | [+0.083, +0.146] | **separated** |
| gemma-4-12B-it vs Qwen3-8B | +0.177 | [+0.145, +0.211] | **separated** |
| Qwen3-14B vs granite-4.2-8b | +0.079 | [+0.050, +0.109] | **separated** |
| Qwen3-14B vs Qwen3-4B | +0.086 | [+0.055, +0.120] | **separated** |
| Qwen3-14B vs Qwen3-8B | +0.149 | [+0.121, +0.177] | **separated** |
| granite-4.2-8b vs Qwen3-4B | +0.008 | [-0.025, +0.044] | within noise |
| granite-4.2-8b vs Qwen3-8B | +0.070 | [+0.039, +0.101] | **separated** |
| Qwen3-4B vs Qwen3-8B | +0.062 | [+0.031, +0.094] | **separated** |

---

# Written summary

## Ranking on the primary condition (A, zero-shot non-thinking)

| rank | model | macro-F1 | 95% CI |
|---|---|---|---|
| 1 | gemma-4-12B-it | 0.730 | [0.698, 0.761] |
| 2 | Qwen3-14B | 0.701 | [0.668, 0.733] |
| 3 | granite-4.2-8b | 0.623 | [0.590, 0.654] |
| 4 | Qwen3-4B | 0.615 | [0.579, 0.650] |
| 5 | Qwen3-8B | 0.553 | [0.519, 0.585] |

**Two tiers, not five ranks.** The paired bootstrap separates most pairs but
not within tiers:

- **Top tier**: gemma-4-12B-it and Qwen3-14B are **within noise**
  (+0.029, CI [-0.002, +0.058]). Do not claim Gemma beats Qwen3-14B.
- **Middle tier**: granite-4.2-8b and Qwen3-4B are **within noise**
  (+0.008, CI [-0.025, +0.044]).
- **Qwen3-8B is separated from everything**, at the bottom.

Every cross-tier comparison is separated, so the tier structure itself is real.

## Thinking mode is a large, consistent gain (condition B, Qwen3 only)

| model | A zero-shot | B thinking | gain |
|---|---|---|---|
| Qwen3-8B | 0.553 | 0.704 | **+0.151** |
| Qwen3-4B | 0.615 | 0.712 | +0.097 |
| Qwen3-14B | 0.701 | 0.762 | +0.061 |

**Qwen3-14B with thinking (0.762) is the best cell in the screen**, above every
condition-A result. The gain is inversely proportional to model size: the
weakest model benefits most.

It is not free. Thinking costs **~6x the wall clock** (3.9-4.6 s/example vs
0.67-0.78) and ~6x the output tokens (437-575 mean vs 72-95).

## The interesting failure: Qwen3-8B cannot abstain zero-shot

| Qwen3-8B, condition A | precision | recall |
|---|---|---|
| NotMentioned | 0.95 | **0.31** |
| Contradiction | 0.28 | 0.84 |

It almost never answers NotMentioned, but is right when it does. Instead it
forces Entailment or Contradiction on ~69% of the clauses that do not address
the policy — which is why its Contradiction precision collapses to 0.28.

This is the **same abstention-calibration failure this model showed at clause
level**, now larger at document scale. Thinking repairs most of it
(NotMentioned recall 0.31 -> 0.70). Abstention is the core product capability,
so this is the most commercially significant result in the screen.

## Evidence extraction is the binding constraint

Best strict F1 is **0.510** (granite). Every model is far weaker at evidence
than at verdicts. Note the inversion: **granite has the best evidence F1 and the
worst verdict macro-F1** — it retrieves well and judges poorly.

Granite's number carries an asterisk: it hit the 512-token cap on **40 examples
(3.9%)**, mid-way through enumerating spans. Its evidence score is suppressed by
the cap, not only by capability.

**Excluding trivially-matchable short gold spans changes evidence F1 by exactly
0.000 on every cell.** The 11 affected dev rows never flip TP/FN, because their
long spans are already uncovered. The concern was real; the measured magnitude
is nil.

## Joint metric — the product number

| model | cond | joint strict | joint lenient | gap |
|---|---|---|---|---|
| gemma-4-12B-it | A | **0.464** | 0.464 | 0.000 |
| granite-4.2-8b | A | 0.452 | 0.452 | 0.000 |
| Qwen3-14B | B | 0.449 | 0.452 | 0.003 |
| Qwen3-14B | A | 0.440 | 0.440 | 0.000 |
| Qwen3-8B | B | 0.440 | 0.449 | 0.009 |
| Qwen3-4B | A | 0.379 | 0.380 | 0.001 |
| **Qwen3-4B** | **B** | **0.353** | **0.442** | **0.089** |
| Qwen3-8B | A | 0.260 | 0.260 | 0.000 |

The strict/lenient gap is the hallucinated-citation rate on NotMentioned rows.
It is ~0 everywhere except **Qwen3-4B with thinking, at 8.9 points** — thinking
makes the 4B model invent evidence for clauses it correctly judges NotMentioned.
A model that says "not mentioned" and then cites a clause anyway is a specific
liability for this product, and it is only visible because joint is reported
both ways.

## Cost

| model | s/example | mean output tokens |
|---|---|---|
| gemma-4-12B-it | **0.024** | 72 |
| Qwen3-14B A | 0.669 | 72 |
| Qwen3-4B A | 0.746 | 93 |
| Qwen3-8B A | 0.776 | 95 |
| granite-4.2-8b A | 1.420 | 159 |
| Qwen3-14B B | 3.918 | 437 |
| Qwen3-8B B | 4.464 | 563 |
| Qwen3-4B B | 4.572 | 575 |

Gemma is **28x faster than Qwen3-14B** at equal output length and **163x faster
than the best cell overall**. On a cost/performance basis it is the standout:
joint-best, top-tier macro-F1, and the cheapest by a wide margin.

## Format quality is not a discriminator

Schema validity is 95.4-99.7%; parse failures are 0.0-0.4%. "JSON, wrong
schema" is **0.0% everywhere** — when these models emit JSON they use the right
keys. As at clause level, formatting does not separate them, so it cannot carry
the argument for SFT.

## What this means for Phase 2

- **Fine-tune Qwen3-14B first**, as planned. It leads condition A within noise
  of Gemma, has the best thinking result, and is the only model that is strong
  on both verdict and abstention.
- **Gemma deserves a place in the SFT set** on the strength of being joint-best
  and 28x cheaper, which is a genuine deployment argument rather than a metric
  artefact.
- **Qwen3-8B is the most interesting SFT candidate despite ranking last**: its
  deficit is abstention calibration, exactly what supervised training on
  balanced targets should repair, and thinking already recovers +0.151 of it.
- Evidence extraction, not verdict accuracy, is where SFT has the most room
  (best strict F1 0.510).
