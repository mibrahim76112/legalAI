# Phase 1 — document-level pre-SFT screen (dev split)

Full NDA as context, one policy question per call, **zero-shot**.
Task supersedes the clause-level screen; those numbers measured an easier
task and are not mixed in here.

**No few-shot by design.** At document level each exemplar carries a full
contract, so three exemplars plus the test contract exceed context.
ContractEval is zero-shot for the same reason. Deliberate, not an omission.

n = 1037 (document, hypothesis) pairs per cell.


## Ranking — macro-F1, condition A (zero-shot non-thinking)

| model | cond | macro-F1 | 95% CI | acc | **Contra PRECISION** | Contra recall | NotMent recall |
|---|---|---|---|---|---|---|---|
| Qwen3-14B | B thinking | **0.762** | [0.728, 0.792] | 0.802 | **0.576** | 0.758 | 0.742 |
| gemma-4-12B-it | A zero-shot | **0.730** | [0.699, 0.762] | 0.792 | **0.450** | 0.716 | 0.697 |
| Qwen3-4B | B thinking | **0.712** | [0.678, 0.746] | 0.770 | **0.444** | 0.674 | 0.759 |
| Qwen3-8B | B thinking | **0.704** | [0.669, 0.736] | 0.757 | **0.464** | 0.684 | 0.700 |
| Qwen3-14B | A zero-shot | **0.701** | [0.669, 0.734] | 0.753 | **0.399** | 0.811 | 0.700 |
| granite-4.2-8b | A zero-shot | **0.623** | [0.590, 0.653] | 0.675 | **0.325** | 0.800 | 0.541 |
| Qwen3-4B | A zero-shot | **0.615** | [0.579, 0.649] | 0.669 | **0.349** | 0.642 | 0.584 |
| Qwen3-8B | A zero-shot | **0.553** | [0.519, 0.585] | 0.623 | **0.282** | 0.842 | 0.314 |

## VETO METRIC — Contradiction precision (the cry-wolf rate)

Recall was the assumed veto metric on the theory that a missed conflict is
catastrophic. The data says that is not the failure mode: **recall is
adequate everywhere (0.64-0.84) while precision is 0.28-0.58**. These models
over-flag conflicts.

A reviewer handed 200 escalations of which 90 are real stops trusting the
flags and re-reads everything, which is the workflow the tool replaces. So
**precision is the commercial bottleneck and the metric SFT must move.**

Dev contains **95 gold Contradictions** of 1037.

| model | cond | flagged | real | false | precision | recall | false-flag rate | over-flag |
|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | B thinking | 125 | 72 | 53 | **0.58** | 0.76 | 42% | 1.3x |
| Qwen3-8B | B thinking | 140 | 65 | 75 | **0.46** | 0.68 | 54% | 1.5x |
| gemma-4-12B-it | A zero-shot | 151 | 68 | 83 | **0.45** | 0.72 | 55% | 1.6x |
| Qwen3-4B | B thinking | 144 | 64 | 80 | **0.44** | 0.67 | 56% | 1.5x |
| Qwen3-14B | A zero-shot | 193 | 77 | 116 | **0.40** | 0.81 | 60% | 2.0x |
| Qwen3-4B | A zero-shot | 175 | 61 | 114 | **0.35** | 0.64 | 65% | 1.8x |
| granite-4.2-8b | A zero-shot | 234 | 76 | 158 | **0.32** | 0.80 | 68% | 2.5x |
| Qwen3-8B | A zero-shot | 284 | 80 | 204 | **0.28** | 0.84 | 72% | 3.0x |

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
| gemma-4-12B-it vs Qwen3-14B | +0.028 | [-0.002, +0.058] | within noise |
| gemma-4-12B-it vs granite-4.2-8b | +0.107 | [+0.076, +0.141] | **separated** |
| gemma-4-12B-it vs Qwen3-4B | +0.115 | [+0.082, +0.146] | **separated** |
| gemma-4-12B-it vs Qwen3-8B | +0.177 | [+0.144, +0.211] | **separated** |
| Qwen3-14B vs granite-4.2-8b | +0.079 | [+0.050, +0.109] | **separated** |
| Qwen3-14B vs Qwen3-4B | +0.087 | [+0.055, +0.120] | **separated** |
| Qwen3-14B vs Qwen3-8B | +0.149 | [+0.121, +0.177] | **separated** |
| granite-4.2-8b vs Qwen3-4B | +0.008 | [-0.025, +0.044] | within noise |
| granite-4.2-8b vs Qwen3-8B | +0.070 | [+0.039, +0.101] | **separated** |
| Qwen3-4B vs Qwen3-8B | +0.062 | [+0.029, +0.093] | **separated** |
