# Base-model screen — results

ContractNLI dev, 978 rows (466 Entailment / 89 Contradiction / 423 NotMentioned).
Greedy, BF16, each model's own chat template, one shared parser, no per-model
prompt tuning. **No training happened.** Array job `21789350`, 8 cells, 1x H100 each.

## Results table

| model | cond | macro-F1 | 95% CI | acc | E P/R/F1 | C P/R/F1 | NM P/R/F1 | JSON | fail | out tok |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | three | **0.802** | [0.771, 0.833] | 0.839 | 0.84/0.91/0.87 | 0.62/0.81/0.70 | 0.92/0.76/0.83 | 100.0% | 0.0% | 10.4 |
| Qwen3-8B | three | **0.786** | [0.754, 0.815] | 0.821 | 0.83/0.89/0.86 | 0.54/0.92/0.68 | 0.93/0.72/0.81 | 100.0% | 0.0% | 10.3 |
| phi-4 | three | **0.783** | [0.751, 0.814] | 0.821 | 0.83/0.91/0.87 | 0.59/0.80/0.68 | 0.89/0.73/0.80 | 100.0% | 0.0% | 12.4 |
| phi-4 | zeros | **0.776** | [0.745, 0.806] | 0.816 | 0.82/0.90/0.86 | 0.53/0.89/0.66 | 0.94/0.70/0.81 | 99.9% | 0.0% | 27.4 |
| Qwen3-14B | zeros | **0.771** | [0.739, 0.799] | 0.812 | 0.84/0.90/0.87 | 0.49/0.94/0.65 | 0.94/0.69/0.79 | 100.0% | 0.0% | 10.3 |
| granite-4.2-8b | three | **0.746** | [0.714, 0.777] | 0.791 | 0.84/0.85/0.84 | 0.46/0.87/0.60 | 0.89/0.72/0.79 | 100.0% | 0.0% | 10.3 |
| granite-4.2-8b | zeros | **0.741** | [0.709, 0.772] | 0.789 | 0.84/0.89/0.86 | 0.46/0.87/0.60 | 0.89/0.66/0.76 | 100.0% | 0.0% | 10.3 |
| Qwen3-8B | zeros | **0.696** | [0.663, 0.727] | 0.738 | 0.76/0.88/0.81 | 0.42/0.96/0.58 | 0.97/0.54/0.69 | 100.0% | 0.0% | 10.2 |
`E` = Entailment, `C` = Contradiction, `NM` = NotMentioned. `out tok` = mean output
tokens (inference-cost proxy). Full per-example outputs in
`/scratch/ibi761/legalai/results/raw__*.jsonl`.

## Ranking (macro-F1, 3-shot)

1. **Qwen3-14B** 0.802
2. **Qwen3-8B** 0.786
3. **phi-4** 0.783
4. granite-4.2-8b 0.746

## The top three are not separable on this dev set

Paired bootstrap, 2000 resamples, same rows for both models in each comparison:

| comparison | diff | 95% CI | verdict |
|---|---|---|---|
| Qwen3-14B vs Qwen3-8B | +0.016 | [-0.008, +0.040] | within noise |
| Qwen3-14B vs phi-4 | +0.019 | [-0.001, +0.038] | within noise |
| Qwen3-8B vs phi-4 | +0.003 | [-0.021, +0.027] | within noise |
| Qwen3-14B vs granite | +0.056 | [+0.033, +0.081] | **separated** |
| Qwen3-8B vs granite | +0.040 | [+0.017, +0.063] | **separated** |
| phi-4 vs granite | +0.037 | [+0.014, +0.061] | **separated** |

Every pair among the top three straddles zero. The single defensible conclusion
is that **Granite is worse than the other three**; the 1.9-point spread across
those three is not a real ordering.

The cause is the dev set, not the models. Macro-F1 weights Contradiction at one
third and dev contains only **89** Contradiction examples, so that third of the
metric is estimated from a small sample and its CI is wide. Separating a
2-point macro-F1 gap here would need roughly 4x the dev set.

## The ranking is NOT stable across conditions

| condition | order |
|---|---|
| zero-shot | phi-4 (0.776) > Qwen3-14B (0.771) > granite (0.741) > **Qwen3-8B (0.696)** |
| 3-shot | **Qwen3-14B (0.802)** > Qwen3-8B (0.786) > phi-4 (0.783) > granite (0.746) |

Qwen3-8B moves from **last to second**. Only Qwen3-14B and Granite hold their
position. Anyone reading the 3-shot ranking as a capability order is reading
prompt-sensitivity.

| model | 0-shot | 3-shot | gain |
|---|---|---|---|
| Qwen3-8B | 0.696 | 0.786 | **+0.090** |
| Qwen3-14B | 0.771 | 0.802 | +0.032 |
| phi-4 | 0.776 | 0.783 | +0.007 |
| granite-4.2-8b | 0.741 | 0.746 | +0.005 |

## The interesting failure: Qwen3-8B cannot abstain zero-shot

Confusion on the 423 NotMentioned rows:

| condition | -> NotMentioned | -> Entailment | -> Contradiction |
|---|---|---|---|
| zero-shot | 228 | 126 | 69 |
| 3-shot | 304 | 80 | 39 |

Zero-shot it forces a verdict on **46%** of clauses that simply do not address
the policy. This is not a reasoning failure: its Contradiction recall is the
highest in the whole screen (0.96 zero-shot, 0.92 3-shot). It has no calibrated
abstention until the exemplars show it that NotMentioned is a permitted answer.

Abstention is the core product capability, so this matters more than its rank.
Three balanced exemplars mostly repair it, and SFT should repair it further, but
it is the model most dependent on being shown the option.

## Commercially weighted diagnostics (3-shot)

| model | Contradiction recall | NotMentioned recall | mean out tok |
|---|---|---|---|
| Qwen3-8B | **0.92** | 0.72 | 10.3 |
| granite-4.2-8b | 0.87 | 0.72 | 10.3 |
| Qwen3-14B | 0.81 | **0.76** | 10.4 |
| phi-4 | 0.80 | 0.73 | 12.4 |

A missed Contradiction auto-clears a conflicting NDA, so recall on that class is
the expensive error. **Qwen3-8B has the best Contradiction recall and Qwen3-14B
the best NotMentioned recall** — the two capabilities trade off against each
other, and they are not held by the same model.

Contradiction precision is 0.46-0.62 everywhere: all four over-predict
Contradiction roughly 1.5-2x its true rate. That is the safe direction for
review (over-flagging costs reviewer time, under-flagging ships a bad NDA), but
it means raw Contradiction counts are not trustworthy before SFT.

## Format quality is not a discriminator

JSON validity is 100% for every model in every condition except one phi-4
zero-shot row (99.9%). **Zero parse failures in 7,824 generations.** phi-4
wraps its JSON in prose or fences (`embedded_json` on 977/978 zero-shot rows)
and pays for it: **27.4 mean output tokens zero-shot vs ~10 for everyone else**,
a 2.7x inference-cost premium that 3-shot removes (12.4).

This kills the "prompting is not enough, we need SFT" argument *on formatting
grounds* — these models already emit clean JSON. The case for SFT has to rest
on the accuracy ceiling (~0.80 macro-F1), on abstention calibration, and on
evidence-span extraction, which no base model can do at all.

## Hardest hypotheses (mean accuracy across all four, 3-shot)

| id | acc | policy |
|---|---|---|
| nda-7 | 68.5% | Receiving Party may share some Confidential Information with some third-parties |
| nda-15 | 68.8% | Agreement shall not grant Receiving Party any right to Confidential Information |
| nda-16 | 69.4% | Receiving Party shall destroy or return some Confidential Information |
| nda-20 | 75.0% | Receiving Party may retain some Confidential Information after return |
| nda-17 | 76.6% | Receiving Party may create a copy of some Confidential Information |

All five are partial-scope or permission policies ("some", "may"). The failure
is consistent across models, so it is a property of the task, and these are
where SFT has the most room.

## Infrastructure

| | |
|---|---|
| Hardware | 1x NVIDIA H100 80GB per cell, 12 CPU, 128G, cuda 12.9 |
| Stack | vLLM 0.25.0, torch 2.11.0, transformers 5.14.1 |
| Generation | 23-73s per 978-row cell |
| Array wall time | 3-9 min per cell including venv build and model load |
| Peak GPU memory | ~74.5 GB every cell |

Peak memory is ~74.5 GB for all four because vLLM preallocates its KV cache at
`gpu_memory_utilization=0.90`. It reflects that setting, **not** model footprint,
and should not be presented as a size comparison. Mean utilization (10-33%) is
likewise dominated by model load: generation itself hits 100% but lasts under a
minute, because 978 short prompts is a tiny batch for an H100.

## Recommendation

**Drop Granite.** It is the only model the data separates, and it is separated
downward against all three others in both conditions.

**The screen cannot pick your top two by macro-F1** — Qwen3-14B, Qwen3-8B and
phi-4 are mutually inseparable. Choose on the tiebreakers instead:

- **Qwen3-14B** — ranks first 3-shot, second zero-shot, best NotMentioned recall,
  most consistent across conditions. The safe pick. ~2x the serving cost of 8B.
- **Qwen3-8B** — best Contradiction recall in the screen and half the parameters,
  so it is the cheapest to fine-tune and serve. But it has the worst zero-shot
  abstention by a wide margin, which is the bet: SFT should fix exactly that.
- **phi-4** — most prompt-robust (+0.007 from exemplars) and best zero-shot, but
  that stability also means the least headroom, and it is 14.7B like the 14B Qwen
  without the Contradiction-recall advantage of the 8B.

If the goal is to learn the most from two parallel fine-tunes, **Qwen3-14B and
Qwen3-8B** is the more informative pair: same family and tokenizer, 2x parameter
difference, and they currently hold opposite strengths (NotMentioned recall vs
Contradiction recall). That isolates model scale, and the post-SFT gap answers
"does 14B earn its serving cost" directly.

Decide on the test split, not this one. Dev's 89 Contradiction examples cannot
resolve 2-point differences; test has 210.
