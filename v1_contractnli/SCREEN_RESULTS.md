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
grounds* — these models already emit clean JSON. Drop it. The case for SFT
rests instead on, strongest first:

1. **Evidence spans** — no base model can produce them at all. This makes the
   span rebuild load-bearing rather than optional.
2. **Abstention calibration** — Qwen3-8B forces a verdict on 46% of silent
   clauses zero-shot; quantify how much SFT improves NotMentioned recall.
3. **The ~0.80 accuracy ceiling** — base models plateau there even with exemplars.
4. **Per-client playbook adaptation** — the generalization story.

A metric that kills one of your own arguments is a working metric; collecting
and reporting it is itself evidence the evaluation is real.

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

Granite was included on the strength of its published IFEval (87.06), the best
of the four, and it finished last. Worth stating plainly in the deck: *a
candidate selected on a published proxy benchmark finished last on our task;
public benchmark scores did not transfer.* That is evidence for measuring on
your own data, and it is more persuasive because it cost us a candidate.

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

**Do not select the winner on test.** An earlier draft of this document advised
deciding on test because it has 210 Contradiction examples rather than 89. That
was wrong and is retracted: choosing the winner on test and then reporting that
same number is selection contamination, reporting the max of two models over the
set it is reported against.

Correct protocol: fine-tune both, **select on post-SFT dev, report both models
on test**, and recommend on cost/performance grounds. "The 8B is within X points
of the 14B at half the serving footprint" is both statistically clean and a
stronger argument than naming one winner and burying the other run.

---

# Verification runs (27 extra cells)

Three decisions flagged **WEAK** during review were tested rather
than argued. One of them changes a recommendation.

## W2. Exemplar choice swamps the gaps the ranking rested on

4 alternative balanced triples (random draws, fixed seeds, same constraints as
the headline triple) x 4 models, 3-shot. With the headline triple that is 5
observations per model.

| model | base | t1 | t2 | t3 | t4 | mean | sd | range |
|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | 0.802 | 0.792 | 0.757 | 0.785 | 0.813 | **0.790** | 0.021 | 0.056 |
| phi-4 | 0.783 | 0.800 | 0.771 | 0.785 | 0.779 | **0.784** | 0.010 | **0.028** |
| Qwen3-8B | 0.786 | 0.774 | 0.734 | 0.765 | 0.765 | **0.765** | 0.019 | 0.052 |
| granite-4.2-8b | 0.746 | 0.732 | 0.641 | 0.679 | 0.738 | **0.707** | 0.045 | 0.105 |

**The concern was justified.** Largest per-model spread from exemplar choice
alone is **0.105**. The published 3-shot gaps were **0.016** and **0.003**. Both
are swamped several times over.

**The order is not stable.** Five triples produced **three distinct orders**.
phi-4 leads in 3 of 5, Qwen3-14B in 2 of 5.

**The specific correction that matters.** Qwen3-8B placed 2nd in the published
ranking. Across the four alternative triples it is **3rd every time**. phi-4
beats it in **4 of 5** triples and is separated in 2:

| triple | phi-4 minus Qwen3-8B | 95% CI | |
|---|---|---|---|
| base | -0.003 | [-0.027, +0.021] | within noise |
| t1 | +0.026 | [+0.004, +0.048] | **phi-4 separated** |
| t2 | +0.038 | [+0.011, +0.063] | **phi-4 separated** |
| t3 | +0.020 | [-0.001, +0.042] | within noise |
| t4 | +0.015 | [-0.008, +0.038] | within noise |

The headline triple was the **only** one where Qwen3-8B edged phi-4, and by
0.003 — inside noise. **Its 2nd place was an artefact of my exemplar choice.**

**Consequence for the published table.** The bootstrap CIs in the results table
capture example sampling only. Total uncertainty is larger. Use the mean across
triples as the more robust estimate, and treat exemplar variance as a real term
in the error budget when comparing base to fine-tuned.

**Secondary finding.** Exemplar robustness is itself a model property worth
reporting: phi-4 varies by 0.028 across triples, Granite by 0.105. A model whose
score swings 10 points on which three examples you picked is a deployment risk
in a prompt-only configuration.

## W1. Granite's exclusion is not a version artefact

Granite 4.1-8b, both conditions, headline triple:

| model | zero-shot | 3-shot | 3-shot C-recall |
|---|---|---|---|
| granite-4.1-8b | 0.698 | 0.739 | 0.899 |
| granite-4.2-8b | 0.741 | 0.746 | 0.865 |

4.1-8b is **worse**, clearly so zero-shot. Both remain separated from the
leaders (vs Qwen3-14B: +0.063 [+0.038, +0.087]; vs phi-4: +0.044 [+0.021,
+0.067]).

**Verdict: "drop Granite" holds**, and my version choice was the one *favourable*
to Granite. The "published benchmarks did not transfer" finding also survives —
it is not an artefact of testing a fresh point release.

## W3. The thinking-mode control was correct, and is now measured

Qwen3-8B, 3-shot, thinking left ON with the cap raised to 1024 tokens:

| | macro-F1 | acc | C-recall | NM-recall | mean out tok | truncated | parse fail |
|---|---|---|---|---|---|---|---|
| thinking OFF (as screened) | **0.786** | 0.821 | 0.921 | 0.719 | 10.3 | 0 | 0 |
| thinking ON | 0.747 | 0.786 | 0.787 | 0.757 | **416.4** | 89 | 3 |

Thinking mode is **worse by 0.039 [-0.064, -0.014], separated**, and costs
**40x the output tokens**. Even at a 1024-token cap, 89 rows still truncated.

So the fairness control did not disadvantage the models it touched; it helped
them, and the asymmetry with phi-4 (which has no such mode) did not bias the
comparison against phi-4. Note thinking *did* improve NotMentioned recall
(0.757 vs 0.719) while hurting Contradiction recall badly (0.787 vs 0.921) —
worth remembering if abstention becomes the binding constraint post-SFT.

## Revised recommendation

Ranking by **mean macro-F1 across 5 triples**, which is the more defensible
estimate:

| model | mean | range | note |
|---|---|---|---|
| Qwen3-14B | 0.790 | 0.056 | leads on mean |
| phi-4 | 0.784 | 0.028 | most exemplar-robust |
| Qwen3-8B | 0.765 | 0.052 | **3rd, not 2nd** |
| granite-4.2-8b | 0.707 | 0.105 | separated, and most fragile |

**Drop Granite** — unchanged, now confirmed across two point releases.

**The fine-tuning pair is now a real trade-off, not a ranking readout.**
Qwen3-8B is measurably behind phi-4, so "take the top two" no longer selects it.
Two defensible pairs:

- **Qwen3-14B + Qwen3-8B** (still recommended). Pairing the two ~14B models
  answers no cost question — both sit in the same serving class. The
  commercially decisive question is whether the *cheap* model can be made good
  enough, and Qwen3-8B is the cheap candidate with the screen's **best
  Contradiction recall (0.921)**. Its weakness is abstention calibration, which
  is precisely what SFT on balanced data is most likely to repair. The
  justification changes though: not "it ranked 2nd" (it did not), but "it is the
  cheap serving candidate whose one weakness is the one SFT should fix."
- **Qwen3-14B + phi-4** if you would rather fine-tune the two strongest and most
  robust base models. Cleaner on current evidence, but confounds family with
  scale, gives you two models in the same cost class, and phi-4's 16K context
  constrains the full-document roadmap.

**Fallback, stated in advance:** if the Qwen3-8B fine-tune underperforms its base
by more than the exemplar spread, phi-4 is the replacement, on the strength of
being 2nd on mean and 1st on robustness.
