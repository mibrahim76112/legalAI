# Three-task evaluation — finished models

All numbers on the **dev** split. The test split has not been touched and is not reported anywhere in this file.

One LoRA adapter per model, trained on the combined three-task dataset at max_seq_len 16,384, one epoch, identical hyperparameters and seed 42 for every model so the comparison is of models rather than of tuning.

Greedy decoding, max 1,024 new tokens, thinking mode off.

## Runs

| model | dataset | adapter | rows | wall clock | sec/example |
|---|---|---|---|---|---|
| Qwen3-4B | 3-task | `Qwen__Qwen3-4B__3task` | 2489 | 101 min | 2.42 |
| Qwen3-8B | 3-task | `Qwen__Qwen3-8B__3task` | 2489 | 104 min | 2.52 |
| Llama-3.1-8B | 3-task | `NousResearch__Meta-Llama-3.1-8B-Instruct__3task` | 2489 | 86 min | 2.06 |
| Qwen3-14B | 3-task | `Qwen__Qwen3-14B__3task` | 2489 | 149 min | 3.60 |
| Qwen3-8B | 2-task control | `Qwen__Qwen3-8B__2task` | 2489 | 102 min | 2.46 |

**Read the 2-task control as a control, not a competitor.** It was trained without risk notes and is scored here on the same dev file, so its Task 3 numbers show what an untrained model produces. Its Task 1 and Task 2 numbers are the meaningful comparison.


## Task 1 — compliance verdict (ContractNLI)

n = 1037 (contract, policy position) pairs. Three classes. Primary metric macro-F1, which weights the rare Contradiction class equally rather than letting the majority classes carry the score.

| model | dataset | macro-F1 | 95% CI | accuracy | schema valid | unparseable |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 3-task | **0.847** | [0.817, 0.874] | 0.889 | 0.998 | 0 |
| Qwen3-14B | 3-task | **0.837** | [0.806, 0.865] | 0.884 | 1.000 | 0 |
| Qwen3-8B | 3-task | **0.826** | [0.795, 0.853] | 0.872 | 1.000 | 0 |
| Qwen3-4B | 3-task | **0.824** | [0.793, 0.852] | 0.869 | 0.997 | 0 |
| Qwen3-8B | 2-task control | **0.822** | [0.791, 0.850] | 0.873 | 1.000 | 0 |

### Per class

| model | Entailment P/R/F1 | Contradiction P/R/F1 | NotMentioned P/R/F1 |
|---|---|---|---|
| Qwen3-4B (3-task) | 0.90/0.92/0.91 | 0.62/0.80/0.70 | 0.90/0.83/0.86 |
| Qwen3-8B (3-task) | 0.91/0.92/0.91 | 0.62/0.81/0.70 | 0.90/0.83/0.86 |
| Llama-3.1-8B (3-task) | 0.92/0.94/0.93 | 0.67/0.81/0.73 | 0.92/0.85/0.88 |
| Qwen3-14B (3-task) | 0.92/0.93/0.92 | 0.64/0.79/0.71 | 0.92/0.84/0.88 |
| Qwen3-8B (2-task control) | 0.91/0.92/0.92 | 0.59/0.80/0.68 | 0.91/0.83/0.87 |

### The metric that mattered — Contradiction precision

The pre-SFT screen established that recall was adequate everywhere (0.64-0.84) while precision sat at 0.28-0.58: these models over-flag conflicts, and a reviewer handed 200 escalations of which 90 are real stops trusting the flags. Precision was therefore the metric SFT had to move. It did.

| model | Contra precision before SFT | after SFT | change | recall after |
|---|---|---|---|---|
| Qwen3-4B | 0.349 | **0.623** | +0.274 | 0.800 |
| Qwen3-8B | 0.282 | **0.616** | +0.334 | 0.811 |
| Llama-3.1-8B | 0.329 | **0.670** | +0.341 | 0.811 |
| Qwen3-14B | 0.399 | **0.641** | +0.242 | 0.789 |

Precision roughly doubled on every model while recall held or improved. That is the cry-wolf rate falling, not a general score drifting upward.


### Evidence extraction

Strict = every gold span recovered as a contiguous normalized substring.

| model | strict F1 | precision | recall | partial coverage |
|---|---|---|---|---|
| Qwen3-4B (3-task) | 0.731 | 0.843 | 0.645 | 0.763 |
| Qwen3-8B (3-task) | 0.739 | 0.848 | 0.655 | 0.768 |
| Llama-3.1-8B (3-task) | 0.747 | 0.862 | 0.660 | 0.774 |
| Qwen3-14B (3-task) | 0.763 | 0.864 | 0.682 | 0.788 |
| Qwen3-8B (2-task control) | 0.753 | 0.850 | 0.676 | 0.783 |

### Is the Task 1 ranking real? Paired bootstrap

Same dev rows for both models, 2000 resamples, percentile CI on the macro-F1 difference. A CI spanning zero means the gap is not separable from noise.

| comparison | mean diff | 95% CI | P(A better) | separated? |
|---|---|---|---|---|
| Llama-3.1-8B vs Qwen3-4B (3-task) | +0.0230 | [+0.0068, +0.0403] | 0.995 | YES |
| Llama-3.1-8B vs Qwen3-8B (3-task) | +0.0208 | [+0.0016, +0.0399] | 0.985 | YES |
| Llama-3.1-8B vs Qwen3-14B (3-task) | +0.0103 | [-0.0084, +0.0300] | 0.858 | no |
| Llama-3.1-8B vs Qwen3-8B (2-task control) | +0.0247 | [+0.0058, +0.0445] | 0.998 | YES |


## Task 2 — clause identification (CUAD)

n = 1098 (contract, category) pairs across 18 categories. Present rate 33.3%.

**Primary metric is macro-F1 on the PRESENT class, averaged over categories.** Pooled accuracy is reported only beside its trivial baseline: a constant-absent predictor scores 66.7%, so an accuracy of 0.88 is a gain of about 21 points over guessing, not 88 points of skill.

| model | dataset | **macro-F1 present** | pooled F1 | precision | recall | pooled acc | trivial acc | unparseable |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | 2-task control | **0.765** | 0.817 | 0.915 | 0.738 | 0.890 | 0.667 | 10 |
| Llama-3.1-8B | 3-task | **0.764** | 0.808 | 0.905 | 0.730 | 0.884 | 0.667 | 11 |
| Qwen3-4B | 3-task | **0.758** | 0.799 | 0.879 | 0.732 | 0.877 | 0.667 | 17 |
| Qwen3-14B | 3-task | **0.754** | 0.802 | 0.898 | 0.724 | 0.881 | 0.667 | 17 |
| Qwen3-8B | 3-task | **0.750** | 0.792 | 0.891 | 0.713 | 0.875 | 0.667 | 16 |

| model | evidence strict F1 | precision | recall | partial coverage |
|---|---|---|---|---|
| Qwen3-4B (3-task) | 0.562 | 1.000 | 0.391 | 0.502 |
| Qwen3-8B (3-task) | 0.590 | 1.000 | 0.418 | 0.520 |
| Llama-3.1-8B (3-task) | 0.614 | 1.000 | 0.443 | 0.547 |
| Qwen3-14B (3-task) | 0.600 | 1.000 | 0.429 | 0.536 |
| Qwen3-8B (2-task control) | 0.595 | 1.000 | 0.423 | 0.543 |

Evidence recall here is capped at **87.65%** by truncation: 92 of 745 gold spans fall outside the 15,360-token input budget and cannot be quoted. That ceiling is not a model failure and is being addressed by the windowed-inference experiment.


### Per category — best model per row is bolded

| category | support present | Qwen3-4B | Qwen3-8B | Llama-3.1-8B | Qwen3-14B | Qwen3-8B (2t) |
|---|---|---|---|---|---|---|
| Anti-Assignment | 44 | **0.966** | 0.953 | 0.953 | 0.952 | **0.966** |
| Audit Rights | 28 | 0.880 | 0.885 | 0.906 | **0.943** | 0.923 |
| Cap On Liability | 29 | 0.881 | 0.881 | **0.947** | 0.929 | 0.929 |
| Change Of Control | 16 | 0.733 | **0.786** | 0.741 | 0.692 | 0.750 |
| Covenant Not To Sue | 10 | 0.625 | 0.625 | **0.667** | 0.533 | 0.571 |
| Exclusivity | 20 | 0.765 | 0.727 | 0.800 | **0.824** | **0.824** |
| Insurance | 20 | **0.824** | **0.824** | **0.824** | **0.824** | 0.788 |
| Ip Ownership Assignment | 13 | 0.700 | 0.727 | 0.667 | 0.600 | **0.762** |
| License Grant | 29 | 0.862 | 0.873 | 0.893 | 0.873 | **0.929** |
| Minimum Commitment | 20 | 0.516 | 0.516 | 0.588 | 0.485 | **0.629** |
| Non-Compete | 15 | **0.667** | 0.636 | 0.571 | 0.636 | 0.500 |
| Non-Transferable License | 12 | 0.720 | 0.692 | **0.833** | 0.720 | 0.762 |
| Notice Period To Terminate Renewal | 14 | 0.833 | **0.846** | 0.833 | 0.800 | 0.833 |
| Post-Termination Services | 21 | 0.722 | 0.562 | 0.611 | **0.737** | 0.588 |
| Renewal Term | 24 | 0.818 | **0.837** | 0.810 | 0.783 | 0.826 |
| Revenue/Profit Sharing | 20 | **0.833** | 0.788 | 0.722 | 0.778 | 0.757 |
| Termination For Convenience | 21 | 0.791 | 0.810 | 0.878 | 0.895 | **0.930** |
| Uncapped Liability | 10 | 0.500 | 0.526 | 0.500 | **0.571** | 0.500 |

### Is the Task 2 ranking real? Paired bootstrap

| comparison | mean diff | 95% CI | P(A better) | separated? |
|---|---|---|---|---|
| Llama-3.1-8B vs Qwen3-4B (3-task) | +0.0070 | [-0.0372, +0.0504] | 0.623 | no |
| Llama-3.1-8B vs Qwen3-8B (3-task) | +0.0148 | [-0.0254, +0.0534] | 0.769 | no |
| Llama-3.1-8B vs Qwen3-14B (3-task) | +0.0105 | [-0.0287, +0.0498] | 0.697 | no |
| Llama-3.1-8B vs Qwen3-8B (2-task control) | +0.0000 | [-0.0432, +0.0439] | 0.502 | no |


## Task 3 — risk note (synthetic)

**This measures imitation fidelity against the Qwen3-32B teacher, not correctness.** There is no human ground truth. A high ROUGE-L means the model writes what the teacher wrote; whether the teacher was right is a separate question that 30 human-reviewed samples in `RISK_NOTE_SAMPLES.md` exist to probe. Do not present these as accuracy.

| model | dataset | ROUGE-L mean | median | one-sentence rate | mean words | ungrounded quotes |
|---|---|---|---|---|---|---|
| Qwen3-4B | 3-task | 0.420 | 0.400 | 0.997 | 33.0 | 1 |
| Qwen3-8B | 3-task | 0.451 | 0.426 | 1.000 | 33.6 | 0 |
| Llama-3.1-8B | 3-task | 0.441 | 0.405 | 1.000 | 34.0 | 1 |
| Qwen3-14B | 3-task | 0.460 | 0.444 | 1.000 | 33.0 | 0 |
| Qwen3-8B | 2-task control | 0.105 | 0.117 | 0.983 | 46.8 | 345 |

The 2-task control scores 0.105 against 0.42-0.46 for the trained models. That gap is the cleanest evidence in this report that the third task was actually learned rather than picked up incidentally.


## The ablation — did the synthetic task cost anything?

Qwen3-8B trained with and without risk notes, same seed, same hyperparameters, scored on the same dev rows.

| metric | 2-task | 3-task | change |
|---|---|---|---|
| ContractNLI macro-F1 | 0.822 | 0.826 | +0.004 |
| ContractNLI Contra precision | 0.594 | 0.616 | +0.022 |
| CUAD macro-F1 present | 0.765 | 0.750 | -0.015 |
| CUAD evidence F1 | 0.595 | 0.590 | -0.005 |
| risk note ROUGE-L | 0.105 | 0.451 | +0.346 |

Adding the third task moves the other two by hundredths in both directions while the third task itself goes from unusable to usable. On this evidence the synthetic task is close to free — but note this is a single seed on one model size, so it bounds the effect rather than proving there is none.


## What these numbers do not say

- **No combined score.** The three tasks are measured in different units against different baselines. Averaging them would produce a number that moves for reasons nobody can attribute.
- **Dev only.** Test is untouched. Selecting a model on dev and then reporting its dev score as the headline would contaminate the number.
- **Task 2 evidence recall is capped at 87.65%** by input truncation, not by model skill.
- **Task 2 per-category results are confounded by training-data loss.** 375 training rows were dropped because their evidence fell outside the 16,384-token window, and the loss ranges from 5.0% to 29.0% by category. The heavy losers — exclusivity, non-compete, minimum commitment, change of control — are under-trained for a data reason, so a weak score there is not purely a model verdict.
- **Task 3 has no ground truth.** Its ceiling is the teacher's ceiling.
