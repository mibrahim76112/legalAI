# SFT methodology

Every figure in this document is read from a run artifact
(`v2_sft/*/run_meta.json`, SLURM job logs, `v2/_memory_probe*.json`) rather
than transcribed. Dev split only; test is untouched.

---

## 1. Framing — one adapter, three tasks

A single LoRA adapter is trained on a mixture of three tasks rather than three
adapters on three tasks. The deliverable is one contract-review assistant, and
three separate adapters would be three models to serve, version and evaluate.

| | input | output | source |
|---|---|---|---|
| Task 1 compliance verdict | full NDA + policy position | `{"verdict", "evidence"[]}` | ContractNLI, human-annotated |
| Task 2 clause identification | full contract + clause category | `{"present", "evidence"[]}` | CUAD, human-annotated |
| Task 3 risk note | one clause + its category | one sentence | **synthetic**, Qwen3-32B teacher |

Tasks 1 and 2 read a whole contract. Task 3 reads a single clause, which makes
the tasks chain — Task 2 locates a clause, Task 3 explains it — and keeps
Task 3 out of the truncation problem the document-level tasks have.

---

## 2. Stack, and why not the usual one

`transformers.Trainer` + `peft`, not TRL / Axolotl / LLaMA-Factory.

The loss-masking and truncation behaviour is the thing most likely to be wrong
in this project, and it is exactly what a wrapper hides. Writing the sequence
construction directly meant it could be asserted rather than assumed. Inference
is vLLM with `TokensPrompt`, tokenizing ourselves to avoid a double-BOS.

---

## 3. Sequence construction

```
input_ids = prompt(add_generation_prompt=True) + target + eos
labels    = [-100] * len(prompt)              + target + eos
```

Loss is computed on the target only. Three decisions here carry weight:

**Prompt truncated from the left, never the target.** Dropping target tokens
would train the model to emit truncated JSON. The head of a contract is the
least load-bearing region for these tasks.

**Thinking mode off**, detected per model rather than assumed. A template that
ignores an unknown kwarg silently produces the wrong thing, so the detector
renders the template both ways and compares the text; a template whose default
is already non-thinking renders identically and is left alone.

**The mask is asserted, not trusted.** Before any GPU time, the trainer checks
on a real example that every prompt label is `-100` and the target is not
entirely masked, and prints the counts:

```
[mask] ok -- first 3170 labels are -100, 74 supervised tokens
```

This is cheap insurance against the single failure that would invalidate every
number downstream while still producing a plausible-looking loss curve.

---

## 4. Hyperparameters — identical for every model

Held constant so that differences are attributable to the model rather than to
tuning.

| parameter | value |
|---|---|
| LoRA rank *r* | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| learning rate | 1e-4 |
| schedule | cosine, warmup ratio 0.03 |
| epochs | 1 |
| per-device batch size | 1 |
| gradient accumulation | 16 (effective batch 16) |
| precision | bf16 |
| gradient checkpointing | on |
| max_seq_len | 16,384 |
| seed | 42 |
| optimizer | AdamW |

Trainable footprint:

| model | total params | LoRA trainable | % trainable | optimizer steps |
|---|---|---|---|---|
| Qwen3-4B | 4,022,468,096 | 33,030,144 | 0.821% | 947 |
| Qwen3-8B | 8,190,735,360 | 43,646,976 | 0.533% | 947 |
| Llama-3.1-8B | 8,030,261,248 | 41,943,040 | 0.522% | 947 |
| Qwen3-14B | 14,768,307,200 | 64,225,280 | 0.435% | 947 |

One epoch was chosen over more because the adapters already reach their best
eval loss by the end of it and the comparison budget was better spent on more
models than on more epochs.

---

## 5. The sequence-length decision — measured, not defaulted

This is the most consequential technical decision in the project and it was
made twice: once wrongly by inheritance, once correctly by measurement.

v1 trained at 8,192, when ContractNLI was the only task. Keeping that would
have been the default. Measuring what it costs on CUAD:

| max_seq_len | train rows truncated | **gold evidence destroyed** | % of evidence-bearing rows |
|---|---|---|---|
| 8,192 | 1,428 | **826** | **38.3%** |
| 16,384 | 755 | 375 | 17.4% |
| 32,768 | 341 | 71 | 3.3% |

At 8,192, 826 rows would carry targets citing text the model cannot see —
training it to fabricate citations, which is the failure the project exists to
reduce. Those rows are **dropped, not truncated**.

Why not 32,768, which loses five times fewer rows? Because it does not fit.
Measured with a real forward, backward and optimizer step at each length:

| model | peak @ 16,384 | @ 32,768 |
|---|---|---|
| Qwen3-14B | 66.8 GiB (91% of an 80 GB H100 reserved) | **OOM** |
| Qwen3-8B | 52.7 GiB | **OOM** |
| Llama-3.1-8B | 46.8 GiB | **OOM** |
| Qwen3-4B | 43.2 GiB | **OOM** |

Even the 4B fails, because memory is dominated by the 151,936-wide logits
tensor (9.3 GiB per fp32 copy at 16,384) rather than by attention. The route to
32,768 is lower memory per token — chunked cross-entropy, 8-bit optimizer
states, QLoRA, two-card sharding — not a bigger window.

**The cost is recorded, not hidden.** The 375 dropped rows are not a uniform
17%: loss ranges from 5.0% (Uncapped Liability) to 29.0% (Non-Transferable
License), concentrated in 67 of 347 contracts, because it tracks contract
length. Commercial terms buried in long agreements lose most; front-of-contract
boilerplate loses least. Weak per-category results on exclusivity or
non-compete are therefore partly a data-availability artifact.

---

## 6. Train-only filtering

The truncation filter applies to **train only**. Dev and test keep every row.

On train, an evidence-truncated row is poison — its target cites invisible
text. On dev/test the same row is a legitimate hard case. Dropping it from
evaluation would curate the benchmark toward short contracts and report an
optimistic number. This was a bug in the first build (32 dev and 74 test rows
were being dropped) caught by reading the build output.

---

## 7. Mixture

| split | total | ContractNLI | CUAD | risk notes |
|---|---|---|---|---|
| train | 15,138 | 7,188 | 5,871 | 2,079 |
| dev | 2,489 | 1,037 | 1,098 | 354 |
| test | 4,538 | 2,091 | 1,836 | 611 |

CUAD is subsampled to roughly 45% of the ContractNLI+CUAD portion of train;
unsubsampled, 18 categories across 347 contracts would swamp ContractNLI.
Dev and test are not subsampled — a mixture ratio is a training choice, and
applying it to evaluation would report a score on a distribution nobody sees.

---

## 8. What was deliberately left out

v1's trainer carried Contradiction oversampling, per-example class weighting
and reasoning-token weighting. All three were dropped here. They were levers
for a single-task precision/recall trade, and carrying them in would confound
the multi-task question with a rebalancing question. The v2 trainer is a
separate file that **imports** from v1 rather than editing it, since v1 is
frozen; it exists because v1's trainer does `ex["verdict"] = r["verdict"]`
unconditionally and CUAD rows have no verdict field.

---

## 9. Experiments run

| experiment | purpose | outcome |
|---|---|---|
| 4 models × 3-task | model comparison at fixed hyperparameters | Llama-3.1-8B best on Task 1; no separation on Task 2 |
| Qwen3-8B × 2-task | **ablation**: isolate the synthetic task's contribution | third task is close to free |
| memory probe, 2 lengths × 4 models | settle 16,384 vs 32,768 | 32,768 unavailable on one card |
| evidence-survival sweep, 3 lengths | quantify truncation damage | 8,192 disqualified |
| 64-row training smoke | validate the never-run trainer end to end | loss mask confirmed on a real example |

**The ablation is the methodological point.** Qwen3-8B trained with and
without risk notes, same seed, same hyperparameters, scored on the same rows:

| metric | 2-task | 3-task | change |
|---|---|---|---|
| ContractNLI macro-F1 | 0.822 | 0.826 | +0.004 |
| Contradiction precision | 0.594 | 0.616 | +0.022 |
| CUAD macro-F1 present | 0.765 | 0.750 | −0.015 |
| risk note ROUGE-L | 0.105 | 0.451 | +0.346 |

Without this control, "we trained on three tasks and it works" is unfalsifiable.

---

## 10. Training curves

All runs completed a full epoch with no dropped rows and monotone eval loss.

| run | steps | train loss | eval loss (step 50 → final) | wall clock |
|---|---|---|---|---|
| Qwen3-14B 3-task | 947 | 0.1081 | 0.1587 → 0.1245 | 6h 35m |
| Qwen3-8B 3-task | 947 | 0.1256 | 0.1892 → 0.1424 | 4h 26m |
| Llama-3.1-8B 3-task | 947 | 0.1194 | 0.1881 → 0.1435 | 3h 50m |
| Qwen3-4B 3-task | 947 | 0.1372 | 0.2044 → 0.1538 | 3h 29m |
| Qwen3-8B 2-task | 817 | 0.0400 | 0.0479 → 0.0291 | 4h 18m |

**Do not put the 2-task loss on a slide beside the others as if it were
better.** It is scored on a different dev set, one containing no risk notes.
Cross-dataset loss comparison is meaningless; the ablation table in §9 is the
valid comparison.

---

## 11. Decisions a reviewer is likely to challenge

| challenge | answer |
|---|---|
| "Why LoRA and not full fine-tuning?" | 0.4-0.8% trainable parameters, one 80 GB card per model, four models compared in a day. Full FT of a 14B needs sharding and a multiple of the budget. |
| "Why r=16?" | Held fixed across models so the comparison is of models. Not swept — and that is stated as a limitation, not presented as tuned. |
| "Why one epoch?" | Eval loss is still falling but flattening; budget was spent on model coverage instead. |
| "Why 16,384?" | Measured OOM at 32,768 on every candidate, and measured 38.3% evidence corruption at 8,192. Neither number is inherited. |
| "Why is Task 3 synthetic?" | No public human-annotated risk-note corpus exists. Provenance, teacher, licence and filters are documented, and the ceiling is the teacher's ceiling. |
| "Is the model ranking real?" | Paired bootstrap, 2,000 resamples, same rows. Llama separates from 4B/8B on Task 1, not from 14B; nothing separates on Task 2. |

---

## 12. Known limitations

- **One seed.** Every run is seed 42. Seed variance is unmeasured, so small
  gaps should be read against the paired bootstrap, not the point estimates.
- **No hyperparameter sweep.** Fixed settings make the model comparison clean
  but mean no model is individually tuned.
- **Train/inference length mismatch.** Models are trained at 16,384 but CUAD
  inputs reach 83,679 tokens. Generalisation past the training length is not
  free, and a windowed-inference experiment is testing how much of that loss is
  recoverable without retraining.
- **Task 3 has no ground truth.** Its metric is imitation fidelity against the
  teacher, never accuracy.
