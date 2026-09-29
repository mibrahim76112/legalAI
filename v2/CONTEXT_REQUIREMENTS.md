# Context-length requirement for the three-task contract assistant

Purpose: establish, from measurement rather than assumption, the minimum
context window a candidate model must support. This is a **hard filter** applied
before any quality comparison — a model that cannot represent the input cannot
be evaluated fairly on it.

## Method

Every row of both built datasets was rendered through the exact production path:
`apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` for
the prompt, and the gold target tokenized with `add_special_tokens=False`.
Tokenizer: `Qwen/Qwen3-4B`. Two quantities per row:

- **prefill** — prompt tokens, what the model must attend over before emitting
- **decode** — gold target tokens, what it must emit
- **total** — prefill + decode, the window the run actually occupies

Decode is measured from the gold target, so it is a *lower* bound on what a
model needs at inference; a verbose model needs more. Counts are recorded in
`_context_measurements.json`.

## Measured distributions (tokens)

| task | rows | prefill p50 | prefill p99 | prefill max | decode p50 | decode p99 | decode max | **total p99** | **total max** |
|---|---|---|---|---|---|---|---|---|---|
| ContractNLI | 10,316 | 2,045 | 6,510 | 11,673 | 55 | 302 | 601 | 6,663 | 11,885 |
| CUAD | 9,180 | 6,956 | 59,169 | 82,451 | 11 | 697 | 3,431 | 59,404 | 83,679 |

The two tasks have **very different shapes**, and that is the whole finding.
ContractNLI feeds one NDA against one policy position: short, tightly bounded,
p99 under 7k. CUAD feeds whole commercial contracts — merger agreements, credit
facilities — and its p99 is **nine times** ContractNLI's. A context budget set on
Task 1 silently destroys Task 2.

Decode is small for both, so context is a **prefill problem**. The single
outlier is CUAD decode max 3,431, where a category matches many clauses and the
target enumerates them all.

## Coverage: what fraction of rows fit

| context window | ContractNLI | CUAD | pooled |
|---|---|---|---|
| 4,096 | 89.0% | 30.4% | 61.4% |
| 8,192 | 99.7% | 55.5% | 78.9% |
| 16,384 | 100.0% | 79.9% | 90.6% |
| 32,768 | 100.0% | 93.0% | 96.7% |
| 65,536 | 100.0% | **99.8%** | 99.9% |
| 131,072 | 100.0% | 100.0% | 100.0% |

## Consequence for model selection

**Minimum admissible context window: 32,768.** Below that a candidate cannot
represent 7% or more of CUAD inputs even in principle, and at 8,192 — a common
window for older 7B-class models — nearly **half of CUAD does not fit**.

A model advertising 4k or 8k is therefore excluded on capability, not on
quality. This is the defensible form of the argument: the exclusion rests on a
measured property of the data, so it survives the question "how do you know it
wouldn't have been better?"

Two cautions on reading the advertised number:

1. Several models advertise a long window reachable only through RoPE scaling
   (YaRN) that is **off by default**. The usable-out-of-the-box window is what
   matters unless scaling is explicitly enabled and evaluated.
2. Advertised context is an *architectural* limit, not a competence claim.
   Fitting 60k tokens and retrieving a clause from the middle of them are
   different things. This filter establishes admissibility only.

## Consequence for training — the more damaging finding

Truncation at training time is not merely lossy, it is **corrupting**. Both
tasks ask the model to quote evidence spans from the contract. When the prompt
is truncated from the left but the target is preserved, the target can cite text
that is no longer present. The model is then explicitly trained to produce
citations it cannot support — that is, trained to hallucinate.

Measured over the 2,155 CUAD train rows that carry evidence:

| max_seq_len | rows truncated | **evidence destroyed** | % of evidence-bearing rows |
|---|---|---|---|
| 8,192 | 1,428 | **826** | **38.3%** |
| 16,384 | 755 | 375 | 17.4% |
| 32,768 | 341 | 71 | 3.3% |

At 8,192 — the v1 setting, chosen when the only task was ContractNLI — more than
a third of CUAD's positive rows would carry unsupported targets. In v1 this same
defect appeared at a scale of 3 rows and those rows were dropped. Here it is 826.

**Decision: train at `max_seq_len = 16384`, and drop rather than truncate any
row whose evidence does not survive.** The reasoning:

- 8,192 is disqualified outright; the corruption is not a rounding error.
- 32,768 is the better *data* choice (71 rows lost instead of 375) but a 14B
  LoRA does not fit one H100 at 32k. The v1 14B run peaked at 49 GB at 8,192,
  and activation memory grows with sequence length.
- 16,384 keeps 82.6% of evidence-bearing CUAD rows intact, fits all four
  candidate sizes on a single card, and costs 375 rows — a loss that is
  **reported, not hidden**.

Note the asymmetry: 16,384 is the *training* budget, constrained by GPU memory.
32,768 is the *inference* admissibility floor, constrained by the data. They are
different numbers answering different questions, and a model trained on
16k-truncated inputs will need evaluation at longer context to be used on the
full CUAD distribution. That limitation is recorded here rather than discovered
later.

## Applying the filter to the candidate pool

Native context read from each model's local `config.json`
(`max_position_embeddings`, and `rope_scaling` where present) — not from a
model card, so the number is the one the weights actually carry.

| model | native context | ContractNLI covered | CUAD covered | CUAD rows that do not fit | admissible |
|---|---|---|---|---|---|
| microsoft/phi-4 | 16,384 | 100.0% | 79.9% | 1,843 | **NO** |
| Qwen3-4B / 8B / 14B / 32B | 40,960 | 100.0% | 96.5% | 324 | yes |
| mistralai/Mixtral-8x22B-Instruct | 65,536 | 100.0% | 99.8% | 21 | yes |
| NousResearch/Meta-Llama-3.1-8B-Instruct | 131,072 | 100.0% | 100.0% | 0 | yes |
| ibm-granite/granite-4.1-8b, 4.2-8b | 131,072 | 100.0% | 100.0% | 0 | yes |
| google/gemma-4-12B-it | 262,144 | 100.0% | 100.0% | 0 | yes |

**phi-4 is excluded on capability.** At 16,384 it cannot represent 1,843 CUAD
inputs — a fifth of the task — regardless of how good it is at the ones it can
read. This is a cleaner exclusion than any quality argument, and it happens to
confirm a decision already taken for a different reason: phi-4 was dropped as
the reasoning validator in favour of Qwen3-32B.

**Qwen3's 40,960 is a real if minor limitation** and should be stated rather
than glossed. It clears the 32,768 floor but leaves 324 CUAD rows (3.5%)
unrepresentable, where Llama, granite and gemma leave none. Qwen3 publishes a
YaRN path to 131,072, but `rope_scaling` is `null` in the shipped config, so
that window is not available without an explicit configuration change and a
separate evaluation. The honest claim is 40,960.

Llama-3.1's 131,072 is itself a RoPE extension — `rope_type: llama3`, factor 8
over an original 8,192 — but unlike Qwen3's it **is enabled in the shipped
config**, so it is what the model does by default.

## Summary of the two numbers

| question | answer | set by |
|---|---|---|
| minimum context a candidate model must support | **32,768** | CUAD input length (93% coverage floor) |
| sequence length to train at | **16,384** | H100 memory for a 14B LoRA |

These differ, and the gap is a known limitation of this run rather than an
oversight: a model fine-tuned on 16k-truncated inputs is being asked at
inference to handle inputs up to 83,679 tokens. Generalization past the training
length is not free. The mitigation applied here is to drop rather than corrupt
the 375 training rows whose evidence would not survive 16,384, so that whatever
the model learns about citation is at least learned from supportable examples.

## The 375 dropped rows are not a uniform 17% — they are a biased 17%

Dropping evidence-truncated rows was presented above as costing "17.4% of
evidence-bearing CUAD rows". Stated that way it sounds like random attrition.
It is not. Broken down by category:

| category | present rows | dropped | % lost |
|---|---|---|---|
| Non-Transferable License | 100 | 29 | **29.0%** |
| Non-Compete | 83 | 23 | 27.7% |
| Minimum Commitment | 106 | 29 | 27.4% |
| Exclusivity | 121 | 33 | 27.3% |
| Revenue/Profit Sharing | 115 | 31 | 27.0% |
| Audit Rights | 143 | 38 | 26.6% |
| Change Of Control | 76 | 19 | 25.0% |
| Ip Ownership Assignment | 89 | 22 | 24.7% |
| License Grant | 174 | 43 | 24.7% |
| Post-Termination Services | 125 | 18 | 14.4% |
| Insurance | 113 | 14 | 12.4% |
| Renewal Term | 116 | 13 | 11.2% |
| Covenant Not To Sue | 73 | 8 | 11.0% |
| Termination For Convenience | 125 | 12 | 9.6% |
| Notice Period To Terminate Renewal | 74 | 7 | 9.5% |
| Cap On Liability | 188 | 16 | 8.5% |
| Anti-Assignment | 254 | 16 | 6.3% |
| Uncapped Liability | 80 | 4 | **5.0%** |
| **total** | **2,155** | **375** | **17.4%** |

The spread is **roughly six-fold**, from 5.0% to 29.0%, and it is not random —
it tracks where in a contract a clause type lives. The heavy losers are
commercial and licensing terms that sit deep in long negotiated agreements.
The light losers are boilerplate that appears near the front of contracts of
every length: anti-assignment, liability caps, uncapped liability.

The drops also **concentrate**: only 67 of 347 train contracts (19.3%) lose any
row at all. One fifth of the contracts absorb the entire loss, which is the
signature of a length effect rather than a content effect.

Consequence: training at 16,384 systematically under-trains exactly the clause
types a commercial reviewer most wants found — exclusivity, non-compete,
minimum commitment, change of control. This does not invalidate the choice,
since the alternative at 8,192 was to train those same categories on fabricated
citations, but it must be reported alongside any per-category result. A weak
score on Exclusivity will be partly a data-availability artifact, not purely a
model failure.

This is the strongest argument for retraining at 32,768 if the memory probe
shows it fits.

## Measured GPU memory — why 16,384 and not 32,768

`v2/probe_memory.py` runs a real LoRA forward, backward and optimizer step at
each length on one H100 80GB (79.2 GiB usable), using the exact v1 training
configuration: bf16, gradient checkpointing on, r=16 over the seven standard
projections, AdamW, per-device batch size 1.

| model | peak @ 16,384 | reserved @ 16,384 | @ 32,768 |
|---|---|---|---|
| Qwen3-14B | 66.8 GiB | 71.9 GiB (91% of device) | **OOM** |
| Qwen3-8B | 52.7 GiB | 56.8 GiB | **OOM** |
| Llama-3.1-8B-Instruct | 46.8 GiB | 50.7 GiB | **OOM** |
| Qwen3-4B | 43.2 GiB | 46.1 GiB | **OOM** |

**32,768 is unavailable to every candidate on a single card**, including the 4B.
That is more emphatic than expected and the reason is not attention — it is the
language-model head. At vocabulary 151,936 the logits tensor alone is about
10 GiB at 16,384 tokens once cross-entropy upcasts it, and cross-entropy
materializes further copies. Doubling the sequence doubles that term before any
activation is counted, which is why even a 4B model with 8 GiB of weights
cannot reach 32,768 this way.

Two honest caveats on these numbers:

1. The probes ran sequentially in one process. Allocator fragmentation
   accumulates across model loads, so the **later** probes are the least
   trustworthy. The decisive measurement — Qwen3-14B — ran first and second,
   before any fragmentation could build up. Qwen3-4B's OOM ran last and is the
   one most likely to be pessimistic.
2. The probe uses random token ids at exactly the target length, which is the
   worst case. Real batches are shorter on average, so steady-state training
   memory will sit below these peaks. The peaks are what matter, since one long
   batch is enough to kill a run.

Neither caveat changes the decision. Even if 4B could reach 32,768 in isolation,
training different models at different sequence lengths would confound the
model comparison with a context-budget comparison. **All four models train at
16,384.**

The route to 32,768 is not a longer context window but lower memory per token:
a chunked or fused cross-entropy that never materializes the full logits
tensor, 8-bit optimizer states, QLoRA, or sharding across two cards. Any of
those would recover the 375 dropped rows. That is the highest-value follow-up
if more GPU time becomes available, and it is a known path rather than an open
research question.
