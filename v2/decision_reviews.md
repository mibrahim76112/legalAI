# v2 decision reviews

Self-adjudication of each v2 decision before it is reported. A judge that never
rules against me is worthless, so weak decisions are rated weak and the test
that would settle them is named.

---

## D1 — Train at `max_seq_len = 16384` rather than 8192 or 32768

**Decision.** Set the training sequence length to 16,384 and *drop* any row
whose gold evidence would not survive left-truncation of the prompt, rather than
training on it with an unsupported target.

**Alternatives considered.**

| option | evidence destroyed | cost |
|---|---|---|
| 8,192 (the v1 setting) | 826 rows / 38.3% | unacceptable corruption |
| 16,384 | 375 rows / 17.4% | fits a 14B LoRA on one H100 |
| 32,768 | 71 rows / 3.3% | 14B LoRA may not fit one H100 |
| window CUAD around the evidence span | 0 | changes the task definition |

**Evidence for.** The measurement is direct: the prompt was rendered through the
production chat template, truncated exactly as the collator truncates it, and
each gold evidence span was checked for survival under the same normalizer used
at scoring time. 826 corrupted rows at 8,192 is not a rounding error, and the
precedent is established — v1 dropped 3 ContractNLI rows for exactly this
defect. Training a model to cite text it cannot see is training it to
hallucinate, which is the failure mode this project exists to reduce.

**Evidence against — and this is the weak part.** The choice of 16,384 over
32,768 rests on a memory claim I have **not measured**: that a 14B LoRA will not
fit one H100 at 32k. What I actually have is a v1 measurement of 49 GB at 8,192
and an extrapolation. 32,768 is the better data choice by a factor of five in
rows preserved. If 14B fits at 32k with gradient checkpointing and FlashAttention,
16,384 is the wrong answer and I picked it on an assumption.

**Second objection.** Dropping 375 rows is not a random 17% sample. The dropped
rows are the *longest* contracts, which are systematically different documents —
credit facilities and merger agreements rather than short services agreements.
The training distribution is therefore shifted toward shorter contracts while
the evaluation distribution is not. This is a selection bias, not a data loss,
and it is the more serious of the two objections because it silently changes
what the model learns.

**Third objection.** Windowing CUAD around the evidence span would preserve
every row and cut the context requirement dramatically. I rejected it because it
changes the task from "read the contract" to "read this excerpt", which requires
the same windowing at inference and makes the CUAD task no longer comparable to
the document-level framing used for ContractNLI. That is a defensible reason,
but it is a *design* reason, not a measurement, and windowing is arguably the
better engineering answer for a clause-extraction task.

**RESOLVED — the assumption was measured and it held.** `v2/probe_memory.py`
ran a real LoRA forward+backward+optimizer step at both lengths on an H100
80GB, with the exact v1 training configuration (bf16, gradient checkpointing,
r=16 over the 7 standard projections, AdamW, batch size 1):

| model | seq 16,384 | seq 32,768 |
|---|---|---|
| Qwen3-14B | **66.8 GiB peak** (71.9 GiB reserved of 79.2) — fits | **OOM** |

So 32,768 is not available for 14B on one card, and 16,384 is correct. Note how
little headroom remains: 71.9 of 79.2 GiB reserved is 91% of the device. A
longer sequence, a larger batch, or a wider LoRA would all push it over.

**Verdict: CONFIRMED, high confidence.** The rejection of 8,192 rests on the
826-row corruption measurement; the choice of 16,384 over 32,768 now rests on a
measured OOM rather than an extrapolation. Both halves are evidence-backed.

**What would still overturn it.** Only a change that frees memory — 8-bit
optimizer states, QLoRA, FSDP or DeepSpeed across two cards, or dropping to
8B-only. Any of those makes 32,768 reachable and would recover 304 training
rows, concentrated in exactly the commercial clause types that matter most.
That is a real avenue, not a closed question, and it is the single highest-value
follow-up if more GPU time becomes available.

**What must be reported regardless.** The per-category breakdown of the 375
dropped rows, so the selection bias is visible rather than buried in an
aggregate percentage.

---

## D2 — Build the two-task dataset now, leave the three-task dataset gated

**Decision.** Assemble ContractNLI + CUAD and proceed, while Task 3 (synthetic
risk notes) stays blocked pending human review of 30 generated samples.

**Evidence for.** The gate exists because risk notes are model-generated and
nobody has yet confirmed they are worth training on. Tasks 1 and 2 are both
human-annotated public benchmarks and need no such approval. Waiting on the gate
to start work that does not depend on it wastes queue time on a contended
cluster, and the user explicitly authorized continuing.

**Evidence against.** A model trained on two tasks is not the deliverable; the
deliverable is three. If the risk notes pass review, the two-task run becomes a
discarded intermediate rather than a result.

**Rebuttal.** It does not become discarded — it becomes the **ablation**. The
two-task model is the control that isolates what the synthetic third task
actually contributes. Without it, any three-task result has no baseline and the
question "did the synthetic data help?" is unanswerable.

**Verdict: SOUND, high confidence.** The work is useful whether or not the gate
opens, which is the property that makes it safe to do while the gate is closed.

**What would overturn it.** Nothing about the risk notes. Only a finding that
the two tasks interfere destructively, which is itself worth knowing.

---

## D3 — Exclude phi-4 on context length

**Decision.** phi-4 is inadmissible because its 16,384 window cannot represent
1,843 CUAD inputs (20.1% of the task).

**Evidence for.** Read from the shipped `config.json`, not a model card. The
coverage figure is computed over the actual built dataset with the actual
tokenizer. The exclusion is a capability claim, not a quality claim, so it does
not require a head-to-head evaluation to support it.

**Evidence against.** One could argue phi-4 should be evaluated on the 79.9% it
*can* read, and that excluding it entirely overstates the case.

**Rebuttal.** A model that silently truncates a fifth of the inputs produces
scores that are not comparable to models that read the whole document. The
comparison would be unfair in phi-4's favour or against it depending on where
the truncation lands, and either way the number would not mean what the table
header says it means.

**Verdict: SOUND, high confidence.** This is the strongest form of exclusion
available — it rests on a measured property of the data rather than on a
contested quality comparison.

**What would overturn it.** Nothing, unless the CUAD task is redefined to use
windowed inputs (see D1's third objection), in which case the context filter
would need recomputing from scratch and phi-4 might re-enter.

---

## D4 — Report Qwen3's context as 40,960, not 131,072

**Decision.** State Qwen3's usable window as the shipped `max_position_embeddings`
of 40,960, not the 131,072 reachable via YaRN.

**Evidence for.** `rope_scaling` is `null` in the shipped config. The 131,072
window requires an explicit configuration change that was not made and was not
evaluated. Claiming it would be claiming a capability the run did not use.

**Note the asymmetry, since it looks inconsistent.** Llama-3.1 is reported at
131,072 even though that is *also* a RoPE extension. The difference is that
Llama ships with `rope_type: llama3` **enabled**, so 131,072 is its default
behaviour, while Qwen3's extension is opt-in. The rule applied is "what the
weights do out of the box", and it is applied consistently.

**Verdict: SOUND, high confidence.**

**What would overturn it.** Enabling YaRN on Qwen3 and evaluating long-context
retrieval at 131,072. If that were done and reported, the larger number would be
the honest one.

---

## D5 — Regenerate the risk notes after finding the teacher was templating

**What was found.** The 40-row risk-note smoke reported **40/40 kept, a 100%
filter pass rate**. That number was meaningless. Measuring the output instead
of trusting the pass rate:

| measure | value |
|---|---|
| notes opening with the identical 6 words | 34 / 40 (85.0%) |
| distinct 4-word openings | 5 / 40 |
| distinct 6-word openings | 6 / 40 |
| notes that were exactly one sentence | 40 / 40 |

Every note was a variation on *"This clause matters to the business because…"*.

**Root cause — two compounding faults, both mine.**

1. The system prompt contained the literal phrase *"explains why this clause
   matters to the business"*. The teacher copied the instruction into the
   answer, which is ordinary instruction-following, not a model defect.
2. Decoding was `temperature=0.0`. One fixed system prompt plus argmax gives
   the same opening on every row by construction. Sampling would have exposed
   the problem immediately.

**Why the filters did not catch it.** All three existing filters are blind to
templating. f1 counts sentences, f2 checks length against the observed p1/p99,
f3 checks that quoted spans occur in the clause. A note can satisfy all three
while being the two-hundredth copy of one sentence stem. Worse, f2 is
**self-calibrating** — bounds taken from the p1/p99 of the very batch being
filtered keep ~98% by definition, so its contribution to the pass rate is close
to tautological. The 100% figure was therefore an artifact of the filters
measuring the wrong thing, not evidence of clean data.

This is the tenth measurement artifact caught in this project before it reached
a conclusion, and the pattern is consistent: **a suspiciously clean number is a
signal to measure something else, not to proceed.**

**Decision.** Regenerate rather than post-filter. Three changes:

- The prompt no longer supplies a stem to copy, and explicitly bans the
  observed opening.
- `temperature=0.7`, `top_p=0.9`, with a **per-request seed** derived from the
  recorded run seed, so variety is restored without losing reproducibility.
- A fourth filter, f4, caps the share of kept notes that may share a 5-word
  opening at 5% (floor 2). Opening diversity is now reported in the manifest
  whether or not f4 fires, so the next reader sees the number rather than
  inferring it from a pass rate.

**Evidence against this decision.** Regenerating costs a GPU run, and one could
argue an 85% templated opening is harmless because the *content* after the stem
still varies and the stem would be learned and then ignored. That argument is
not safe: the downstream task is a free-text risk note, where the model's output
distribution is the product, and a model trained to open every note identically
produces a visibly robotic deliverable.

**Verdict: SOUND, high confidence** on regenerating. **The fix itself is
UNVERIFIED** until the second smoke reports its diversity numbers — the prompt
change is a hypothesis about the teacher's behaviour, not yet a measurement.

**What would overturn it.** The second smoke still showing low opening
diversity. If banning the stem merely moves the teacher to a different fixed
stem, the problem is decoding-level and the answer is higher temperature or a
varied prompt, not a reworded instruction.

---

## D6 — The truncation filter applies to train only, never to dev or test

**What was found.** The first build of the combined dataset applied the
evidence-survival filter to all three splits, dropping 32 dev rows and 74 test
rows alongside the 375 train rows.

**Why that was wrong.** Those dropped rows are the *longest* contracts. Removing
them from the evaluation set curates the benchmark toward documents the model
finds easier and reports an optimistic number. A model that cannot handle an
83,000-token contract must be scored on that contract, not excused from it.

The asymmetry is the point: on **train** an evidence-truncated row is poison,
because its target cites text the model cannot see and it actively teaches
fabrication. On **dev/test** the same row is a legitimate hard case.

**Decision.** Filter train only. Dev and test keep every row. The identities of
the dropped train rows are written to `combined/_dropped_train_rows.json` so the
selection bias can be inspected rather than assumed uniform.

**Verdict: SOUND, high confidence.** This was a bug in my own builder, caught by
reading the build output rather than by a test, which is worth noting as a gap:
there is no assertion anywhere that evaluation splits are unfiltered.

**What would overturn it.** Nothing. The only remaining question is how skewed
the 375 dropped train rows are by category, which is measured separately.
