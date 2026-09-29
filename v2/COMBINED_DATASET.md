# The combined three-task dataset

One LoRA adapter, three tasks, one sequence format. Built by
`v2/build_combined.py`, seed 20260919, `max_seq_len` 16,384.

## Composition

| split | total | ContractNLI | CUAD | risk notes | dropped |
|---|---|---|---|---|---|
| train | 15,138 | 7,188 | 5,871 | 2,079 | 375 |
| dev | 2,489 | 1,037 | 1,098 | 354 | 0 |
| test | 4,538 | 2,091 | 1,836 | 611 | 0 |

A two-task control set is built by the same script with `--risk-notes` omitted,
at `v2/combined_2task/` (13,059 / 2,135 / 3,927). It exists so the question
"did the synthetic task help, hurt, or do nothing?" has an answer instead of a
guess. Without it, a three-task result has no baseline.

## The three tasks

| | input | output | source |
|---|---|---|---|
| 1. compliance verdict | full NDA + policy position | verdict + evidence spans | ContractNLI (human-annotated) |
| 2. clause identification | full contract + clause category | evidence spans or none | CUAD (human-annotated) |
| 3. risk note | one clause + its category | one sentence | **synthetic**, Qwen3-32B |

Tasks 1 and 2 read a whole contract. Task 3 reads a single clause, by design:
it makes the tasks chain — Task 2 locates a clause, Task 3 explains it — and it
keeps Task 3 out of the truncation problem the document-level tasks have.

## Decisions that shaped the file, and why

**Mixture ratio — the cap never bound, so NO subsampling happened.**
The builder caps CUAD at 45% of the ContractNLI+CUAD portion of train, which
works out to `int(0.45/0.55 x 7188) = 5881` rows. CUAD supplies 5,871 after the
truncation filter. 5,871 is not greater than 5,881, so the random subsample
branch never executed and **every surviving CUAD row is in the training set**.
The realised CUAD share is 38.8% of the three-task train file.

This is worth stating explicitly because the code contains an `rng.shuffle` and
a slice that look like random subsampling and are never reached. Anyone reading
the builder would reasonably assume rows were discarded at random. They were
not. Dev and test are not capped at all — a mixture ratio is a training choice,
and applying it to evaluation would report a score on a distribution nobody
will ever see.

**Truncation filter is train-only.** 375 train rows whose gold evidence falls
outside the 16,384-token window are dropped rather than truncated, because a
kept row would teach the model to cite text it cannot see. Dev and test keep
every row. Dropping a long contract from the evaluation set would curate the
benchmark toward easy documents and report an optimistic number; a model that
cannot handle an 83,000-token contract must be scored on it, not excused from
it. The dropped row identities are in `combined/_dropped_train_rows.json`.

**The dropped rows are biased, and it is recorded.** Loss ranges from 5.0%
(Uncapped Liability) to 29.0% (Non-Transferable License) — a six-fold spread
that tracks contract length, concentrated in 67 of 347 contracts. Commercial
terms lose most; front-of-contract boilerplate loses least. Any weak
per-category result on exclusivity, non-compete, minimum commitment or change
of control is therefore partly a data-availability artifact. Full table in
`CONTEXT_REQUIREMENTS.md`.

**16,384 is a memory limit, not a data choice.** Measured: Qwen3-14B peaks at
66.8 GiB at 16,384 on an 80 GB H100 and OOMs at 32,768. So do 8B, Llama-3.1-8B
and even 4B, because the cost is dominated by the 151,936-wide logits tensor
rather than by attention. All four models train at the same length so the
comparison is between models, not between context budgets.

## Sequence construction

Unchanged from v1 and shared with it by import, not by copy:
`prompt(add_generation_prompt=True) + target + eos`, with loss masked to the
target only — every prompt token carries label `-100`. The trainer asserts this
on the first example before any GPU time is spent, and prints the counts.

`v2/train_multitask.py` exists rather than reusing v1's trainer because that one
does `ex["verdict"] = r["verdict"]` unconditionally and CUAD rows have no
verdict field. v1 is frozen, so it is imported from, never edited. The v2
trainer is also deliberately simpler: no Contradiction oversampling, no class
weighting, no reasoning-token weighting. Those were levers for a single-task
precision/recall trade, and carrying them in would confound the multi-task
question with a rebalancing question.

## What this dataset cannot tell you

- Task 3 has **no human-verified ground truth**. A model scoring well on it is
  imitating Qwen3-32B, not demonstrating legal understanding. Thirty samples
  are laid out in `RISK_NOTE_SAMPLES.md` for human review.
- Test is untouched and stays untouched. Model selection happens on dev.
- The three tasks are evaluated on different metrics and are not commensurable.
  There is no single "combined score" here, and one should not be invented.
