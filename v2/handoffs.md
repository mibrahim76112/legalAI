# v2 handoff log — what was done, why, what broke, what fixed it

Chronological. Every figure here is traceable to a file in `v2/` or a job log
in `/scratch/ibi761/legalai/logs/`. Nothing is transcribed from memory.

---

## Step 1 — Measure the context requirement before choosing models

**Why.** Model selection needed a defensible filter. "This model is better" is
contestable; "this model cannot represent a fifth of the inputs" is not.

**Process.** Rendered every row of both built datasets through the production
chat template and tokenizer, recording prompt tokens (prefill) and gold target
tokens (decode) separately. Raw counts in `_context_measurements.json`.

**Result.** The two tasks have completely different shapes. ContractNLI p99 is
6,663 tokens; CUAD p99 is 59,404 and its max is 83,679. A context budget chosen
by looking at Task 1 silently destroys Task 2. Decode is small everywhere, so
this is a prefill problem.

**Consequence.** Minimum admissible context is 32,768 (93% CUAD coverage).
phi-4 at 16,384 is excluded on capability, covering only 79.9% of CUAD.
Qwen3's honest number is 40,960, not the 131,072 it can reach via YaRN, because
`rope_scaling` is null in the shipped config. Llama-3.1 **is** reported at
131,072 because its RoPE extension ships enabled. Same rule, applied both ways.

---

## Step 2 — Discover that 8,192 would have trained the model to hallucinate

**Why.** v1 trained at 8,192 when ContractNLI was the only task. The obvious
move was to keep it.

**Process.** For each CUAD train row carrying evidence, truncated the prompt
exactly as the collator does and checked whether every gold evidence span
survived, using the same normalizer as the scoring code.

**What was found.**

| max_seq_len | rows truncated | evidence destroyed | % of evidence rows |
|---|---|---|---|
| 8,192 | 1,428 | **826** | **38.3%** |
| 16,384 | 755 | 375 | 17.4% |
| 32,768 | 341 | 71 | 3.3% |

At 8,192, 826 rows would carry targets citing text the model cannot see. That
is training it to fabricate citations — the exact failure the project exists to
reduce. In v1 the same defect appeared at a scale of 3 rows and those rows were
dropped; here it is 826.

**Decision.** Drop rather than truncate, and move to 16,384.

---

## Step 3 — Stop asserting the memory limit and measure it

**What was wrong with Step 2.** The choice of 16,384 over 32,768 rested on an
assumption I had not tested: that a 14B LoRA would not fit at 32k. 32,768 is
the better data choice by five times the rows preserved. Self-review rated the
decision PROVISIONAL and named the test that would settle it.

**Process.** `v2/probe_memory.py` runs a real forward, backward and optimizer
step at each length with the exact v1 training config.

| model | peak @ 16,384 | @ 32,768 |
|---|---|---|
| Qwen3-14B | 66.8 GiB (91% of device reserved) | **OOM** |
| Qwen3-8B | 52.7 GiB | **OOM** |
| Llama-3.1-8B | 46.8 GiB | **OOM** |
| Qwen3-4B | 43.2 GiB | **OOM** |

**Result.** The assumption held, and more strongly than expected — even 4B
cannot reach 32,768, because memory is dominated by the 151,936-wide logits
tensor rather than attention. 16,384 is now measured, not assumed.

**Caveat recorded.** The probes ran sequentially in one process, so allocator
fragmentation makes the later results the least trustworthy. The decisive 14B
measurement ran first.

---

## Step 4 — Catch my own builder curating the evaluation set

**What broke.** The first combined-dataset build applied the evidence-survival
filter to all three splits, dropping 32 dev and 74 test rows.

**Why that is wrong.** Those are the longest contracts. Removing them from
evaluation reports an optimistic number on an easier distribution. The
asymmetry is the whole point: on train such a row is poison because it teaches
fabrication; on dev/test it is a legitimate hard case.

**Fix.** Filter train only. Dev and test keep every row. Dropped train row
identities written to `combined/_dropped_train_rows.json`.

**Gap this exposed.** Nothing in the codebase asserts that evaluation splits
are unfiltered. It was caught by reading build output, not by a test.

---

## Step 5 — Show that the 375 dropped rows are biased, not random

**Why.** "17.4% lost" sounds like random attrition. It is not, and reporting it
that way would be misleading.

**Result.** Per-category loss ranges from 5.0% (Uncapped Liability) to 29.0%
(Non-Transferable License), a six-fold spread, concentrated in 67 of 347
contracts. Heavy losers are commercial and licensing terms buried deep in long
negotiated agreements; light losers are front-of-contract boilerplate.

**Consequence, stated up front rather than discovered later.** Training at
16,384 under-trains exactly the clause types a commercial reviewer most wants
found. A weak score on exclusivity or non-compete will be partly a
data-availability artifact, not purely a model failure.

---

## Step 6 — A 100% filter pass rate that meant nothing

**What happened.** The first risk-note smoke reported 40/40 kept. Rather than
accept it, I measured the output.

| measure | value |
|---|---|
| notes opening with the identical 6 words | 34/40 (**85.0%**) |
| distinct 4-word openings | 5/40 |

Every note was a rewording of *"This clause matters to the business because…"*.

**Root cause, two parts, both mine.** The system prompt contained the literal
phrase "explains why this clause matters to the business", so the teacher
copied the instruction into its answer. And decoding was `temperature=0.0`, so
one fixed prompt plus argmax gives the same opening by construction.

**Why no filter caught it.** All three filters were blind to templating — they
check sentence count, length and quote groundedness. Worse, the length filter
takes its bounds from the p1/p99 of the very batch it filters, so it keeps ~98%
by definition. The 100% pass rate was an artifact of measuring the wrong thing.

**Fix.** Prompt rewritten to supply no stem and to ban the observed opening;
decoding moved to temperature 0.7 / top-p 0.9 with a per-request seed derived
from the run seed, so variety is restored without losing reproducibility; and a
fourth filter caps the share of kept notes sharing a 5-word opening at 5%.

**Verification.**

| measure | before | after (120-row smoke) | full run (3,160 rows) |
|---|---|---|---|
| kept | 100.0% | 96.7% | 96.3% |
| distinct 5-word openings | 6/40 (15%) | 104/116 (90%) | 2,049/3,044 (67%) |
| most common opening | 85.0% | 2.6% | 2.6% |

f4 never fired in either run — the generation fix solved it at source and the
filter is a regression guard, not a crutch. The pass rate went **down**, and
that is the improvement: a filter rejecting nothing is usually measuring the
wrong property.

---

## Step 7 — Add Llama to the pre-SFT screen

**Why.** The screen contained no Llama, which is the first thing a reviewer
would ask about. Meta's repo is gated and compute nodes cannot accept a licence
interactively, so the NousResearch mirror was used — and verified as a mirror
rather than assumed, by checking architecture, layer count, RoPE config and
vocabulary against the Meta release.

**Result.** Macro-F1 0.592, fifth of six, CI overlapping Qwen3-4B and Granite.
Adding it changed no selection decision.

**The finding worth keeping.** Llama fails in the *opposite* direction to Qwen.
Every Qwen model over-flags contradictions (Qwen3-8B raises 284 flags for 95
real conflicts). Llama raises the joint-fewest flags but catches only half the
real conflicts, recall 0.505 against a field minimum otherwise of 0.642. It
also returns no evidence at all on 34.2% of rows that have gold evidence, two
and a half times the next-worst model. Over-flagging wastes reviewer time;
under-flagging lets a conflicting NDA through. Those are not interchangeable,
and they need different fixes.

---

## Step 8 — Build both datasets and launch training

Three-task train 15,138 rows (7,188 ContractNLI / 5,871 CUAD / 2,079 risk
notes); two-task control 13,059. The control exists so the contribution of the
synthetic task is measurable rather than assumed.

`v2/train_multitask.py` was written rather than reusing v1's trainer, which
does `ex["verdict"] = r["verdict"]` unconditionally and would crash on CUAD
rows. v1 is imported from, never edited.

A 64-row smoke ran first and confirmed the loss mask on a real example: 3,170
prompt labels at -100, 74 supervised tokens. The trainer asserts this before
any long run starts.

Launched: Qwen3-4B, Qwen3-8B, Llama-3.1-8B and Qwen3-14B on the three-task set,
plus Qwen3-8B on the two-task control.

---

## Open items

- Human review of `RISK_NOTE_SAMPLES.md` (30 notes, all 18 categories).
- Evaluation of the trained adapters on dev. Test stays untouched.
- The 32,768 question is not closed. Chunked cross-entropy, 8-bit optimizer
  states, QLoRA or two-card sharding would each make it reachable and recover
  375 training rows concentrated in the most commercially important clause
  types. Highest-value follow-up if GPU time allows.
