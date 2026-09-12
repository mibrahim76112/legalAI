# Decision reviews — self-adjudication

Every decision on this project, judged against its own alternatives *after*
seeing the results. Companion to `handoffs.md`: that file records what I did and
why at the time; this one asks whether the choice still holds now that there is
evidence.

A review that never rules against the author is decoration. Three decisions
below are rated **WEAK**, and one of them undermines the screen's only
statistically firm conclusion.

## Rubric

| verdict | meaning |
|---|---|
| **UPHELD** | Correct, and the evidence now supports it. |
| **UPHELD / CAVEAT** | Right call, but it carries a limitation that must be stated when the number is used. |
| **WEAK** | Defensible when made, but rests on an untested assumption that could change a stated conclusion. Needs verification. |
| **OVERTURNED** | Wrong. Corrected, with the correction recorded. |

Each entry states what evidence would change the verdict, because a judgment
with no falsifier is an opinion.

---

# WEAK — needs verification before the deck

## W1. Granite 4.2-8b instead of 4.1-8b

**Decision.** Used `ibm-granite/granite-4.2-8b`.

**Why then.** The brief said "current exact ID". 4.2-8b is the newer release
(2026-08-07 vs 2026-04-06) and both are instruct-tuned.

**Judgment: WEAK.** This is the most consequential unilateral choice I made, and
I under-weighted it at the time.

Granite is **the only model the screen separates**. "Drop Granite" is the single
firm conclusion the whole exercise produced, and it rests on a version I picked
on a one-word reading of the brief. The alternative had roughly **5x the
downloads** (218,996 vs 41,752) and four more months of maturity, which
correlates with fixed chat templates and generation configs — exactly the
failure class the brief warned about.

If 4.1-8b scores materially higher, two claims weaken at once: "drop Granite",
and the "public benchmarks did not transfer" story built on its IFEval of 87.06,
since that score was published against a Granite 4.x line rather than this
specific point release.

**What would overturn it.** Run 4.1-8b, both conditions. If it lands inside the
top three's confidence band, the exclusion becomes a version artefact rather
than a model finding.

**Cost.** 2 cells, ~6 minutes of queue. Cheapest test in this document relative
to what it protects.

---

## W2. A single exemplar triple, with variance never measured

**Decision.** One 3-shot triple, chosen as the median-length train example per
class, shared by all four models.

**Why then.** Determinism and defensibility — no cherry-picking, no seed
lottery, identical inputs across models.

**Judgment: WEAK. This is the most serious methodological hole in the screen.**

The reasoning was sound for *fairness between models* and I still defend that
part. But it silently assumed the *level* of each score is insensitive to which
triple was drawn, and I never tested that assumption.

The screen's own data argues against it. Qwen3-8B gained **+0.090** macro-F1
from those three examples. A condition with that much leverage is not plausibly
invariant to exemplar choice. Now compare that to the gaps the ranking rests on:

| pair | gap |
|---|---|
| Qwen3-14B vs Qwen3-8B | 0.016 |
| Qwen3-8B vs phi-4 | 0.003 |

If swapping the triple moves scores by even 0.01-0.02 — well below the +0.090
sensitivity already demonstrated — it swamps both gaps. The 3-shot ranking
order could be an artefact of my exemplar pick rather than a property of the
models.

This does **not** damage the reported conclusion, because I declined to present
an ordering among the top three. It damages any *future* attempt to read that
order as real, and it means my reported bootstrap CIs understate total
uncertainty: they capture example sampling only, not exemplar choice.

**What would overturn it.** 3-5 different balanced triples across all four
models, 3-shot. Report the spread of macro-F1 per model. If the spread is small
relative to 0.016, the single-triple design is vindicated and the CIs stand. If
it is comparable or larger, exemplar variance belongs in the error bars and the
honest statement becomes stronger, not weaker.

**Cost.** 4 models x 4 extra triples = 16 cells, ~10 minutes each in parallel.
The most informative remaining experiment in the project.

---

## W3. Thinking mode disabled, counterfactual never run

**Decision.** Request direct-answer mode where the template exposes it (Qwen3
x2, Granite). Built `--allow-thinking` to measure the alternative.

**Why then.** A `<think>` block under a 128-token cap truncates before the JSON,
scoring a capable model as unparseable — a harness artefact, and the brief's
explicit guardrail.

**Judgment: JUSTIFIED BUT UNVERIFIED.**

The mechanism is real and partly confirmed: mean output settled at ~10 tokens
with zero truncation, which is direct-answer behaviour. But I asserted the
counterfactual — that leaving thinking on would have *hurt* — and then never
measured it. I built the flag and did not use it.

There is also an asymmetry I did not acknowledge in the results document. Two
Qwen models and Granite were intervened on; phi-4 has no such switch and ran
untouched. The intervention is uniform in *intent* but not in *effect*, and
phi-4 finished within noise of both Qwen models. A reviewer can reasonably ask
whether the intervention helped or hindered the models it touched, and right now
I cannot answer with data.

**What would overturn it.** One cell: Qwen3-8B, 3-shot, `--allow-thinking`, with
`max_new_tokens` raised enough to let a think block complete. If accuracy is
equal or better, the fairness control cost those models something.

**Cost.** 1-2 cells, ~6 minutes. Converts a plausible argument into a measured one.

---

# OVERTURNED — already corrected

## O1. "Decide finally on test"

**Original.** Advised selecting the winner on the test split because it has 210
Contradiction examples rather than 89.

**Judgment: OVERTURNED.** Selecting on test and reporting that same number is
selection contamination — reporting the max of candidates chosen over the set
they are reported against. It contradicted the purpose of the entire screen. The
irony is that I built a paired bootstrap specifically to avoid over-reading
small differences, then advised an inflation error in the closing line.

**Correction.** Select on post-SFT dev, report **both** models on test,
recommend on cost/performance. Retracted in `SCREEN_RESULTS.md` rather than
quietly edited, so the reasoning stays auditable.

## O2. Adjacent-pairs-only bootstrap

**Original.** Compared only neighbouring ranked models.

**Judgment: OVERTURNED.** The task is to select the top **two**, so the binding
comparison is rank 1 against rank 3, which adjacency never tests. I matched the
test to the shape of my output table instead of to the decision.

**Correction.** Full pairwise matrix, both conditions. This produced the actual
finding — the top three are *mutually* inseparable — which adjacency-only had
made look like a cleaner ordering than exists.

---

# UPHELD WITH CAVEAT

## C1. Macro-F1, 3-shot, as the ranking metric

**UPHELD** — specified by the brief, and correct for a 9.1%-Contradiction set
where accuracy tracks the majority classes.

**Caveat that must travel with it.** There is no single "the ranking". Zero-shot
and 3-shot produce **different orders**, and Qwen3-8B moves from last to second.
Quoting the 3-shot order without the zero-shot order beside it overstates how
settled the result is. Compounded by W2: exemplar variance is not in the CIs.

## C2. Qwen3-14B + Qwen3-8B as the fine-tuning pair

**UPHELD** — same family and tokenizer isolates scale; they hold opposite
strengths (NotMentioned recall 0.76 vs Contradiction recall 0.92), so the
post-SFT comparison is informative either way.

**Caveat.** Not chosen on the ranking metric, because the ranking metric could
not separate them. Chosen on tiebreakers and on what the experiment teaches. Say
that out loud in the deck; presenting them as "the top two by macro-F1" would
misrepresent a 0.016 gap that straddles zero. Both also inherit any Qwen
family-level weakness, which a Qwen-only shortlist cannot detect.

## C3. `gpu_memory_utilization=0.90`

**UPHELD** for throughput, but it cost an infrastructure datapoint: peak memory
is ~74.5 GB for **all four** models because vLLM preallocates its KV cache, so
the number reflects my flag, not model footprint. Already stated honestly in
`SCREEN_RESULTS.md`. A real per-model figure would need
`torch.cuda.max_memory_allocated()` or a lower utilization setting.

---

# UPHELD

| decision | why it holds |
|---|---|
| **Dev for the screen, test reserved** | Preserved the only uncontaminated set. Survived my own O1 error against it. |
| **Verify eval set is row-identical to `sft_data/baseline/valid.jsonl`** | Checked on four axes (length, `chunk_id` order, gold order, exact user-turn reconstruction). Makes "measure once, use twice" auditable, not aspirational. |
| **Import the SFT builder's loader instead of reimplementing** | Caught that my own ids (`h00`-`h16`) would have made per-hypothesis analysis un-joinable against the SFT work — a defect that surfaces weeks later. |
| **Paired, not unpaired, bootstrap** | Both models scored identical rows; pairing removes example-difficulty variance. Unpaired would be both wrong and less sensitive. |
| **Refuse to present an ordering among the top three** | The evidence genuinely does not support one. This is the single most defensible thing in the screen. |
| **Smoke-test gate before 8 cells** | Paid for itself three times: the opencv install failure, the zeroed GPU sampling, and the over-provisioned walltime. On a shared cluster, 8 jobs failing identically wastes others' time. |
| **Separate `json_valid` from `parsed_verdict`** | They answer different questions. Revealed phi-4's `embedded_json` habit and its 27.4-token zero-shot cost, which the accuracy number alone hid. |
| **Count parse failures as wrong, report separately** | Dropping them would reward refusal; hiding them would confuse formatting with reasoning. |
| **Pass token ids, not strings, to vLLM** | `apply_chat_template` already emits BOS; a rendered string risks a second one. Removes a silent, model-dependent corruption. |
| **Verify library APIs against the installed artifact** | Caught the removed `resume_download` kwarg and that `max_model_len` reaches vLLM only via `**kwargs`. Either would have failed all eight jobs. |
| **Check the empty-premise leak before trusting any metric** | The one bug that would have inflated scores instead of crashing. NotMentioned premises turned out to be the *longest*. |
| **Node-local venv in `$SLURM_TMPDIR`** | Eight jobs importing one venv over a networked FS hammers the metadata server. |
| **System prompt as a flag, not a silent choice** | Either hardcoded answer is wrong for one of the two purposes; a flag serves both and makes the discrepancy visible in code. |
| **Tag outputs by system prompt** | The `sft` rerun would otherwise have silently overwritten the `spec` ranking of record. |

---

# Standing verdict

The screen's **reported** conclusions are sound, because the headline finding is
a refusal to rank rather than a ranking, and a refusal is robust to W2 and W3.

What is not yet safe is anything that treats the 3-shot order as real, or treats
Granite's exclusion as a model-level fact rather than a point-release fact.

Priority if there is queue budget:

1. **W1 — Granite 4.1-8b** (~6 min). Protects the only firm conclusion.
2. **W2 — exemplar sensitivity** (~10 min, 16 cells). Determines whether the
   ranking metric is stable at all, and whether the CIs are honest.
3. **W3 — thinking-mode counterfactual** (~6 min). Converts an argument into a
   measurement.
