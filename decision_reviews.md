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

---

# Added after the `--system sft` rerun

## O3. `json_valid` conflated "not JSON" with "JSON, wrong key"

**Original.** A single `json_valid` flag, set only when the reply was JSON
carrying the `verdict` key. Reported in `SCREEN_RESULTS.md` as "JSON validity".

**Judgment: OVERTURNED.** The flag measured *schema compliance* and was labelled
*JSON validity*. Under the `spec` prompt the two coincide, so the original
screen's 100% figures were correct and the defect stayed invisible. The `sft`
rerun separated them and the mislabel became a false conclusion: every zero-shot
cell reported **0.0% "JSON validity"** while simultaneously reporting **0 parse
failures**, which is self-contradictory on its face.

**What was actually happening.** The models emitted well-formed JSON with an
invented key, because the `sft` prompt asks for "a single JSON object" without
naming the field:

| prompt | zero-shot JSON-parseable | zero-shot schema-valid | keys invented |
|---|---|---|---|
| `spec` (names `{"verdict": ...}`) | 100% | 100% | — |
| `sft` ("a single JSON object") | 100% | **0%** | `Relation`, `relationship`, `relation`, `result` |

All four models, independently, reached for `Relation`/`relationship`. Granite
used `relation` on 978/978 rows.

**Fix.** Split the metric into `json_parseable` (well-formed JSON at all) and
`schema_valid` (JSON carrying `verdict`), both recomputed from `raw_output` in
`harness_core.json_shape()`. **No job was re-run** — every raw output was
archived, so all sixteen cells were re-scored offline. That is the payoff for
saving raw outputs rather than only metrics.

**What this rescues.** The parser's fallback tiers. `bare_label` recovered the
verdict from the JSON *value* even when the key was wrong, so accuracy stayed
measurable (76-81% zero-shot). A strict-only parser would have recorded 0%
accuracy across eight cells and I would have concluded four capable models were
broken — the exact failure the brief's guardrail warned about.

**Consequence for the deck (revises 15a).** The "prompting is not enough"
argument is not dead after all; it was aimed at the wrong target. The finding is
sharper than the original one:

> Base models do not fail at JSON. They fail at *unspecified schema*. Told the
> exact shape, all four comply 100% of the time. Told only "return JSON", all
> four invent a field name and none produce the required schema.

**Actionable, and it affects your pipeline.** `02_build_sft_data.py`'s system
prompt does not name the `verdict` key. Its training *targets* do use
`{"verdict": ...}`, so fine-tuning will teach the schema regardless. But that
prompt produces 0% schema compliance from a base model, which matters for the
prompt-only fallback path and for any base-vs-tuned comparison that uses it.
Adding the key to the prompt costs one line and removes the whole effect.

## C4. The ranking order is prompt-dependent too

Under the `sft` prompt the 3-shot order becomes Qwen3-8B (0.793) > phi-4
(0.790) > Qwen3-14B (0.790) — all three still mutually within noise, Granite
still separated downward (0.755).

So the *conclusions* are robust to the prompt: three inseparable leaders, one
separated laggard. The *order among the leaders* changed for the second time,
now under prompt variation as well as condition variation. Third independent
reason not to read that order as real.

Zero-shot macro-F1 also fell under the less specific prompt (Qwen3-14B
0.771 -> 0.708; Qwen3-8B 0.696 -> 0.662), so schema under-specification costs
accuracy, not just format.

---

# Verdicts after the verification runs (27 cells)

## W1 -> UPHELD (tested)

Granite 4.1-8b scored **lower** than 4.2-8b (3-shot 0.739 vs 0.746; zero-shot
0.698 vs 0.741) and remains separated from the leaders. My version choice was
the one *favourable* to Granite, so "drop Granite" and the benchmark-non-transfer
finding both stand. The WEAK rating was appropriate caution; the decision
survived it.

## W2 -> CONFIRMED WEAK. The concern was real and it corrected a recommendation.

Exemplar spread reaches **0.105** per model against published gaps of **0.016**
and **0.003**. Five triples produced **three distinct orders**.

The concrete damage: **Qwen3-8B's 2nd place was an artefact of my exemplar
choice.** It is 3rd in all four alternative triples; phi-4 beats it in 4 of 5
and is separated in 2. The headline triple was the only one where Qwen3-8B came
out ahead, by 0.003, inside noise.

**What I got right.** Refusing to present an ordering among the top three. That
refusal is what kept the published conclusion correct despite the design flaw
underneath it. Had I ranked them confidently, the deck would now contain a claim
the follow-up experiment refutes.

**What I got wrong.** Treating "identical exemplars across models" as
sufficient. It secures fairness *between* models but says nothing about the
stability of the *level*, and I reported CIs that omit this variance term.
Single-triple few-shot evaluation is not adequate for gaps under ~0.05 on a set
this size, and I should have known that from the +0.090 few-shot gain the first
run already showed me.

**Standing fix.** Report mean across triples, with the spread, as the estimate.
Treat any single-triple few-shot number as a point sample.

## W3 -> UPHELD (tested)

Thinking mode is worse by **0.039 [-0.064, -0.014], separated**, at **40x** the
output tokens, with 89/978 still truncating at a 1024-token cap. The control
helped the models it touched rather than penalising them, and created no bias
against phi-4. Asserted correctly, and now measured rather than argued.

## Net

Of the three WEAK ratings, two were over-caution and one was a genuine defect
that changed a recommendation. That ratio is the argument for running the checks:
the two cheap ones cost ~6 minutes each to convert into evidence, and the third
caught a flaw in my own methodology that no amount of re-reading would have
surfaced.

**Unchanged by all of it:** drop Granite; the top group is inseparable on a
single triple; the refusal to rank was the right call. **Changed:** Qwen3-8B is
3rd on merit, not 2nd, so its place in the fine-tuning pair now rests on serving
cost and Contradiction recall rather than on rank.
