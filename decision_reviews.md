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

---

# Judge pass — document-level phases (Phase 0 + Phase 1 harness)

Run before Phase 1 results exist, so this audits the *setup*. Claims were
re-verified against the data rather than restated.

## D1. Gemma 4 train/inference template mismatch -> **CRITICAL, affects Phase 2**

**Found.** For each model I checked whether the inference prompt is a literal
prefix of what SFT would train on:

| model | inference prompt is a prefix of training render |
|---|---|
| Qwen3-8B | yes |
| granite-4.2-8b | yes |
| **gemma-4-12B-it** | **NO** |

Gemma's training render is `<\|turn>model\n{"verdict": ...}<turn\|>`, but the
inference prompt is `<\|turn>model\n<\|channel>thought\n<channel\|>`. Fine-tuning
on the former and serving with the latter means the model has never seen the
empty thought channel that immediately precedes its answer at inference.

**Why it matters.** This is precisely the guardrail's "train and infer at the
same template settings — a mismatch is silent and would invalidate everything".
It would not crash. Gemma would simply underperform for a reason invisible in
the metrics, and we would have written it off as a weak model.

**Fix for Phase 2 (applies to ALL models, not just Gemma).** Do not build
training sequences with `apply_chat_template(full_messages)`. Build them as

    prompt = apply_chat_template(msgs[:-1], add_generation_prompt=True,
                                 enable_thinking=False)
    sequence = prompt + target + eos

so alignment is structural rather than coincidental, and prompt-token loss
masking falls out of the same boundary. Qwen3 and Granite happen to be safe
today; constructing it this way stops that being luck.

**Phase 1 is unaffected** — it is inference only.

## D2. Whitespace-only gold evidence span -> FIXED

ContractNLI carries exactly **one** whitespace-only annotated span in 11,973:
dev doc 582, nda-19, span 68 = a single `" "`.

Two distinct harms, both silent:
- **Training**: puts `""` into the target evidence array, teaching the model to
  emit empty strings.
- **Evaluation**: an empty string can never be "covered", so that example could
  never be a TP for *any* model. A guaranteed, invisible FN.

Dropped, with the drop recorded in the manifest. That example now carries 5
spans instead of 6. It is the only reading consistent with "the exact
sentence(s) from the contract that justify your answer".

## D3. Phase 0 integrity assumptions -> UPHELD (all verified, none assumed)

| assumption | verified |
|---|---|
| `annotation_sets[0]` is not lossy | every document in all 3 splits has exactly **1** annotation set |
| dev's `labels` dict valid for all splits | key sets identical; **0** hypotheses differ in text |
| all 17 hypotheses per document | 423/61/123 documents, all with exactly 17 |
| no duplicate (doc, hypothesis) pairs | 0 in every split |
| Entailment/Contradiction always carry evidence | 0 without |
| NotMentioned never carries evidence | 0 with |

Taking `[0]` would have been a real risk had any document carried two
annotators; it does not, so the choice is safe rather than merely convenient.

## D4. Context budget -> UPHELD, re-verified against the CURRENT prompt

The 8192/16384 decision was made with a slightly different system prompt
(`'no related clause'` vs `'an empty evidence list'`), so it was re-measured
with the prompt actually in use, on the real built data, per model:

- **Eval: 0 prompts over budget** on any model, split or condition. Worst case
  is 8,125 tokens against a 15,872 budget (16384 − 512). Inference never
  truncates, as required.
- **Train: 31 of 7,188** sequences exceed 8192 (0.43%), which reconciles with
  the earlier ~39-over figure after the 3 evidence-losing pairs were dropped.
  All 31 keep their gold evidence.

## D5. Very short gold spans -> CAVEAT, deliberately not filtered

1–4 character gold spans (section numbers: `2.1`, `4.3`, `8.2`) occur in 0.4%
of train, **1.8% of dev**, 1.1% of test gold-bearing pairs.

They cut both ways under strict matching: `2.1` appears all over a contract so
it can be matched trivially, but a model that quotes the clause text without its
section number fails to cover it and loses the whole example to FN.

Not filtered — the brief specifies resolving the annotated spans, and removing
them would be inventing a convention. Flagged so the strict number is read with
it in mind; the partial-credit variant is the check against it.

## D6. My own error — cancelled running jobs

I queried job state and issued `scancel` in the same command, so I killed two
smoke tasks that had just transitioned to RUNNING rather than the pending ones I
meant to replace. A few GPU-minutes wasted on a shared cluster. Check state,
then act on it; do not combine the two.

## Standing verdict

Setup is sound to proceed. One critical issue found (D1) that would have
silently invalidated Gemma's post-SFT numbers, and one data defect fixed (D2).
Neither affects Phase 1, which is inference only.

---

# D7. Loss masking — verified before the first fine-tune

**Same bug class as D1: silent, invisible in metrics.** Unmasked prompt loss
does not crash and the loss curve still looks plausible; the run is simply
wasted on contract reconstruction.

**Measured, not estimated.** Over 200 training examples:

| | |
|---|---|
| prompt tokens | 491,056 |
| target tokens | 13,620 |
| target share of tokens | **2.70%** |
| loss that would go to contract reconstruction if unmasked | **97.3%** |

**Verification is token-level, not a claim.** `verify_loss_masking.py` prints the
label tensor around the boundary and asserts three things per model: every label
before the target start is `-100`, no label after it is, and labels equal
input_ids after the boundary. Example (Qwen3-14B, boundary at 1859):

```
[ 1857] input= 151668  label=   -100              '</think>'
[ 1858] input=    271  label=   -100                  '\n\n'
[ 1859] input=   4913  label=   4913                    '{"'  <-- TARGET STARTS
[ 1860] input=    423  label=    423                   'ver'
```

**It also closes D1 structurally.** The same check asserts the masked prefix IS
the inference prompt, byte for byte. All five available models pass, Gemma
included — its train prompt now ends `<|turn>model\n<|channel>thought\n<channel|>`,
exactly what inference emits. Phase 2 imports `doc_sft_seq.py`, so the code that
was verified is the code that trains.

Truncation at `max_seq_len` cuts the PROMPT from the left, never the target:
dropping target tokens would train the model to emit malformed JSON.

# D8. Trivially-matchable short spans — now measured, not footnoted

D5 flagged these as a caveat. A caveat with an unknown magnitude is not a
finding, so every evidence metric is now computed twice and the difference is
reported per cell.

**The effect is SIGNED and genuinely goes both ways**, which is why assuming a
direction would have been wrong:

- *Inflates* when the model emits `2.1` anywhere — the span is covered for free.
- *Deflates* when the model quotes the clause text but omits its section number
  — under strict all-spans-covered matching that loses the **whole example** to FN.

On a hand-built fixture the second effect dominated, moving F1 from 0.500 to
1.000 once trivial spans were removed. So the honest label is a signed delta,
not "inflation".

Headline numbers keep these spans (they are real annotations, and filtering
them would invent a convention). The report adds a table with F1 excluding
them plus the delta, so the statement becomes "excluding trivially-matchable
short spans moves evidence F1 by X points" with X measured per model.

Examples whose gold evidence is *entirely* trivial are excluded from the
variant rather than reclassified as gold-empty — reclassifying would turn a
TP/FN into an FP/TN and silently change what is being measured.

---

# D9. The veto metric was wrong — precision, not recall

**Original.** Contradiction *recall* was carried forward from the clause-level
screen as the veto metric, on the theory that a missed conflict auto-clears a
bad NDA.

**Judgment: OVERTURNED by the data.**

| | range across all 8 cells |
|---|---|
| Contradiction recall | 0.64 – 0.84 (adequate everywhere) |
| Contradiction **precision** | **0.28 – 0.58** |

Recall was never the failure mode. **Over-flagging is.** Dev holds 95 gold
Contradictions; Qwen3-8B flags **284** (3.0x), Granite **234** (2.5x).

| model | cond | flagged | real | false-flag rate |
|---|---|---|---|---|
| Qwen3-14B | thinking | 125 | 72 | 42% |
| gemma-4-12B-it | zero-shot | 151 | 68 | 55% |
| granite-4.2-8b | zero-shot | 234 | 76 | 68% |
| Qwen3-8B | zero-shot | 284 | 80 | **72%** |

**Why it matters more than the ranking.** A reviewer handed 200 escalations of
which 90 are real stops trusting the flags and re-reads every clause — which is
exactly the workflow the product replaces. Precision is therefore the commercial
bottleneck, and moving it is the result SFT has to deliver.

Evaluation priority is now: **1. Contradiction precision · 2. evidence F1 /
joint · 3. macro-F1** (headline, least diagnostic).

# D10. Two reported numbers were artifacts

## Granite's schema validity was cap-suppressed — CONFIRMED

| | |
|---|---|
| truncated at 512 | 40 |
| `not_json` | 48 |
| **overlap** | **40 (100% of truncations)** |
| genuinely malformed | 8 (0.8%) |
| truncated rows earning an evidence TP | **0 of 40** (24 had gold evidence) |

So Granite's 95.4% schema validity is **4.6% = 3.9% cap artifact + 0.8% real**.
Its true formatting failure rate is 0.8%, and its evidence F1 of 0.510 is a
**floor**, not a measurement. The 8 genuine failures are a specific bug: Granite
omits commas between evidence array elements.

**Action:** re-running **all five** condition-A cells at `max_new_tokens=1024`,
not just Granite, so the "identical cap across models" fairness control survives
while the cap stops being a confound.

## Gemma's 0.024 s/example is NOT yet explained — do not publish it

Gemma and Qwen3-14B-A both emit **72 mean output tokens**, yet Gemma is 28x
faster. Hypotheses tested and their status:

| hypothesis | result |
|---|---|
| larger batch / more concurrency | **rejected** — Gemma had the *lowest* (8.81x vs 16-27x) |
| prefix caching on for Gemma only | **rejected** — enabled for both |
| smaller text tower | **rejected** — ~11.7B dense, comparable to Qwen3-14B |
| sliding-window attention | **partial** — 40 of 48 layers use window 1024 vs full attention on all 40 of Qwen's; helps, but does not account for 28x |

The gap is in **decode throughput** (3,112 tok/s vs ~110-120 for every other
cell), not prefill, which sliding-window attention explains least well.

**Verdict: unexplained. Withheld from any cost claim.** The 1024-token re-run
measures all five cells under identical conditions and will show whether this
reproduces (architectural) or vanishes (scheduling artifact).

# D11. Llama-3.1-8B-Instruct never ran

Still `GatedRepoError: 403`. The token authenticates (`mibrahim7611`, read
scope) and model metadata is readable, so the credential is fine — the licence
has not been granted for that account. Re-checked at the end of Phase 1;
unchanged.

This costs more than one row: Llama was the only candidate with an **external
published number** to sanity-check the whole harness against. Without it there
is no independent anchor confirming our absolute numbers are in the right range,
only internal consistency. Worth fixing before Phase 3.
