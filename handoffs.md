# Handoff log — base-model selection screen

A decision log, not a summary. For every step: what was chosen, why, what
actually happened, what broke, and what changed as a result. Results themselves
live in `SCREEN_RESULTS.md`.

Session: 2026-09-12. Cluster: Nibi (Alliance Canada). Account: `def-ekram_gpu`.

---

## Step 0 — Survey the cluster before writing anything

**Decision.** Spend the first actions on environment recon rather than code.

**Why.** This is a shared HPC system with enforced cgroup limits on login nodes
and a documented set of practices. Guessing at module names, storage locations
or GPU flavours wastes queue time and risks disrupting other users. Cheaper to
read the environment than to debug assumptions inside a failed job.

**Process.** Checked login host (`l3.nibi.sharcnet`), `sshare` for the account,
`diskusage_report` for quota, `module avail` for the Python/CUDA/Arrow stack,
and the read-only Alliance wiki copy at `/cvmfs/.../custom/docs/en/Nibi.md` for
node characteristics and GPU request syntax.

**What this established.**

| | |
|---|---|
| GPU nodes | 36 nodes, 8x H100 80GB SXM, 112 cores, 2000G RAM |
| Per-GPU bundle | ~14 cores, ~250G RAM |
| Scratch | 1 TB soft quota, 13 GB used |
| Internet | available on login node |

**What failed.** Nothing, but one early grep for vLLM wheels looked in
`gentoo2023/x86-64-v4/` and came back empty, which briefly suggested vLLM was
unavailable. A wider `find` showed the wheels live in `gentoo2023/generic/`.

**Improvement.** Stopped narrowing searches by guessed architecture directory;
searched the whole wheelhouse tree instead.

---

## Step 1 — The prepared data did not exist

**Decision.** Stop and ask rather than invent a dataset.

**Why.** The brief stated training data was "already prepared". It was not on
disk. The premise construction is the single choice that determines whether the
screen measures anything, and I could not recover it from an empty directory.

**Process.** Searched `/home/ibi761`, `/scratch/ibi761` and the `def-ekram`
project space to depth 6 for `*contract*nli*`, `*.jsonl`, `train.json`,
`dev.json`, plus any HF cache. Found only `t2.py` (an inspection script),
a `.gitignore` listing `contract_nli/` and `flat_data/`, and an unrelated MARL
project in scratch. No HF cache at all.

**Why I did not just pick a construction.** The two plausible readings diverge
badly:

- Premise = gold evidence spans. ContractNLI only has evidence spans for
  Entailment and Contradiction. NotMentioned would get an **empty premise**,
  leaking the label outright and inflating every score.
- Premise = full document. Honest, but 2-5k tokens per row, which changes
  prompt budget, GPU sizing and the whole cost story.

Guessing wrong produced either a worthless screen or the wrong infrastructure.

**What failed.** My assumption that a stated precondition was true. I should
have verified the input data existed as step 0, before any environment work.

**Improvement.** Asked one blocking question with the search evidence attached,
while starting the model prefetch in parallel because it was independent of the
answer. No idle waiting.

---

## Step 2 — Discarding my own data prep after the `git pull`

**Decision.** Throw away my `prepare_data.py` and rebuild it to import the
user's `02_build_sft_data.py`.

**Why.** The pull revealed the real pipeline: `01_inspect_data.py`,
`02_build_sft_data.py`, `03_sanity_check.py`. My first version had invented its
own conventions. Two would have caused silent damage later:

| mine (wrong) | theirs (authoritative) |
|---|---|
| hypothesis ids `h00`-`h16` (sorted text) | `nda-N` from `trismik/ContractNLI` labels |
| no premise filter | `.strip()` + `MIN_PREMISE_CHARS = 20` |
| no chunk id | `chunk_id` = sha1(premise)[:12] |

Different hypothesis ids would have made per-hypothesis error analysis
un-joinable against the SFT work. That is a defect you would only discover weeks
later, when comparing base to fine-tuned per policy.

**Process.** Rewrote `prepare_data.py` to `importlib`-load the SFT builder
(numeric filename, so a plain import will not do) and call its own `load_rows()`
and `hypothesis_ids()`. One definition of the row set, not two.

**What failed.** Verified rather than assumed. I checked the rebuilt eval set
against `sft_data/baseline/valid.jsonl` on four axes: length, `chunk_id`
sequence, gold sequence, and exact reconstruction of the user turn. All four
matched — 978 rows, identical order.

**Why that check matters.** These base numbers are the control column of the
final fine-tuned-vs-base table. If the base model and the fine-tuned model score
even slightly different row sets, every delta in that table is contaminated.
The check makes the claim "measure once, use twice" verifiable rather than
aspirational.

**Improvement.** Read the existing scripts before writing a parallel
implementation. When the repo has a convention, import it; do not re-derive it.

---

## Step 3 — Eval split: dev, not test

**Decision.** Score the screen on dev (978 rows), reserve test (1991 rows).

**Why.** The brief referenced "dev set size" in its guardrails, and test must
stay unused so it can serve as the clean headline number after fine-tuning.
Spending test here would have cost the only uncontaminated set.

**Correction recorded (2026-09-12).** I initially closed `SCREEN_RESULTS.md`
with "decide finally on test, which has 210 Contradiction examples rather than
89." **That was wrong and has been retracted.** Selecting the winner on test and
then reporting that same number is selection contamination — it reports the max
of two models chosen on the set it is reported against, which is precisely the
error the rest of this screen was built to avoid.

**Adopted instead.** Fine-tune both candidates, report **both** test numbers
honestly, and recommend one on cost/performance grounds. "The 8B is within X
points of the 14B at half the serving footprint" is both cleaner statistically
and a stronger consulting argument than presenting one winner and burying the
other run.

---

## Step 4 — Few-shot exemplar selection

**Decision.** One exemplar per class, from TRAIN only, chosen as the
**median-length** example of its class. Same three for all four models.

**Why median-length rather than random.** A seed-based pick is reproducible but
arbitrary: it can land on an unusually short or long clause and quietly advantage
or handicap every model. The median is deterministic, defensible in a client
deck ("we did not cherry-pick"), and stable if the source is re-fetched.

**Why three and balanced.** Per the brief: a single exemplar primes the label
distribution and would distort Contradiction recall. Balanced coverage shows the
model that all three verdicts, including abstention, are permitted answers.

**What the data turned up.** 5 premises appear in **both** train and dev. The
exemplar pool explicitly excludes any premise present in the eval split, so no
exemplar can echo a scored row. Without that filter, one of the three shots
could have been a near-duplicate of a graded example.

| class | hypothesis | chars |
|---|---|---|
| Entailment | nda-1 | 464 |
| Contradiction | nda-20 | 439 |
| NotMentioned | nda-12 | 534 |

---

## Step 5 — Verifying the NotMentioned leak was absent

**Decision.** Before building anything, check the premise-length distribution
per class.

**Why.** This was the failure mode that would have invalidated the entire screen
while still producing plausible-looking numbers — the most dangerous kind of bug.

**Result.** No empty premises in any class. NotMentioned premises are in fact
the **longest** (median 534 chars vs 466 for Entailment). 4,703 unique premises
across 9,788 rows, and 1,717 premises appear with more than one label because
the same clause entails one policy and contradicts another. That is correct
clause-level structure, not leakage.

**Improvement.** Adopted as a standing habit: check for the degenerate-input
case before trusting any metric, because a leak inflates scores rather than
crashing.

---

## Step 6 — Choosing the Granite model ID

**Decision.** `ibm-granite/granite-4.2-8b`.

**Why.** The brief said "check the `ibm-granite` HF org for the current exact
ID". Queried the HF API and ranked candidates. Two were viable:

| id | created | downloads | note |
|---|---|---|---|
| `granite-4.1-8b` | 2026-04-06 | 218,996 | more established |
| `granite-4.2-8b` | 2026-08-07 | 41,752 | **current** |

Both are `text-generation` + `conversational` (instruct, not base — Granite 4.x
dropped the `-instruct` suffix; `-base` is the separate pretrained variant).
Picked 4.2 because the brief asked for current. Excluded `-GGUF` (wrong format),
`-base` (not instruct), `-fp8` (would break the BF16-for-all fairness control)
and `granite-guardian` (a different model class).

**Residual risk, stated plainly.** 4.2-8b is newer and less battle-tested. If
you want the higher-download variant, `models.txt` is one line and the rerun is
~6 minutes. Given Granite finished last by a separated margin, re-running on
4.1-8b is the one cheap check that could change a conclusion.

---

## Step 7 — Prefetch to `$SCRATCH`

**Decision.** A dedicated login-node script, with an allow/ignore pattern set and
a dry run before committing the download.

**Why.** Compute nodes have no outbound network, so every model must be cached
first or all eight jobs fail identically. Storage goes to scratch, not home:
home has a 50 GB quota and is backed up, and 93 GB of re-downloadable weights
belong in neither.

**Why patterns matter.** Several repos ship GGUF conversions, ONNX exports,
duplicate `.bin` copies and `original/` mirrors. Pulling everything would have
roughly doubled the transfer for files vLLM never reads.

**Process.** Ran `snapshot_download(dry_run=True)` across all four first:

| model | files | size |
|---|---|---|
| Qwen3-14B | 15 | 29.6 GB |
| Qwen3-8B | 12 | 16.4 GB |
| granite-4.2-8b | 13 | 17.6 GB |
| phi-4 | 14 | 29.3 GB |
| **total** | | **92.9 GB** |

Confirmed none were gated (no `HF_TOKEN` present — the repo's `.env` does not
exist) and that 93 GB fit the 1 TB scratch quota, before starting.

**What failed.** `snapshot_download(resume_download=True)` — the parameter was
**removed** in `huggingface_hub` 1.x (installed: 1.30.0). It would have raised
`TypeError` on the very first model.

**How it was caught.** Inspected the live signature with `inspect.signature`
rather than trusting recalled API knowledge. Resume is automatic in 1.x, so the
fix was deletion.

**Improvement.** Verify library signatures against the installed version, not
memory. Applied this again later to vLLM (Step 11).

**Also added.** Post-download verification per model: `config.json` present, at
least one `.safetensors` shard, and a tokenizer (mandatory — the harness calls
`apply_chat_template`). Writes `prefetch_manifest.json`. A silently incomplete
cache would otherwise surface as a confusing in-job failure.

---

## Step 8 — The system-prompt conflict

**Decision.** Make the system prompt a flag. Default to the brief's; do not
silently pick one.

**Why.** `02_build_sft_data.py` bakes a **different string** into its training
targets than the brief specifies:

- brief (`spec`): `...Respond with JSON only, in the form {"verdict": "..."}.`
- SFT (`sft`): `...Answer with a single JSON object and nothing else.`

This matters differently per purpose. For **ranking**, it is harmless — all four
models share one prompt, so the comparison is internally fair. For the **base
column of the final table**, it is a genuine confound: if the fine-tuned model
was trained on `sft` and the base measured on `spec`, part of any base-to-tuned
delta is just the prompt change.

**Why a flag rather than a choice.** Either hardcoded answer is wrong for one of
the two purposes. A flag serves both and makes the discrepancy visible in the
code instead of buried in a commit message.

**Follow-up (done).** Reran the full screen with `--system sft` (job
`21790522`) so the base column is prompt-matched to the fine-tune. Cost: ~6
minutes of queue. Removes the objection entirely rather than arguing about it.

**Defect found while doing it.** Output filenames were
`raw__<model>__<condition>.jsonl` with no prompt tag, so the `sft` run would
have **silently overwritten** the `spec` results — the ranking of record. Fixed
by adding the system tag to every output filename, giving each prompt its own
results directory, and making `aggregate.py` key cells on
`(model, condition, system)` so two prompts can never merge into one row.
Regression-checked the aggregator against the existing `spec` results first and
confirmed identical numbers before submitting.

---

## Step 9 — Parser design

**Decision.** A single shared parser with three tiers, reporting *how* each
output parsed, and separating JSON validity from correctness.

**Why tiers.** A model that reasons correctly but formats loosely must not be
scored as wrong — that would measure format compliance and call it capability.
So: strict JSON, then JSON embedded in prose or fences, then a bare label.

**Why `json_valid` is tracked separately from `parsed_verdict`.** They answer
different questions. `json_valid` is the "can it follow the output contract"
evidence; `parsed_verdict` is "did it get the answer right". A bare label parses
successfully but is explicitly **not** counted as JSON-valid, so the format
metric stays honest.

**Why `<think>` blocks are stripped before parsing.** A reasoning model may
mention "Contradiction" while deliberating and then answer "Entailment". Parsing
the raw string would score the deliberation. The think block is removed first,
and there is a unit test for exactly that case.

**Why parse failures are counted as wrong but reported separately.** A failure
is not a correct answer, so it cannot be dropped from the denominator — dropping
it would reward models that refuse. But it is reported as its own rate so a
formatting problem is never mistaken for a reasoning problem.

**Process.** 18 unit tests written and run **before** any GPU time: fenced JSON,
extra keys, case variants, `not_mentioned` spellings, `neutral`, think-wrappers,
empty, `None`, and an invalid verdict value.

**What failed.** A quoting typo in the test file (`'...'"` mismatch) caused a
`SyntaxError`. Fixed; 18/18 passed.

**Payoff.** In production the parser hit **zero failures across 7,824
generations**. The tier reporting also earned its keep: it revealed that phi-4
uses `embedded_json` on 977/978 zero-shot rows, which is what explains its 27.4
mean output tokens versus ~10 for the others. That is a cost finding the
accuracy number alone would have hidden.

---

## Step 10 — Chat templates: two bugs that would have faked a capability gap

**Decision.** Render each model with its own template via
`apply_chat_template`, and validate the rendering on the login node with
tokenizers only, before any GPU work.

**Why validate first.** The brief's own guardrail: systematically malformed
output most likely means a chat-template bug, not a bad model. Tokenizer-only
work is lightweight and login-node appropriate, so this check was nearly free.

**Bug 1 — every prompt measured as 2 tokens.**

`apply_chat_template(..., tokenize=True)` returns a `BatchEncoding` in
transformers 5.x, not a list of ids. `len()` of it counts dict keys
(`input_ids`, `attention_mask`) — hence exactly 2 for every model and condition.

*How caught.* The number was impossible. A clause plus a policy cannot be two
tokens. Printed `type()` and keys rather than rationalising it.

*Fix.* Extract `input_ids`, and unwrap a batch dimension if present.

*Consequence if shipped.* The `max_model_len` guard would have compared 2
against 8192 and passed; truncation would then have appeared as mysteriously bad
accuracy on long clauses only.

**Bug 2 — the thinking-mode probe reported false support.**

I detected `enable_thinking` support by calling the template with the kwarg and
catching `TypeError`. Templates **silently ignore unknown kwargs** instead of
raising, so phi-4 — which has no thinking mode — was reported as supporting one.

*How caught.* The probe claimed all four models supported it, including phi-4,
which is not a reasoning model. An implausibly uniform result.

*Fix.* Detect by **comparing rendered text** with and without the kwarg. Probe
once per tokenizer, not per example.

*Verified result.* Qwen3-8B, Qwen3-14B and Granite 4.2 expose the switch; phi-4
genuinely does not.

**Why disabling thinking is the fair choice.** Qwen3's template defaults to
emitting a `<think>` block. Under the shared 128-token cap the JSON would often
be truncated away, scoring a capable model as unparseable — a harness artefact,
exactly what the brief warned about. Direct-answer mode is requested where the
template supports it, and `--allow-thinking` is kept to measure the default
behaviour deliberately. Confirmed in production: mean output ~10 tokens, zero
truncation.

**Validated prompt budget** (after the fix):

| condition | median | p95 | max |
|---|---|---|---|
| zero-shot | ~220 | ~470 | 616 |
| 3-shot | ~636 | ~890 | 1040 |

Comfortably inside `max_model_len=8192`.

---

## Step 11 — Inference engine: vLLM, verified against the wheel

**Decision.** vLLM from the Alliance wheelhouse, with `transformers` batched
`generate` as the documented fallback.

**Why vLLM.** Continuous batching over ~1000 short prompts is far faster than
naive batched generate, and the brief asked for it with a stated fallback.

**Why verified rather than trusted.** After the `resume_download` removal
(Step 7), I stopped relying on recalled APIs. Inspected the **actual wheel**
(`vllm-0.25.0+computecanada`) as a zip archive, without installing it:

| checked | result |
|---|---|
| `TokensPrompt` importable from top-level `vllm` | yes (in `__all__`) |
| `LLM.__init__` accepts `dtype`, `gpu_memory_utilization`, `trust_remote_code`, `enforce_eager` | yes, explicit params |
| `max_model_len` | **not** an explicit param — reaches `EngineArgs` via `**kwargs` |

The `max_model_len` finding is the kind of thing that fails eight jobs at once.
Confirmed `EngineArgs.max_model_len` exists before relying on the passthrough.

**Why pass token ids rather than strings.** `apply_chat_template` already emits
the model's BOS token. Handing vLLM a rendered string makes it re-tokenize and
potentially prepend BOS a second time — a subtle, model-dependent corruption.
Tokenizing once and passing `TokensPrompt` removes the class of bug and gives
exact prompt-token counts for the cost table.

---

## Step 12 — Environment build: the failure that justified the smoke test

**Decision.** Build the venv **inside** each job in `$SLURM_TMPDIR`, from the
wheelhouse only.

**Why node-local rather than one shared venv on scratch.** Eight concurrent jobs
importing from a single venv on a networked filesystem hammers the metadata
server — disruptive on a shared cluster and slower for the jobs. Node-local is
faster and immune to that contention. Cost is ~2-3 min of build per job, which
is acceptable at this scale.

**What failed.** The first smoke job (`21788758`) died in 17 seconds.
`pip install --no-index vllm` failed building `opencv-noinstall`: vLLM depends on
`opencv-python-headless>=4.13.0`, and the wheelhouse deliberately ships a **dummy
wheel that errors out**, instructing you to load the OpenCV *module* instead,
before activating the venv.

**Second failure while fixing the first.** Added `opencv/4.14.0` to the module
line and it silently did not load — `module list` showed no opencv, and
`import cv2` failed. `module spider opencv/4.13.0` explained why: opencv is
available under `StdEnv/2023 gcc/12.3` with **cuda/12.9 or 13.2**, and I had
pinned **cuda/12.6**. The load failed as an unsatisfiable combination rather
than as a loud error.

**Fix.** `StdEnv/2023 gcc/12.3 python/3.11 cuda/12.9 opencv/4.13.0 arrow/17.0.0`,
verified on the login node by actually importing `cv2` (4.13.0) before
resubmitting. Second smoke job completed in 4m27s.

**Improvement.** Never trust a module load that produces no output — check
`module list` or import the thing. Silent no-ops are worse than errors.

---

## Step 13 — The smoke test as a gate

**Decision.** One model, 50 examples, both conditions, before submitting eight
cells. Did not submit the array in parallel to save queue time.

**Why.** Two reasons. First, on a shared cluster, eight jobs failing identically
on the same install bug wastes other people's GPU time, and the opencv failure
proves that was the live risk, not a hypothetical one. Second, the brief asked
for parsing to be clean on 50 examples before scaling.

**Tradeoff acknowledged.** The queue showed `ReqNodeNotAvail`, so gating cost
real wall-clock. Accepted, because submitting broken jobs costs more.

**Result.** Clean on the first gated attempt: 100% JSON validity, zero parse
failures, all `strict_json`, zero truncation. Zero-shot 68%, 3-shot 78% on 50
rows — 3-shot helping was the expected direction, a sanity signal that the
conditions were actually different.

**What the smoke test also revealed.**

1. **Generation was far faster than budgeted** — 1-2s per 50 examples. The 2h
   walltime request was wasteful on a shared scheduler, so I cut it to 1h, which
   also schedules sooner. Measured, then adjusted.
2. **The GPU evidence was worthless.** `nvidia-smi` ran *after* the Python
   process exited and reported `0 MiB, 0%`. You explicitly asked for utilization
   evidence for the infrastructure slide, so a post-hoc zero was a silent
   failure to deliver. Fixed by sampling every 5s in the background during
   generation and reporting the peak.

**Improvement.** The gate paid for itself twice: once on the install bug, once
on two quality defects that would have reached the deck.

---

## Step 14 — Metrics and the noise question

**Decision.** Rank on macro-F1 (3-shot). Report everything else as diagnostics.
Add per-cell bootstrap CIs **and** a paired bootstrap between models.

**Why macro-F1 for ranking.** Dev is 9.1% Contradiction. Accuracy would be
dominated by the two majority classes, and a model that never predicted
Contradiction could still look respectable.

**Why paired rather than unpaired bootstrap.** Both models scored **identical
rows**. Pairing removes example-difficulty variance, which otherwise dominates
the comparison and produces needlessly wide intervals. An unpaired test here
would have been both wrong and less sensitive.

**Why this was built before results arrived.** The whole point was to avoid
over-reading a small gap. Deciding the noise threshold after seeing the ranking
invites motivated reasoning.

**Process — validated on synthetic data first.** Generated fake outputs for
three models at 82%, 80% and 55% accuracy, one with 20% deliberate format
breakage. The aggregator correctly reported the 82-vs-80 pair as `WITHIN NOISE`,
separated the broken model, counted parse failures as wrong, and surfaced them
as their own 19-20% rate. That test told me *before* the real run that a
~2-point gap would be unresolvable on 978 rows — which is exactly what happened.

**What failed in my own analysis.** The first aggregation compared only
**adjacent** ranked pairs. But the task is to pick the top **two**, so the
decision hinges on rank 1 vs rank 3, which adjacency never tests.

**Fix.** Ran the full pairwise matrix, all six comparisons, both conditions.
That is what established the actual conclusion: the top three are **mutually
inseparable**, and Granite is the only model the data separates. Adjacent-only
would have implied a cleaner ordering than exists.

**Improvement.** Match the statistical test to the decision being made, not to
the shape of the output table.

---

## Step 15 — Findings that overturned prior assumptions

Three results changed conclusions rather than confirming them. Recording them
because the reasoning, not just the number, is what belongs in the deck.

### 15a. The JSON-validity argument is dead

**Assumption going in.** JSON validity would be evidence for "why SFT rather
than prompting".

**What the data said.** ~100% JSON validity for every model in every condition
(one phi-4 row at 99.9%), and **zero parse failures in 7,824 generations**.
Modern instruct models simply do not fail at JSON.

**Action.** Drop the argument. Revised SFT justification, strongest first:

1. **Evidence spans** — no base model can produce them at all. Now the strongest
   justification, which makes the span rebuild **load-bearing, not optional**.
2. **Abstention calibration** — Qwen3-8B forces a verdict on 46% of silent
   clauses zero-shot. Quantify SFT's improvement in NotMentioned recall.
3. **The ~0.80 accuracy ceiling** — base models plateau there even with
   exemplars.
4. **Per-client playbook adaptation** — the generalization story.

**Why this is still a win.** A metric that kills your own argument is a working
metric. Collecting it and reporting it honestly is the evidence that the
evaluation is real.

### 15b. Public benchmark scores did not transfer

Granite 4.2-8b was included on the strength of its published IFEval (87.06), and
it finished **last** (0.746) — the only model separated from the field, downward,
against all three others in both conditions.

Worth a line in the deck as-is: *a candidate selected on a published proxy
benchmark finished last on our task; public benchmark scores did not transfer.*
It is evidence for the "measure it yourself" methodology, and more persuasive
because it cost something.

### 15c. The ranking is not stable across conditions

| condition | order |
|---|---|
| zero-shot | phi-4 > Qwen3-14B > granite > **Qwen3-8B (last)** |
| 3-shot | **Qwen3-14B** > Qwen3-8B (2nd) > phi-4 > granite |

Qwen3-8B moves from last to second on the largest few-shot gain in the screen
(+0.090). Reporting the 3-shot ranking alone would have presented prompt
sensitivity as a capability order. This is why both conditions were measured
rather than just the ranking one.

---

## Step 16 — Model pair for fine-tuning

**Decision.** Qwen3-14B and Qwen3-8B.

**Why.** Same family and tokenizer, so the comparison isolates **scale** rather
than confounding it with tokenizer and pretraining differences. "Does 14B earn
its serving cost" is the commercially relevant question. They also currently
hold opposite strengths — Qwen3-14B the best NotMentioned recall (0.76),
Qwen3-8B the best Contradiction recall (0.92) — so the post-SFT comparison is
informative either way it lands.

**Counter-argument, stated.** Both inherit any family-level weakness, and a
Qwen-only shortlist cannot reveal that. Judged weaker than the scale-isolation
benefit.

**Why not phi-4.** Most prompt-robust (+0.007 from exemplars) and best
zero-shot, but that stability also implies the least headroom. It is 14.7B —
comparable to the 14B Qwen — without the 8B's Contradiction-recall advantage.
Its 16K context would also constrain the roadmap, which matters once full-document
review and evidence spans enter scope.

**Not chosen on macro-F1.** The screen could not separate these three. The pair
was chosen on tiebreakers and on what the experiment would *teach*, and that
reasoning is recorded here precisely because the headline metric did not settle it.

---

## Step 17 — Resource discipline

Choices made to respect the shared cluster, recorded because they were
deliberate:

- **Login node kept light.** Prefetch (network I/O) and tokenizer-only template
  validation. No model loading, no GPU work, no builds.
- **No tight scheduler polling.** All waits used a 90-120s loop, above the 60s
  floor, with one background waiter per job rather than several.
- **Right-sized requests.** 1 GPU, 12 CPU (within the ~14-core per-GPU bundle),
  128G (within ~250G), walltime cut 2h->1h once generation was measured.
- **Storage placement.** 93 GB of weights and all raw outputs to scratch, never
  home (50 GB quota, backed up). Checked quota before downloading.
- **Test jobs limited.** One 50-example smoke job, not a series.
- **Bundled work.** Both conditions in a single job per model rather than
  sixteen short jobs.

---

## Open items

1. **`--system sft` rerun** — job `21790522`, submitted. Gives the
   prompt-matched base column. Aggregate into the final table when it lands.
2. **Granite 4.1-8b re-check** (optional) — the one cheap test that could
   overturn a conclusion, since Granite is the only separated result.
3. **Evidence-span rebuild** — now the primary SFT justification (15a), so it is
   load-bearing for the deck rather than a nice-to-have.
4. **Post-SFT protocol** — select on post-SFT **dev**; report **both** models on
   test; recommend on cost/performance. Do not select on test (Step 3).
5. **Hardest hypotheses** — nda-7, nda-15, nda-16, nda-20, nda-17 all sit at
   68-77% across all four models. All are partial-scope or permission policies
   ("some", "may"). Consistent across models, so it is a task property and the
   place SFT has the most room.
