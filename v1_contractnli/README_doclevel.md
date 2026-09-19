# Document-level NDA playbook-compliance review

Supersedes the clause-level screen (`SCREEN_RESULTS.md`), which measured a
different, easier task. Those numbers are archived and are **not** mixed into
these tables.

Design follows ContractEval (NLLP 2025, arXiv 2508.03080): full contract as
context, one question per inference call, zero-shot. Our task is a superset —
they do extraction only, we do extraction **plus** three-way judgment.

## Settled parameters

| | |
|---|---|
| Training `max_seq_len` | 8192 (p99 of total = 6,743 tokens) |
| Inference context | 16384 — eval inputs are never truncated |
| Dropped pairs | doc 622/nda-10, doc 622/nda-19, doc 162/nda-19 (all train) |
| Target | `{"verdict": "...", "evidence": [...]}`, empty array for NotMentioned |
| Policy question | the official `labels[hid].hypothesis`, verbatim |

The system prompt says **"or an empty evidence list if none apply"** so that
prompt and target agree. Training on `[]` while the prompt asks for a sentinel
string would teach the model to disobey an explicit instruction.

## Models

| model | licence | gated | role |
|---|---|---|---|
| Qwen/Qwen3-14B | apache-2.0 | no | A + B |
| Qwen/Qwen3-8B | apache-2.0 | no | A + B |
| Qwen/Qwen3-4B | apache-2.0 | no | A + B |
| ibm-granite/granite-4.2-8b | apache-2.0 | no | A |
| google/gemma-4-12B-it | **apache-2.0** | no | A |
| meta-llama/Llama-3.1-8B-Instruct | llama3.1 | **yes — blocked** | A |

**Gemma choice.** Gemma 4 12B (`google/gemma-4-12B-it`) **is Apache 2.0** and
ungated. Gemma 3 12B is under the restrictive `gemma` licence *and* gated. So
Gemma 4 is used, and it is used for every phase — no mid-run switching.

**Llama status.** The HF token authenticates (`mibrahim7611`, read scope) and
model metadata is readable, but file download returns **403**, meaning the
licence has not been accepted for this account. Accept it at the model page,
then re-run the prefetch; the cell slots in without other changes.

## Conditions

- **A — zero-shot, non-thinking.** All models. Primary condition, used for ranking.
- **B — zero-shot, thinking enabled.** Qwen3 only (3 cells). N/A elsewhere.

No few-shot: each document-level exemplar carries a full contract, so three
exemplars plus the test contract exceed context.

## Chat template findings (verified before any GPU work)

All five available models expose `enable_thinking`, and all five prefill an
**empty reasoning block into the PROMPT** when it is disabled, so generation
begins after it and the parser sees clean output.

| model | default | non-thinking prompt ends with |
|---|---|---|
| Qwen3 14B/8B/4B | thinking | `<think>\n\n</think>\n\n` |
| granite-4.2-8b | thinking | `<think></think>` |
| gemma-4-12B-it | **non-thinking** | `<\|channel>thought\n<channel\|>` |

**Detector bug found and fixed.** Probing `enable_thinking=False` alone cannot
detect a template whose default is already non-thinking: Jinja treats an
undefined variable as falsy, so `render(undefined) == render(False)`. Gemma 4 is
exactly that case (`{%- if not enable_thinking -%}`) and was initially
misreported as having no toggle. Detection now probes **both** directions.

Gemma 4 also speaks in channel syntax (`<|channel>thought ... <channel|>`), which
the parser strips before looking for JSON.

## Run it

```bash
# 0. prefetch (LOGIN NODE - compute nodes have no internet)
source /scratch/ibi761/legalai/envs/prefetch/bin/activate
export HF_HOME=/scratch/ibi761/legalai/hf_home
python prefetch_models.py --models-file models_doc.txt

# 1. build document-level data + verify evidence is verbatim
python build_doc_sft.py

# 2. verify chat templates
python verify_templates.py

# 3. smoke test (2 cells, 24 examples) before the array
sbatch smoke_phase1.sh

# 4. Phase 1 screen on dev (8 cells)
sbatch run_phase1.sh

# 5. report
python report_phase1.py --dir /scratch/ibi761/legalai/doc_results --split dev
```

## Evidence matching

Normalization is lowercase, whitespace runs collapsed to one space, strip. No
other normalization.

- A gold span is **covered** if its normalized text is a contiguous substring of
  the concatenated normalized prediction.
- **TP** gold non-empty and every span covered · **FN** gold non-empty and ≥1 not
  covered (includes empty prediction) · **FP** gold empty and prediction
  non-empty · **TN** both empty (excluded from F1/F2, counted in verdict metrics).

Reported strict *and* partial-credit (mean fraction of gold spans covered).
Strict is the ContractEval-comparable figure.

**Joint metric is reported two ways.** The brief's definition — verdict correct
AND all gold spans covered — is vacuously true when gold evidence is empty, so a
model that invents citations on a NotMentioned row still scores. `joint_lenient`
is that literal reading; `joint_strict` additionally requires no invented
evidence. The gap between them is the hallucinated-citation rate.

## Files

| file | role |
|---|---|
| `build_doc_sft.py` | Phase 0 data + evidence-verbatim verification |
| `doc_harness.py` | prompts, parsing, evidence matching |
| `test_doc_harness.py` | 19 unit tests — run before spending GPU hours |
| `doc_metrics.py` | all metrics, recomputable from raw JSONL |
| `doc_eval.py` | one (model, condition) cell |
| `verify_templates.py` | chat-template pre-flight |
| `run_phase1.sh` | 8-cell array |
| `report_phase1.py` | tables, generated programmatically |
