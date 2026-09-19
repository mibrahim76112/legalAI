# Arm B plan — reasoning-first targets (NOT YET IMPLEMENTED, awaiting approval)

## Two corrections to the brief before anything else

### 1. The Llama-3.3-70B fallback is unavailable

| model | gated | licence | weights |
|---|---|---|---|
| Qwen/Qwen3-235B-A22B-Instruct-2507 | **no** | apache-2.0 | 470 GB |
| meta-llama/Llama-3.3-70B-Instruct | **manual** | llama3.3 | 141 GB |
| Qwen/Qwen3-32B | no | apache-2.0 | 66 GB |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | no | apache-2.0 | 61 GB |

Llama-3.3-70B is gated the same way Llama-3.1-8B is — the account authenticates
but the licence grant is not approved, which has blocked Llama all project. The
named fallback would fail identically.

**Proposed substitute fallback: `Qwen/Qwen3-32B`** (dense, 66 GB, 1-2 GPUs,
ungated). Same family as the primary teacher, so a fallback does not also change
model family. `Qwen3-30B-A3B-Instruct-2507` is the cheaper MoE alternative.

### 2. `normalize()` is not in `run_eval.py`

`run_eval.py` is the superseded **clause-level** script and contains no
normalizer. The document-level function Arm A actually evaluates with is
**`doc_harness.norm`**:

```python
def norm(s):
    return " ".join((s or "").lower().split())
```

lowercase + collapse whitespace + strip, exactly as the brief describes. Arm B
will import that symbol. No second normalizer will be written.

## Teacher feasibility (measured, not assumed)

| | |
|---|---|
| Qwen3-235B-A22B BF16 | 470 GB weights |
| scratch free | 867 GB of 1024 GB quota |
| 8x H100 on one node | schedulable, ~30 min queue (`--test-only`) |
| 4x / 2x H100 | schedulable, ~5 min queue |
| train pairs to generate | 7,188 |

235B-A22B is MoE with 22B active, so 8-way tensor parallel on one node fits with
KV headroom. Viable, but it is a 470 GB download and an 8-GPU allocation on a
contended cluster. **Recommendation: start with Qwen3-32B** to validate the whole
pipeline end to end on a few hundred pairs, then run the full pass on 235B only
if the 32B rationales are visibly weaker. Cheaper to fail small.

## File layout

```
gen_reasoning.py        Pass 1 + Pass 2 teacher generation (vLLM offline)
reasoning_filters.py    the 4 string filters; imports doc_harness.norm
build_armb_data.py      kept rationales -> doc_sft_armb/{train,valid,test}.jsonl
train_sft.py            UNCHANGED, gains --target-schema {armA,armB}
doc_harness.py          UNCHANGED (norm, parser gains optional reasoning key)
prefetch_teacher.py     login-node download of the teacher

reasoning/
  pass1_raw.jsonl       every Pass-1 generation, unfiltered
  disagreements.jsonl   teacher verdict != gold  (label-audit artefact)
  pass2_raw.jsonl       conditioned regenerations
  kept.jsonl            survived all filters
  rejected.jsonl        one row per rejection, with reason
  REPORT.md             the numbers you approve on
```

Arm A files are not modified. Arm B data lands in a separate directory.

## Teacher prompt — Pass 1 (blind, no gold label)

> SYSTEM
> You are a senior contracts lawyer reviewing a Non-Disclosure Agreement against
> a stated policy position. Work only from the contract text supplied. Respond
> with JSON only, in the form
> `{"reasoning": "...", "verdict": "...", "evidence": [...]}`.
>
> The `reasoning` field must, in this order:
> 1. name the governing clause by its section number or heading;
> 2. quote the operative language verbatim from the contract, in quotation marks;
> 3. apply that language to the stated position and state which verdict follows.
>
> Quote only text that appears in the contract. Do not paraphrase inside
> quotation marks. If no clause addresses the position, say so explicitly and
> return an empty evidence list.
>
> `verdict` is exactly one of Entailment, Contradiction, NotMentioned.
> `evidence` is the list of verbatim sentence(s) that justify the verdict, or []
> for NotMentioned.
>
> USER
> CONTRACT: {full document text}
>
> POSITION: {hypothesis verbatim from labels dict}

`reasoning` is first in the schema and first in generation order, so the verdict
is conditioned on it. That ordering is the whole point of the arm.

## Teacher prompt — Pass 2 (conditioned, disagreements only)

Identical, with one inserted line before USER:

> The correct verdict for this position is **{gold_verdict}**. Derive that
> verdict from the contract. If the contract genuinely does not support it, say
> so in the reasoning rather than inventing support.

The escape clause matters: without it the teacher will manufacture justification
for a label that may itself be wrong, which is exactly what the disagreements
file exists to detect. Rows are tagged `conditioned: true` for ablation.

## Filters — thresholds derived, not hardcoded

All four are pure string checks. No LLM judge.

| # | filter | rule |
|---|---|---|
| 1 | quotes grounded | every `"..."` span inside `reasoning`, once `norm()`-ed, is a substring of `norm(contract)` |
| 2 | evidence matches gold | the Arm A rule exactly: gold non-empty AND every gold span is a contiguous substring of the concatenated normalized prediction |
| 3 | verdict consistency | the verdict named in the reasoning text equals the JSON `verdict` field |
| 4 | length sane | see derivation below |

**Filter 4 derivation.** I cannot set these before Pass 1 exists. The rule I will
apply, stated now so it is not chosen after seeing what it excludes:

- compute the token-length distribution of Pass-1 `reasoning` fields;
- floor = **p1**, and additionally reject anything under 20 tokens, which cannot
  contain a clause name plus a verbatim quote plus an application;
- ceiling = **p99**, and additionally reject anything that hits `max_new_tokens`
  (truncated mid-generation, so the JSON is unparseable anyway);
- report both derived numbers in REPORT.md alongside the histogram.

Filter 1 has a known failure mode I will measure rather than paper over: a
teacher that quotes across a page break or normalises a ligature will fail a
substring test that a human would pass. If rejection-by-filter-1 exceeds ~15% I
will report the rate and show examples before changing anything.

## Pre-approval report (Part 1 deliverable)

1. Pass-1 agreement with gold: overall and per verdict class.
2. Filter rejections by reason, as counts and as % of Pass 1.
3. Final yield: kept / 7,188.
4. Token-length distribution of the new Arm B targets, with the derived
   `max_seq_len` and how many examples truncate there.
5. 20 random kept rationales printed in full.

## Part 2 — Arm B training, controlled

Identical to Arm A: LoRA r=16, alpha=32, dropout 0.05, targets
`q,k,v,o,gate,up,down_proj` on all layers, lr 1e-4 cosine, warmup 0.03, 1 epoch,
batch 1 x grad-accum 16, **seed 42**, same data order.

Two changes, both mandated by the brief:

| | Arm A | Arm B |
|---|---|---|
| target schema | `{verdict, evidence}` | `{reasoning, verdict, evidence}` |
| `max_seq_len` | 8192 | **derived in Part 1** |

Sequence construction is unchanged and stays
`prompt(add_generation_prompt=True) + target + eos` with loss masked to the
target — the `doc_sft_seq.build_example` path that was verified token-by-token.
This is the Gemma template alignment fix and it will not be regressed.

`max_seq_len` will be set to the smallest power of two covering **p99** of Arm B
total sequence length, matching the method used for Arm A's 8192, and I will
report the truncation count at that value. If p99 exceeds 16384 I will report it
and ask rather than silently truncating reasoning.

Model order: **Qwen3-4B first**, alone. Extend only once it is clean.

## Part 3 — Eval

Same harness, same metrics. Dev and test targets stay Arm A schema — **no teacher
output enters dev or test**. Arm B's generated reasoning at inference is parsed
and discarded as unscored free text; verdict and evidence are scored against gold
exactly as in Arm A.

Additions:
- output tokens per query, mean and p95, for the cost/accuracy frontier;
- **Contradiction precision** called out separately;
- paired bootstrap Arm A vs Arm B on identical test items.

**Pre-registered prediction, recorded before the run:** reasoning-first should
improve Contradiction precision, Arm A's worst metric. Arm A base was 0.28-0.58
and post-SFT 0.60-0.69. If Arm B does not beat the matched Arm A fine-tune on
that metric, the report will say so plainly and the arm will be reported as a
negative result.

## SLURM scripts

| script | resources | purpose |
|---|---|---|
| `prefetch_teacher.py` | login node | download teacher weights |
| `gen_reasoning_p1.sh` | 4x H100 (32B) or 8x (235B), 1 node, 6h | Pass 1 over 7,188 pairs |
| `gen_reasoning_p2.sh` | same, 2h | Pass 2 over disagreements only |
| `train_armb.sh` | 1x H100, 8h | Qwen3-4B Arm B |
| `eval_armb.sh` | 1x H100, 3h | merge adapter + dev/test eval |

All inherit `setup_env.sh`, which already exports
`VLLM_WORKER_MULTIPROC_METHOD=spawn` and pins `VLLM_CACHE_ROOT`,
`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_DIR` under
`$SLURM_TMPDIR`, and asserts every module actually loaded.

## Open questions for you

1. **Teacher fallback**: approve `Qwen3-32B` in place of the gated Llama-3.3-70B?
2. **Start small**: validate the pipeline on Qwen3-32B before committing a 470 GB
   download and an 8-GPU allocation for 235B?
3. **Pass 2 scope**: if Pass-1 disagreement is very high (>40%), conditioned rows
   would dominate the training set. Cap them at some fraction, or keep all?
