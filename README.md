# AI Contract Review Assistant

A fine-tuned LLM that reviews contracts three ways at once: checks them against
a playbook of standing requirements, finds the clauses that matter, and explains
each one in a sentence a non-lawyer can act on. Every answer is anchored to
quoted contract text, so a reviewer can verify rather than trust.

Built as a technical assessment. One LoRA adapter handles all three tasks, and
there's a Next.js interface in `webapp/` that runs on real model output — every
finding on screen came from the fine-tuned adapter, nothing is mocked.

## The three tasks

| task | input | output | source |
|---|---|---|---|
| Compliance | contract + playbook position | verdict + the language that justifies it | ContractNLI |
| Clause ID | contract + clause category | present/absent + the clause text | CUAD, 18 categories |
| Risk note | one clause | one plain-English sentence | synthetic, Qwen3-32B teacher |

22,165 examples over 1,117 contracts. Split by contract, so no document appears
in two splits.

| split | ContractNLI | CUAD | Risk notes | total |
|---|---|---|---|---|
| train | 7,188 | 5,871 | 2,079 | 15,138 |
| dev | 1,037 | 1,098 | 354 | 2,489 |
| test | 2,091 | 1,836 | 611 | 4,538 |

## Results

Four models trained with identical hyperparameters. Llama-3.1-8B won on every
task. Numbers below are dev; test was run once, at the end, on the selected
model only.

**Compliance classification**

| model | accuracy | macro-F1 |
|---|---|---|
| Qwen3-4B | 86.9% | 82.4% |
| Qwen3-8B | 87.2% | 82.6% |
| **Llama 3.1-8B** | **88.9%** | **84.7%** |
| Qwen3-14B | 88.4% | 83.7% |

**Clause identification**

| model | accuracy | macro-F1 |
|---|---|---|
| Qwen3-4B | 87.7% | 85.5% |
| Qwen3-8B | 87.5% | 85.1% |
| **Llama 3.1-8B** | **88.4%** | **86.3%** |
| Qwen3-14B | 88.1% | 85.8% |

**Evidence extraction** (token F1 / span recall @ Jaccard 0.5)

| model | ContractNLI | CUAD |
|---|---|---|
| Qwen3-4B | 80.1% / 72.3% | 57.1% / 41.2% |
| Llama 3.1-8B | 81.2% / 73.6% | 60.3% / 44.7% |
| Qwen3-14B | 82.5% / 75.0% | 59.0% / 44.4% |

**Risk notes** (ROUGE-L against the teacher — imitation fidelity, not
correctness): 0.42 to 0.46 across models.

**What fine-tuning bought**, over the same base models prompted zero-shot:

| | Qwen3-4B | Qwen3-8B | Llama 3.1-8B |
|---|---|---|---|
| ContractNLI classification | +19.1 | +17.2 | +26.7 |
| ContractNLI evidence | +34.9 | +41.0 | +34.8 |
| CUAD classification | +20.3 | +16.9 | +18.0 |
| CUAD evidence | +20.0 | +17.8 | +24.8 |
| Risk notes | +13.7 | +17.1 | +19.5 |

Absolute F1 points, not percentages. Evidence gains are the largest — roughly
double — which is worth noting because people tend to assume fine-tuning helps
classification more than extraction.

## Cost and latency

Measured on one H100, single request, prefix caching off.

| model | TTFT | TPOT | VRAM (GB) | train time (h) |
|---|---|---|---|---|
| Qwen3-4B | 1.49 s | 12.7 ms | 9.7 | 3.49 |
| Qwen3-8B | 1.57 s | 15.2 ms | 17.5 | 4.43 |
| Llama 3.1-8B | 1.35 s | 14.4 ms | 17.0 | 3.83 |
| Qwen3-14B | 2.31 s | 20.8 ms | 30.0 | 6.58 |

Qwen3-14B costs 72% more training time and more than double the memory to score
below Llama. Scale bought nothing here.

## Training

LoRA, r=16, alpha=32, dropout 0.05, over the seven standard projections. One
epoch, lr 1e-4 cosine with 3% warmup, effective batch 16, bf16, gradient
checkpointing, max sequence length 16,384. Same settings for every model so the
comparison is between models rather than between tunings.

Loss is plain token-level cross-entropy on the target only — every prompt token
carries `-100`. There is no custom objective; the F1 metrics are a downstream
readout, not something the model optimises.

Sequence length was measured, not defaulted. At 8,192 tokens the gold evidence
falls outside the window on 826 CUAD training rows, which would teach the model
to cite text it cannot see; those rows are dropped rather than truncated. 32,768
OOMs on every candidate including the 4B, because memory is dominated by the
151,936-wide logits tensor rather than by attention.

## Layout

```
v2/
  build_*.py            dataset construction (seeds recorded in manifests)
  train_multitask.py    LoRA SFT over the three-task mixture
  eval_multitask.py     evaluation; --rescore re-scores archived predictions offline
  evidence_eval.py      strict / token-F1 / span-recall evidence metrics
  rag_retrieval.py      BM25, dense, RRF and cross-encoder reranking comparison
  measure_ttft.py       latency at concurrency 1
  near_dup_check.py     MinHash near-duplicate screen across splits
  jobs/                 SLURM scripts
  *.md                  findings, decisions, methodology

webapp/                 Next.js review interface, with API routes that call the
                        model for live extraction. See webapp/README.md.

v1_contractnli/         earlier single-task work. Frozen; imported from, never edited.
```

Datasets and model weights are not in the repo. Rebuild them with the `build_*`
scripts — seeds are fixed (20260919 for data, 42 for training) so the splits
reproduce exactly. The v1 training data is rebuilt separately; the order and the
three ablation directories that cannot be reproduced are described in
`v1_contractnli/ARCHIVE_NOTE.md`.

## Running it

```bash
module load python/3.11                  # or any 3.11
pip install -r requirements.txt          # torch, transformers, peft, vllm, sentence-transformers
python v2/build_cuad.py
python v2/gen_risk_notes.py --teacher Qwen/Qwen3-32B --tp 2
python v2/build_combined.py --risk-notes v2/risk_notes
sbatch v2/jobs/train_multitask.sh        # set MODEL=...
sbatch v2/jobs/eval_multitask.sh
```

The demo:

```bash
cd webapp && npm install && npm run dev
```

## Reading order

`v2/SFT_METHODOLOGY.md` for the training approach, `v2/RESULTS_THREE_TASK.md`
for full results with significance testing, `v2/CONTRACTNLI_EVALUATION.md` and
`v2/EVIDENCE_EVALUATION.md` for the per-task breakdowns, and
`v2/TASK_EXAMPLES.md` for what the inputs and outputs actually look like.

Slides: `Contract_2.pptx`.
