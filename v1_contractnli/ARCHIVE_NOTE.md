# v1 — ContractNLI document-level playbook compliance (archived)

Archived on project pivot. All jobs terminated, tree committed clean first.

## What is here

Everything from the v1 project: the clause-level screen, the document-level
Arm A / Arm B work, the STaR reasoning pipeline, EVADE preparation, all
diagnostics and findings. `git log --follow <file>` still works — files were
moved with `git mv`, so per-file history is intact.

## Headline results

| | |
|---|---|
| best config | Qwen3-14B, Arm A schema, 7,188 rows |
| macro-F1 (dev) | 0.839 |
| Contradiction precision | 0.650 |
| evidence F1 | 0.772 |
| hallucination rate | 0.3% (base models: 17.7–19.3%) |

**Test split was never touched.** Every number is dev.

## Negative results, all measured

- Reasoning in the target (Arm B) lost on 12/12 paired-bootstrap comparisons.
- Evidence-first key order: no effect, trending negative.
- LoRA rank 16 → 64: no effect (eval gap 0.011).
- Oracle evidence: perfect retrieval does **not** improve verdicts (±0.008).
- CUAD: rejected by the Phase 0 decision rule (14–16% vs a 40% threshold).

## Left unfinished

- EVADE validation — 7,188 conditioned rationales generated (3h02m GPU),
  never validated. Mixtral-8x22B was cached for this; job cancelled pending.
- Token-level reasoning reweighting — wired into `train_sft.py`, launched,
  cancelled at 13 min. No results.
- 3TF hybrid arm — designed, template verified, not built.
- Phase 3 test-split evaluation — never run.

## Scratch artifacts (OUTSIDE this repo, untouched)

`/scratch/ibi761/legalai/` — 490 GB total:

| path | size | contents |
|---|---|---|
| `hf_home/` | 456 GB | 9 model checkpoints incl. Mixtral-8x22B, Qwen3-32B |
| `sft_out/` | 28 GB | all LoRA adapters |
| `doc_results*/` | 57 MB | every raw prediction JSONL |
| `envs/prefetch/` | 278 MB | login-node venv |

Raw predictions are the valuable part — every metric in the reports recomputes
from them with no GPU.
