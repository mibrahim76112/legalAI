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

## Rebuilding the training data

The `doc_sft*/` and `reasoning/` JSONL files are not tracked — they are derived
from ContractNLI and came to 618 MB, which is a lot to clone for something a
build script reproduces. Run these from inside `v1_contractnli/`, in order; the
later steps read the earlier ones.

| step | command | writes |
|---|---|---|
| 1 | `python build_doc_sft.py` | `doc_sft/` — downloads ContractNLI from the Hub |
| 2 | `python build_oracle_eval.py` | `doc_sft_oracle/` — reads `doc_sft/valid.jsonl` |
| 3 | `bash gen_reasoning.sh` | `reasoning/` — **needs a GPU**, ~3h |
| 4 | `python build_armb_data.py` | `doc_sft_armb/` — reads `reasoning/pass1_raw.jsonl` and `doc_sft/train.jsonl` |

Everything except step 3 is CPU text processing. Seeds and the drop list are
fixed in the scripts, so the splits come back identical — check a rebuild
against the tracked `doc_sft/_manifest.json` and `doc_sft_armb/_manifest.json`,
which record the per-split class counts, the yield, and the dropped rows.

Three ablation directories are **not** reproducible from what is committed.
`doc_sft_evfirst/` (evidence-first key order) and `doc_sft_amatch/` were built
by editing the builder rather than by a flag, and those edits were not kept;
`doc_sft_armb2/` was most likely `build_armb_data.py --f2-strict --out
doc_sft_armb2`, but it has no manifest to confirm it. None of them are needed to
reproduce the headline — they back the negative results above, and those numbers
are already written down in `ARMB_RESULTS.md` and `TRAINING_DIAGNOSTICS.md`.

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
