# Base-model screen — NDA playbook-compliance reviewer

Pre-SFT screen of four base models on ContractNLI. **No training happens here.**
Output: a ranked shortlist for fine-tuning, plus the base-model baseline rows of
the final benchmark table.

## What is measured

| | |
|---|---|
| Task | clause + policy -> `Entailment` / `Contradiction` / `NotMentioned` |
| Eval set | ContractNLI **dev**, 978 rows (47.6% E, 9.1% C, 43.3% NM) |
| Conditions | **A** zero-shot, **B** 3-shot balanced (one exemplar per class, TRAIN only) |
| Ranking metric | **macro-F1, condition B** |
| Diagnostics | per-class P/R/F1, JSON validity, parse-failure rate, mean output tokens |

Evidence-span extraction is deliberately excluded: a base model cannot know the
span format, so including it would measure format-guessing, not task ability.

The eval set is built by `prepare_data.py`, which **imports `02_build_sft_data.py`'s
own loader**. It is verified row-for-row identical to `sft_data/baseline/valid.jsonl`
(same order, same `chunk_id`, same golds), so these base numbers are a valid
control column for the post-SFT comparison.

## Fairness controls

Held fixed across all four models:
greedy decoding (temperature 0) · BF16 · one shared `max_new_tokens` (128) ·
**each model's own chat template** via `apply_chat_template` · one shared parser ·
no per-model prompt tuning.

Two things worth knowing:

- **Thinking mode.** Qwen3 and Granite 4.2 default their templates to a reasoning
  mode that emits a `<think>` block before the answer. Under a shared token cap
  that truncates before the JSON appears, which would score a capable model as
  unparseable — a harness artefact, not a capability gap. The harness detects the
  switch by *comparing rendered text* (templates silently ignore unknown kwargs,
  so a `TypeError` probe reports false support) and requests direct-answer mode.
  phi-4 has no such mode. Override with `--allow-thinking` to measure the default.
- **System prompt.** The brief's prompt and the one baked into
  `02_build_sft_data.py` are **different strings**. The screen uses the brief's
  (`--system spec`, default) since all four share it and ranking is unaffected.
  For a prompt-matched base column in the final table, re-run with `--system sft`.

## Run it

Everything below assumes `cd /project/6030214/ibrahim/legalAI`.

### 1. Prefetch models — LOGIN NODE (compute nodes have no internet)

```bash
module load python/3.11
source /scratch/ibi761/legalai/envs/prefetch/bin/activate
export HF_HOME=/scratch/ibi761/legalai/hf_home
python prefetch_models.py            # ~93 GB into $SCRATCH, re-runnable
```

### 2. Build the eval set — LOGIN NODE

```bash
python 02_build_sft_data.py --variant baseline   # needs network (hypothesis ids)
python prepare_data.py                            # -> data/
```

### 3. Smoke test — 50 examples, one model, one GPU

```bash
sbatch smoke_test.sh
```
Confirms vLLM works on an H100 and that parsing is clean **before** committing
eight jobs. Check with `python inspect_raw.py --dir /scratch/ibi761/legalai/results_smoke`.

### 4. Full screen — 8 cells (4 models x 2 conditions), all parallel

```bash
sbatch run_all.sh
```
Array task `i` -> model `i/2` of `models.txt`, condition `i%2`
(`0`=zeroshot, `1`=threeshot). 1x H100, 12 CPU, 128G, 2h per cell.

### 5. Aggregate

```bash
python aggregate.py --dir /scratch/ibi761/legalai/results --out results
```

## Reading the output

`aggregate.py` reports a bootstrap 95% CI per cell and a **paired** bootstrap
between adjacently-ranked models. Pairing matters: both models scored identical
rows, so pairing removes example-difficulty variance.

Treat `WITHIN NOISE` literally. Dev has only **89 Contradiction examples**, and
macro-F1 weights that class at one third, so its CI is wide. A one-point macro-F1
gap on this set is not a basis for preferring one model over another — use the
diagnostics (Contradiction recall, JSON validity, output cost) to break such ties.

## Files

| file | role |
|---|---|
| `prefetch_models.py` | login-node download of all four models to `$SCRATCH` |
| `prepare_data.py` | frozen eval set + shared 3-shot exemplars |
| `harness_core.py` | prompts + answer parser (identical for all models) |
| `test_parser.py` | parser unit tests — run before spending GPU hours |
| `run_eval.py` | one (model, condition) cell via vLLM |
| `setup_env.sh` | in-job venv from the Alliance wheelhouse |
| `smoke_test.sh` | 50-example debug job |
| `run_all.sh` | 8-cell SLURM array |
| `inspect_raw.py` | parse-quality inspector |
| `aggregate.py` | metrics, results table, bootstrap noise analysis |

Raw outputs land in `/scratch/ibi761/legalai/results/raw__<model>__<condition>.jsonl`,
one row per example with `raw_output`, `parsed_verdict`, `gold_verdict`,
`chunk_id` and `hypothesis_id` for per-hypothesis error analysis.
