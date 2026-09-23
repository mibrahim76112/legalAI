# Local inference (Mac, MLX)

Llama-3.1-8B-Instruct 4-bit + the cluster's three-task LoRA adapter,
converted from PEFT. Needs `macbundle/` (from the cluster) alongside.

## One-time setup

```bash
venv-mlx/bin/python inference/convert_peft_to_mlx.py macbundle/adapter adapters/llama_3task_mlx
venv-mlx/bin/pip install pypdf
```

The 4-bit base (~4.5 GB) downloads on first run. If it stalls, set
`HF_HUB_DISABLE_XET=1`.

## Run the app

```bash
venv-mlx/bin/python inference/server.py     # terminal 1: wait for "model ready"
cd webapp && npm run dev                    # terminal 2: http://localhost:3000/reviews/new
```

Only one model process at a time: on 16 GB, a second copy pushes the Mac
into swap. Stop the server before running the scripts below.

## Scripts

| script | what |
|---|---|
| `run_file.py FILE --out X.json` | review one PDF/DOCX/TXT without the server |
| `validate_drift.py --out X.jsonl` | 4-bit vs cluster bf16 on `validation_60.jsonl` |

## Results so far (smoke checks, not project results)

* Drift, 60 rows: labels agree with the cluster on 23/24 ContractNLI and
  22/24 CUAD. Risk notes are never word-identical (free prose), token
  Jaccard 0.60.
* `test_nda.pdf`: 13/13 against `test_nda_expected.md` once PDF line wraps
  are rejoined; 12/13 before.
* Speed: ~2 min for a short NDA (17 positions); a contract near 16k tokens
  pays ~90 s of prefill per window, then a few seconds per question.

## Design choices to know

* **Windowing:** contracts over ~15k tokens are split into windows with
  1,024 tokens of overlap. Across windows Contradiction beats Entailment
  beats NotMentioned, and a clause is present if any window finds it.
* **Task-3 prompt:** the short system prompt in the validation rows, not
  `prompts.TASK3_SYSTEM`, which agreed with the cluster less.
* Jobs live in the server's memory; restarting it loses them.
