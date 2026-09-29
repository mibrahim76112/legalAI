# Local inference bundle

## The adapter is PEFT, not MLX

`adapter/adapter_model.safetensors` — 448 tensors, keys shaped
`base_model.model.model.layers.N.<proj>.lora_{A,B}.weight`. So it is the
second of your two cases: convert to MLX format, or merge into the base and
quantize.

Config: r=16, alpha=32, dropout 0.05, over
`q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`,
base `NousResearch/Meta-Llama-3.1-8B-Instruct` (a verified mirror of the Meta
release, used because the Meta repo is gated).

Trained in **bf16** on the full-precision base, so your instinct is right: a
4-bit base will drift. `validation_60.jsonl` is there to measure how much.

## Contents

| path | what |
|---|---|
| `adapter/` | the trained adapter, 161 MB |
| `lib/prompts.py` | all three task prompts + the 17 positions and 18 categories |
| `lib/spans.py` | span text → character offsets, standalone |
| `lib/text_norm.py` | the single normalizer everything shares |
| `lib/reasoning_filters.py` | `_grounded`, the substring primitive `spans.py` needs |
| `lib/doc_harness.py` | `parse_output`, `strip_wrappers`, `_find_json` — output parsing |
| `validation_60.jsonl` | 60 dev examples with the cluster's bf16 output |

## Two corrections to your plan

**The span routine is not in `build_demo_data.py`.** That file only calls it.
It lives in `v2/cuad_windowed.py` and depends on `text_norm.py` and
`reasoning_filters.py`. `lib/spans.py` is that routine extracted with its
dependencies, so you don't need the build script at all.

**`norm_with_map` must agree with `norm` exactly.** It rebuilds the normalized
string while tracking a per-character index back into the original, which is
the only reason offsets are recoverable. If you touch it, assert
`norm_with_map(s)[0] == norm(s)`.

Validation on the cluster: 745/745 gold spans round-tripped, plus 227/227
sampled from window-overlap regions, where round-trip means slicing the
**original** contract at the computed offsets and matching after normalization.

## Whatever adapters are already on your Mac are not from this project

There is no Qwen2.5-3B anywhere in this work, and no 20-iteration run. The only
smoke adapter here is Qwen3-4B over 64 rows. Ignore what you found.

## Prompts: use them exactly

The model was trained on these strings, including the JSON shape stated in the
system message. Deviating puts it off-distribution.

- Task 1 → `{"verdict": "Entailment|Contradiction|NotMentioned", "evidence": [...]}`
- Task 2 → `{"present": true|false, "evidence": [...]}`
- Task 3 → one sentence of prose, no JSON

`apply_chat_template(msgs, add_generation_prompt=True)`, greedy, max 1024 new
tokens. For Qwen-family templates also pass `enable_thinking=False`; Llama's
template ignores it. Detect support by rendering both ways and comparing the
text, because a template that silently swallows an unknown kwarg looks
supported when probed with a try/except.

## Measuring the 4-bit drift

`validation_60.jsonl` is stratified on purpose: 24 ContractNLI (8 of each
verdict), 24 CUAD (12 present, 12 absent), 12 risk notes. Each row carries
`messages`, `gold_target`, and `cluster_bf16_output`.

Agreement with `cluster_bf16_output` is the number you want — it isolates
quantization from model error. Compare against the cluster's dev figures:

| metric | bf16 on full dev |
|---|---|
| ContractNLI macro-F1 | 0.847 |
| ContractNLI Contradiction precision | 0.670 |
| CUAD present-class F1 | 0.808 |

Do not report local numbers as the project's results. 60 rows is a smoke check,
not an evaluation, and the split is dev.

## Context

ContractNLI prompts are p50 2,272 and max 6,520 tokens, so they fit anywhere.
CUAD reaches 83,679 tokens on the worst contract. Training and evaluation both
ran at 16,384 total with 1,024 reserved for generation. On 16 GB, cap context
at 16,384 and expect long CUAD documents to truncate from the left, which is
what the cluster did too.
