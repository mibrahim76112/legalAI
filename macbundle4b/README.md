# Qwen3-4B bundle — the fast local option

Same three tasks, same prompts, roughly half the memory and about twice the
speed of the 8B on a 16 GB Mac.

`adapter/` is **PEFT format** (`adapter_model.safetensors`), base
`Qwen/Qwen3-4B`, r=16, alpha=32 over the seven standard projections. Convert to
MLX or merge-and-quantize, same as the 8B.

## Why this is a reasonable downgrade

Measured on the identical 599-row stratified subset, both fine-tuned:

| metric | Qwen3-4B | Llama-3.1-8B |
|---|---|---|
| ContractNLI macro-F1 | 0.859 | 0.875 |
| Contradiction precision | 0.882 | 0.907 |
| **CUAD present-class F1** | **0.778** | 0.757 |
| CUAD accuracy | 0.820 | 0.816 |
| risk note ROUGE-L | 0.415 | 0.450 |

The 4B is **better on CUAD clause identification** and behind on the other two.
On full dev the ContractNLI gap (0.824 vs 0.847) IS separated by paired
bootstrap, so the 8B is genuinely stronger there; the CUAD gap is not separated
in either direction.

Local memory: 4-bit weights ~2.5 GB against ~5 GB for the 8B.

## Prompts and span location

`lib/` is identical to the 8B bundle: `prompts.py` (all three tasks, the 17
playbook positions, the 18 clause categories), `spans.py` (quoted text ->
character offsets), plus `text_norm.py`, `reasoning_filters.py`,
`doc_harness.py`.

One difference that matters: **Qwen3's chat template supports
`enable_thinking`, and Llama's does not.** Pass `enable_thinking=False` for
this model. The adapter was trained with thinking off; leaving it on puts the
model off-distribution and it will emit `<think>` blocks ahead of the JSON.

## Validation

`validation_60.jsonl` carries **Qwen3-4B's** cluster bf16 output in
`cluster_bf16_output`, not the 8B's. Agreement with that field isolates
quantization drift from model error. Do not compare it against the 8B bundle's
file — different model, different predictions.

Full-dev bf16 reference for this model: ContractNLI macro-F1 0.824,
Contradiction precision 0.623, CUAD present-class F1 0.799.
