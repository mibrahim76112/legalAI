# Deploying the adapter behind a GPU endpoint

Local MLX on a 16 GB Mac runs at 6.5 tok/s (Llama-8B) or 10.8 tok/s
(Qwen3-4B), and the pipeline asks the model one question at a time. A GPU
endpoint fixes both: faster tokens, and many questions in flight at once.

## Which GPU

FP8 needs Ada or newer; the L4 qualifies.

| model | bf16 | FP8 | L4 24 GB ($0.80/h) |
|---|---|---|---|
| Llama-3.1-8B | 16 GB | 8.5 GB | both fit; bf16 is tight with a 16k KV cache |
| Qwen3-4B | 8 GB | 4 GB | both fit easily |

An L40S (48 GB, ~$1.80/h) buys nothing for a single-contract demo. Reach for it
only for long contexts at high concurrency.

## Path A: serve the adapter, no merge, no quantization

Upload the 127 MB (4B) or 161 MB (8B) adapter and let vLLM apply it to the
stock base model:

    vllm serve Qwen/Qwen3-4B --enable-lora \
      --lora-modules contract=/repo/adapter \
      --max-model-len 16384 --max-lora-rank 16

Fewest moving parts and nothing to re-validate beyond the endpoint itself.
Start here.

## Path B: merge and quantize to FP8

Roughly 1.5-2x the throughput of bf16 and frees GPU memory. Run
`merge_and_quantize.py` on the cluster or in Colab — not on the Mac, which
has no CUDA and would need the multi-GB bf16 base first.

    pip install torch transformers peft llmcompressor
    python merge_and_quantize.py --adapter macbundle4b/adapter --out qwen3-4b-contract-fp8

FP8_DYNAMIC is data-free, so there is no calibration set to get wrong. Only if
FP8 costs too much accuracy is W4A16 + GPTQ worth the calibration data (use
contracts from `sft_data`, never generic text).

## Push and create the endpoint

    hf auth login
    hf upload <you>/qwen3-4b-contract-fp8 ./qwen3-4b-contract-fp8 --private

At ui.endpoints.huggingface.co: New -> your repo -> engine vLLM -> AWS L4 x1 ->
Protected -> container arg `--max-model-len 16384`. Scale-to-zero is fine for
rehearsals; turn it off on demo day so the first request isn't a cold start.

## Prompt parity, or the numbers stop meaning anything

* The system prompts must stay byte-identical to `macbundle*/lib/prompts.py`.
* Task 3 uses the SHORT prompt (`TASK3_SYSTEM_TRAINED` in
  `inference/pipeline.py`), not `prompts.TASK3_SYSTEM`.
* `temperature=0`, `max_tokens=1024`.
* Qwen3 only: `chat_template_kwargs={"enable_thinking": False}`. The adapter
  was trained with thinking off; leaving it on emits `<think>` blocks and the
  JSON parser rejects them. Llama's template ignores the flag.

## Validate the endpoint before trusting it

Same check as the local build, against the same reference:

    python inference/validate_drift.py --backend remote \
      --endpoint https://<id>.endpoints.huggingface.cloud/v1 \
      --data macbundle4b/validation_60.jsonl --out results/drift_endpoint.jsonl

Compare label agreement with `cluster_bf16_output` to the local figures in
`inference/README.md`. A quantized endpoint that agrees less than local 4-bit
is a regression, however fast it is.

## One non-technical point

Contracts currently never leave the Mac. A Protected endpoint is token-gated,
but the text does travel to HF's cloud and sits in vLLM's memory there. Fine
for test NDAs; a disclosure decision for a real client contract.
