"""Merge a PEFT LoRA adapter into its base model and quantize to FP8.

Run on a CUDA box (cluster or Colab), not the Mac: it needs the bf16 base in
PyTorch. FP8_DYNAMIC is data-free, so there is no calibration set.

    python deploy/merge_and_quantize.py --adapter macbundle4b/adapter \
        --out qwen3-4b-contract-fp8

Skip this entirely if you serve the adapter with vLLM's --enable-lora
(see README.md, path A).
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="folder with adapter_config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merged", default=None, help="where to keep the bf16 merge (default: <out>-bf16)")
    ap.add_argument("--scheme", default="FP8_DYNAMIC", choices=["FP8_DYNAMIC", "none"],
                    help="none = stop after the bf16 merge")
    args = ap.parse_args()

    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    cfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
    base = cfg["base_model_name_or_path"]
    merged = args.merged or f"{args.out}-bf16"
    print(f"base {base}  r={cfg['r']} alpha={cfg['lora_alpha']}")

    # dtype=, not torch_dtype=, which current Transformers deprecates
    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter, dtype="bfloat16",
                                                     device_map="auto")
    model = model.merge_and_unload()
    model.save_pretrained(merged)
    # tokenizer from the base: the adapter never changes it
    AutoTokenizer.from_pretrained(base).save_pretrained(merged)
    print(f"merged -> {merged}")

    if args.scheme == "none":
        return

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    oneshot(model=merged,
            recipe=QuantizationModifier(targets="Linear", scheme=args.scheme,
                                        ignore=["lm_head"]),
            output_dir=args.out)
    print(f"quantized -> {args.out}")
    print("sanity check:  vllm serve ./" + args.out + " --max-model-len 16384")


if __name__ == "__main__":
    main()
