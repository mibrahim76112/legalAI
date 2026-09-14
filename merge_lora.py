#!/usr/bin/env python3
"""
Merge a LoRA adapter into its base model so the result can be served by vLLM
exactly like a base model.

Merging rather than vLLM's runtime LoRA: it keeps Phase 3 inference code
identical for base and fine-tuned models (one --model path, no LoRARequest
branch), which is what makes base-vs-tuned a clean comparison. It also sidesteps
vLLM LoRA support gaps for unusual architectures like Gemma4Unified.
"""
import argparse, json, shutil
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    import transformers
    from peft import PeftModel

    last = None
    base = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                     "AutoModelForVision2Seq"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            base = cls.from_pretrained(args.base, dtype=torch.bfloat16,
                                       trust_remote_code=True)
            print(f"[load] {cls_name} -> {type(base).__name__}")
            break
        except Exception as e:
            last = f"{cls_name}: {type(e).__name__}: {str(e)[:110]}"
    if base is None:
        raise SystemExit(f"could not load base. last: {last}")

    print(f"[merge] applying adapter {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(str(out))
    # multimodal repos need the processor config alongside, or vLLM refuses
    for extra in ("processor_config.json", "preprocessor_config.json",
                  "chat_template.jinja"):
        src = Path(args.base) / extra
        if src.exists():
            shutil.copy(src, out / extra)
    print(f"[done] merged -> {out}")


if __name__ == "__main__":
    main()
