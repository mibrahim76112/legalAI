#!/usr/bin/env python3
"""Measure peak GPU memory for a LoRA fwd+bwd at several sequence lengths.

Settles decision D1: the choice of max_seq_len=16384 over 32768 rested on an
UNMEASURED assumption that a 14B LoRA will not fit one H100 at 32k. This
replaces the assumption with a number.

Config mirrors v1_contractnli/train_sft.py exactly: bf16, gradient
checkpointing on, LoRA r=16 alpha=32 over the 7 standard projections, AdamW,
per-device batch size 1. Attention implementation is left at the transformers
default, as in v1 -- overriding it here would measure a config v1 never used.
"""
import os, json, gc, argparse, traceback
import torch


def probe(model_id, seq_len, lora_r=16, lora_alpha=32):
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "v1_contractnli"))
    sys.path.insert(0, str(_P(__file__).resolve().parent))
    from model_loading import load_model
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = load_model(model_id, torch.bfloat16)
    # NemotronH: transformers 5.14 builds a DynamicCache from config.layer_types,
    # which for nemotron_h contains 27 'linear_attention', 25 'mlp' and 4
    # 'full_attention' entries. DYNAMIC_LAYER_TYPE_MAPPING has no 'mlp' key, so
    # the forward dies with KeyError: 'mlp'. The construction is guarded by
    # `if use_cache and past_key_values is None`, so disabling the cache avoids
    # it entirely -- and the cache is useless during checkpointed training
    # anyway. Set explicitly rather than relying on
    # gradient_checkpointing_enable() to do it.
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            try:
                model.config.text_config.use_cache = False
            except Exception:
                pass
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    tgts = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
            "down_proj", "in_proj", "out_proj"]
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    tgts = [t for t in tgts if t in present] or ["q_proj", "v_proj"]
    print(f"[lora] targets resolved: {tgts}")
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=tgts))
    model.cuda(); model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    V = int(getattr(model.config, "vocab_size", tok.vocab_size) or tok.vocab_size)
    ids = torch.randint(0, max(V - 1, 2), (1, seq_len), device="cuda")
    labels = ids.clone()
    labels[:, : seq_len // 2] = -100          # prompt masked, as in production

    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    out.loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    del model, opt, out, ids, labels
    gc.collect(); torch.cuda.empty_cache()
    return peak, reserved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seq-lens", nargs="+", type=int, default=[16384, 32768])
    ap.add_argument("--out", default="v2/_memory_probe.json")
    args = ap.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"[gpu] {torch.cuda.get_device_name(0)}  {total:.1f} GiB\n")
    res = []
    for m in args.models:
        for sl in args.seq_lens:
            try:
                peak, resv = probe(m, sl)
                fits = "FITS"
                print(f"{m:22s} seq={sl:6d}  peak={peak:6.1f} GiB  "
                      f"reserved={resv:6.1f} GiB  {fits}")
                res.append({"model": m, "seq_len": sl, "peak_gib": round(peak, 2),
                            "reserved_gib": round(resv, 2), "fits": True,
                            "gpu_total_gib": round(total, 1)})
            except torch.cuda.OutOfMemoryError:
                print(f"{m:22s} seq={sl:6d}  OOM")
                res.append({"model": m, "seq_len": sl, "fits": False,
                            "error": "OutOfMemoryError",
                            "gpu_total_gib": round(total, 1)})
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"{m:22s} seq={sl:6d}  ERROR {type(e).__name__}: {e}")
                traceback.print_exc()
                res.append({"model": m, "seq_len": sl, "fits": None,
                            "error": f"{type(e).__name__}: {str(e)[:200]}"})
                gc.collect(); torch.cuda.empty_cache()
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
