#!/usr/bin/env python3
"""Model loading for v2, wrapping v1's loader instead of editing it.

Two problems v1's loader does not cover:

1. NemotronH. transformers 5.14 supports `nemotron_h` natively, but the NVIDIA
   repo also ships custom modeling code, and `trust_remote_code=True` makes
   that remote code win. The remote class does not register with the Auto
   classes, so every AutoModel* attempt fails with "Unrecognized configuration
   class transformers_modules.nvidia...". Trying the NATIVE path first fixes it.

2. v1's loader reports only the LAST exception across four Auto classes, which
   hid the real cause above. This records every attempt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))


def load_model(model_id, dtype=None, quantization_config=None, **kw):
    """Native implementation first, remote-code fallback second."""
    import torch
    from transformers import AutoModelForCausalLM
    if dtype is None:
        dtype = torch.bfloat16
    common = dict(dtype=dtype, **kw)
    if quantization_config is not None:
        common["quantization_config"] = quantization_config
        common.setdefault("device_map", {"": 0})
    else:
        common.setdefault("device_map", None)

    attempts = []
    # 1. native, no remote code -- required for NemotronH
    try:
        m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False,
                                                 **common)
        print(f"[load] native AutoModelForCausalLM -> {type(m).__name__}")
        return m
    except Exception as e:
        attempts.append(f"native CausalLM: {type(e).__name__}: {str(e)[:160]}")

    # 2. the v1 ladder (CausalLM -> ImageTextToText -> Vision2Seq -> Seq2Seq)
    #    with remote code, which is what gemma-4-12B-it needs.
    import transformers
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                     "AutoModelForVision2Seq", "AutoModelForSeq2SeqLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            m = cls.from_pretrained(model_id, trust_remote_code=True, **common)
            print(f"[load] {cls_name} (remote code) -> {type(m).__name__}")
            return m
        except Exception as e:
            attempts.append(f"{cls_name}: {type(e).__name__}: {str(e)[:160]}")

    raise SystemExit("could not load " + model_id + "\n  " + "\n  ".join(attempts))
