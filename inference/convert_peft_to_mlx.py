"""Convert the cluster's PEFT LoRA adapter to the layout mlx-lm loads.

PEFT stores lora_A as (r, in) and lora_B as (out, r) and computes
x @ A.T @ B.T * (alpha / r). mlx-lm's LoRALinear stores lora_a as (in, r) and
lora_b as (r, out) and computes x @ lora_a @ lora_b * scale. So each matrix is
transposed and scale = alpha / r.

    python inference/convert_peft_to_mlx.py macbundle/adapter adapters/llama_3task_mlx
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

KEY_RE = re.compile(
    r"^base_model\.model\.(model\.layers\.\d+\.(?:self_attn|mlp)\.\w+_proj)\.lora_([AB])\.weight$"
)


def main(src, dst):
    src, dst = Path(src), Path(dst)
    peft = json.loads((src / "adapter_config.json").read_text())
    assert peft["peft_type"] == "LORA" and not peft.get("use_dora")
    assert not peft.get("use_rslora"), "rslora uses alpha/sqrt(r)"
    tensors = load_file(str(src / "adapter_model.safetensors"))

    out, layers, modules = {}, set(), set()
    for k, v in tensors.items():
        m = KEY_RE.match(k)
        if not m:
            raise ValueError(f"unexpected key {k}")
        path, ab = m.groups()
        out[f"{path}.lora_{ab.lower()}"] = np.ascontiguousarray(v.T)
        layers.add(int(path.split(".")[2]))
        modules.add(".".join(path.split(".")[3:]))

    r = peft["r"]
    cfg = {
        "fine_tune_type": "lora",
        "num_layers": max(layers) + 1,
        "lora_parameters": {
            "rank": r,
            "scale": peft["lora_alpha"] / r,
            "dropout": 0.0,
            "keys": sorted(modules),
        },
        "base_model_peft": peft["base_model_name_or_path"],
    }
    assert layers == set(range(cfg["num_layers"])), "adapter must cover every layer"
    for key, a in out.items():
        if key.endswith("lora_a"):
            assert a.shape[1] == r, (key, a.shape)

    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "adapters.safetensors"))
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{len(out)} tensors, {cfg['num_layers']} layers, modules {cfg['lora_parameters']['keys']}, "
          f"scale {cfg['lora_parameters']['scale']} -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
