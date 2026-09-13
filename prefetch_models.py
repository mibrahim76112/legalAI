#!/usr/bin/env python3
"""
Prefetch all candidate base models to $SCRATCH so compute nodes (no internet)
can load them offline.

RUN THIS ON A LOGIN NODE (compute nodes have no outbound network).

  export HF_HOME=$SCRATCH/legalai/hf_home
  python prefetch_models.py

Re-runnable: snapshot_download resumes and skips already-complete files.
"""
import os
import sys
import json
import argparse
from pathlib import Path


def load_dotenv(path=".env"):
    """Gated repos (e.g. meta-llama/*) need HF_TOKEN; the repo keeps it in .env."""
    p = Path(path)
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return os.environ.get("HF_TOKEN")

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

# Weight shards we want; everything else (GGUF, ONNX, duplicate .bin, the
# `original/` mirrors some repos ship) is excluded so we don't burn quota.
ALLOW = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "vocab*",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.*",
    "preprocessor_config.json",
    # Multimodal repos (gemma-4-*) ship processor_config.json instead, and
    # vLLM's multimodal loader refuses to start without the processor files.
    # Matching only "preprocessor_config.json" silently omitted it.
    "processor_config.json",
    "*_config.json",
]
IGNORE = ["original/*", "*.gguf", "*.onnx", "*.pth", "consolidated*", "*.bin"]

# Files that must exist afterwards or the model is unusable offline.
REQUIRED = ["config.json"]


def read_models(path: Path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-file", default="models.txt")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    args = ap.parse_args()

    if not args.hf_home:
        sys.exit("HF_HOME not set. export HF_HOME=$SCRATCH/legalai/hf_home")

    token = load_dotenv() or os.environ.get("HF_TOKEN")
    os.environ["HF_HOME"] = args.hf_home
    # Must be OFF here - this is the one place we are allowed to hit the network.
    os.environ["HF_HUB_OFFLINE"] = "0"
    Path(args.hf_home).mkdir(parents=True, exist_ok=True)

    models = read_models(Path(args.models_file))
    print(f"HF_HOME  = {args.hf_home}")
    print(f"models   = {len(models)}\n")

    manifest, failures = {}, []
    for i, mid in enumerate(models):
        print(f"[{i+1}/{len(models)}] {mid}", flush=True)
        try:
            local = snapshot_download(
                repo_id=mid,
                allow_patterns=ALLOW,
                ignore_patterns=IGNORE,
                token=token,
                max_workers=4,   # gentle on the shared login node; resume is automatic in hub>=1.0
            )
        except GatedRepoError:
            print(f"    GATED - token present={bool(token)}; a 403 here means the "
                  f"licence has not been accepted for this account: {mid}", flush=True)
            failures.append((mid, "gated")); continue
        except RepositoryNotFoundError:
            print(f"    NOT FOUND: {mid}", flush=True)
            failures.append((mid, "not_found")); continue
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            failures.append((mid, repr(e))); continue

        p = Path(local)
        missing = [f for f in REQUIRED if not (p / f).exists()]
        shards = sorted(p.glob("*.safetensors"))
        size_gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9

        # A tokenizer is mandatory: the harness calls apply_chat_template().
        has_tok = any((p / n).exists() for n in
                      ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"))
        if not has_tok:
            missing.append("tokenizer*")

        if missing or not shards:
            print(f"    INCOMPLETE - missing={missing} shards={len(shards)}", flush=True)
            failures.append((mid, f"incomplete: {missing}")); continue

        print(f"    OK  {len(shards)} shards, {size_gb:.1f} GB -> {local}", flush=True)
        manifest[mid] = {"path": str(local), "shards": len(shards), "size_gb": round(size_gb, 1)}

    # Merge rather than replace: a partial run (e.g. --models-file with one
    # extra candidate) must not erase the record of the already-cached models.
    out = Path(args.hf_home) / "prefetch_manifest.json"
    merged = {}
    if out.exists():
        try:
            merged = json.loads(out.read_text())
        except Exception:
            merged = {}
    merged.update(manifest)
    out.write_text(json.dumps(merged, indent=2))
    print(f"\nmanifest -> {out}")
    print(f"total {sum(m['size_gb'] for m in merged.values()):.1f} GB ({len(merged)} models cached)")

    if failures:
        print("\nFAILURES:")
        for mid, why in failures:
            print(f"  {mid}: {why}")
        sys.exit(1)
    print("\nAll models cached. Jobs can run with HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
