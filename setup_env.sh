#!/bin/bash
# Build the vLLM venv inside a job, from the Alliance wheelhouse only.
#
# Built in $SLURM_TMPDIR (node-local) rather than a shared venv on /scratch:
# eight concurrent jobs importing from one networked venv hammers the metadata
# server, and node-local is both faster and immune to that contention.
#
#   source setup_env.sh
set -euo pipefail

module purge
# opencv module is REQUIRED before the venv: vllm depends on
# opencv-python-headless, and the wheelhouse ships a dummy wheel that errors
# out telling you to load the module instead. opencv needs cuda/12.9 (not 12.6).
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.9 opencv/4.13.0 arrow/17.0.0

VENV=${SLURM_TMPDIR:-/tmp/$USER}/vllm_env
if [ ! -x "$VENV/bin/python" ]; then
    echo "[setup_env] building venv at $VENV"
    virtualenv --no-download "$VENV"
    source "$VENV/bin/activate"
    pip install --no-index --upgrade pip
    # vllm pulls torch/xformers etc. from the wheelhouse as dependencies
    # pillow/torchvision: gemma-4-12B-it is Gemma4UnifiedForConditionalGeneration
    # (text+vision+audio), so transformers gates its processor behind
    # is_vision_available() even for pure-text use. Without them vLLM refuses
    # to build the model at all.
    pip install --no-index vllm transformers jinja2 pillow torchvision
else
    source "$VENV/bin/activate"
fi

# --- per-job, node-local caches + spawn -------------------------------------
# Two failures in the first Phase 1 launch came from here, not from the models:
#   1. "Cannot re-initialize CUDA in forked subprocess" - vLLM's default fork
#      start method breaks when CUDA is already initialised (seen on MIG).
#   2. "OSError: [Errno 116] Stale file handle" inside torch inductor - the
#      compile caches defaulted to $HOME on a networked filesystem, and two
#      array tasks landed on the SAME node and raced each other.
# Pinning every cache under $SLURM_TMPDIR makes them node-local and per-job.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
_C=${SLURM_TMPDIR:-/tmp/$USER}/cache
mkdir -p "$_C"/{vllm,inductor,triton,flashinfer}
export VLLM_CACHE_ROOT="$_C/vllm"
export TORCHINDUCTOR_CACHE_DIR="$_C/inductor"
export TRITON_CACHE_DIR="$_C/triton"
export FLASHINFER_WORKSPACE_DIR="$_C/flashinfer"

python -c "import vllm, torch, transformers; print(
    f'[setup_env] vllm={vllm.__version__} torch={torch.__version__} '
    f'transformers={transformers.__version__} cuda={torch.cuda.is_available()}')"
