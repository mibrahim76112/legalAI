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
    pip install --no-index vllm transformers jinja2
else
    source "$VENV/bin/activate"
fi

python -c "import vllm, torch, transformers; print(
    f'[setup_env] vllm={vllm.__version__} torch={torch.__version__} '
    f'transformers={transformers.__version__} cuda={torch.cuda.is_available()}')"
