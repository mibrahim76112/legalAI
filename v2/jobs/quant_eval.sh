#!/bin/bash
#SBATCH --job-name=q
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=1:10:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
# Inference-time quantization of the FROZEN BASE. The adapter is untouched, so
# this isolates post-training quantization from anything about training.
# Same 599-row subset as the base-model baseline, so all three precisions and
# the zero-shot numbers sit on identical examples.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
BASE=${BASE:?}; ADAPTER=${ADAPTER:?}; TAG=${TAG:?}; QUANT=${QUANT:-}
nvidia-smi --query-gpu=name,memory.total --format=csv
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python v2/eval_multitask.py --base "$BASE" --adapter "$ADAPTER" --tag "$TAG" \
    --data v2/baseline_subset.jsonl \
    --out-dir /scratch/ibi761/legalai/v2_baseline \
    --gpu-mem-util 0.90 ${QUANT:+--quantization "$QUANT"}
echo "=== done ==="
