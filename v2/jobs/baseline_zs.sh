#!/bin/bash
#SBATCH --job-name=zs
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=1:10:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
#   sbatch --job-name=zs_q4 --export=ALL,BASE=Qwen/Qwen3-4B,TAG=zs_q4 v2/jobs/baseline_zs.sh
# 40GB MIG slice: schedules far faster than a full H100 and fits an 8B at 16k.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
BASE=${BASE:?}; TAG=${TAG:?}
nvidia-smi --query-gpu=name,memory.total --format=csv
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python v2/eval_multitask.py --base "$BASE" --tag "$TAG" \
    --data v2/baseline_subset.jsonl \
    --out-dir /scratch/ibi761/legalai/v2_baseline \
    --gpu-mem-util 0.90 --max-new-tokens 512
echo "=== done ==="
