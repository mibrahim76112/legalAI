#!/bin/bash
#SBATCH --job-name=onedocS
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=40G
#SBATCH --time=0:18:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
# Deliberately tiny so it can drop into a short backfill gap: a 20GB MIG slice,
# 3 CPUs, 18 minutes. Qwen3-4B is ~8GB in bf16, leaving ~10GB of KV cache which
# is enough for 70 prompts at 16k with eager execution.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=name,memory.total --format=csv
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python v2/run_one_document.py --gpu-mem-util 0.92
