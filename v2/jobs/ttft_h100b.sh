#!/bin/bash
#SBATCH --job-name=ttfth2
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=0:20:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
# The other two models, same protocol: concurrency 1, prefix caching OFF,
# full H100 so the numbers match the q4/llama run exactly.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=name,memory.total --format=csv
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
S=/scratch/ibi761/legalai/v2_sft
python v2/measure_ttft.py --base Qwen/Qwen3-8B \
  --adapter $S/Qwen__Qwen3-8B__3task/final --tag h100_q8 --n 6 --max-model-len 16384
python v2/measure_ttft.py --base Qwen/Qwen3-14B \
  --adapter $S/Qwen__Qwen3-14B__3task/final --tag h100_q14 --n 6 --max-model-len 16384
