#!/bin/bash
#SBATCH --job-name=ttft
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=40G
#SBATCH --time=0:25:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=name,memory.total --format=csv
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
S=/scratch/ibi761/legalai/v2_sft
python v2/measure_ttft.py --base Qwen/Qwen3-4B \
  --adapter $S/Qwen__Qwen3-4B__3task/final --tag q4 --n 8 --max-model-len 8192
python v2/measure_ttft.py --base NousResearch/Meta-Llama-3.1-8B-Instruct \
  --adapter $S/NousResearch__Meta-Llama-3.1-8B-Instruct__3task/final \
  --tag llama --n 8 --max-model-len 8192
