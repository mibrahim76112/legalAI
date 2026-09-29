#!/bin/bash
#SBATCH --job-name=ttft
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
S=/scratch/ibi761/legalai/v2_sft
python v2/measure_ttft.py --base Qwen/Qwen3-4B \
  --adapter $S/Qwen__Qwen3-4B__3task/final --tag q4 --n 10
python v2/measure_ttft.py --base NousResearch/Meta-Llama-3.1-8B-Instruct \
  --adapter $S/NousResearch__Meta-Llama-3.1-8B-Instruct__3task/final --tag llama --n 10
