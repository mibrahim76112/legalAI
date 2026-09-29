#!/bin/bash
#SBATCH --job-name=win_D
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python v2/cuad_windowed.py --base Qwen/Qwen3-4B \
    --adapter /scratch/ibi761/legalai/v2_sft/Qwen__Qwen3-4B__3task_win/final \
    --tag q4_wintrain_win
