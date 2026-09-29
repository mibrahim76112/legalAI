#!/bin/bash
#SBATCH --job-name=onedocmig
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=0:35:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
# 70 prompts on a 40GB MIG slice: both task heads over both demo documents.
# MIG queues far shorter than a full H100 and 40GB is ample for a 4B at 16k.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=name,memory.total --format=csv
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python v2/run_one_document.py --gpu-mem-util 0.85
