#!/bin/bash
#SBATCH --job-name=ckptsweep
#SBATCH --account=def-ekram_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/sweep_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/sweep_%j.out
# CHUNK 1. The diagnostic says 1 epoch overshoots on eval LOSS. But loss here is
# 93% JSON scaffolding + evidence copying and only 7% verdict, so it is a
# distrusted proxy -- tuning against it would be optimising the wrong thing.
# This run keeps every checkpoint so we can score them on the REAL metrics and
# find out (a) where the task optimum actually is and (b) whether eval loss
# predicts it. Only then is hyperparameter tuning justified.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SFT_DIR=/scratch/ibi761/legalai/sft_sweep
python train_sft.py --model Qwen/Qwen3-8B --seed 42 --variant sweep \
       --eval-steps 25 --save-steps 50 --save-total-limit 20
echo "=== sweep training done ==="
