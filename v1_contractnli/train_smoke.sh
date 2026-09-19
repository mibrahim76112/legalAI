#!/bin/bash
#SBATCH --job-name=sft_smoke
#SBATCH --account=def-ekram_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/sftsmoke_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/sftsmoke_%j.out
# Validate the whole SFT path on 64 examples before committing an 8h job:
# model load (gemma is Gemma4UnifiedForConditionalGeneration, not a plain
# CausalLM), LoRA attach, masked-loss forward/backward, checkpoint save.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SFT_DIR=/scratch/ibi761/legalai/sft_smoke
python train_sft.py --model google/gemma-4-12B-it --limit 64 --eval-n 16 \
       --epochs 1 --grad-accum 4 --seed 42
echo "=== smoke done ==="
