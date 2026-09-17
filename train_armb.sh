#!/bin/bash
#SBATCH --job-name=armb
#SBATCH --account=def-ekram_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/armb_%x_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/armb_%x_%j.out
# Arm B: reasoning-first targets. IDENTICAL to Arm A except --train/--valid and
# the target schema -- same LoRA r/alpha/dropout/targets, same LR/schedule/
# epochs/batch, same seed 42, same max_seq_len 8192 (Arm B p99 is 6480, so 8192
# still covers p99; 11 rows of 2733 truncate prompt-left, target preserved).
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
MODEL=${MODEL:?}
DATA=${DATA:-doc_sft_armb}
VAR=${VAR:-armb}
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SFT_DIR=/scratch/ibi761/legalai/sft_out
python train_sft.py --model "$MODEL" --seed 42 --variant "$VAR" \
       --train "$DATA/train.jsonl" --valid "$DATA/valid.jsonl" \
       --max-seq-len 8192
echo "=== done $MODEL $VAR ==="
