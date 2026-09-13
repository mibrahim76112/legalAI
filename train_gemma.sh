#!/bin/bash
#SBATCH --job-name=sft_gemma
#SBATCH --account=def-ekram_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/sft_gemma_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/sft_gemma_%j.out
# PHASE 2, model 1 of 3: gemma-4-12B-it.
# Gemma goes first deliberately -- it is top-tier on Phase 1 AND it is the model
# whose train/infer template alignment needed the doc_sft_seq.py fix, so a
# successful run validates that fix before the other two depend on it.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=name,memory.total --format=csv
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SFT_DIR=/scratch/ibi761/legalai/sft_out
GPULOG=/scratch/ibi761/legalai/logs/sftgpu_gemma_$SLURM_JOB_ID.csv
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 15 > "$GPULOG" &
S=$!
python train_sft.py --model google/gemma-4-12B-it --seed 42
kill $S 2>/dev/null || true
echo "=== done ==="
