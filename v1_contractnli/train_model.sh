#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --account=def-ekram_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/sft_%x_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/sft_%x_%j.out
# Generic Phase 2 LoRA SFT. Same hyperparameters and seed for every model so
# adaptation gain is comparable across them:
#   sbatch --job-name=q14 --export=ALL,MODEL=Qwen/Qwen3-14B train_model.sh
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
MODEL=${MODEL:?set MODEL}
echo "=== SFT $MODEL on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SFT_DIR=/scratch/ibi761/legalai/sft_out
GPULOG=/scratch/ibi761/legalai/logs/sftgpu_${SLURM_JOB_NAME}_$SLURM_JOB_ID.csv
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 30 > "$GPULOG" &
S=$!
python train_sft.py --model "$MODEL" --seed 42
kill $S 2>/dev/null || true
echo "=== done $MODEL ==="
