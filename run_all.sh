#!/bin/bash
#SBATCH --job-name=nli_screen
#SBATCH --account=def-ekram_gpu
#SBATCH --array=0-7
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/screen_%A_%a.out
#SBATCH --error=/scratch/ibi761/legalai/logs/screen_%A_%a.out

# 4 models x 2 conditions = 8 independent cells, one GPU each, all in parallel.
#   task id -> model = id/2 (line of models.txt), condition = id%2
set -euo pipefail
cd /project/6030214/ibrahim/legalAI

MODEL=$(sed -n "$(( SLURM_ARRAY_TASK_ID / 2 + 1 ))p" models.txt)
CONDS=(zeroshot threeshot)
COND=${CONDS[$(( SLURM_ARRAY_TASK_ID % 2 ))]}

echo "=== task $SLURM_ARRAY_TASK_ID : $MODEL / $COND ==="
echo "=== node $(hostname) job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
# GPU evidence for the infrastructure slide
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

source setup_env.sh

export HF_HOME=/scratch/ibi761/legalai/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RESULTS_DIR=/scratch/ibi761/legalai/results

python run_eval.py --model "$MODEL" --condition "$COND"

echo "=== peak GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
echo "=== done $MODEL / $COND ==="
