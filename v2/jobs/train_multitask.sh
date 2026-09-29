#!/bin/bash
#SBATCH --job-name=v2sft
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=11:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
#   sbatch --job-name=v2_q14 --export=ALL,MODEL=Qwen/Qwen3-14B v2/jobs/train_multitask.sh
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
MODEL=${MODEL:?set MODEL}
MSL=${MSL:-16384}
EXTRA=${EXTRA:-}
echo "=== v2 multitask SFT $MODEL msl=$MSL on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SFT_DIR=/scratch/ibi761/legalai/v2_sft
GPULOG=/scratch/ibi761/legalai/logs/v2gpu_${SLURM_JOB_NAME}_$SLURM_JOB_ID.csv
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 60 > "$GPULOG" &
S=$!
python v2/train_multitask.py --model "$MODEL" --max-seq-len "$MSL" --seed 42 $EXTRA
kill $S 2>/dev/null || true
echo "=== done $MODEL ==="
