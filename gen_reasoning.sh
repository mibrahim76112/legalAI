#!/bin/bash
#SBATCH --job-name=reasongen
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/reasongen_%x_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/reasongen_%x_%j.out
# Arm B teacher generation. Qwen3-32B on 4x H100, tensor-parallel.
#   sbatch --job-name=p1 --export=ALL,PHASE=1 gen_reasoning.sh
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=name,memory.total --format=csv
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python gen_reasoning.py --teacher "${TEACHER:-Qwen/Qwen3-32B}" \
       --pass "${PHASE:?set PHASE}" --tp 4 ${LIMIT:+--limit $LIMIT}
echo "=== done pass ${PHASE} ==="
