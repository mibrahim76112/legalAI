#!/bin/bash
#SBATCH --job-name=reasongen
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/reasongen_%x_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/reasongen_%x_%j.out
# Arm B teacher generation. Qwen3-32B is 66GB, so 2x H100 (160GB) is ample:
# ~33GB of weights per card leaves ~47GB each for KV cache at 16k context.
# 2 GPUs also schedules immediately where 4 waits ~26 min on this queue.
#   sbatch --job-name=p1 --export=ALL,PHASE=1 gen_reasoning.sh
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=name,memory.total --format=csv
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python gen_reasoning.py --teacher "${TEACHER:-Qwen/Qwen3-32B}" \
       --pass "${PHASE:?set PHASE}" --tp "${NGPU:-2}" ${LIMIT:+--limit $LIMIT} \
       ${ALLROWS:+--all-rows} ${TAG:+--tag $TAG}
echo "=== done pass ${PHASE} ==="
