#!/bin/bash
#SBATCH --job-name=evval
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=880G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/evval_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/evval_%j.out
# EVADE validation. Mixtral-8x22B (281GB BF16) tensor-parallel over 8 H100s:
# ~35GB of weights per card, leaving ~45GB each for KV at 16k context.
# 2h walltime, not 4 -- 7188 short generations (~200 output tokens each) should
# finish well inside an hour, and the shorter ask schedules materially better
# against a 1755-job pending queue.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=index,name,memory.total --format=csv
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python validate_evade.py --tp 8 ${LIMIT:+--limit $LIMIT} \
       --rationales "${RATS:-reasoning/pass2_raw_cond.jsonl}"
echo "=== validation done ==="
