#!/bin/bash
#SBATCH --job-name=risknote
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/risknote_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/risknote_%j.out
# Task 3 risk notes. Teacher Qwen3-32B (apache-2.0, cached, our own hardware --
# no frontier API). One note per present=true CUAD row; splits inherited from
# the source contract.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python v2/gen_risk_notes.py --teacher Qwen/Qwen3-32B --tp 2 ${LIMIT:+--limit $LIMIT}
echo "=== done ==="
