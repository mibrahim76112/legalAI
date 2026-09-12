#!/bin/bash
#SBATCH --job-name=nli_check
#SBATCH --account=def-ekram_gpu
#SBATCH --array=0-2
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/check_%A_%a.out
#SBATCH --error=/scratch/ibi761/legalai/logs/check_%A_%a.out

# Verification cells for decision_reviews.md:
#   0,1 -> W1: granite-4.1-8b, both conditions. Tests whether "drop Granite"
#          is a model finding or an artefact of picking the 4.2 point release.
#     2 -> W3: Qwen3-8B 3-shot with thinking mode LEFT ON and a raised token
#          cap. Tests the counterfactual I asserted but never measured.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI

source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node $(hostname) task $SLURM_ARRAY_TASK_ID ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

case $SLURM_ARRAY_TASK_ID in
  0) export RESULTS_DIR=/scratch/ibi761/legalai/results
     python run_eval.py --model ibm-granite/granite-4.1-8b --condition zeroshot ;;
  1) export RESULTS_DIR=/scratch/ibi761/legalai/results
     python run_eval.py --model ibm-granite/granite-4.1-8b --condition threeshot ;;
  2) export RESULTS_DIR=/scratch/ibi761/legalai/results_thinking
     # thinking left on; cap raised so a think block can finish and the JSON
     # still fits, otherwise this would measure truncation, not capability
     python run_eval.py --model Qwen/Qwen3-8B --condition threeshot \
            --allow-thinking --max-new-tokens 1024 --max-model-len 4096 ;;
esac
echo "=== done task $SLURM_ARRAY_TASK_ID ==="
