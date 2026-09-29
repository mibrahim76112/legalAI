#!/bin/bash
#SBATCH --job-name=lbase
#SBATCH --account=def-ekram_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/llama_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/llama_%j.out
# Fills the Llama gap in the v1 Phase 1 table. Same harness, same settings as
# every other condition-A cell: zero-shot non-thinking, greedy, BF16,
# max_new_tokens 1024, dev split. v1 code is READ but not modified; raw output
# goes to scratch and the write-up to v2/.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI/v1_contractnli
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export RESULTS_DIR=/scratch/ibi761/legalai/doc_results
python doc_eval.py --model NousResearch/Meta-Llama-3.1-8B-Instruct \
       --condition nothink --split dev --max-new-tokens 1024 --tag mnt1024
echo "=== done ==="
