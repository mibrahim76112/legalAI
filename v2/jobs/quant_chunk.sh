#!/bin/bash
#SBATCH --job-name=qc
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=0:55:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
QUANT=${QUANT:?}; CHUNK=${CHUNK:?}; TAG=${TAG:?}
source v1_contractnli/setup_env.sh
[ "$QUANT" = "bitsandbytes" ] && pip install --no-index bitsandbytes >/dev/null 2>&1
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
S=/scratch/ibi761/legalai/v2_sft
python v2/eval_multitask.py \
  --base NousResearch/Meta-Llama-3.1-8B-Instruct \
  --adapter $S/NousResearch__Meta-Llama-3.1-8B-Instruct__3task/final \
  --tag "$TAG" --data v2/chunks/dev_$CHUNK.jsonl \
  --out-dir /scratch/ibi761/legalai/v2_baseline --quantization "$QUANT"
