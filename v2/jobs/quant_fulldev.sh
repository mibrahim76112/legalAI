#!/bin/bash
#SBATCH --job-name=qfull
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=3:30:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
# Full dev (2,489 rows) so a 0.01 difference is interpretable. bf16 on these
# same rows already exists as raw__llama_3task.jsonl, so this completes a
# three-way on identical examples at 4x the sample of the 599-row run and
# 15x the 160-row run.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
QUANT=${QUANT:?}; TAG=${TAG:?}
source v1_contractnli/setup_env.sh
[ "$QUANT" = "bitsandbytes" ] && pip install --no-index bitsandbytes >/dev/null 2>&1
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
S=/scratch/ibi761/legalai/v2_sft
python v2/eval_multitask.py \
  --base NousResearch/Meta-Llama-3.1-8B-Instruct \
  --adapter $S/NousResearch__Meta-Llama-3.1-8B-Instruct__3task/final \
  --tag "$TAG" --data v2/combined/dev.jsonl \
  --out-dir /scratch/ibi761/legalai/v2_baseline --quantization "$QUANT"
echo "=== done ==="
