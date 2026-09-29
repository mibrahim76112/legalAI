#!/bin/bash
#SBATCH --job-name=qnf4
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=0:25:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
# Deliberately small: 160 rows, 25 minutes, so it fits short backfill gaps.
# The identical rows are rescored offline at bf16 and FP8 for a clean 3-way.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# setup_env.sh does not install bitsandbytes; vLLM's NF4 path imports it at
# engine init and dies with ModuleNotFoundError without it.
pip install --no-index bitsandbytes
python -c "import bitsandbytes; print('[quant] bitsandbytes', bitsandbytes.__version__)"
python v2/eval_multitask.py \
  --base NousResearch/Meta-Llama-3.1-8B-Instruct \
  --adapter /scratch/ibi761/legalai/v2_sft/NousResearch__Meta-Llama-3.1-8B-Instruct__3task/final \
  --tag qs_llama_nf4 --data v2/quant_subset.jsonl \
  --out-dir /scratch/ibi761/legalai/v2_baseline \
  --quantization bitsandbytes --gpu-mem-util 0.90
echo "=== done ==="
