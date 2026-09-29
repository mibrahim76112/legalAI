#!/bin/bash
#SBATCH --job-name=qnf4
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
# Full H100, short walltime. The 2g.20gb slices are NOT memory-isolated on this
# cluster -- two foreign processes holding 12.5 GiB each showed up inside a
# 19.62 GiB allocation and OOMed my engine twice.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source v1_contractnli/setup_env.sh
pip install --no-index bitsandbytes >/dev/null 2>&1
python -c "import bitsandbytes as b; print('[quant] bitsandbytes', b.__version__)"
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
S=/scratch/ibi761/legalai/v2_sft
python v2/eval_multitask.py \
  --base NousResearch/Meta-Llama-3.1-8B-Instruct \
  --adapter $S/NousResearch__Meta-Llama-3.1-8B-Instruct__3task/final \
  --tag qs_llama_nf4 --data v2/quant_subset.jsonl \
  --out-dir /scratch/ibi761/legalai/v2_baseline --quantization bitsandbytes
echo "=== done ==="
