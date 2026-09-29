#!/bin/bash
#SBATCH --job-name=ttfth100
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=name,memory.total --format=csv
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
S=/scratch/ibi761/legalai/v2_sft
for spec in "q4:Qwen/Qwen3-4B:Qwen__Qwen3-4B__3task" \
            "llama:NousResearch/Meta-Llama-3.1-8B-Instruct:NousResearch__Meta-Llama-3.1-8B-Instruct__3task"; do
  IFS=':' read -r tag base dir <<< "$spec"
  python v2/measure_ttft.py --base "$base" --adapter "$S/$dir/final" \
     --tag "h100_$tag" --n 6 --max-model-len 16384 || echo "SKIP $tag"
done
