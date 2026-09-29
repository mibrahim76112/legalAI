#!/bin/bash
#SBATCH --job-name=probenem
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=name,memory.total --format=csv
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# Nemotron at 16384 (the standard budget) and, since Gemma-4 OOMed at 16384,
# Gemma-4 at 8192 purely to establish WHERE its ceiling is -- not to train it
# there, which would confound the model comparison with a context comparison.
python v2/probe_memory.py --models nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --seq-lens 16384 --out v2/_memory_probe_nemotron.json
python v2/probe_memory.py --models google/gemma-4-12B-it \
    --seq-lens 8192 4096 --out v2/_memory_probe_gemma_short.json
