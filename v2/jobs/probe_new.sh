#!/bin/bash
#SBATCH --job-name=probenew
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
# Probe the two NEW candidates before committing multi-hour training runs.
# Gemma-4 has a 262,144 vocab (16.0 GiB per fp32 logits copy at 16,384 vs
# Qwen's 9.3) and Qwen3-14B already peaked at 91% of the card.
# Nemotron is a hybrid Mamba-Transformer: this also confirms it LOADS and that
# the LoRA target list resolves to the mixer projections, not just 4 attention
# layers.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=name,memory.total --format=csv
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python v2/probe_memory.py \
    --models nvidia/NVIDIA-Nemotron-Nano-9B-v2 google/gemma-4-12B-it \
    --seq-lens 16384 \
    --out v2/_memory_probe_new.json
