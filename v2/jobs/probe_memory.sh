#!/bin/bash
#SBATCH --job-name=memprobe
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
# Settles decision D1: does a 14B LoRA fit one H100 at seq_len 32768?
# 14B is probed FIRST because it is the decisive case; if the job runs out of
# wall time the answer that matters is already recorded.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=name,memory.total --format=csv
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python v2/probe_memory.py \
    --models Qwen/Qwen3-14B Qwen/Qwen3-8B NousResearch/Meta-Llama-3.1-8B-Instruct Qwen/Qwen3-4B \
    --seq-lens 16384 32768 \
    --out v2/_memory_probe.json
