#!/bin/bash
#SBATCH --account=def-ekram
#SBATCH --job-name=wincost
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/ibi761/legalai/logs/%x-%j.out
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source /scratch/ibi761/legalai/envs/prefetch/bin/activate
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python v2/window_cuad.py --base Qwen/Qwen3-8B --cost-only --tag cost
