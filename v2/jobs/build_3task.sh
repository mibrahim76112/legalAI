#!/bin/bash
#SBATCH --account=def-ekram
#SBATCH --job-name=build3
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/ibi761/legalai/logs/%x-%j.out
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source /scratch/ibi761/legalai/envs/prefetch/bin/activate
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python v2/build_combined.py --risk-notes v2/risk_notes --out v2/combined
