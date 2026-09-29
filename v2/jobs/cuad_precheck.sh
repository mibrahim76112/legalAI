#!/bin/bash
#SBATCH --account=def-ekram
#SBATCH --job-name=cuadpre
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/ibi761/legalai/logs/%x-%j.out
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source /scratch/ibi761/legalai/envs/prefetch/bin/activate
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "######## COST PRE-CHECK ########"
python v2/cuad_windowed.py --cost-only
echo; echo "######## COORDINATE MAPPING VALIDATION ########"
python v2/validate_coord_mapping.py
