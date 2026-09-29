#!/bin/bash
#SBATCH --account=def-ekram
#SBATCH --job-name=neardup
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/scratch/ibi761/legalai/logs/%x-%j.out
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source /scratch/ibi761/legalai/envs/prefetch/bin/activate
export TOKENIZERS_PARALLELISM=false
python v2/near_dup_check.py
