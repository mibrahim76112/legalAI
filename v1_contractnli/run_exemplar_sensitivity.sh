#!/bin/bash
#SBATCH --job-name=nli_exvar
#SBATCH --account=def-ekram_gpu
#SBATCH --array=0-15
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/exvar_%A_%a.out
#SBATCH --error=/scratch/ibi761/legalai/logs/exvar_%A_%a.out

# W2: exemplar sensitivity. 4 models x 4 alternative balanced triples, 3-shot.
# Combined with the existing median-length triple this gives 5 observations per
# model, so we can report the spread of macro-F1 attributable to exemplar
# choice rather than assuming it is zero.
#   task -> model = id/4 (line of models.txt), triple = id%4 + 1
set -euo pipefail
cd /project/6030214/ibrahim/legalAI

MODEL=$(sed -n "$(( SLURM_ARRAY_TASK_ID / 4 + 1 ))p" models.txt)
T=$(( SLURM_ARRAY_TASK_ID % 4 + 1 ))

echo "=== task $SLURM_ARRAY_TASK_ID : $MODEL / triple t$T ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RESULTS_DIR=/scratch/ibi761/legalai/results_exvar

python run_eval.py --model "$MODEL" --condition threeshot \
       --fewshot-file "data/fewshot_t${T}.json" --tag "t${T}"

echo "=== done $MODEL / t$T ==="
