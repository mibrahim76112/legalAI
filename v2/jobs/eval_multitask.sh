#!/bin/bash
#SBATCH --job-name=v2eval
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
#   sbatch --job-name=ev_q4 --export=ALL,BASE=Qwen/Qwen3-4B,ADAPTER=/path/final,TAG=q4_3task v2/jobs/eval_multitask.sh
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
BASE=${BASE:?set BASE}
ADAPTER=${ADAPTER:-}
TAG=${TAG:-}
DATA=${DATA:-v2/combined/dev.jsonl}
EXTRA=${EXTRA:-}
echo "=== eval $BASE adapter=$ADAPTER on $(hostname) ==="
source v1_contractnli/setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python v2/eval_multitask.py --base "$BASE" ${ADAPTER:+--adapter "$ADAPTER"} \
    ${TAG:+--tag "$TAG"} --data "$DATA" $EXTRA
echo "=== done ==="
