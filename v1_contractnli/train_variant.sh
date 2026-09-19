#!/bin/bash
#SBATCH --job-name=sftvar
#SBATCH --account=def-ekram_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/sft_%x_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/sft_%x_%j.out
# Contradiction-recall experiment. Baseline Gemma SFT traded recall
# (0.716 -> 0.547) for precision (0.463 -> 0.693). Two independent levers,
# same model, same seed, everything else identical to the baseline run:
#   OVERSAMPLE=4  -> data-side: Contradiction 11.7% -> 34.6% of train rows
#   CLASSW=3.0    -> loss-side: Contradiction examples weighted 3x
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
MODEL=${MODEL:?}; VARIANT=${VARIANT:?}
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SFT_DIR=/scratch/ibi761/legalai/sft_out
python train_sft.py --model "$MODEL" --seed 42 --variant "$VARIANT" \
       --oversample-contradiction "${OVERSAMPLE:-1}" \
       --class-weight-contradiction "${CLASSW:-1.0}"
echo "=== done $MODEL / $VARIANT ==="
