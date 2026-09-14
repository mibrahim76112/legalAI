#!/bin/bash
#SBATCH --job-name=sft_eval
#SBATCH --account=def-ekram_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/sfteval_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/sfteval_%j.out
# Merge the LoRA adapter, then evaluate on dev with the SAME harness and the
# SAME settings as the Phase 1 base run, so the delta is attributable to SFT.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
MODEL=${MODEL:?set MODEL}
SLUG=$(echo "$MODEL" | tr '/' '_')
ADAPTER=/scratch/ibi761/legalai/sft_out/$(echo "$MODEL" | sed 's|/|__|')/adapter
MERGED=$SLURM_TMPDIR/merged_$SLUG
BASEDIR=$(python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL', local_files_only=True))")
echo "base=$BASEDIR"; echo "adapter=$ADAPTER"
python merge_lora.py --base "$BASEDIR" --adapter "$ADAPTER" --out "$MERGED"
export RESULTS_DIR=/scratch/ibi761/legalai/doc_results_sft
python doc_eval.py --model "$MERGED" --condition nothink --split dev \
       --max-new-tokens 1024 --tag sft
echo "=== done ==="
