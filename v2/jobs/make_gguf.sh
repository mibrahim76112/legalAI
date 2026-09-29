#!/bin/bash
#SBATCH --account=def-ekram
#SBATCH --job-name=mkgguf
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/ibi761/legalai/logs/%x-%j.out
# Two deliverables for running on a MacBook:
#   A. adapter-only GGUF  (~250MB)  -> use with a locally pulled Qwen3-4B
#   B. merged model GGUF Q8_0 (~4.3GB) -> self-contained, no base needed
# NF4/bitsandbytes is CUDA-only and cannot run on Apple Silicon; GGUF + Metal
# is the equivalent path there.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
module purge
module load StdEnv/2023 gcc/12.3 python/3.11 arrow/21.0.0
VENV=$SLURM_TMPDIR/gg
virtualenv --no-download $VENV && source $VENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch transformers peft safetensors sentencepiece gguf numpy
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1
L=/scratch/ibi761/legalai/gguf/lcpp
OUT=/scratch/ibi761/legalai/gguf/out
mkdir -p $OUT
BASE=Qwen/Qwen3-4B
ADPT=/scratch/ibi761/legalai/v2_sft/Qwen__Qwen3-4B__3task/final

echo "########## A. adapter -> GGUF ##########"
python $L/convert_lora_to_gguf.py "$ADPT" --base "$BASE" --outtype f16 \
  --outfile $OUT/contract-review-lora-f16.gguf
ls -lh $OUT/contract-review-lora-f16.gguf

echo "########## B. merge then GGUF Q8_0 ##########"
python v1_contractnli/merge_lora.py --base "$BASE" --adapter "$ADPT" \
  --out $SLURM_TMPDIR/merged
python $L/convert_hf_to_gguf.py $SLURM_TMPDIR/merged --outtype q8_0 \
  --outfile $OUT/contract-review-4b-q8_0.gguf
ls -lh $OUT/
echo "=== done ==="
