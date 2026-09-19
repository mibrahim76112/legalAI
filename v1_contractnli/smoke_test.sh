#!/bin/bash
#SBATCH --job-name=nli_smoke
#SBATCH --account=def-ekram_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/smoke_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/smoke_%j.out

# Proves the vLLM install works on a real GPU and that parsing is clean,
# on 50 examples with ONE model, before we commit eight jobs to the queue.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi
echo "=== GPU topology ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

source setup_env.sh

export HF_HOME=/scratch/ibi761/legalai/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RESULTS_DIR=/scratch/ibi761/legalai/results_smoke

for COND in zeroshot threeshot; do
  echo ""
  echo "############ SMOKE: Qwen/Qwen3-8B / $COND ############"
  python run_eval.py --model Qwen/Qwen3-8B --condition "$COND" --limit 50
done

echo ""
echo "=== parse summary ==="
python inspect_raw.py --dir "$RESULTS_DIR"

echo ""
echo "=== peak GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
