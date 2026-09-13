#!/bin/bash
#SBATCH --job-name=doc_a1024
#SBATCH --account=def-ekram_gpu
#SBATCH --array=0-4
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/a1024_%A_%a.out
#SBATCH --error=/scratch/ibi761/legalai/logs/a1024_%A_%a.out
# Condition A re-run at max_new_tokens=1024.
#
# At 512, granite truncated 40/1037 (3.9%) mid-enumeration. ALL 40 scored
# not_json and 0 of 40 earned an evidence TP despite 24 having gold evidence,
# so both its schema-validity and its evidence F1 were cap-suppressed rather
# than measured. Qwen3-4B also truncated 1.4%.
#
# Re-running EVERY condition-A cell (not just granite) keeps the "identical
# max_new_tokens across models" fairness control intact while removing the cap
# as a confound. Tagged mnt1024 so the 512 results remain for comparison.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
LINE=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" cells_a1024.txt)
MODEL=$(echo "$LINE" | awk '{print $1}'); COND=$(echo "$LINE" | awk '{print $2}')
echo "=== $MODEL / $COND / mnt=1024 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export RESULTS_DIR=/scratch/ibi761/legalai/doc_results
SLUG=$(echo "$MODEL" | tr '/' '_')
GPULOG="$RESULTS_DIR/gpu__${SLUG}__${COND}__dev__mnt1024.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > "$GPULOG" &
S=$!
python doc_eval.py --model "$MODEL" --condition "$COND" --split dev \
       --max-new-tokens 1024 --tag mnt1024
kill $S 2>/dev/null || true
echo "=== done ==="
