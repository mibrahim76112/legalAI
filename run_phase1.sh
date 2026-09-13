#!/bin/bash
#SBATCH --job-name=doc_p1
#SBATCH --account=def-ekram_gpu
#SBATCH --array=0-7
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/p1_%A_%a.out
#SBATCH --error=/scratch/ibi761/legalai/logs/p1_%A_%a.out

# PHASE 1: document-level pre-SFT screen on the DEV split.
# Cells listed in cells_phase1.txt, one line per (model, condition).
set -euo pipefail
cd /project/6030214/ibrahim/legalAI

LINE=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" cells_phase1.txt)
MODEL=$(echo "$LINE" | awk '{print $1}')
COND=$(echo "$LINE" | awk '{print $2}')
SPLIT=${SPLIT:-dev}

echo "=== task $SLURM_ARRAY_TASK_ID : $MODEL / $COND / $SPLIT ==="
echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RESULTS_DIR=/scratch/ibi761/legalai/doc_results

# Sample the GPU DURING generation; querying after the process exits reports 0.
SLUG=$(echo "$MODEL" | tr '/' '_')
mkdir -p "$RESULTS_DIR"
GPULOG="$RESULTS_DIR/gpu__${SLUG}__${COND}__${SPLIT}.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
           --format=csv -l 5 > "$GPULOG" &
SAMPLER=$!

python doc_eval.py --model "$MODEL" --condition "$COND" --split "$SPLIT"

kill $SAMPLER 2>/dev/null || true
python - "$GPULOG" <<'EOF'
import sys, csv, re
rows=list(csv.DictReader(open(sys.argv[1])))
def col(k):
    out=[]
    for r in rows:
        v=(r.get(k) or "").split()
        if v and re.match(r'^\d+(\.\d+)?$', v[0]): out.append(float(v[0]))
    return out
mem,util=col(" memory.used [MiB]"),col(" utilization.gpu [%]")
if mem: print(f"=== GPU during run: peak_mem={max(mem)/1024:.1f}GB "
              f"peak_util={max(util):.0f}% mean_util={sum(util)/len(util):.0f}% "
              f"samples={len(rows)}")
EOF
echo "=== done $MODEL / $COND ==="
