#!/bin/bash
#SBATCH --job-name=nli_screen
#SBATCH --account=def-ekram_gpu
#SBATCH --array=0-7
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/screen_%A_%a.out
#SBATCH --error=/scratch/ibi761/legalai/logs/screen_%A_%a.out

# 4 models x 2 conditions = 8 independent cells, one GPU each, all in parallel.
#   task id -> model = id/2 (line of models.txt), condition = id%2
set -euo pipefail
cd /project/6030214/ibrahim/legalAI

MODEL=$(sed -n "$(( SLURM_ARRAY_TASK_ID / 2 + 1 ))p" models.txt)
CONDS=(zeroshot threeshot)
COND=${CONDS[$(( SLURM_ARRAY_TASK_ID % 2 ))]}

echo "=== task $SLURM_ARRAY_TASK_ID : $MODEL / $COND ==="
echo "=== node $(hostname) job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
# GPU evidence for the infrastructure slide
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

source setup_env.sh

export HF_HOME=/scratch/ibi761/legalai/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RESULTS_DIR=/scratch/ibi761/legalai/results

# Sample the GPU DURING generation. Querying after the run reports 0 MiB /0%
# because the process has already exited - that is what the smoke test showed.
SLUG=$(echo "$MODEL" | tr '/' '_')
GPULOG="$RESULTS_DIR/gpu__${SLUG}__${COND}.csv"
mkdir -p "$RESULTS_DIR"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
           --format=csv -l 5 > "$GPULOG" &
SAMPLER=$!

python run_eval.py --model "$MODEL" --condition "$COND"

kill $SAMPLER 2>/dev/null || true
echo "=== GPU during run (peak) ==="
python - "$GPULOG" <<'EOF'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("  no samples"); raise SystemExit
def col(k):
    return [float(r[k].split()[0]) for r in rows if r[k].strip() and r[k].split()[0].replace('.','',1).isdigit()]
mem, util = col(" memory.used [MiB]"), col(" utilization.gpu [%]")
print(f"  samples={len(rows)} peak_mem={max(mem):.0f} MiB "
      f"peak_util={max(util):.0f}% mean_util={sum(util)/len(util):.0f}%")
EOF
echo "=== done $MODEL / $COND ==="
