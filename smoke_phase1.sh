#!/bin/bash
#SBATCH --job-name=doc_smoke
#SBATCH --account=def-ekram_gpu
#SBATCH --array=0-1
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=124G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/dsmoke_%A_%a.out
#SBATCH --error=/scratch/ibi761/legalai/logs/dsmoke_%A_%a.out
# Gate before the 8-cell array. Task 0: Qwen3-8B (baseline sanity).
# Task 1: Gemma 4 -- channel syntax (most likely to surprise the parser) AND the
# largest MIG candidate at 24GB of weights.
#
# Runs on a 3g.40gb MIG instance rather than a full H100: these are single-GPU
# inference jobs whose measured mean utilization on the clause-level screen was
# 10-33%, which is the Alliance doc's stated criterion for using an instance.
# A 3g.40gb costs 6.1 RGU against 12.2 for a full H100, so it bills half the
# priority and schedules sooner. This smoke also VALIDATES that vLLM works on
# MIG at a realistic memory level before Phase 1 commits to it.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export RESULTS_DIR=/scratch/ibi761/legalai/doc_results_smoke
case $SLURM_ARRAY_TASK_ID in
  0) M=Qwen/Qwen3-8B ;;
  1) M=google/gemma-4-12B-it ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv
python doc_eval.py --model "$M" --condition nothink --split dev --limit 24 --tag smoke
echo "--- raw samples ---"
python - <<EOF
import json,glob
for fp in glob.glob("$RESULTS_DIR/raw__*smoke*.jsonl"):
    rows=[json.loads(l) for l in open(fp)]
    print(fp.split('/')[-1], len(rows))
    for r in rows[:4]:
        print("  gold:",r["gold_verdict"],"| parsed:",r["parsed_verdict"],
              "| json:",r["json_state"],"| nev:",len(r["parsed_evidence"]))
        print("  RAW:",repr(r["raw_output"][:220]))
EOF
