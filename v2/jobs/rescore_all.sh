#!/bin/bash
#SBATCH --account=def-ekram
#SBATCH --job-name=rescore
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/ibi761/legalai/logs/%x-%j.out
# Rescore ARCHIVED raw outputs offline -- no GPU, no regeneration, so the
# predictions being scored are byte-identical to the ones already reported.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source /scratch/ibi761/legalai/envs/prefetch/bin/activate
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for f in /scratch/ibi761/legalai/v2_results/raw__*.jsonl; do
  t=$(basename "$f" .jsonl); t=${t#raw__}
  case "$t" in smoke_*) continue;; esac
  echo "### $t"
  python v2/eval_multitask.py --base Qwen/Qwen3-4B --rescore "$f" --tag "$t" \
      --out-dir /scratch/ibi761/legalai/v2_results >/dev/null
done
echo "=== done ==="
