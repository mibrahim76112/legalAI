#!/bin/bash
#SBATCH --account=def-ekram
#SBATCH --job-name=evidev
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/ibi761/legalai/logs/%x-%j.out
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source /scratch/ibi761/legalai/envs/prefetch/bin/activate
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
R=/scratch/ibi761/legalai/v2_results
for t in q4_3task q8_3task llama_3task q14_3task llama_3task_TEST; do
  echo "############ $t"
  python v2/evidence_eval.py --raw $R/raw__$t.jsonl --tag $t
done
echo "=== done ==="
