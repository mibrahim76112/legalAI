#!/bin/bash
#SBATCH --job-name=gemma_smoke
#SBATCH --account=def-ekram_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=0:45:00
#SBATCH --output=/scratch/ibi761/legalai/logs/gsmoke_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/gsmoke_%j.out
# Gemma-only retry. First attempt failed because the prefetch allow-list matched
# "preprocessor_config.json" but this repo ships "processor_config.json", so
# vLLM's multimodal loader had no processor. File is now cached; this also
# tests the pillow/torchvision additions.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export RESULTS_DIR=/scratch/ibi761/legalai/doc_results_smoke
python -c "
from transformers import AutoProcessor
import glob
p=glob.glob('$HF_HOME/hub/models--google--gemma-4-12B-it/snapshots/*/')[0]
try: print('AutoProcessor OK:', type(AutoProcessor.from_pretrained(p)).__name__)
except Exception as e: print('AutoProcessor FAILED:', type(e).__name__, str(e)[:200])
"
python doc_eval.py --model google/gemma-4-12B-it --condition nothink --split dev --limit 16 --tag smoke
python - <<'EOF'
import json,glob
for fp in glob.glob("/scratch/ibi761/legalai/doc_results_smoke/raw__google*smoke*.jsonl"):
    rows=[json.loads(l) for l in open(fp)]
    print(f"\n{fp.split('/')[-1]}  n={len(rows)}")
    ok=sum(1 for r in rows if r["schema_valid"])
    print(f"  schema_valid {ok}/{len(rows)} | parse_fail "
          f"{sum(1 for r in rows if r['parsed_verdict'] is None)}")
    for r in rows[:3]:
        print(f"  gold={r['gold_verdict']:14s} parsed={str(r['parsed_verdict']):14s} "
              f"json={r['json_state']} nev={len(r['parsed_evidence'])}")
        print(f"    RAW: {r['raw_output'][:170]!r}")
EOF
