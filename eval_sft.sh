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
VARIANT=${VARIANT:-}
SUF=${VARIANT:+__$VARIANT}
ADAPTER=/scratch/ibi761/legalai/sft_out/$(echo "$MODEL" | sed 's|/|__|')${SUF}/adapter
MERGED=$SLURM_TMPDIR/merged_$SLUG
# Resolve the cached snapshot by path, NOT snapshot_download(local_files_only):
# that helper treats a snapshot as incomplete if ANY repo file is absent, and
# the prefetch allow-list deliberately skips .gitattributes and README.md.
BASEDIR=$(python - <<'EOF'
import glob, os, sys
repo = os.environ["MODEL"].replace("/", "--")   # quoted heredoc: read from env, not $MODEL
pat = os.path.join(os.environ["HF_HOME"], "hub", f"models--{repo}", "snapshots", "*")
hits = [d for d in glob.glob(pat) if os.path.isfile(os.path.join(d, "config.json"))]
if not hits:
    sys.exit(f"no cached snapshot with config.json under {pat}")
print(sorted(hits)[-1])
EOF
)
echo "base=$BASEDIR"; echo "adapter=$ADAPTER"
python merge_lora.py --base "$BASEDIR" --adapter "$ADAPTER" --out "$MERGED"
export RESULTS_DIR=/scratch/ibi761/legalai/doc_results_sft
python doc_eval.py --model "$MERGED" --condition nothink --split "${SPLIT:-dev}" \
       --data-dir "${DATA_DIR:-doc_sft}" --max-new-tokens "${MNT:-1024}" \
       --tag "sft${VARIANT:+_$VARIANT}"
echo "=== done ==="
