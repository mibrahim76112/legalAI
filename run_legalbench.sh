#!/bin/bash
#SBATCH --job-name=lb
#SBATCH --account=def-ekram_gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ibi761/legalai/logs/lb_%x_%j.out
#SBATCH --error=/scratch/ibi761/legalai/logs/lb_%x_%j.out
# Held-out generalisation: LegalBench CUAD (38 tasks). Nothing from CUAD was
# used in training. Deliberately EXCLUDES legalbench's 14 contract_nli_* tasks,
# which share our training source and would not support a held-out claim.
#
# MODEL may be a HF id (base) or a merged fine-tuned path. For a fine-tune, set
# SFT_MODEL + VARIANT and the adapter is merged first, exactly as in eval_sft.sh.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
source setup_env.sh
export HF_HOME=/scratch/ibi761/legalai/hf_home TOKENIZERS_PARALLELISM=false
export LB_DIR=/scratch/ibi761/legalai/legalbench_results
TARGET="${MODEL:?}"
# MERGE=1 selects the fine-tuned path; VARIANT may be EMPTY (Arm A adapters
# live at sft_out/<model>/adapter with no suffix), so keying off VARIANT alone
# silently evaluated the BASE model under a fine-tuned label.
if [ "${MERGE:-0}" = "1" ]; then
  SUF="${VARIANT:+__${VARIANT}}"
  ADAPTER=/scratch/ibi761/legalai/sft_out/$(echo "$MODEL" | sed 's|/|__|')${SUF}/adapter
  BASEDIR=$(HF_HUB_OFFLINE=1 python - <<'EOF'
import glob, os, sys
repo = os.environ["MODEL"].replace("/", "--")
pat = os.path.join(os.environ["HF_HOME"], "hub", f"models--{repo}", "snapshots", "*")
hits = [d for d in glob.glob(pat) if os.path.isfile(os.path.join(d, "config.json"))]
print(sorted(hits)[-1])
EOF
)
  TARGET=$SLURM_TMPDIR/merged_lb
  HF_HUB_OFFLINE=1 python merge_lora.py --base "$BASEDIR" --adapter "$ADAPTER" --out "$TARGET"
fi
python eval_legalbench.py --model "$TARGET" --label "${LABEL:?}" \
       --limit-per-task "${NPT:-80}"
echo "=== done $LABEL ==="
