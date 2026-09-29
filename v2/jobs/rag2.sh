#!/bin/bash
#SBATCH --account=def-ekram
#SBATCH --job-name=rag2
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/ibi761/legalai/logs/%x-%j.out
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
module purge; module load StdEnv/2023 gcc/12.3 python/3.11 arrow/21.0.0
V=$SLURM_TMPDIR/rag; virtualenv --no-download $V && source $V/bin/activate
pip install --no-index --upgrade pip >/dev/null
pip install --no-index torch sentence_transformers transformers numpy requests \
    huggingface_hub scipy scikit-learn >/dev/null
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "########## A. parent-child chunking (bge-base, no rerank) ##########"
python v2/rag_retrieval.py --parent-child --no-rerank --out v2/_rag_parentchild.json
echo "########## B. BGE-M3 dense, flat chunking (no rerank) ##########"
python v2/rag_retrieval.py --model BAAI/bge-m3 --no-rerank --out v2/_rag_bgem3.json
echo "########## C. BGE-M3 + parent-child (best of both) ##########"
python v2/rag_retrieval.py --model BAAI/bge-m3 --parent-child --no-rerank \
    --out v2/_rag_m3_pc.json
