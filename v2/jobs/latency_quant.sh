#!/bin/bash
#SBATCH --job-name=latq
#SBATCH --account=def-ekram_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=0:50:00
#SBATCH --output=/scratch/ibi761/legalai/logs/%x_%j.out
# ONE job, steps run SEQUENTIALLY. The previous attempt submitted two jobs at
# once and SLURM placed both on the same 20GB slice; they each took ~12.5 GiB
# of a 19.62 GiB device and OOMed. Nothing here runs concurrently.
set -euo pipefail
cd /project/6030214/ibrahim/legalAI
nvidia-smi --query-gpu=name,memory.total --format=csv
source v1_contractnli/setup_env.sh
pip install --no-index bitsandbytes >/dev/null 2>&1 || true
export HF_HOME=/scratch/ibi761/legalai/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/scratch/ibi761/legalai/v2_sft

echo "########## 1. LATENCY: Qwen3-4B (bf16, concurrency 1) ##########"
python v2/measure_ttft.py --base Qwen/Qwen3-4B \
  --adapter $S/Qwen__Qwen3-4B__3task/final --tag q4 --n 6 --max-model-len 8192

echo "########## 2. LATENCY: Llama-3.1-8B (bf16, concurrency 1) ##########"
python v2/measure_ttft.py --base NousResearch/Meta-Llama-3.1-8B-Instruct \
  --adapter $S/NousResearch__Meta-Llama-3.1-8B-Instruct__3task/final \
  --tag llama --n 6 --max-model-len 8192

echo "########## 3. QUALITY: Llama-3.1-8B at NF4 4-bit ##########"
python v2/eval_multitask.py \
  --base NousResearch/Meta-Llama-3.1-8B-Instruct \
  --adapter $S/NousResearch__Meta-Llama-3.1-8B-Instruct__3task/final \
  --tag qs_llama_nf4 --data v2/quant_subset.jsonl \
  --out-dir /scratch/ibi761/legalai/v2_baseline \
  --quantization bitsandbytes --gpu-mem-util 0.85

echo "########## 4. LATENCY: Llama at NF4 (does 4-bit change speed?) ##########"
python v2/measure_ttft.py --base NousResearch/Meta-Llama-3.1-8B-Instruct \
  --adapter $S/NousResearch__Meta-Llama-3.1-8B-Instruct__3task/final \
  --tag llama_nf4 --n 6 --max-model-len 8192 || echo "SKIP nf4 latency"
echo "=== done ==="
