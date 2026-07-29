#!/bin/bash
# Phase 1: Memory Manager GRPO, 4xH200 DDP + colocate vLLM + frozen AA vLLM.
# Usage: bash scripts/run_mm_4gpu.sh [extra args passed to the python script]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_h200.sh

mkdir -p logs
LOG="logs/mm_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== MM phase 4-GPU @ $(date) ==="
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
  python -c "import torch, vllm, trl, transformers, accelerate; print('torch', torch.__version__, '| vllm', vllm.__version__, '| trl', trl.__version__, '| transformers', transformers.__version__, '| accelerate', accelerate.__version__)"
} 2>&1 | tee "$LOG"

accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --machine_rank 0 \
  --mixed_precision bf16 \
  scripts/train_memory_r1_rl_tracked.py \
  --phase mm \
  "$@" \
  2>&1 | tee -a "$LOG"

echo "MM log: $LOG"
