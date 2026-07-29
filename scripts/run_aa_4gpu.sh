#!/bin/bash
# Phase 2: Answer Agent GRPO, 4xH200 DDP + colocate vLLM (no frozen AA needed).
# Usage: bash scripts/run_aa_4gpu.sh [extra args passed to the python script]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_h200.sh

mkdir -p logs
LOG="logs/aa_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== AA phase 4-GPU @ $(date) ==="
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
  python -c "import torch, vllm, trl, transformers, accelerate; print('torch', torch.__version__, '| vllm', vllm.__version__, '| trl', trl.__version__, '| transformers', transformers.__version__, '| accelerate', accelerate.__version__)"
} 2>&1 | tee "$LOG"

accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --machine_rank 0 \
  --mixed_precision bf16 \
  scripts/train_memory_r1_rl_tracked.py \
  --phase aa \
  "$@" \
  2>&1 | tee -a "$LOG"

echo "AA log: $LOG"
