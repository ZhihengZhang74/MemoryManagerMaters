#!/bin/bash
# Smoke test: single GPU, few steps, verifies imports / vLLM startup / data loading.
# Usage: bash scripts/run_smoke.sh [aa|mm] [steps]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_h200.sh

PHASE="${1:-aa}"
STEPS="${2:-2}"

mkdir -p logs
LOG="logs/smoke_${PHASE}_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== smoke ${PHASE} ${STEPS} steps @ $(date) ==="
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
  python -c "import torch, vllm, trl, transformers, accelerate; print('torch', torch.__version__, '| vllm', vllm.__version__, '| trl', trl.__version__, '| transformers', transformers.__version__, '| accelerate', accelerate.__version__)"
} 2>&1 | tee "$LOG"

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --machine_rank 0 \
  --mixed_precision bf16 \
  scripts/train_memory_r1_rl_tracked.py \
  --phase "$PHASE" \
  --max-steps "$STEPS" \
  --eval-every 0 \
  --checkpoint-every 0 \
  2>&1 | tee -a "$LOG"

echo "Smoke log: $LOG"
