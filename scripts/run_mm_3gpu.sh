#!/bin/bash
# Phase 1: Memory Manager GRPO, 3-GPU layout: 2 DDP ranks + 1 aux GPU.
# GPU 0/1 = training (DDP + colocate policy vLLM)
# GPU 2   = shared frozen engine: frozen AA scoring + LLM judge (always awake)
# Effective batch stays 128 = 2 (micro-bs) x 2 (ranks) x 32 (grad accum).
#
# Usage: bash scripts/run_mm_3gpu.sh [em|llm|mm_delta] [extra args...]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_h200.sh

MODE="${1:-em}"
shift || true

mkdir -p logs
LOG="logs/mm_${MODE}_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== MM phase 3-GPU (2 DDP + 1 aux) reward=${MODE} @ $(date) ==="
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
  python -c "import torch, vllm, trl, transformers, accelerate; print('torch', torch.__version__, '| vllm', vllm.__version__, '| trl', trl.__version__, '| transformers', transformers.__version__, '| accelerate', accelerate.__version__)"
} 2>&1 | tee "$LOG"

accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --machine_rank 0 \
  --mixed_precision bf16 \
  scripts/train_memory_r1_rl_tracked.py \
  --phase mm \
  --reward-mode "$MODE" \
  --aux-gpu 2 \
  --grad-accum 32 \
  --vllm-util 0.13 \
  --vllm-aa-util 0.35 \
  --optim adamw_bnb_8bit \
  "$@" \
  2>&1 | tee -a "$LOG"

echo "MM log: $LOG"
