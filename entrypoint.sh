#!/bin/bash
# SageMaker training entry point for Memory-R1 (arXiv:2508.19828).
# Faithful reproduction: RL from base model, no SFT warmstart.
# Paper order: Phase 1 = RL MM, Phase 2 = RL AA.
# Uses DeepSpeed ZeRO-3 + CPU offload for full fine-tuning on 4x A10G.
set -euo pipefail

echo "============================================"
echo "Memory-R1 SageMaker Training"
echo "============================================"
echo "Phase: ${SM_HP_PHASE:-all}"
echo "Max steps: ${SM_HP_MAX_STEPS:-200}"
echo "Instance: $(hostname)"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
echo "GPUs: $NUM_GPUS"
nvidia-smi 2>/dev/null || true
echo "============================================"

# ---- Setup ----
INPUT_DIR="${SM_CHANNEL_TRAINING:-/opt/ml/input/data/training}"
MODEL_DIR="${SM_MODEL_DIR:-/opt/ml/model}"
PHASE="${SM_HP_PHASE:-all}"
MAX_STEPS="${SM_HP_MAX_STEPS:-200}"
EVAL_EVERY="${SM_HP_EVAL_EVERY:-20}"
CHECKPOINT_EVERY="${SM_HP_CHECKPOINT_EVERY:-50}"
BASE_MODEL="${SM_HP_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

export HF_HOME="/tmp/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME" "$MODEL_DIR"

# ---- Always use code from the fresh tar ----
if [ -f "$INPUT_DIR/agents-memory-v2.tar" ]; then
    echo "Extracting code from agents-memory-v2.tar..."
    tar -xf "$INPUT_DIR/agents-memory-v2.tar" -C /tmp/
    PROJECT_DIR="/tmp/agents-memory"
elif [ -d "/tmp/agents-memory" ]; then
    PROJECT_DIR="/tmp/agents-memory"
else
    PROJECT_DIR="$INPUT_DIR"
fi

# ---- Use prep channel for pre-downloaded model weights ----
PREP_DIR="${SM_CHANNEL_PREP:-}"
if [ -n "$PREP_DIR" ]; then
    echo "Prep channel found at: $PREP_DIR"
    if [ -f "$PREP_DIR/model.tar.gz" ]; then
        echo "Extracting model weights from prep channel..."
        mkdir -p /tmp/prep_extracted
        tar -xzf "$PREP_DIR/model.tar.gz" -C /tmp/prep_extracted/
        PREP_DIR="/tmp/prep_extracted"
    fi
    if [ -d "$PREP_DIR/base_model" ]; then
        BASE_MODEL="$PREP_DIR/base_model"
        echo "Using pre-cached model from: $BASE_MODEL"
    fi
fi

# ---- Use prev channel for outputs from previous jobs ----
PREV_DIR="${SM_CHANNEL_PREV:-}"
if [ -n "$PREV_DIR" ]; then
    echo "Prev channel found at: $PREV_DIR"
    if [ -f "$PREV_DIR/model.tar.gz" ]; then
        echo "Extracting previous job outputs..."
        mkdir -p /tmp/prev_extracted
        tar -xzf "$PREV_DIR/model.tar.gz" -C /tmp/prev_extracted/
        PREV_DIR="/tmp/prev_extracted"
    fi
    if [ -d "$PREV_DIR" ]; then
        mkdir -p "$PROJECT_DIR/models"
        cp -r "$PREV_DIR"/* "$PROJECT_DIR/models/" 2>/dev/null || true
        echo "Previous outputs loaded into $PROJECT_DIR/models/"
    fi
fi

echo "Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# ---- Install uv + Python 3.13 ----
echo ""
echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo ""
echo "=== Setting up Python 3.13 venv ==="
uv venv --python 3.13 .venv
source .venv/bin/activate
python --version

echo ""
echo "=== Installing dependencies ==="
uv pip install torch --index-url https://download.pytorch.org/whl/cu129
uv pip install flash-attn --no-build-isolation 2>&1 | tail -3 || echo "flash-attn not available, continuing"
uv pip install deepspeed
uv pip install -e ".[training]"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import trl; print(f'TRL {trl.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"

# ---- Check data ----
DATA_DIR="$PROJECT_DIR/data/r1_training"
if [ ! -d "$DATA_DIR" ] || [ ! -f "$DATA_DIR/answer_agent_train.jsonl" ]; then
    echo "ERROR: Training data not found at $DATA_DIR"
    find "$PROJECT_DIR" -maxdepth 3 -type f | head -50
    exit 1
fi
echo "Training data found:"
wc -l "$DATA_DIR"/*.jsonl

# ---- Phase functions ----
# Paper order: Phase 1 = RL MM (frozen base AA), Phase 2 = RL AA (frozen trained MM)
# No SFT warmstart — RL directly from base model.

run_rl_mm() {
    echo ""
    echo "============================================"
    echo "PHASE 1: RL Memory Manager GRPO ($MAX_STEPS steps)"
    echo "  From base model (no SFT warmstart)"
    echo "  Frozen AA = base model (untrained)"
    echo "============================================"
    # DeepSpeed requires accelerate launch
    accelerate launch --num_processes $NUM_GPUS --mixed_precision bf16 \
        scripts/train_memory_r1_rl_tracked.py \
        --phase mm \
        --base-model "$BASE_MODEL" \
        --max-steps "$MAX_STEPS" \
        --eval-every "$EVAL_EVERY" \
        --checkpoint-every "$CHECKPOINT_EVERY"
    echo "RL MM complete."
}

run_rl_aa() {
    echo ""
    echo "============================================"
    echo "PHASE 2: RL Answer Agent GRPO ($MAX_STEPS steps)"
    echo "  From base model (no SFT warmstart)"
    echo "  Uses frozen trained MM from Phase 1"
    echo "============================================"
    # Find frozen MM from Phase 1
    FROZEN_MM=""
    for p in "$PROJECT_DIR/models/memory-r1-rl/memory_manager_rl/best" \
             "$PROJECT_DIR/models/memory-r1-rl/memory_manager_rl/final"; do
        if [ -d "$p" ]; then
            FROZEN_MM="$p"
            break
        fi
    done
    if [ -n "$FROZEN_MM" ]; then
        echo "Using frozen MM from: $FROZEN_MM"
    else
        echo "WARNING: No trained MM found. AA will use base model memories."
    fi

    accelerate launch --num_processes $NUM_GPUS --mixed_precision bf16 \
        scripts/train_memory_r1_rl_tracked.py \
        --phase aa \
        --base-model "$BASE_MODEL" \
        --max-steps "$MAX_STEPS" \
        --eval-every "$EVAL_EVERY" \
        --checkpoint-every "$CHECKPOINT_EVERY"
    echo "RL AA complete."
}

run_eval() {
    echo ""
    echo "============================================"
    echo "EVAL: Answer Agent - val + test"
    echo "============================================"

    # Find the AA model (RL-trained or base)
    AA_MODEL=""
    for p in "$PROJECT_DIR/models/memory-r1-rl/answer_agent_rl/best" \
             "$PROJECT_DIR/models/memory-r1-rl/answer_agent_rl/final"; do
        if [ -d "$p" ]; then
            AA_MODEL="$p"
            break
        fi
    done

    if [ -z "$AA_MODEL" ]; then
        echo "ERROR: No AA model found."
        find "$PROJECT_DIR/models" -maxdepth 4 -type d 2>/dev/null
        exit 1
    fi
    echo "Evaluating: $AA_MODEL"

    echo "--- Val set (81 QAs) ---"
    python -u scripts/eval_sft.py \
        --agent aa \
        --base-model "$BASE_MODEL" \
        --aa-adapter "$AA_MODEL" \
        --split val \
        --output "$PROJECT_DIR/models/eval_rl_val_results.json"

    echo "--- Test set (1307 QAs) ---"
    python -u scripts/eval_sft.py \
        --agent aa \
        --base-model "$BASE_MODEL" \
        --aa-adapter "$AA_MODEL" \
        --split test \
        --output "$PROJECT_DIR/models/eval_rl_test_results.json"

    echo "Eval complete."
}

# ---- Run requested phase ----
case "$PHASE" in
    rl_mm)    run_rl_mm ;;
    rl_aa)    run_rl_aa ;;
    eval)     run_eval ;;
    all)      run_rl_mm; run_rl_aa ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid: rl_mm, rl_aa, eval, all"
        exit 1 ;;
esac

# ---- Copy outputs to model dir ----
echo ""
echo "============================================"
echo "Copying outputs to $MODEL_DIR"
echo "============================================"

if [ -d "$PROJECT_DIR/models" ]; then
    cp -r "$PROJECT_DIR/models/"* "$MODEL_DIR/" 2>/dev/null || true
fi
find "$PROJECT_DIR" -name "metrics.jsonl" -exec cp {} "$MODEL_DIR/" \; 2>/dev/null || true
find "$PROJECT_DIR" -name "eval_*_results.json" -exec cp {} "$MODEL_DIR/" \; 2>/dev/null || true

echo "Final output contents:"
find "$MODEL_DIR" -type f | head -30

echo ""
echo "============================================"
echo "TRAINING COMPLETE"
echo "============================================"
