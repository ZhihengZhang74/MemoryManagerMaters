#!/bin/bash
# SageMaker PREP job: install deps + download model weights via uv.
# Uses uv to install Python 3.13 and all dependencies.
set -euo pipefail

echo "============================================"
echo "Memory-R1 PREP: Dependencies + Model Download"
echo "============================================"

INPUT_DIR="${SM_CHANNEL_TRAINING:-/opt/ml/input/data/training}"
MODEL_DIR="${SM_MODEL_DIR:-/opt/ml/model}"
BASE_MODEL="${SM_HP_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

export HF_HOME="/tmp/hf_cache"
mkdir -p "$HF_HOME" "$MODEL_DIR"

# ---- Extract project tar ----
if [ -d "/tmp/agents-memory" ]; then
    PROJECT_DIR="/tmp/agents-memory"
elif [ -f "$INPUT_DIR/agents-memory-v2.tar" ]; then
    echo "Extracting tar..."
    tar -xf "$INPUT_DIR/agents-memory-v2.tar" -C /tmp/
    PROJECT_DIR="/tmp/agents-memory"
else
    echo "ERROR: Could not find project tar"
    exit 1
fi

echo "Project dir: $PROJECT_DIR"
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
# Install PyTorch with CUDA first (matches DLC's cu129 drivers)
uv pip install torch --index-url https://download.pytorch.org/whl/cu129
uv pip install -e ".[training]"

echo ""
echo "=== Verifying imports ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import trl; print(f'TRL {trl.__version__}')"
python -c "import peft; print(f'PEFT {peft.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"

# ---- Download model weights ----
echo ""
echo "=== Downloading model: $BASE_MODEL ==="
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os

model_id = '$BASE_MODEL'
save_dir = '$MODEL_DIR/base_model'
os.makedirs(save_dir, exist_ok=True)

print(f'Downloading tokenizer: {model_id}')
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(save_dir)
print(f'Tokenizer saved to {save_dir}')

print(f'Downloading model: {model_id} (this takes a few minutes...)')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.save_pretrained(save_dir)
print(f'Model saved to {save_dir}')
"

# ---- Copy project (code + data) to output ----
echo ""
echo "=== Packaging project files ==="
mkdir -p "$MODEL_DIR/project"
cp -r "$PROJECT_DIR/scripts" "$MODEL_DIR/project/"
cp -r "$PROJECT_DIR/src" "$MODEL_DIR/project/"
cp -r "$PROJECT_DIR/data" "$MODEL_DIR/project/"
cp "$PROJECT_DIR/pyproject.toml" "$MODEL_DIR/project/"
cp "$PROJECT_DIR/entrypoint.sh" "$MODEL_DIR/project/" 2>/dev/null || true
cp -r "$PROJECT_DIR/sagemaker" "$MODEL_DIR/project/" 2>/dev/null || true

# Save frozen requirements for reproducibility
uv pip freeze > "$MODEL_DIR/requirements_frozen.txt"

echo ""
echo "=== Output contents ==="
du -sh "$MODEL_DIR"/*
find "$MODEL_DIR" -maxdepth 3 -type f | head -40

echo ""
echo "============================================"
echo "PREP COMPLETE"
echo "============================================"
