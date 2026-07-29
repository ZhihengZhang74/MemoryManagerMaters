#!/bin/bash
# Common environment for Memory-R1 4xH200 DDP + vLLM GRPO training.
# Source this from launch scripts: source scripts/env_h200.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate r1vllm

# Fully offline — all models and data are local
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/home/zhangzhiheng/.cache/huggingface

# Local imports (repo not pip-installed: pyproject requires python>=3.13)
export PYTHONPATH=/home/zhangzhiheng/memory-r1/src:${PYTHONPATH:-}

export BASE_MODEL=/home/zhangzhiheng/models/Qwen2.5-7B-Instruct

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=0   # H200 NVLink: keep P2P enabled
export OMP_NUM_THREADS=8
# Do NOT set CUDA_VISIBLE_DEVICES — accelerate assigns GPUs per rank
