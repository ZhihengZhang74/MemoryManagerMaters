#!/usr/bin/env python3
"""Memory-R1 GRPO Training with Comprehensive Metrics Tracking.

Two-phase GRPO reinforcement learning for the Memory-R1 system:
  Phase 1 (AA): Train Answer Agent with direct EM reward
  Phase 2 (MM): Train Memory Manager with indirect EM via frozen AA

Based on arXiv:2508.19828, Appendix D hyperparameters.

Usage:
    uv run python scripts/train_memory_r1_rl_tracked.py --phase aa --max-steps 10
    uv run python scripts/train_memory_r1_rl_tracked.py --phase mm --frozen-aa-path models/memory-r1-rl/adapter_answer_agent_rl/best
    uv run python scripts/train_memory_r1_rl_tracked.py --phase both
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Add src/ to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents_memory.rl_callbacks import MemoryR1MetricsCallback
from agents_memory.rl_eval import evaluate_aa, evaluate_mm
from agents_memory.rl_rewards import (
    MMRewardComputer,
    extract_answer_from_completion,
    compute_em,
    compute_f1,
)

# ---------------------------------------------------------------------------
# Constants (Paper Appendix D)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "r1_training"
OUTPUT_DIR = Path(__file__).parent.parent / "models"

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# GRPO hyperparameters (Paper Appendix D, Figure 7)
GRPO_GROUP_SIZE = 8
GRPO_KL_COEFF = 0.01
RL_LEARNING_RATE = 1e-6           # Paper: PPO actor LR = 1e-6
MAX_COMPLETION_TOKENS_AA = 2048   # Paper: max response length = 2048
MAX_COMPLETION_TOKENS_MM = 2048   # Paper: max response length = 2048
GENERATION_TEMPERATURE = 1.0      # Paper: τ=1.0 for exploration during training
MAX_SEQ_LENGTH = 4096             # Paper: max prompt length = 4096
PER_DEVICE_BATCH_SIZE = 2         # Paper: micro-batch = 2 per GPU
GRADIENT_ACCUMULATION = 16        # With 4 GPUs (g5.12xlarge): 2 * 4 * 16 = 128 effective batch

# Paper uses full fine-tuning (no LoRA). DeepSpeed ZeRO-3 + CPU offload for memory.
DEEPSPEED_CONFIG = str(Path(__file__).parent.parent / "sagemaker" / "ds_zero3_offload.json")


# ---------------------------------------------------------------------------
# Data loading (matches base script format)
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_rl_dataset_aa(path: Path, tokenizer, max_seq_length: int) -> Dataset:
    """Load AA data for GRPO. Transforms messages format into prompt + gold_answer.

    Input JSONL: {"messages": [{"role":"user","content":...}, {"role":"assistant","content":...}]}
    Output Dataset columns: "prompt" (str with chat template), "gold_answer" (str)
    """
    examples = load_jsonl(path)
    dataset_examples = []
    for example in examples:
        user_message = example["messages"][0]["content"]
        assistant_message = example["messages"][1]["content"]

        # Extract gold answer after **Answer:** marker
        gold_answer = assistant_message.split("**Answer:**")[-1].strip()

        # Apply chat template with generation prompt
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )

        if len(tokenizer.encode(prompt)) > max_seq_length:
            continue

        dataset_examples.append({
            "prompt": prompt,
            "gold_answer": gold_answer,
        })

    return Dataset.from_list(dataset_examples)


def load_rl_dataset_mm(path: Path, tokenizer, max_seq_length: int) -> Dataset:
    """Load MM data for GRPO. Builds running memory bank and pairs with QA.

    Each example gets: prompt, memory_bank_state (JSON), qa_pairs (JSON)
    """
    mm_chatml = load_jsonl(path)
    split = "val" if "_val" in Path(path).name else "train"
    mm_raw = load_jsonl(DATA_DIR / f"memory_manager_{split}_raw.jsonl")
    aa_raw = load_jsonl(DATA_DIR / f"answer_agent_{split}_raw.jsonl")

    assert len(mm_chatml) == len(mm_raw), (
        f"ChatML ({len(mm_chatml)}) and raw ({len(mm_raw)}) MM data must align"
    )

    running_bank: list[dict] = []
    next_id = 0
    dataset_examples = []

    for chatml, raw in zip(mm_chatml, mm_raw):
        dia_id = raw["turn"]["dia_id"]
        speaker = raw["turn"]["speaker"]
        operations = raw["operations"]

        user_message = chatml["messages"][0]["content"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )

        memory_bank_state = copy.deepcopy(running_bank)

        # Find QA pairs whose evidence overlaps this turn
        qa_pairs = [qa for qa in aa_raw if dia_id in qa.get("evidence_refs", [])]

        # Advance running bank with gold operations
        for op in operations:
            event = op.get("event", "NONE").upper()
            if event == "ADD":
                running_bank.append({
                    "id": str(next_id),
                    "text": op["text"],
                    "speaker": speaker,
                    "evidence_ref": dia_id,
                })
                next_id += 1
            elif event == "UPDATE":
                for mem in running_bank:
                    if mem["id"] == op["id"]:
                        mem["text"] = op["text"]
                        break
            elif event == "DELETE":
                running_bank = [m for m in running_bank if m["id"] != op["id"]]

        if not qa_pairs:
            continue

        if len(tokenizer.encode(prompt)) > max_seq_length:
            continue

        dataset_examples.append({
            "prompt": prompt,
            "memory_bank_state": json.dumps(memory_bank_state),
            "qa_pairs": json.dumps(qa_pairs),
        })

    return Dataset.from_list(dataset_examples)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def detect_device() -> tuple[str, torch.dtype]:
    """Detect best available device and dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def setup_model_for_grpo(
    model_name: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """Load base model in full bf16 for full fine-tuning. No LoRA (paper doesn't use it).

    DeepSpeed ZeRO-3 handles memory sharding across GPUs.
    Don't set device_map — DeepSpeed manages device placement.
    """
    device, dtype = detect_device()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Full fine-tuning: {trainable:,} / {total:,} params (100%)")

    return model, tokenizer, device


def load_frozen_aa(
    model_name: str,
    trained_model_path: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load frozen Answer Agent for MM reward computation.

    If trained_model_path is provided, loads the full fine-tuned model.
    Otherwise loads the base model (for Phase 1: MM training with base AA).
    """
    device, dtype = detect_device()

    load_path = trained_model_path if trained_model_path else model_name
    print(f"  Loading frozen AA from: {load_path}")

    model = AutoModelForCausalLM.from_pretrained(
        load_path, torch_dtype=dtype, trust_remote_code=True, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        trained_model_path or model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer


# ---------------------------------------------------------------------------
# Reward functions (GRPOTrainer interface: completions is list[str])
# ---------------------------------------------------------------------------

def aa_em_reward(completions: list[str], gold_answer: list[str], **kwargs) -> list[float]:
    """Paper Eq. 4: Pure binary EM reward for Answer Agent."""
    rewards = []
    for completion, gold in zip(completions, gold_answer):
        predicted = extract_answer_from_completion(completion)
        rewards.append(compute_em(predicted, gold))
    return rewards


def aa_f1_reward(completions: list[str], gold_answer: list[str], **kwargs) -> list[float]:
    """Token-level F1 reward (informational, weight=0)."""
    rewards = []
    for completion, gold in zip(completions, gold_answer):
        predicted = extract_answer_from_completion(completion)
        rewards.append(compute_f1(predicted, gold))
    return rewards


# ---------------------------------------------------------------------------
# Phase 1: Answer Agent GRPO
# ---------------------------------------------------------------------------

def train_aa(args: argparse.Namespace) -> Path:
    """Phase 1: Train Answer Agent with GRPO. Reward = pure EM (Paper Eq. 4)."""
    print("\n" + "=" * 60)
    print("PHASE 1: Answer Agent GRPO Training")
    print("=" * 60)

    output_dir = OUTPUT_DIR / "memory-r1-rl" / "adapter_answer_agent_rl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # No SFT warmstart — paper starts RL from base model directly
    model, tokenizer, device = setup_model_for_grpo(args.base_model)

    # Load data
    train_path = DATA_DIR / "answer_agent_train.jsonl"
    train_dataset = load_rl_dataset_aa(train_path, tokenizer, MAX_SEQ_LENGTH)
    print(f"  Training examples: {len(train_dataset)}")

    val_data = None
    val_path = DATA_DIR / "answer_agent_val.jsonl"
    if val_path.exists():
        val_data = load_rl_dataset_aa(val_path, tokenizer, MAX_SEQ_LENGTH)
        val_data = list(val_data)
        print(f"  Validation examples: {len(val_data)}")

    # GRPO config (Paper Appendix D) with DeepSpeed ZeRO-3 for full fine-tuning
    grpo_config = GRPOConfig(
        output_dir=str(output_dir / "trainer_output"),
        num_generations=GRPO_GROUP_SIZE,
        max_completion_length=MAX_COMPLETION_TOKENS_AA,
        temperature=GENERATION_TEMPERATURE,
        beta=GRPO_KL_COEFF,
        learning_rate=RL_LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.max_steps + 1,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        deepspeed=DEEPSPEED_CONFIG,
        lr_scheduler_type="constant",  # Paper Appendix D: "constant warmup schedule"
        warmup_steps=0,
    )

    metrics_callback = MemoryR1MetricsCallback(
        metrics_path=output_dir / "metrics.jsonl",
        phase="aa",
        eval_fn=evaluate_aa if val_data else None,
        eval_kwargs={"val_dataset": val_data, "device": device} if val_data else {},
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=output_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[aa_em_reward],
        processing_class=tokenizer,
        callbacks=[metrics_callback],
    )

    print("\nStarting AA GRPO training (full fine-tuning, DeepSpeed ZeRO-3)...")
    trainer.train()

    # Save full model (not adapter)
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nFinal AA model saved to {final_dir}")

    return output_dir


# ---------------------------------------------------------------------------
# Phase 2: Memory Manager GRPO
# ---------------------------------------------------------------------------

def train_mm(args: argparse.Namespace) -> Path:
    """Phase 2: Train Memory Manager with GRPO. Reward = indirect EM via frozen AA."""
    print("\n" + "=" * 60)
    print("PHASE 1: Memory Manager GRPO Training (paper trains MM first)")
    print("=" * 60)

    output_dir = OUTPUT_DIR / "memory-r1-rl" / "memory_manager_rl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen AA — paper uses base (untrained) AA for Phase 1
    frozen_aa_path = args.frozen_aa_path
    frozen_aa, tokenizer = load_frozen_aa(
        args.base_model,
        trained_model_path=frozen_aa_path,  # None = base model
    )

    # Setup MM model — no SFT warmstart, train from base model
    model, tokenizer, device = setup_model_for_grpo(args.base_model)

    # Load data
    train_path = DATA_DIR / "memory_manager_train.jsonl"
    train_dataset = load_rl_dataset_mm(train_path, tokenizer, MAX_SEQ_LENGTH)
    print(f"  Training examples: {len(train_dataset)}")

    val_data = None
    val_path = DATA_DIR / "memory_manager_val.jsonl"
    if val_path.exists():
        val_data = load_rl_dataset_mm(val_path, tokenizer, MAX_SEQ_LENGTH)
        val_data = list(val_data)
        print(f"  Validation examples: {len(val_data)}")

    # GRPO config with DeepSpeed ZeRO-3
    grpo_config = GRPOConfig(
        output_dir=str(output_dir / "trainer_output"),
        num_generations=GRPO_GROUP_SIZE,
        max_completion_length=MAX_COMPLETION_TOKENS_MM,
        temperature=GENERATION_TEMPERATURE,
        beta=GRPO_KL_COEFF,
        learning_rate=RL_LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.max_steps + 1,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        deepspeed=DEEPSPEED_CONFIG,
        lr_scheduler_type="constant",  # Paper Appendix D: "constant warmup schedule"
        warmup_steps=0,
    )

    mm_reward = MMRewardComputer(
        frozen_aa_model=frozen_aa,
        tokenizer=tokenizer,
        max_new_tokens=MAX_COMPLETION_TOKENS_AA,
        device=device,
    )

    eval_fn = None
    eval_kwargs = {}
    if val_data:
        eval_fn = evaluate_mm
        eval_kwargs = {"frozen_aa": frozen_aa, "val_dataset": val_data, "device": device}

    metrics_callback = MemoryR1MetricsCallback(
        metrics_path=output_dir / "metrics.jsonl",
        phase="mm",
        eval_fn=eval_fn,
        eval_kwargs=eval_kwargs,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=output_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[mm_reward],
        processing_class=tokenizer,
        callbacks=[metrics_callback],
    )

    print("\nStarting MM GRPO training (full fine-tuning, DeepSpeed ZeRO-3)...")
    trainer.train()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nFinal MM model saved to {final_dir}")

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory-R1 GRPO Training with Metrics Tracking"
    )
    parser.add_argument(
        "--phase", choices=["aa", "mm", "both"], required=True,
        help="Training phase: aa (Answer Agent), mm (Memory Manager), or both",
    )
    parser.add_argument(
        "--base-model", default=DEFAULT_BASE_MODEL,
        help=f"Base model name or path (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--frozen-aa-path", default=None,
        help="Path to frozen AA adapter for MM training (default: auto from Phase 1)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=200,
        help="Maximum training steps (default: 200, per Paper Figure 7)",
    )
    parser.add_argument(
        "--eval-every", type=int, default=50,
        help="Run validation every N steps (default: 50)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=100,
        help="Save checkpoint every N steps (default: 100)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Memory-R1 GRPO Training")
    print(f"  Phase: {args.phase}")
    print(f"  Base model: {args.base_model}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Eval every: {args.eval_every} steps")
    print(f"  Checkpoint every: {args.checkpoint_every} steps")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Paper order: MM first (frozen base AA), then AA (frozen trained MM)
    if args.phase in ("mm", "both"):
        mm_output = train_mm(args)
        print(f"\nMM training complete. Output: {mm_output}")

    if args.phase in ("aa", "both"):
        train_aa(args)

    print("\nAll training complete.")


if __name__ == "__main__":
    main()
