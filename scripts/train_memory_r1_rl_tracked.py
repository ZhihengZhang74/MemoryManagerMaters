#!/usr/bin/env python3
"""Memory-R1 GRPO Training with Comprehensive Metrics Tracking.

Two-phase GRPO reinforcement learning for the Memory-R1 system:
  Phase 1 (MM): Train Memory Manager with indirect EM via frozen AA
  Phase 2 (AA): Train Answer Agent with direct EM reward

Based on arXiv:2508.19828, Appendix D hyperparameters.

Architecture (4xH200, pure DDP + per-rank colocated vLLM):
  - Policy rollouts: TRL colocate vLLM (one engine per rank, sleep mode)
  - MM reward scoring: one additional self-managed frozen-AA vLLM per rank
  - No DeepSpeed by default (optional ZeRO-2 via --use-zero2)

Usage:
    accelerate launch --num_processes 4 scripts/train_memory_r1_rl_tracked.py --phase mm
    accelerate launch --num_processes 4 scripts/train_memory_r1_rl_tracked.py --phase aa
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

DEFAULT_BASE_MODEL = os.environ.get(
    "BASE_MODEL", "/home/zhangzhiheng/models/Qwen2.5-7B-Instruct"
)

# GRPO hyperparameters (Paper Appendix D, Figure 7)
GRPO_GROUP_SIZE = 8
GRPO_KL_COEFF = 0.01
RL_LEARNING_RATE = 1e-6           # Paper: PPO actor LR = 1e-6
MAX_COMPLETION_TOKENS_AA = 2048   # Paper: max response length = 2048
MAX_COMPLETION_TOKENS_MM = 2048   # Paper: max response length = 2048
GENERATION_TEMPERATURE = 1.0      # Paper: τ=1.0 for exploration during training
MAX_SEQ_LENGTH = 4096             # Paper: max prompt length = 4096
PER_DEVICE_BATCH_SIZE = 2         # Paper: micro-batch = 2 per GPU
GRADIENT_ACCUMULATION = 16        # With 4 GPUs: 2 * 4 * 16 = 128 effective batch

VLLM_MAX_MODEL_LEN = MAX_SEQ_LENGTH + MAX_COMPLETION_TOKENS_AA  # 6144

ZERO2_CONFIG = str(Path(__file__).parent.parent / "configs" / "ds_zero2.json")


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_rank0() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def rank0_print(*args_, **kwargs_):
    if is_rank0():
        print(*args_, **kwargs_)


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

def setup_model_for_grpo(
    model_name: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """Load base model in full bf16 for full fine-tuning. No LoRA (paper doesn't use it).

    Pure DDP: accelerate places the model on this rank's GPU; don't set device_map.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    rank0_print(f"  Full fine-tuning: {trainable:,} / {total:,} params (100%)")

    return model, tokenizer, "cuda"


def load_frozen_aa_vllm(
    model_name: str,
    trained_model_path: str | None = None,
    gpu_memory_utilization: float = 0.14,
):
    """Load frozen Answer Agent as a per-rank vLLM engine for MM reward scoring.

    Runs in its own vLLM EngineCore subprocess (default executor), pinned to
    this rank's GPU via a temporary CUDA_VISIBLE_DEVICES override. This keeps
    it fully isolated from TRL's in-process colocated policy engine — sharing
    external_launcher state would make the policy engine's sleep(level=2)
    discard the frozen AA weights (vLLM's sleep allocator is a process-wide
    singleton). Sleeps at level 1 (weights offloaded to CPU) between calls.

    The torchelastic/distributed env vars set by `accelerate launch` are
    temporarily scrubbed while the engine spawns: if the EngineCore subprocess
    inherits TORCHELASTIC_USE_AGENT_STORE, torch makes every rank a TCPStore
    *client* (nobody hosts the store), so the engine's private
    init_process_group would block for 600s connecting to its own unused port.
    """
    from vllm import LLM

    load_path = trained_model_path if trained_model_path else model_name
    rank0_print(f"  Loading frozen AA vLLM from: {load_path}")

    # Pin to this rank's GPU (resolve BEFORE scrubbing LOCAL_RANK below).
    device_index = local_rank()

    # Env vars the spawned EngineCore must NOT inherit from accelerate/torchrun.
    scrub_vars = [
        "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_ERROR_FILE", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
        "ROLE_NAME", "MASTER_ADDR", "MASTER_PORT",
    ]
    saved_env = {k: os.environ.pop(k) for k in scrub_vars if k in os.environ}
    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    try:
        llm = LLM(
            model=load_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=VLLM_MAX_MODEL_LEN,
            dtype="bfloat16",
            enable_sleep_mode=True,
            seed=0,
            max_num_batched_tokens=4096,
            trust_remote_code=True,
        )
    finally:
        if prev_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
        os.environ.update(saved_env)
    llm.sleep(level=1)
    return llm


def load_frozen_aa_hf(
    model_name: str,
    trained_model_path: str | None = None,
) -> AutoModelForCausalLM:
    """HF fallback for --no-vllm debugging. Pinned to this rank's GPU."""
    load_path = trained_model_path if trained_model_path else model_name
    rank0_print(f"  Loading frozen AA (HF) from: {load_path}")

    model = AutoModelForCausalLM.from_pretrained(
        load_path, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(f"cuda:{local_rank()}")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def print_memory_snapshot(tag: str) -> None:
    if is_rank0() and torch.cuda.is_available():
        dev = local_rank()
        free, total = torch.cuda.mem_get_info(dev)
        print(
            f"  [mem:{tag}] GPU{dev} used={(total - free) / 1e9:.1f}GB / {total / 1e9:.1f}GB "
            f"(torch alloc={torch.cuda.memory_allocated(dev) / 1e9:.1f}GB)"
        )


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
# GRPO config builder (shared AA/MM)
# ---------------------------------------------------------------------------

def build_grpo_config(args: argparse.Namespace, output_dir: Path, max_completion: int) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(output_dir / "trainer_output"),
        num_generations=GRPO_GROUP_SIZE,
        max_completion_length=max_completion,
        temperature=GENERATION_TEMPERATURE,
        beta=GRPO_KL_COEFF,
        learning_rate=RL_LEARNING_RATE,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.max_steps + 1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # required for DDP
        remove_unused_columns=False,
        report_to="none",
        lr_scheduler_type="constant",  # Paper Appendix D: "constant warmup schedule"
        warmup_steps=0,
        # --- GRPO objective fidelity (paper) ---
        loss_type="grpo",              # trl 1.9.1 defaults to "dapo" otherwise
        scale_rewards="group",         # A = (r - mean) / std within group
        # --- vLLM colocated rollouts ---
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_util,
        # Sleep mode is OFF by default: trl 1.9.1 pairs sleep(level=2) with a
        # per-step reload_weights() from disk (vllm#29341 workaround), which
        # both costs ~2.3s/step and overwrites the freshly synced policy
        # weights with the base checkpoint. Memory fits without it (see plan §2).
        vllm_enable_sleep_mode=args.vllm_sleep,
        vllm_tensor_parallel_size=1,
        vllm_max_model_length=VLLM_MAX_MODEL_LEN,
        # --- DDP / perf ---
        optim=args.optim,
        deepspeed=ZERO2_CONFIG if args.use_zero2 else None,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
    )


# ---------------------------------------------------------------------------
# Phase 2: Answer Agent GRPO
# ---------------------------------------------------------------------------

def train_aa(args: argparse.Namespace) -> Path:
    """Phase 2: Train Answer Agent with GRPO. Reward = pure EM (Paper Eq. 4)."""
    rank0_print("\n" + "=" * 60)
    rank0_print("PHASE 2: Answer Agent GRPO Training")
    rank0_print("=" * 60)

    output_dir = OUTPUT_DIR / "memory-r1-rl" / "adapter_answer_agent_rl"
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    # No SFT warmstart — paper starts RL from base model directly
    model, tokenizer, device = setup_model_for_grpo(args.base_model)

    # Load data
    train_path = DATA_DIR / "answer_agent_train.jsonl"
    train_dataset = load_rl_dataset_aa(train_path, tokenizer, MAX_SEQ_LENGTH)
    rank0_print(f"  Training examples: {len(train_dataset)}")

    val_data = None
    val_path = DATA_DIR / "answer_agent_val.jsonl"
    if val_path.exists():
        val_data = list(load_rl_dataset_aa(val_path, tokenizer, MAX_SEQ_LENGTH))
        rank0_print(f"  Validation examples: {len(val_data)}")

    grpo_config = build_grpo_config(args, output_dir, MAX_COMPLETION_TOKENS_AA)

    metrics_callback = MemoryR1MetricsCallback(
        metrics_path=output_dir / "metrics.jsonl",
        phase="aa",
        eval_fn=evaluate_aa if val_data else None,
        eval_kwargs={"val_dataset": val_data, "device": device} if val_data else {},
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=output_dir,
    )

    print_memory_snapshot("before_trainer_init")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[aa_em_reward],
        processing_class=tokenizer,
        callbacks=[metrics_callback],
    )

    print_memory_snapshot("after_trainer_init")
    rank0_print("\nStarting AA GRPO training (full fine-tuning, DDP + colocate vLLM)...")
    trainer.train()

    # Save full model (rank0-safe via trainer.save_model)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    if is_rank0():
        tokenizer.save_pretrained(str(final_dir))
        print(f"\nFinal AA model saved to {final_dir}")

    return output_dir


# ---------------------------------------------------------------------------
# Phase 1: Memory Manager GRPO
# ---------------------------------------------------------------------------

def train_mm(args: argparse.Namespace) -> Path:
    """Phase 1: Train Memory Manager with GRPO. Reward = indirect EM via frozen AA."""
    rank0_print("\n" + "=" * 60)
    rank0_print("PHASE 1: Memory Manager GRPO Training (paper trains MM first)")
    rank0_print("=" * 60)

    output_dir = OUTPUT_DIR / "memory-r1-rl" / "memory_manager_rl"
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Setup MM policy model — no SFT warmstart, train from base model
    model, policy_tokenizer, device = setup_model_for_grpo(args.base_model)

    # Frozen AA tokenizer (same base model family; separate name for clarity)
    aa_tokenizer = AutoTokenizer.from_pretrained(
        args.frozen_aa_path or args.base_model, trust_remote_code=True,
    )
    if aa_tokenizer.pad_token is None:
        aa_tokenizer.pad_token = aa_tokenizer.eos_token

    # Load frozen AA — paper uses base (untrained) AA for Phase 1
    if args.no_vllm:
        frozen_aa_llm = None
        frozen_aa_model = load_frozen_aa_hf(args.base_model, args.frozen_aa_path)
    else:
        frozen_aa_llm = load_frozen_aa_vllm(
            args.base_model,
            trained_model_path=args.frozen_aa_path,  # None = base model
            gpu_memory_utilization=args.vllm_aa_util,
        )
        frozen_aa_model = None

    print_memory_snapshot("after_frozen_aa_init")

    # Load data
    train_path = DATA_DIR / "memory_manager_train.jsonl"
    train_dataset = load_rl_dataset_mm(train_path, policy_tokenizer, MAX_SEQ_LENGTH)
    rank0_print(f"  Training examples: {len(train_dataset)}")

    val_data = None
    val_path = DATA_DIR / "memory_manager_val.jsonl"
    if val_path.exists():
        val_data = list(load_rl_dataset_mm(val_path, policy_tokenizer, MAX_SEQ_LENGTH))
        rank0_print(f"  Validation examples: {len(val_data)}")

    grpo_config = build_grpo_config(args, output_dir, MAX_COMPLETION_TOKENS_MM)

    mm_reward = MMRewardComputer(
        tokenizer=aa_tokenizer,
        frozen_aa_llm=frozen_aa_llm,
        frozen_aa_model=frozen_aa_model,
        max_new_tokens=args.frozen_aa_max_tokens,
        device=device,
    )

    eval_fn = None
    eval_kwargs = {}
    if val_data:
        eval_fn = evaluate_mm
        eval_kwargs = {
            "frozen_aa": frozen_aa_llm if frozen_aa_llm is not None else frozen_aa_model,
            "frozen_aa_is_vllm": frozen_aa_llm is not None,
            "val_dataset": val_data,
            "device": device,
            "max_new_tokens_aa": args.frozen_aa_max_tokens,
        }

    metrics_callback = MemoryR1MetricsCallback(
        metrics_path=output_dir / "metrics.jsonl",
        phase="mm",
        eval_fn=eval_fn,
        eval_kwargs=eval_kwargs,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=output_dir,
        reward_computer=mm_reward,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[mm_reward],
        processing_class=policy_tokenizer,
        callbacks=[metrics_callback],
    )

    print_memory_snapshot("after_trainer_init")
    rank0_print("\nStarting MM GRPO training (full fine-tuning, DDP + colocate vLLM)...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    if is_rank0():
        policy_tokenizer.save_pretrained(str(final_dir))
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
        help="Training phase: aa (Answer Agent), mm (Memory Manager), or both. "
        "Prefer running mm and aa as two separate launches for clean GPU memory.",
    )
    parser.add_argument(
        "--base-model", default=DEFAULT_BASE_MODEL,
        help=f"Base model name or path (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--frozen-aa-path", default=None,
        help="Path to frozen AA model for MM training (default: base model, per paper)",
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
    # --- vLLM / memory pressure knobs ---
    parser.add_argument(
        "--vllm-util", type=float, default=0.14,
        help="GPU memory utilization for the colocated policy vLLM (default: 0.14)",
    )
    parser.add_argument(
        "--vllm-aa-util", type=float, default=0.14,
        help="GPU memory utilization for the frozen AA vLLM in MM phase (default: 0.14)",
    )
    parser.add_argument(
        "--vllm-sleep", action="store_true",
        help="Enable sleep mode for the colocated policy vLLM. Off by default: "
        "trl 1.9.1 reloads weights from disk every step when enabled (slow, and "
        "clobbers the synced policy weights with the base checkpoint)",
    )
    parser.add_argument(
        "--optim", default="adamw_torch",
        help="Optimizer (default: adamw_torch; use adamw_bnb_8bit to cut 45GB)",
    )
    parser.add_argument(
        "--use-zero2", action="store_true",
        help="Enable DeepSpeed ZeRO-2 sharding (L3 memory fallback)",
    )
    parser.add_argument(
        "--no-vllm", action="store_true",
        help="Disable vLLM entirely; fall back to HF generate (debug only, slow)",
    )
    parser.add_argument(
        "--frozen-aa-max-tokens", type=int, default=512,
        help="Max new tokens for frozen AA reward scoring (default: 512)",
    )
    parser.add_argument(
        "--per-device-batch-size", type=int, default=PER_DEVICE_BATCH_SIZE,
        help=f"Per-device micro batch size (default: {PER_DEVICE_BATCH_SIZE})",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=GRADIENT_ACCUMULATION,
        help=f"Gradient accumulation steps (default: {GRADIENT_ACCUMULATION})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rank0_print("Memory-R1 GRPO Training")
    rank0_print(f"  Phase: {args.phase}")
    rank0_print(f"  Base model: {args.base_model}")
    rank0_print(f"  Max steps: {args.max_steps}")
    rank0_print(f"  Eval every: {args.eval_every} steps")
    rank0_print(f"  Checkpoint every: {args.checkpoint_every} steps")
    rank0_print(f"  Output: {OUTPUT_DIR}")
    rank0_print(
        "  vLLM: "
        + ("disabled" if args.no_vllm
           else f"colocate (sleep mode {'on' if args.vllm_sleep else 'off'})")
    )
    rank0_print(f"  Optimizer: {args.optim}  ZeRO-2: {args.use_zero2}")
    rank0_print(f"  World size: {os.environ.get('WORLD_SIZE', '1')}")
    if torch.cuda.is_available():
        rank0_print(f"  GPU: {torch.cuda.get_device_name(local_rank())}")
        props = torch.cuda.get_device_properties(local_rank())
        rank0_print(f"  GPU memory: {props.total_memory / 1e9:.1f} GB")
    rank0_print()

    if args.phase == "both":
        rank0_print(
            "WARNING: --phase both keeps MM-phase engines resident during AA. "
            "Prefer two separate launches (run_mm_4gpu.sh then run_aa_4gpu.sh)."
        )

    # Paper order: MM first (frozen base AA), then AA
    if args.phase in ("mm", "both"):
        mm_output = train_mm(args)
        rank0_print(f"\nMM training complete. Output: {mm_output}")

    if args.phase in ("aa", "both"):
        train_aa(args)

    rank0_print("\nAll training complete.")


if __name__ == "__main__":
    main()
