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
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Add src/ to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents_memory.prompts_r1 import MEMORY_MANAGER_TYPED_PROMPT
from agents_memory.rl_callbacks import MemoryR1MetricsCallback
from agents_memory.rl_eval import evaluate_aa, evaluate_mm
from agents_memory.rl_grpo import TypedTwoLevelGRPOTrainer
from agents_memory.rl_judge import LLMJudge
from agents_memory.rl_rewards import (
    AAJudgeReward,
    MMRewardComputer,
    extract_answer_from_completion,
    compute_em,
    compute_f1,
    parse_mm_atomic,
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
    Output Dataset columns: "prompt" (chat-templated), "gold_answer", "question"
    (question comes from the aligned *_raw.jsonl; needed by the LLM judge).
    """
    examples = load_jsonl(path)
    raw_path = path.with_name(path.name.replace(".jsonl", "_raw.jsonl"))
    raw = load_jsonl(raw_path) if raw_path.exists() else [{}] * len(examples)
    assert len(raw) == len(examples), (
        f"ChatML ({len(examples)}) and raw ({len(raw)}) AA data must align"
    )
    dataset_examples = []
    for example, raw_example in zip(examples, raw):
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
            "question": raw_example.get("question", ""),
        })

    return Dataset.from_list(dataset_examples)


def _build_typed_prompt(raw: dict, op_focus: str, tokenizer, max_seq_length: int) -> str | None:
    """Build a single chat-templated typed MM prompt for one op_focus.

    Returns None if the templated prompt exceeds max_seq_length.
    Reuses the same related_memories / new_facts rendering as prepare_r1_data.py.
    """
    turn = raw["turn"]
    related = raw.get("related_memories") or []
    related_memories = (
        json.dumps(related, indent=2) if related else "No existing memories yet."
    )
    new_facts = f"- [{turn['speaker']}] ({turn['timestamp']}) {turn['text']}"
    content = MEMORY_MANAGER_TYPED_PROMPT.format(
        op_focus=op_focus,
        related_memories=related_memories,
        new_facts=new_facts,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if len(tokenizer.encode(prompt)) > max_seq_length:
        return None
    return prompt


def load_rl_dataset_mm(
    path: Path, tokenizer, max_seq_length: int, mm_gen_mode: str = "standard",
) -> Dataset:
    """Load MM data for GRPO. Builds running memory bank and pairs with QA.

    Each example gets: prompt, memory_bank_state (JSON), qa_pairs (JSON).

    In typed mode (mm_gen_mode="typed") each example additionally carries the
    four typed prompts (prompt_add / prompt_update / prompt_delete / prompt_none)
    for the 4x2 two-level advantage design; `prompt` is kept as the prompt_none
    placeholder for backward compatibility. Typed prompts are rendered from the
    raw `related_memories` / turn fields so the op_focus instruction is present.
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

        if mm_gen_mode == "typed":
            typed_prompts = {}
            for focus in ("ADD", "UPDATE", "DELETE", "NONE"):
                p = _build_typed_prompt(raw, focus, tokenizer, max_seq_length)
                if p is None:
                    break  # too long; skip this turn entirely
                typed_prompts[f"prompt_{focus.lower()}"] = p
            if len(typed_prompts) != 4:
                continue
            dataset_examples.append({
                "prompt": typed_prompts["prompt_none"],
                "prompt_add": typed_prompts["prompt_add"],
                "prompt_update": typed_prompts["prompt_update"],
                "prompt_delete": typed_prompts["prompt_delete"],
                "prompt_none": typed_prompts["prompt_none"],
                "memory_bank_state": json.dumps(memory_bank_state),
                "qa_pairs": json.dumps(qa_pairs),
            })
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
    device_index: int | None = None,
    keep_awake: bool = False,
):
    """Load a frozen base-model vLLM engine (frozen AA scoring and/or LLM judge).

    Two placement modes:
    - device_index=None (colocated 4-GPU layout): pinned to this rank's own
      GPU, sleeps at level 1 between calls to give memory back to training.
    - device_index=K (dedicated aux-GPU layout, e.g. 2+1): every rank pins its
      engine to GPU K, which runs no training; pass keep_awake=True so the
      engine never sleeps (no wake/sleep latency, no cumem churn).

    Runs in its own vLLM EngineCore subprocess (default executor). This keeps
    it fully isolated from TRL's in-process colocated policy engine — sharing
    external_launcher state would make the policy engine's sleep(level=2)
    discard the frozen AA weights (vLLM's sleep allocator is a process-wide
    singleton).

    The torchelastic/distributed env vars set by `accelerate launch` are
    temporarily scrubbed while the engine spawns: if the EngineCore subprocess
    inherits TORCHELASTIC_USE_AGENT_STORE, torch makes every rank a TCPStore
    *client* (nobody hosts the store), so the engine's private
    init_process_group would block for 600s connecting to its own unused port.
    """
    from vllm import LLM

    load_path = trained_model_path if trained_model_path else model_name
    rank0_print(f"  Loading frozen aux vLLM from: {load_path}")

    # Pin target GPU (resolve BEFORE scrubbing LOCAL_RANK below).
    shared_aux = device_index is not None
    rank = local_rank()
    if device_index is None:
        device_index = rank

    # Shared aux GPU: ranks must create their engines ONE AT A TIME (file
    # gate below), and the memory budget must be derived from a live free-
    # memory measurement. vLLM 0.19.1 imposes two competing constraints:
    #   1. KV sizing:      util*total - used_by_others - weights - overhead >= 1 seq
    #   2. startup check:  util*total <= free memory at engine startup
    # A fixed util cannot satisfy both for the 2nd+ engine, so compute
    # util = (used_by_others + weights + overhead + KV target) / total,
    # capped just under the startup limit.
    effective_util = gpu_memory_utilization
    seq_path = None
    if shared_aux:
        seq_path = Path(tempfile.gettempdir()) / f"memr1_aux_seq_{os.getppid()}"
        while True:
            done = int(seq_path.read_text()) if seq_path.exists() else 0
            if done >= rank:
                break
            time.sleep(2.0)
        # Query free memory via NVML (no CUDA context side-effect on aux GPU)
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            free_b, total_b = float(info.free), float(info.total)
        finally:
            pynvml.nvmlShutdown()
        used_others = total_b - free_b
        weights_b = 16e9    # 7.6B bf16 weights + load margin
        overhead_b = 4e9    # CUDA graphs / activations / context
        kv_target_b = 20e9  # per-engine KV cache target
        want = used_others + weights_b + overhead_b + kv_target_b
        cap = used_others + free_b * 0.92  # stay under startup free-mem check
        effective_util = round(min(want, cap) / total_b, 3)
    rank0_print(
        f"  Aux engine GPU: {device_index}  keep_awake: {keep_awake}  "
        f"util: {effective_util:.3f}"
    )

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
            gpu_memory_utilization=effective_util,
            max_model_len=VLLM_MAX_MODEL_LEN,
            dtype="bfloat16",
            enable_sleep_mode=not keep_awake,
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
        # Let the next rank start creating its engine (success or failure,
        # so a crash doesn't leave the others waiting until the job timeout).
        if seq_path is not None:
            seq_path.write_text(str(rank + 1))
    if not keep_awake:
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


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _build_typed_sft_dataset(tokenizer) -> Dataset:
    """Build typed SFT data from gold ops (online, no new data file).

    For each turn's each gold op (event=E): one sample = (prompt_E, atomic JSON).
    For each turn's missing type T: sample (prompt_T, {"op":"NONE"}), balanced
    so NONE samples per type <= 2x positive samples of that type (fixed seed).
    """
    import random

    raw_rows = load_jsonl(DATA_DIR / "memory_manager_train_raw.jsonl")
    rng = random.Random(42)

    ALL_TYPES = ("ADD", "UPDATE", "DELETE", "NONE")
    samples: list[dict] = []

    # Track positive sample counts per type for NONE balancing
    type_pos_counts: dict[str, int] = {t: 0 for t in ALL_TYPES}
    none_candidates: dict[str, list[dict]] = {t: [] for t in ALL_TYPES}

    for raw in raw_rows:
        # Build typed prompts for this turn
        typed_prompts: dict[str, str | None] = {}
        for focus in ALL_TYPES:
            p = _build_typed_prompt(raw, focus, tokenizer, MAX_SEQ_LENGTH)
            typed_prompts[focus] = p
        if any(v is None for v in typed_prompts.values()):
            continue

        # Determine which types appear as gold ops this turn
        present_types: set[str] = set()
        for op in raw.get("operations", []):
            event = str(op.get("event", "NONE")).upper()
            if event in ("ADD", "UPDATE", "DELETE"):
                present_types.add(event)
                # Build the atomic JSON completion for this gold op
                if event == "ADD":
                    atomic = {"op": "ADD", "text": op.get("text", "")}
                elif event == "UPDATE":
                    atomic = {"op": "UPDATE", "id": str(op.get("id", "")), "text": op.get("text", "")}
                elif event == "DELETE":
                    atomic = {"op": "DELETE", "id": str(op.get("id", ""))}
                completion_str = json.dumps(atomic)
                prompt_msg = [{"role": "user", "content": _strip_chat_template(typed_prompts[event])}]
                comp_msg = [{"role": "assistant", "content": completion_str}]
                samples.append({"prompt": prompt_msg, "completion": comp_msg})
                type_pos_counts[event] += 1

        # For missing types, create NONE samples (balanced later)
        for t in ALL_TYPES:
            if t not in present_types:
                none_atomic = json.dumps({"op": "NONE"})
                prompt_msg = [{"role": "user", "content": _strip_chat_template(typed_prompts[t])}]
                comp_msg = [{"role": "assistant", "content": none_atomic}]
                none_candidates[t].append({"prompt": prompt_msg, "completion": comp_msg})

    # Balance NONE samples: per type <= 2x positive samples of that type
    for t in ALL_TYPES:
        cap = 2 * type_pos_counts.get(t, 0)
        candidates = none_candidates[t]
        if len(candidates) > cap:
            candidates = rng.sample(candidates, cap)
        samples.extend(candidates)

    rng.shuffle(samples)
    return Dataset.from_list(samples)


def _strip_chat_template(prompt_str: str) -> str:
    """Extract the user content from a chat-templated prompt string.

    The typed prompts from _build_typed_prompt are already chat-templated; for
    SFT's conversational format we need the raw user content so SFTTrainer can
    re-apply the template. This is a best-effort extraction: strip the system/
    generation markers added by Qwen2.5's chat template.
    """
    # Qwen2.5 chat template wraps with <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant
    marker = "<|im_start|>user\n"
    if marker in prompt_str:
        start = prompt_str.index(marker) + len(marker)
        end = prompt_str.rfind("<|im_end|>")
        if end > start:
            return prompt_str[start:end].strip()
    return prompt_str


def sft_mm_coldstart(args: argparse.Namespace, output_dir: Path) -> str:
    """Optional SFT cold-start for the Memory Manager before GRPO.

    Teaches the policy the distilled ADD/UPDATE/DELETE output format from the
    gold operations (memory_manager_train.jsonl, ChatML) so GRPO only has to
    refine decisions rather than learn the format from scratch. Runs under the
    same accelerate DDP process group, saves to output_dir/sft_init, and
    returns that path for GRPO to load. Only assistant tokens are supervised.

    In typed mode (--mm-gen-mode typed), SFT data is constructed online from
    the raw gold ops: for each gold op (event=E) one sample (prompt_E, atomic
    JSON); for each turn's missing types T a (prompt_T, {"op":"NONE"}) sample,
    balanced so NONE samples per type <= 2x positive samples (fixed seed).
    """
    from trl import SFTConfig, SFTTrainer

    sft_dir = output_dir / "sft_init"
    rank0_print("\n" + "-" * 60)
    rank0_print(f"MM SFT cold-start: {args.mm_sft_epochs} epoch(s) -> {sft_dir}")
    rank0_print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mm_gen_mode == "typed":
        train_ds = _build_typed_sft_dataset(tokenizer)
    else:
        rows = load_jsonl(DATA_DIR / "memory_manager_train.jsonl")
        # Conversational prompt/completion format: TRL applies the chat template to
        # each side and, with completion_only_loss, supervises ONLY the assistant
        # completion. (Qwen2.5's template lacks {% generation %} markers, so
        # assistant_only_loss would mask everything to zero - this format is the
        # robust alternative.)
        train_ds = Dataset.from_list([
            {"prompt": [r["messages"][0]], "completion": [r["messages"][1]]}
            for r in rows
        ])
    rank0_print(f"  SFT examples: {len(train_ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True,
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir / "sft_trainer_output"),
        num_train_epochs=args.mm_sft_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.mm_sft_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        max_length=VLLM_MAX_MODEL_LEN,
        packing=False,
        completion_only_loss=True,  # supervise only the gold-ops completion
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model(str(sft_dir))
    if is_rank0():
        tokenizer.save_pretrained(str(sft_dir))
        print(f"  SFT cold-start model saved to {sft_dir}")

    # Free the SFT trainer/model before GRPO reloads from disk.
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _barrier()
    return str(sft_dir)


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
    """Phase 2: Train Answer Agent with GRPO.

    Reward per --reward-mode:
    - em (paper Eq. 4): binary exact match
    - llm: 6-level graded score from a frozen judge on the aux GPU
    """
    rank0_print("\n" + "=" * 60)
    rank0_print(f"PHASE 2: Answer Agent GRPO Training (reward={args.reward_mode})")
    rank0_print("=" * 60)

    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    output_dir = OUTPUT_DIR / "memory-r1-rl" / f"answer_agent_rl_{args.reward_mode}{tag_suffix}"
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

    # Reward selection: EM (paper) or frozen LLM judge on the aux GPU
    if args.reward_mode == "llm":
        judge_llm = load_frozen_aa_vllm(
            args.base_model,
            gpu_memory_utilization=args.vllm_aa_util,
            device_index=args.aux_gpu,
            keep_awake=args.aux_gpu is not None,
        )
        judge = LLMJudge(
            llm=judge_llm, tokenizer=tokenizer,
            max_tokens=args.judge_max_tokens,
            manage_sleep=args.aux_gpu is None,
            continuous=args.judge_continuous,
        )
        reward_fn = AAJudgeReward(judge)
        print_memory_snapshot("after_judge_init")
    else:
        reward_fn = aa_em_reward

    metrics_callback = MemoryR1MetricsCallback(
        metrics_path=output_dir / "metrics.jsonl",
        phase="aa",
        eval_fn=evaluate_aa if val_data else None,
        eval_kwargs={"val_dataset": val_data, "device": device} if val_data else {},
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=output_dir,
        reward_computer=reward_fn if args.reward_mode == "llm" else None,
    )

    print_memory_snapshot("before_trainer_init")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
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
    """Phase 1: Train Memory Manager with GRPO.

    Reward = frozen AA answers questions from the updated memory bank, then
    per --reward-mode the answers are scored by EM (paper) or the LLM judge.
    The frozen AA and the judge share ONE aux vLLM engine (same base model).
    """
    rank0_print("\n" + "=" * 60)
    rank0_print(f"PHASE 1: Memory Manager GRPO Training (reward={args.reward_mode})")
    rank0_print("=" * 60)

    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    output_dir = OUTPUT_DIR / "memory-r1-rl" / f"memory_manager_rl_{args.reward_mode}{tag_suffix}"
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
    _barrier()  # ensure output_dir exists before SFT/GRPO writes on any rank

    # Optional SFT cold-start: policy starts from distilled-ops weights, not base.
    # (Frozen AA / judge below always stay the base model, per paper.)
    grpo_init_model = args.base_model
    if args.mm_sft_epochs > 0:
        grpo_init_model = sft_mm_coldstart(args, output_dir)

    # Setup MM policy model (base, or SFT cold-start weights if enabled)
    model, policy_tokenizer, device = setup_model_for_grpo(grpo_init_model)

    # Frozen AA tokenizer (same base model family; separate name for clarity)
    aa_tokenizer = AutoTokenizer.from_pretrained(
        args.frozen_aa_path or args.base_model, trust_remote_code=True,
    )
    if aa_tokenizer.pad_token is None:
        aa_tokenizer.pad_token = aa_tokenizer.eos_token

    # Load frozen AA — paper uses base (untrained) AA for Phase 1
    aux_dedicated = args.aux_gpu is not None
    if args.no_vllm:
        frozen_aa_llm = None
        frozen_aa_model = load_frozen_aa_hf(args.base_model, args.frozen_aa_path)
    else:
        frozen_aa_llm = load_frozen_aa_vllm(
            args.base_model,
            trained_model_path=args.frozen_aa_path,  # None = base model
            gpu_memory_utilization=args.vllm_aa_util,
            device_index=args.aux_gpu,
            keep_awake=aux_dedicated,
        )
        frozen_aa_model = None

    # LLM judge shares the frozen AA engine (same frozen base model).
    # Needed by the graded "llm" reward, the "mm_delta" reward, and the
    # "mm_smooth" reward (which forces the continuous 2-decimal judge).
    judge = None
    if args.reward_mode in ("llm", "mm_delta", "mm_smooth"):
        if frozen_aa_llm is None:
            raise ValueError(f"--reward-mode {args.reward_mode} requires vLLM (drop --no-vllm)")
        judge = LLMJudge(
            llm=frozen_aa_llm, tokenizer=aa_tokenizer,
            max_tokens=args.judge_max_tokens,
            manage_sleep=False,  # MMRewardComputer owns wake/sleep around all calls
            continuous=args.judge_continuous or args.reward_mode == "mm_smooth",
        )

    print_memory_snapshot("after_frozen_aa_init")

    # Load data
    train_path = DATA_DIR / "memory_manager_train.jsonl"
    train_dataset = load_rl_dataset_mm(train_path, policy_tokenizer, MAX_SEQ_LENGTH, mm_gen_mode=args.mm_gen_mode)
    rank0_print(f"  Training examples: {len(train_dataset)}")

    val_data = None
    val_path = DATA_DIR / "memory_manager_val.jsonl"
    if val_path.exists():
        val_data = list(load_rl_dataset_mm(val_path, policy_tokenizer, MAX_SEQ_LENGTH, mm_gen_mode=args.mm_gen_mode))
        rank0_print(f"  Validation examples: {len(val_data)}")

    grpo_config = build_grpo_config(args, output_dir, MAX_COMPLETION_TOKENS_MM)

    mm_reward = MMRewardComputer(
        tokenizer=aa_tokenizer,
        frozen_aa_llm=frozen_aa_llm,
        frozen_aa_model=frozen_aa_model,
        max_new_tokens=args.frozen_aa_max_tokens,
        device=device,
        manage_sleep=not aux_dedicated,
        judge=judge,
        use_delta=args.reward_mode in ("mm_delta", "mm_smooth"),
        w_after=args.mm_w_after,
        w_delta=args.mm_w_delta,
        correct_threshold=args.mm_correct_threshold,
        delta_keep_correct=args.mm_delta_keep,
        use_tanh_delta=args.reward_mode == "mm_smooth",
        tanh_tau=args.mm_tanh_tau,
    )

    eval_fn = None
    eval_kwargs = {}
    if val_data:
        eval_fn = evaluate_mm
        eval_kwargs = {
            "frozen_aa": frozen_aa_llm if frozen_aa_llm is not None else frozen_aa_model,
            "frozen_aa_is_vllm": frozen_aa_llm is not None,
            "frozen_aa_manage_sleep": not aux_dedicated,
            "val_dataset": val_data,
            "device": device,
            "max_new_tokens_aa": args.frozen_aa_max_tokens,
            "mm_gen_mode": args.mm_gen_mode,
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

    trainer = TypedTwoLevelGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[mm_reward],
        processing_class=policy_tokenizer,
        callbacks=[metrics_callback],
        mm_gen_mode=args.mm_gen_mode,
        advantage_mode=args.advantage_mode,
        adv_global_weight=args.adv_global_weight,
        adv_local_weight=args.adv_local_weight,
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
        "--max-steps", type=int, default=100,
        help="Maximum training steps (default: 100)",
    )
    parser.add_argument(
        "--eval-every", type=int, default=10,
        help="Run validation every N steps (default: 10)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=100,
        help="Save checkpoint every N steps (default: 100)",
    )
    parser.add_argument(
        "--mm-sft-epochs", type=int, default=0,
        help="MM SFT cold-start epochs before GRPO (0=off, default). Teaches the "
        "distilled ADD/UPDATE/DELETE format from gold ops; MM phase only",
    )
    parser.add_argument(
        "--mm-sft-lr", type=float, default=1e-5,
        help="Learning rate for the MM SFT cold-start (default: 1e-5)",
    )
    parser.add_argument(
        "--reward-mode", choices=["em", "llm", "mm_delta", "mm_smooth"], default="em",
        help="Reward scoring: em = binary exact match (paper Eq. 4, default); "
        "llm = 3-level graded score {0.0,0.5,1.0} from a frozen LLM judge "
        "(add --judge-continuous for the 2-decimal continuous variant); "
        "mm_delta = w_after*R_after + w_delta*R_delta (MM only), R_after is the "
        "judge score on the post-op bank, R_delta maps the before/after "
        "correctness transition (baseline = pre-op bank); "
        "mm_smooth = w_after*s_after + w_delta*tanh((s_after-s_before)/tau) "
        "(MM only) on continuous judge scores, no hard threshold",
    )
    parser.add_argument(
        "--mm-w-after", type=float, default=0.7,
        help="mm_delta: weight on R_after (default: 0.7)",
    )
    parser.add_argument(
        "--mm-w-delta", type=float, default=0.3,
        help="mm_delta: weight on R_delta (default: 0.3)",
    )
    parser.add_argument(
        "--mm-correct-threshold", type=float, default=0.5,
        help="mm_delta: judge score >= this counts as 'correct' for the delta "
        "transition (default: 0.5)",
    )
    parser.add_argument(
        "--mm-delta-keep", type=float, default=0.3,
        help="mm_delta: reward for right->right (keep correct) transition "
        "(default: 0.3; wrong->right=+1, right->wrong=-1, wrong->wrong=0)",
    )
    parser.add_argument(
        "--mm-tanh-tau", type=float, default=0.25,
        help="mm_smooth: smoothing temperature of tanh((s_after-s_before)/tau) "
        "(default: 0.25; one judge level of improvement ~ tanh(0.8) = 0.66)",
    )
    parser.add_argument(
        "--judge-continuous", action="store_true",
        help="Switch the LLM judge to 5-band rubric + 2-decimal continuous "
        "score in [0,1] (default off = legacy 3-level judge). Auto-enabled "
        "by --reward-mode mm_smooth",
    )
    parser.add_argument(
        "--run-tag", default="",
        help="Optional experiment tag appended to the output dir name, e.g. "
        "--run-tag smooth -> answer_agent_rl_llm_smooth/ (default: empty)",
    )
    parser.add_argument(
        "--aux-gpu", type=int, default=None,
        help="GPU index for the shared frozen engine (frozen AA + judge). "
        "E.g. 2 in the 2+1 layout (ranks on GPU 0/1, aux on GPU 2). "
        "Default None = colocate on each rank's GPU with sleep mode (4-GPU layout)",
    )
    parser.add_argument(
        "--judge-max-tokens", type=int, default=8,
        help="Max new tokens for the LLM judge (choice-constrained, default: 8)",
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
    parser.add_argument(
        "--mm-gen-mode", choices=["standard", "typed"], default="standard",
        help="standard=现状整库输出同prompt采8; typed=原子操作+4类型提示词x2（默认 standard）",
    )
    parser.add_argument(
        "--advantage-mode", choices=["group", "twolayer"], default="group",
        help="group=现状单层; twolayer=0.5全局+0.5局部（默认 group；twolayer 要求 typed）",
    )
    parser.add_argument(
        "--adv-global-weight", type=float, default=0.5,
        help="twolayer: 全局 advantage 权重（默认 0.5）",
    )
    parser.add_argument(
        "--adv-local-weight", type=float, default=0.5,
        help="twolayer: 局部 advantage 权重（默认 0.5）",
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
    rank0_print(f"  Reward mode: {args.reward_mode}")
    rank0_print(f"  MM gen mode: {args.mm_gen_mode}  Advantage mode: {args.advantage_mode}")
    rank0_print(
        "  Aux engine: "
        + (f"dedicated GPU {args.aux_gpu} (always awake)" if args.aux_gpu is not None
           else "colocated per-rank (sleep level 1)")
    )
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
