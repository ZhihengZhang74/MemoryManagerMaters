#!/usr/bin/env python3
"""RL training for Memory-R1 agents (Memory Manager and Answer Agent).
GUIDED IMPLEMENTATION -- follow the TODOs and comments to build this yourself.

This script extends the SFT training (train_memory_r1.py) with reinforcement learning,
following the Memory-R1 paper: https://arxiv.org/abs/2508.19828

The paper's key insight: SFT teaches the model to imitate teacher-generated labels,
but RL teaches it to *optimize for downstream task performance* -- i.e., the Memory
Manager learns operations that actually help the Answer Agent answer correctly.

Two RL algorithms are supported:
  - PPO (Proximal Policy Optimization) -- Schulman et al. 2017
  - GRPO (Group Relative Policy Optimization) -- Shao et al. 2024 (DeepSeek-Math)

Prerequisites:
  1. Run prepare_r1_data.py to generate training data:
       cd agents-memory && uv run python scripts/prepare_r1_data.py
     This creates data/r1_training/ with JSONL files and memory banks.
     Needs OPENAI_API_KEY in .env (for embeddings). Use --skip-teacher to
     avoid teacher model API calls.
  2. Run train_memory_r1.py to get SFT checkpoints (RL starts from SFT):
       uv run python scripts/train_memory_r1.py --agent both
  3. Install training extras: uv sync --extra training

Usage:
    # Answer Agent with GRPO (recommended starting point -- simpler reward loop)
    uv run python scripts/train_memory_r1_rl.py --agent aa --method grpo

    # Answer Agent with PPO
    uv run python scripts/train_memory_r1_rl.py --agent aa --method ppo

    # Memory Manager with GRPO (harder -- needs frozen AA for reward)
    uv run python scripts/train_memory_r1_rl.py --agent mm --method grpo

    # Smoke test
    uv run python scripts/train_memory_r1_rl.py --agent aa --method grpo --smoke-test
"""

# ============================================================
# SECTION 1: Imports
# ============================================================
# LEARNING NOTE: RL training uses different trainers than SFT.
#
# For PPO, you need:
#   - trl.PPOConfig and trl.PPOTrainer
#   - trl.AutoModelForCausalLMWithValueHead (wraps model with a value head)
#   PPO needs a "critic" (value function) to estimate advantages. trl handles
#   this by attaching a small linear head on top of the LM that predicts
#   expected future reward.
#
# For GRPO, you need:
#   - trl.GRPOConfig and trl.GRPOTrainer
#   GRPO does NOT need a value head. Instead, it samples G completions per prompt,
#   scores them all, and uses the group statistics (mean, std) as the baseline.
#   This is simpler to implement and often works just as well.
#
# Read the trl docs to understand the API:
#   - PPO:  https://huggingface.co/docs/trl/ppo_trainer
#   - GRPO: https://huggingface.co/docs/trl/grpo_trainer
#
# Also read Search-R1 (https://arxiv.org/abs/2501.05366) -- it uses a very
# similar pattern: RL to teach an LLM to use an external tool (search vs memory).

import argparse
import copy
import json
import re
import string
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from agents_memory.prompts_r1 import ANSWER_AGENT_PROMPT

# TODO: Import the RL trainers from trl.
# For PPO you'll need: PPOConfig, PPOTrainer
# For GRPO you'll need: GRPOConfig, GRPOTrainer
# For PPO specifically, you also need: AutoModelForCausalLMWithValueHead
#
# Hint: check what's available with `from trl import ...`
# The trl version in pyproject.toml is >=0.27.2 which has both.
#
# YOUR CODE HERE:
from trl import PPOConfig, PPOTrainer
from trl import GRPOConfig, GRPOTrainer
from trl import AutoModelForCausalLMWithValueHead


# ============================================================
# SECTION 2: Configuration
# ============================================================

DATA_DIR = Path(__file__).parent.parent / "data" / "r1_training"
OUTPUT_DIR = Path(__file__).parent.parent / "models"

# Model defaults (same as SFT script)
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SMOKE_TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# LoRA config -- same as SFT (paper Appendix D)
LORA_R = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# RL-specific hyperparameters (from paper Appendix D)
# LEARNING NOTE: RL learning rates are typically LOWER than SFT because you're
# fine-tuning an already-trained model. Too high = catastrophic forgetting.
RL_LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 4096

# PPO hyperparameters
PPO_EPOCHS = 4               # Number of PPO optimization passes per batch of rollouts
PPO_CLIP_EPSILON = 0.2       # Clipping threshold for policy ratio (paper Eq. 2)
PPO_BATCH_SIZE = 16           # Number of rollouts to collect before updating
PPO_MINI_BATCH_SIZE = 4       # Mini-batch size within each PPO epoch
PPO_KL_PENALTY = "kl"         # KL penalty type ("kl", "abs", "mse", or "full")
PPO_TARGET_KL = 6.0           # Target KL divergence -- training stops early if exceeded

# GRPO hyperparameters
GRPO_GROUP_SIZE = 8           # G in paper Eq. 3 -- increased from 4 for more reward variance
GRPO_KL_COEFF = 0.01          # beta in paper Eq. 3 -- KL penalty against reference policy
GRPO_BATCH_SIZE = 8           # Prompts per batch (must be >= num_generations for GRPO)

# Generation settings for rollouts
GENERATION_MAX_NEW_TOKENS = 512
GENERATION_TEMPERATURE = 0.7  # >0 needed for diverse sampling in GRPO
GENERATION_TOP_P = 0.9


@dataclass
class RLTrainingArgs:
    """Training arguments for RL fine-tuning."""
    agent: str = "aa"              # "mm" or "aa" (train one at a time for RL)
    method: str = "grpo"           # "ppo" or "grpo"
    model_name: str = DEFAULT_MODEL
    sft_adapter_path: str = ""     # Path to SFT adapter to initialize from
    aa_adapter_path: str = ""      # Path to frozen AA adapter (for MM reward computation)
    output_dir: str = str(OUTPUT_DIR)
    max_seq_length: int = MAX_SEQ_LENGTH
    learning_rate: float = RL_LEARNING_RATE
    num_episodes: int = 3          # Number of passes over the dataset
    lora_r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    use_4bit: bool = True
    bf16: bool = True
    smoke_test: bool = False
    max_steps: int = -1


def resolve_rl_output_path(args: RLTrainingArgs, agent: str) -> Path:
    """Return and create an agent-specific output directory for RL adapters."""
    name = "adapter_answer_agent_rl" if agent == "aa" else "adapter_memory_manager_rl"
    out = Path(args.output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


# ============================================================
# SECTION 3: Reward Functions
# ============================================================
# LEARNING NOTE: The paper uses a single, simple reward signal: Exact Match (EM).
# This is the same metric used in SQuAD and other QA benchmarks.
#
# Paper Section 3.1 (Eq. 4):
#   R_answer = EM(y_pred, y_gold)
#
# where EM is 1 if the normalized prediction matches the normalized gold answer,
# and 0 otherwise. "Normalized" means: lowercased, articles removed, punctuation
# removed, whitespace collapsed.
#
# Why EM and not something fancier?
# - It's cheap to compute (no LLM judge needed during training)
# - It's deterministic (no noise in the reward signal)
# - The paper shows it's sufficient to learn good memory behaviors
#
# For the Memory Manager, the reward is INDIRECT:
#   MM performs operation -> memory bank changes -> frozen AA answers question -> EM reward
# This is what makes MM training harder -- it needs to learn how its actions
# affect a downstream model's performance (credit assignment problem).
#
# For the Answer Agent, the reward is DIRECT:
#   AA generates answer -> EM against gold -> reward
# This is simpler -- the reward directly reflects the quality of the output.


def normalize_answer(s: str) -> str:
    """Normalize answer string for Exact Match comparison.

    Standard SQuAD normalization. Steps (order matters!):
      1. lower(s)             -- "The Dog" -> "the dog"
      2. remove_articles(s)   -- "the dog" -> " dog"  (regex \\b(a|an|the)\\b)
      3. remove_punctuation(s)-- strip all string.punctuation chars
      4. whitespace_collapse(s)-- " dog" -> "dog"  (" ".join(s.split()))

    Why this order?
      - Lowercase FIRST so the article regex matches "The", "A", "AN" etc.
      - Remove articles BEFORE collapsing whitespace because removal leaves
        gaps (e.g., "the dog" -> " dog") that collapse cleans up.
      - Punctuation removal is independent but do it before collapse so
        trailing punctuation doesn't leave extra spaces.

    TODO: Implement using the 4 helper functions below.

    YOUR CODE HERE:
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", "", text)

    def whitespace_collapse(text: str) -> str:
        return " ".join(text.split())

    def remove_punctuation(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    # YOUR CODE HERE: call the 4 helpers in order: lower -> remove_articles ->
    # remove_punctuation -> whitespace_collapse. Return the result.
    
    text = lower(s)
    text = remove_articles(text)
    text = remove_punctuation(text)
    text = whitespace_collapse(text)
    return text

def compute_em_reward(predicted: str, gold: str) -> float:
    """Compute Exact Match reward (binary 0.0 or 1.0).

    Paper Eq. 4: R_answer = EM(y_pred, y_gold)

    TODO: Implement this function.
    Steps:
      1. Normalize both strings using normalize_answer()
      2. Compare the normalized strings
      3. Return 1.0 if they match exactly, 0.0 otherwise

    YOUR CODE HERE:
    """
    predicted = normalize_answer(predicted)
    gold = normalize_answer(gold)
    return float(predicted == gold)


def compute_f1_reward(predicted: str, gold: str) -> float:
    """Compute token-level F1 as a softer reward signal.

    LEARNING NOTE: The paper uses EM, but you might want to experiment with F1
    as a reward too. F1 gives partial credit for partial matches, which can help
    with training stability (less sparse reward signal).

    F1 = 2 * (precision * recall) / (precision + recall)
    where precision = |common tokens| / |predicted tokens|
    and   recall    = |common tokens| / |gold tokens|

    TODO (OPTIONAL / STRETCH GOAL): Implement token-level F1.
    Steps:
      1. Normalize both strings with normalize_answer()
      2. Split both into token lists on whitespace
      3. Compute the set of common tokens (use collections.Counter for multiset)
      4. precision = num_common / len(pred_tokens)  (handle zero division)
      5. recall    = num_common / len(gold_tokens)  (handle zero division)
      6. f1 = 2 * precision * recall / (precision + recall)  (handle zero division)
      7. Return f1 as float

    YOUR CODE HERE:
    """
    from collections import Counter

    predicted = normalize_answer(predicted)
    gold = normalize_answer(gold)
    predicted_tokens = predicted.split()
    gold_tokens = gold.split()
    if not predicted_tokens or not gold_tokens:
        return 0.0
    # Multiset intersection to handle duplicate tokens correctly
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(predicted_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_combined_reward(predicted: str, gold: str, em_weight: float = 0.2) -> float:
    """Combined F1 + EM reward for less sparse GRPO signal.

    F1 provides continuous partial credit (0-1), ensuring GRPO gets variance
    across the group even when no completion is an exact match.
    EM bonus rewards exact matches more strongly.

    Formula: reward = (1 - em_weight) * F1 + em_weight * EM
    Default: 0.8 * F1 + 0.2 * EM
    """
    f1 = compute_f1_reward(predicted, gold)
    em = compute_em_reward(predicted, gold)
    return (1 - em_weight) * f1 + em_weight * em


# ============================================================
# SECTION 4: RL Data Loading
# ============================================================
# LEARNING NOTE: RL data is fundamentally different from SFT data.
#
# In SFT, each example is (input, target_output) and the model learns to
# reproduce the target. The loss is cross-entropy on the target tokens.
#
# In RL, each example is just a PROMPT. The model generates its own completion,
# then gets a reward based on quality. The loss is a policy gradient that
# increases the probability of high-reward completions.
#
# So we need to load the same JSONL files but extract ONLY the prompts (user
# messages), not the target outputs (assistant messages).
#
# For the Answer Agent:
#   - Prompt = ANSWER_AGENT_PROMPT with retrieved memories + question
#   - The model generates: selected memories + answer
#   - Reward = EM(extracted_answer, gold_answer)
#   - We also need to store gold_answer alongside each prompt for reward computation
#
# For the Memory Manager:
#   - Prompt = MEMORY_MANAGER_PROMPT with related memories + new facts
#   - The model generates: JSON with memory operations
#   - Reward = (apply ops to memory bank -> AA answers question -> EM)
#   - We need to store the associated QA pair and memory bank state
#
# ALL DATA IS GENERATED by prepare_r1_data.py. Run it first:
#   cd agents-memory && uv run python scripts/prepare_r1_data.py
# It needs OPENAI_API_KEY in .env. Use --skip-teacher to avoid teacher LLM calls.


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
    """Load Answer Agent data for RL training.

    Each example becomes a prompt (no target) + gold answer for reward.

    The SFT JSONL files (generated by prepare_r1_data.py) have this structure:
      {"messages": [
          {"role": "user", "content": "<ANSWER_AGENT_PROMPT with memories + question>"},
          {"role": "assistant", "content": "<selected memories + **Answer:** <answer>>"}
      ]}

    For RL, we need:
      - "prompt": the user message formatted with chat template + generation prompt
      - "gold_answer": extracted from the assistant message (after "**Answer:**")

    TODO: Implement this function.
    Steps:
      1. Load the JSONL file using load_jsonl(path)
      2. For each example:
         a. Extract the user message: ex["messages"][0]
         b. Apply the chat template with add_generation_prompt=True:
              tokenizer.apply_chat_template(
                  [ex["messages"][0]], tokenize=False, add_generation_prompt=True
              )
            This adds the assistant turn prefix so the model knows to generate.
         c. Extract the gold answer from ex["messages"][1]["content"]:
            Look for "**Answer:**" and take everything after it, stripped.
            Fallback: if pattern not found, use the full assistant content.
         d. Check token length: len(tokenizer.encode(prompt))
            Skip if > max_seq_length
      3. Return a Dataset with columns: "prompt" (str), "gold_answer" (str)

    YOUR CODE HERE:
    """
    examples = load_jsonl(path)
    dataset_examples = []
    for example in examples:
      # Extract the user message and assistant message
      user_message = example["messages"][0]["content"]
      assistant_message = example["messages"][1]["content"]

      # Extract the gold answer from the assistant message
      gold_answer = assistant_message.split("**Answer:**")[-1].strip()

      # Apply chat template to tell the model to generate an answer
      # Add generation prompt basically adds a prefix to the chat that indicates the start of an assistant response.
      prompt = tokenizer.apply_chat_template(
          [{"role": "user", "content": user_message}],
          tokenize=False, 
          add_generation_prompt=True
      )
      
      # Check token length and skip if too long
      if len(tokenizer.encode(prompt)) > max_seq_length:
        continue

      # Add the example to the dataset
      dataset_examples.append({
        "prompt": prompt,
        "gold_answer": gold_answer
      })
    return Dataset.from_list(dataset_examples)


def load_rl_dataset_mm(path: Path, tokenizer, max_seq_length: int) -> Dataset:
    """Load Memory Manager data for RL training.

    This is trickier because the MM reward requires running the frozen AA.
    Each MM example needs:
      - "prompt": the MM prompt (user message with chat template)
      - "memory_bank_state": the current memory bank (for applying operations)
      - "qa_pairs": associated QA pairs (to evaluate with frozen AA)

    LEARNING NOTE: The paper's MM training loop (Section 3.1) works like this:
      1. MM sees a new dialogue turn + related memories
      2. MM generates an operation (ADD/UPDATE/DELETE/NOOP) + content
      3. The operation is APPLIED to the memory bank
      4. The updated memory bank is used to answer associated QA pairs
      5. The EM score on those QA pairs becomes the reward

    This creates a complex environment where each action (memory operation)
    changes the state (memory bank) and the reward is delayed (comes from QA).

    For simplicity in a first implementation, you can use the pre-built memory
    banks from prepare_r1_data.py and pair each MM prompt with the QA pairs
    that depend on the memories it manages.

    TODO: Implement this function (hardest data loader -- do AA first).

    Steps:
      1. Load the MM JSONL with load_jsonl(path)
         Each example has {"messages": [{"role":"user","content":...}, {"role":"assistant","content":...}]}
      2. Load the AA JSONL too (you need the QA pairs for reward computation)
         AA data lives at DATA_DIR / "answer_agent_train.jsonl" (or val)
         NOTE: all these files are generated by running prepare_r1_data.py:
           cd agents-memory && uv run python scripts/prepare_r1_data.py          # needs OPENAI_API_KEY
           uv run python scripts/prepare_r1_data.py --skip-teacher  # no teacher LLM calls
         Run that first if data/r1_training/ is empty.
      3. Load the pre-built memory bank from:
         DATA_DIR / "memory_banks" / "train_memory_bank.json"
         This gives you the full memory bank state that prepare_r1_data.py built.
      4. For each MM example:
         a. Extract the user message -> apply chat template with add_generation_prompt=True
            This becomes the "prompt" column.
         b. Snapshot the memory bank state at this point in the conversation.
            Hint: the MM examples are sequential (turn by turn). You can track a
            running memory bank by applying the gold operations from previous turns.
         c. Find associated QA pairs -- these are questions whose evidence_refs
            overlap with memories managed by this turn. For a simpler first pass,
            you can just pair each MM example with ALL QA pairs (less precise but
            functional).
         d. Store: prompt, memory_bank_state (as JSON string), qa_pairs (as JSON string),
            gold_operations (from assistant message, for logging/debugging)
      5. Filter by token length, return Dataset with columns:
         "prompt", "memory_bank_state", "qa_pairs"

    Simplification for v1: skip step 4b/4c and just store the full memory bank
    + all QA pairs for every MM example. It's less efficient but gets you running.

    YOUR CODE HERE:
    """
    # Load ChatML MM data (for prompts) and raw MM data (for dia_id + operations)
    mm_chatml = load_jsonl(path)
    split = "train" if "train" in str(path) else "val"
    mm_raw = load_jsonl(Path(DATA_DIR) / f"memory_manager_{split}_raw.jsonl")
    aa_raw = load_jsonl(Path(DATA_DIR) / f"answer_agent_{split}_raw.jsonl")

    assert len(mm_chatml) == len(mm_raw), (
        f"ChatML ({len(mm_chatml)}) and raw ({len(mm_raw)}) MM data must align"
    )

    # Build running memory bank turn-by-turn, snapshot before each turn
    running_bank: list[dict] = []
    next_id = 0
    dataset_examples = []

    for chatml, raw in zip(mm_chatml, mm_raw):
        dia_id = raw["turn"]["dia_id"]
        speaker = raw["turn"]["speaker"]
        operations = raw["operations"]

        # 4a: Build prompt from ChatML user message
        user_message = chatml["messages"][0]["content"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False, add_generation_prompt=True,
        )

        # 4b: Snapshot the memory bank BEFORE this turn's operations
        memory_bank_state = copy.deepcopy(running_bank)

        # 4c: Overlap match -- QA pairs whose evidence_refs include this turn's dia_id
        qa_pairs = [qa for qa in aa_raw if dia_id in qa["evidence_refs"]]

        # Apply gold operations to advance the running bank for future turns
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

        # Extract gold operations JSON for logging/debugging
        gold_operations = json.dumps({"memory": operations})

        # Skip turns with no associated QA pairs (no reward signal possible)
        if not qa_pairs:
            continue

        # Check token length
        if len(tokenizer.encode(prompt)) > max_seq_length:
            continue

        dataset_examples.append({
            "prompt": prompt,
            "memory_bank_state": json.dumps(memory_bank_state),
            "qa_pairs": json.dumps(qa_pairs),
            "gold_operations": gold_operations,
        })

    return Dataset.from_list(dataset_examples)


# ============================================================
# SECTION 5: Model Setup for RL
# ============================================================
# LEARNING NOTE: The RL model starts from the SFT checkpoint, NOT from scratch.
# This is crucial -- without SFT warmup, the model would generate garbage
# and never get positive rewards, making RL training impossible.
#
# The paper's training pipeline:
#   Base model (Qwen-2.5-7B) -> SFT fine-tuning -> RL fine-tuning
#
# For PPO, the model needs a "value head" -- an extra linear layer that
# predicts the expected reward from each state. trl provides
# AutoModelForCausalLMWithValueHead which handles this.
#
# For GRPO, no value head is needed because advantages are computed from
# the group of sampled completions (group-relative advantage).


def detect_device(verbose: bool = True) -> tuple[str, torch.dtype]:
    """Detect the best available device and dtype."""
    if torch.cuda.is_available():
        if verbose:
            print("  Device: CUDA GPU")
        return "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if verbose:
            print("  Device: Apple MPS")
        return "mps", torch.float16
    else:
        if verbose:
            print("  Device: CPU")
        return "cpu", torch.float32


def setup_model_for_ppo(args: RLTrainingArgs):
    """Load model with value head for PPO training.

    TODO: Implement this function.
    Steps:
      1. Detect device and dtype using detect_device()
      2. Set up BitsAndBytesConfig if use_4bit and CUDA (same as SFT script):
           BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
      3. Load the BASE model using AutoModelForCausalLM.from_pretrained(args.model_name, ...)
      4. Load the tokenizer with AutoTokenizer.from_pretrained(args.model_name, ...)
         Set pad_token = eos_token if pad_token is None.
      5. Create a LoRA config (same params as SFT):
           LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=LORA_TARGET_MODULES, task_type=TaskType.CAUSAL_LM, bias="none")
      6. IMPORTANT: Wrap the model with AutoModelForCausalLMWithValueHead:
           model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model, peft_config=lora_config)
         This adds a linear layer on top that predicts V(s) for advantage estimation.
         See: https://huggingface.co/docs/trl/models#trl.AutoModelForCausalLMWithValueHead
      7. If you have an SFT adapter, load its weights:
           model.pretrained_model.load_adapter(args.sft_adapter_path, "default")
      8. Return model, tokenizer, device

    HINT: AutoModelForCausalLMWithValueHead.from_pretrained() can accept
    peft_config= directly, so you don't need to call get_peft_model() separately.

    YOUR CODE HERE:
    """
    # Detect device and dtype
    device, dtype = detect_device()

    # Set up quantization config
    quantization_config = None
    if args.use_4bit and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    
    # Set up model kwargs
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["device_map"] = device if device != "cpu" else None
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # Wrap the model with AutoModelForCausalLMWithValueHead
    # This adds a value head for advantage estimation in PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model, peft_config=lora_config)

    # If you have an SFT adapter, load its weights
    if args.sft_adapter_path:
        model.pretrained_model.load_adapter(args.sft_adapter_path, "default")

    return model, tokenizer, device

def setup_model_for_grpo(args: RLTrainingArgs):
    """Load model for GRPO training (no value head needed).

    TODO: Implement this function.
    Steps:
      1. Detect device and dtype using detect_device()
      2. Set up BitsAndBytesConfig if use_4bit and CUDA (same as SFT script)
      3. Load the base model with AutoModelForCausalLM.from_pretrained(args.model_name, ...)
      4. Load the tokenizer, set pad_token = eos_token if needed
      5. Create LoRA config (same as SFT) and apply it:
           from peft import get_peft_model
           model = get_peft_model(model, lora_config)
      6. If you have an SFT adapter, load its weights:
           model.load_adapter(args.sft_adapter_path, "default")
         Or alternatively, just pass the SFT adapter path as the model_name to
         from_pretrained() and it will load the base model + adapter together.
      7. Return model, tokenizer, device

    LEARNING NOTE: GRPO is simpler than PPO for setup because it doesn't need
    a value head. The advantage is estimated by sampling G completions and
    normalizing their rewards:
      A_i = (r_i - mean(r)) / std(r)    (paper Eq. 3)

    This means: if your completion got a higher reward than average in the group,
    it gets positive advantage (reinforced). If lower, negative (discouraged).

    YOUR CODE HERE:
    """
    device, dtype = detect_device()
    quantization_config = None
    if args.use_4bit and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["device_map"] = device if device != "cpu" else None
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # Apply LoRA config via get_peft_model
    model = get_peft_model(model, lora_config)
    if args.sft_adapter_path:
        model.load_adapter(args.sft_adapter_path, "default")
    return model, tokenizer, device

# ============================================================
# SECTION 6: Memory Bank Operations (Environment Step)
# ============================================================
# LEARNING NOTE: This is the "environment" in the RL formulation.
#
# When the Memory Manager generates an operation, we need to actually apply
# it to the memory bank. This is like taking an action in an environment
# and observing the next state.
#
# The Memory Manager outputs JSON like:
#   {"memory": [{"id": "42", "text": "...", "event": "ADD"}]}
#
# We parse this, apply the operation, and return the updated memory bank.
# The updated bank is then used by the frozen Answer Agent to generate
# rewards.
#
# This section is only needed for Memory Manager RL training.
# If you're only doing Answer Agent, you can skip this and come back later.


def parse_mm_output(generated_text: str) -> list[dict]:
    """Parse Memory Manager output into a list of operations.

    The MM should output JSON like:
      {"memory": [{"id": "0", "text": "Alice likes coffee", "event": "ADD"}]}

    But since this is model-generated during RL, the text is often malformed:
      - Extra text before/after the JSON (reasoning, explanation, markdown fences)
      - Truncated JSON (hit max_new_tokens mid-object)
      - Wrong keys, missing fields, or completely garbled output

    This function is called by compute_mm_reward() to turn the MM's free-form
    generation into structured operations that apply_memory_operations() can
    execute on the memory bank. If parsing fails, return [] (= NOOP), meaning
    the memory bank stays unchanged and the reward reflects that.

    TODO: Implement this function.

    Steps:
      1. Extract JSON from the generated text.
         The model may wrap the JSON in explanation text or markdown fences.
         Use re.search(r'\\{.*\\}', generated_text, re.DOTALL) to grab the
         outermost curly-brace block. If no match, return [].

      2. Parse the extracted string with json.loads().
         If it raises json.JSONDecodeError (malformed/truncated), return [].

      3. Get the operations list from parsed["memory"].
         The top-level key is "memory" and its value is a list.
         If the key is missing or the value isn't a list, return [].

      4. Validate each operation. Keep only dicts that have ALL of:
           - "event": str, one of "ADD", "UPDATE", "DELETE", "NONE"
           - "text":  str  (the memory content; needed for ADD and UPDATE)
           - "id":    str  (memory ID; needed for UPDATE and DELETE)
         Filter out any invalid entries. It's fine if the filtered list is empty.

      5. Return the filtered list of valid operation dicts.

    Example:
      Input:  'Here is my update:\n{"memory": [{"id": "3", "text": "...", "event": "UPDATE"}]}'
      Output: [{"id": "3", "text": "...", "event": "UPDATE"}]

      Input:  'I think we should keep everything'  (no JSON)
      Output: []  (treated as NOOP)

    YOUR CODE HERE:
    """
    # Step 1: Extract outermost JSON object from potentially noisy text
    json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
    if not json_match:
        return []

    # Step 2: Parse JSON
    json_str = json_match.group(0)
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    # Step 3: Extract "memory" key
    operations = parsed.get("memory", [])
    if not isinstance(operations, list):
        return []

    # Step 4: Filter to only valid operations (don't reject all for one bad op)
    valid_events = {"ADD", "UPDATE", "DELETE", "NONE"}
    valid_ops = [
        op for op in operations
        if isinstance(op, dict)
        and op.get("event") in valid_events
        and isinstance(op.get("text"), str)
        and isinstance(op.get("id"), str)
    ]

    return valid_ops


def apply_memory_operations(
    memory_bank: list[dict],
    operations: list[dict],
) -> list[dict]:
    """Apply Memory Manager operations to a memory bank.

    This is the environment step: given the current memory bank and the MM's
    chosen operations, produce the next memory bank state.

    Operations:
      - ADD:    append new entry to memory bank
      - UPDATE: replace text of existing entry (matched by id)
      - DELETE: remove entry from memory bank (matched by id)
      - NONE/NOOP: no change

    TODO: Implement this function.
    Steps:
      1. import copy; bank = copy.deepcopy(memory_bank)  (don't mutate input!)
      2. For each operation in operations:
         event = op.get("event", "NONE").upper()
         if event == "ADD":
             bank.append({"id": op["id"], "text": op["text"]})
         elif event == "UPDATE":
             for entry in bank:
                 if entry["id"] == op["id"]:
                     entry["text"] = op["text"]
                     break
         elif event == "DELETE":
             bank = [e for e in bank if e["id"] != op["id"]]
         # NONE: do nothing
      3. Return bank

    LEARNING NOTE: In a real system you'd also need to update embeddings
    for ADD/UPDATE operations (for future retrieval). For training, you can
    skip this since the reward comes from a frozen AA that receives the
    memories directly.

    YOUR CODE HERE:
    """
    bank = copy.deepcopy(memory_bank)
    for op in operations:
      event = op.get("event", "NONE").upper()
      if event == "ADD":
        bank.append({"id": op["id"], "text": op["text"]})
      elif event == "UPDATE":
        for entry in bank:
          if entry["id"] == op["id"]:
            entry["text"] = op["text"]
            break
      elif event == "DELETE":
        bank = [e for e in bank if e["id"] != op["id"]]
    return bank


# ============================================================
# SECTION 7: Answer Agent RL Training
# ============================================================
# LEARNING NOTE: Start here! The Answer Agent RL loop is simpler than the
# Memory Manager because the reward is DIRECT (no frozen model needed).
#
# The training loop for Answer Agent:
#   1. Sample a batch of prompts (question + retrieved memories)
#   2. Generate completions (selected memories + answer)
#   3. Extract the answer from each completion (after "**Answer:**")
#   4. Compute EM reward against gold answer
#   5. Update the policy using PPO or GRPO
#
# Paper Section 3.2:
#   y ~ pi_theta(. | q, M_ret)
#   R = EM(y_pred, y_gold)


def extract_answer_from_completion(completion: str) -> str:
    """Extract the answer portion from an Answer Agent completion.

    The AA is prompted to output: relevant memories first, then "**Answer:** <answer>"
    We need to extract just the answer part for reward computation.

    TODO: Implement this function.
    Steps:
      1. Look for "**Answer:**" in the completion (case-insensitive).
         Use: re.search(r'\\*\\*Answer:\\*\\*\\s*(.*)', completion, re.IGNORECASE | re.DOTALL)
      2. If found, return the captured group stripped of whitespace.
      3. If not found, try a simpler fallback: look for "Answer:" without bold markers.
      4. If still not found, return the last non-empty line of the completion
         (the model might have just output the answer directly).
      5. Always strip the result.

    YOUR CODE HERE:
    """
    # Try bold markdown format: **Answer:** <text>
    answer_match = re.search(r'\*\*Answer:\*\*\s*(.*)', completion, re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    # Fallback: plain "Answer:" without bold markers
    answer_match = re.search(r'Answer:\s*(.*)', completion, re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    # Last resort: return the last non-empty line
    lines = [l.strip() for l in completion.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def train_aa_grpo(args: RLTrainingArgs):
    """Train Answer Agent using GRPO.

    LEARNING NOTE: This is the recommended starting point for understanding
    RL training. GRPO is simpler than PPO (no value head, no GAE).

    Paper Section 3.2 + Eq. 3:
    For each prompt (q, M_ret):
      1. Sample G candidate answers {y_i} from the policy
      2. Compute reward for each: r_i = EM(y_i, y_gold)
      3. Compute group-relative advantage: A_i = (r_i - mean(r)) / std(r)
      4. Update policy to increase probability of high-advantage completions

    The GRPOTrainer from trl handles steps 1-4 internally. You just need to:
      - Provide a reward function
      - Configure the trainer
      - Call trainer.train()

    TODO: Implement this function.
    Steps:
      1. Load the RL dataset:
           train_path = DATA_DIR / "answer_agent_train.jsonl"
           train_ds = load_rl_dataset_aa(train_path, tokenizer, args.max_seq_length)
      2. Set up the model:
           model, tokenizer, device = setup_model_for_grpo(args)
      3. Create a reward function. GRPOTrainer expects:
           def reward_fn(completions: list[str], **kwargs) -> list[float]
         where kwargs contains extra Dataset columns (like "gold_answer").
         Inside:
           a. For each completion, call extract_answer_from_completion(text)
           b. Compute compute_em_reward(extracted_answer, gold_answer)
           c. Return list of float rewards
         Example:
           def reward_fn(completions, gold_answer, **kwargs):
               rewards = []
               for comp, gold in zip(completions, gold_answer):
                   pred = extract_answer_from_completion(comp)
                   rewards.append(compute_em_reward(pred, gold))
               return rewards
      4. Configure GRPOConfig:
           GRPOConfig(
               output_dir=str(output_path),
               num_generations=GRPO_GROUP_SIZE,
               max_completion_length=GENERATION_MAX_NEW_TOKENS,
               temperature=GENERATION_TEMPERATURE,
               beta=GRPO_KL_COEFF,
               learning_rate=args.learning_rate,
               per_device_train_batch_size=GRPO_BATCH_SIZE,
               num_train_epochs=args.num_episodes,
               max_steps=args.max_steps if args.max_steps > 0 else -1,
               logging_steps=1,
               report_to="none",
           )
      5. Create GRPOTrainer:
           trainer = GRPOTrainer(
               model=model,
               args=grpo_config,
               train_dataset=train_ds,
               reward_funcs=reward_fn,   # <-- note: "reward_funcs" not "reward_fn"
               processing_class=tokenizer,
           )
      6. trainer.train()
      7. Save the adapter:
           model.save_pretrained(str(output_path))
           tokenizer.save_pretrained(str(output_path))

    RESOURCES:
      - trl GRPOTrainer: https://huggingface.co/docs/trl/grpo_trainer
      - GRPO paper: https://arxiv.org/abs/2402.03300 (DeepSeek-Math)
      - Search-R1: https://arxiv.org/abs/2501.05366 (similar RL-for-tools pattern)

    YOUR CODE HERE:
    """
    train_path = DATA_DIR / "answer_agent_train.jsonl"
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    # Load the RL dataset
    train_ds = load_rl_dataset_aa(train_path, tokenizer, args.max_seq_length)
    # Set up the model
    model, tokenizer, device = setup_model_for_grpo(args)
    # Create a reward function
    def reward_fn(completions: list[str], **kwargs) -> list[float]:
      rewards = []
      for completion, gold_answer in zip(completions, kwargs["gold_answer"]):
        extracted_answer = extract_answer_from_completion(completion)
        r = compute_combined_reward(extracted_answer, gold_answer)
        rewards.append(r)
      return rewards
    
    output_path = resolve_rl_output_path(args, "aa")

    # Configure GRPOConfig
    grpo_config = GRPOConfig(
      output_dir=str(output_path),
      num_generations=GRPO_GROUP_SIZE,
      max_completion_length=GENERATION_MAX_NEW_TOKENS,
      temperature=GENERATION_TEMPERATURE,
      beta=GRPO_KL_COEFF,
      learning_rate=args.learning_rate,
      per_device_train_batch_size=GRPO_BATCH_SIZE,
      num_train_epochs=args.num_episodes,
      max_steps=args.max_steps if args.max_steps > 0 else -1,
      logging_steps=1,
      report_to="none",
      remove_unused_columns=False,
    )
    # Create GRPOTrainer
    trainer = GRPOTrainer(
      model=model,

      args=grpo_config,
      train_dataset=train_ds,
      reward_funcs=reward_fn,
      processing_class=tokenizer,
    )
    # Train the model
    trainer.train()
    # Save the adapter
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

def train_aa_ppo(args: RLTrainingArgs):
    """Train Answer Agent using PPO.

    LEARNING NOTE: PPO is the classic RL algorithm. It's more complex than
    GRPO but gives you finer-grained control over training.

    Paper Section 3.2 + Eq. 2:
    The clipped PPO objective:
      J(theta) = E[min(rho * A, clip(rho, 1-eps, 1+eps) * A)]
    where:
      rho = pi_theta(y|q,M) / pi_old(y|q,M)  (importance ratio)
      A = advantage estimated from reward (using GAE)
      eps = clip epsilon (0.2)

    The PPO training loop:
      1. Collect rollouts: for each prompt, generate a completion
      2. Compute rewards for each completion
      3. Estimate advantages using the value head (GAE)
      4. Update policy and value head using clipped objective
      5. Repeat

    TODO: Implement this function.
    Steps:
      1. Load the RL dataset:
           train_ds = load_rl_dataset_aa(train_path, tokenizer, args.max_seq_length)
      2. Set up the model WITH VALUE HEAD:
           model, tokenizer, device = setup_model_for_ppo(args)
      3. Configure PPOConfig:
           PPOConfig(
               batch_size=PPO_BATCH_SIZE,
               mini_batch_size=PPO_MINI_BATCH_SIZE,
               ppo_epochs=PPO_EPOCHS,
               learning_rate=args.learning_rate,
               cliprange=PPO_CLIP_EPSILON,
               target_kl=PPO_TARGET_KL,
               log_with=None,
           )
      4. Create PPOTrainer:
           trainer = PPOTrainer(
               model=model,
               config=ppo_config,
               dataset=train_ds,
               tokenizer=tokenizer,
           )
      5. Implement the training loop manually (PPO doesn't auto-generate):
           for epoch in range(args.num_episodes):
               for batch in trainer.dataloader:
                   # a. Tokenize prompts
                   query_tensors = [tokenizer.encode(p, return_tensors="pt").squeeze()
                                    for p in batch["prompt"]]
                   # b. Generate completions
                   response_tensors = [trainer.generate(q, max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                                                        temperature=GENERATION_TEMPERATURE)
                                       for q in query_tensors]
                   # c. Decode to text
                   response_texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                                     for r in response_tensors]
                   # d. Compute rewards
                   rewards = []
                   for resp_text, gold in zip(response_texts, batch["gold_answer"]):
                       pred = extract_answer_from_completion(resp_text)
                       r = compute_em_reward(pred, gold)
                       rewards.append(torch.tensor(r))
                   # e. PPO step
                   stats = trainer.step(query_tensors, response_tensors, rewards)
                   trainer.log_stats(stats, batch, rewards)
      6. Save the adapter.

    KEY DIFFERENCE from GRPO: In PPO, you manage the generation loop yourself.
    The trainer.step() takes (queries, responses, rewards) and does the
    policy update. In GRPO, the trainer handles generation internally.

    RESOURCES:
      - trl PPOTrainer: https://huggingface.co/docs/trl/ppo_trainer
      - PPO paper: https://arxiv.org/abs/1707.06347
      - RLHF blog: https://huggingface.co/blog/rlhf

    YOUR CODE HERE:
    """
    train_path = DATA_DIR / "answer_agent_train.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    train_ds = load_rl_dataset_aa(train_path, tokenizer, args.max_seq_length)
    model, tokenizer, device = setup_model_for_ppo(args)
    output_path = resolve_rl_output_path(args, "aa")

    ppo_config = PPOConfig(
               batch_size=PPO_BATCH_SIZE,
               mini_batch_size=PPO_MINI_BATCH_SIZE,
               ppo_epochs=PPO_EPOCHS,
               learning_rate=args.learning_rate,
               cliprange=PPO_CLIP_EPSILON,
               target_kl=PPO_TARGET_KL,
               log_with=None,
    )
    trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        dataset=train_ds,
        tokenizer=tokenizer,
    )
    for epoch in range(args.num_episodes):
        for batch in trainer.dataloader:
            # a. Tokenize prompts
            query_tensors = [tokenizer.encode(p, return_tensors="pt").squeeze() for p in batch["prompt"]]
            # b. Generate completions
            response = [trainer.generate(q, 
            max_new_tokens=GENERATION_MAX_NEW_TOKENS, 
            temperature=GENERATION_TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id)
             for q in query_tensors]
            
            # c. Decode to text
            response_texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response]
            # d. Compute rewards
            rewards = []
            for resp_text, gold in zip(response_texts, batch["gold_answer"]):
              pred = extract_answer_from_completion(resp_text)
              r = compute_em_reward(pred, gold)
              rewards.append(r)
            # e. PPO step
            stats = trainer.step(query_tensors, response, rewards)
            trainer.log_stats(stats, batch, rewards)
    # Save the adapter
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
# ============================================================
# SECTION 8: Memory Manager RL Training
# ============================================================
# LEARNING NOTE: Memory Manager RL is the hardest part of this implementation.
#
# The key challenge: the MM's reward is INDIRECT. The MM doesn't directly
# answer questions -- it manages memories that the AA uses to answer.
# This creates a two-step reward:
#   1. MM takes action (memory operation)
#   2. Memory bank updates
#   3. AA uses updated bank to answer a question
#   4. EM(AA's answer, gold) becomes MM's reward
#
# Paper Section 3.1:
#   (o, m') ~ pi_theta(. | x, M_old)
#   Apply operation to memory bank -> M_new
#   Frozen AA answers using M_new
#   R = EM(y_pred, y_gold)
#
# You need a FROZEN Answer Agent to compute rewards. "Frozen" means its
# weights don't change during MM training -- it's just used for evaluation.
#
# This is similar to how in actor-critic RL, the critic evaluates the actor's
# actions. Here, the AA is like a fixed critic for the MM.


def load_frozen_answer_agent(adapter_path: str, model_name: str):
    """Load a frozen (non-trainable) Answer Agent for MM reward computation.

    TODO: Implement this function.
    Steps:
      1. Load the base model:
           model = AutoModelForCausalLM.from_pretrained(model_name, ...)
      2. Load the SFT (or RL-trained) AA adapter:
           from peft import PeftModel
           model = PeftModel.from_pretrained(model, adapter_path)
      3. Set model to eval mode:
           model.eval()
      4. Disable gradients on all parameters:
           for param in model.parameters():
               param.requires_grad = False
      5. Load tokenizer:
           tokenizer = AutoTokenizer.from_pretrained(model_name, ...)
      6. Return model, tokenizer

    LEARNING NOTE: The frozen AA should be the best AA you have. If you've
    already trained an RL AA, use that. Otherwise, use the SFT AA.

    YOUR CODE HERE:
    """
    # Load with 4-bit quantization on GPU (same as training model)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    for param in model.parameters():
      param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

def compute_mm_reward(
    generated_operations: str,
    memory_bank: list[dict],
    qa_pairs: list[dict],
    frozen_aa_model,
    frozen_aa_tokenizer,
) -> float:
    """Compute reward for Memory Manager by running frozen Answer Agent.

    This is the environment step + reward computation:
      1. Parse MM's generated operations
      2. Apply operations to memory bank
      3. For each associated QA pair:
         a. Retrieve memories from updated bank
         b. Run frozen AA to generate answer
         c. Compute EM against gold
      4. Average EM across QA pairs = reward

    TODO: Implement this function.
    Steps:
      1. ops = parse_mm_output(generated_operations)
      2. updated_bank = apply_memory_operations(memory_bank, ops)
      3. total_reward = 0.0
         for qa in qa_pairs:
           a. Format the AA prompt using ANSWER_AGENT_PROMPT from prompts_r1.py:
                from agents_memory.prompts_r1 import ANSWER_AGENT_PROMPT
                memories_text = format memories from updated_bank (see prepare_r1_data.py
                for how format_aa_chatml formats retrieved memories)
                prompt = ANSWER_AGENT_PROMPT.format(memories=memories_text, question=qa["question"])
           b. Tokenize and generate with frozen model:
                inputs = frozen_aa_tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = frozen_aa_model.generate(**inputs, max_new_tokens=256, temperature=0)
                response = frozen_aa_tokenizer.decode(outputs[0], skip_special_tokens=True)
           c. pred = extract_answer_from_completion(response)
              total_reward += compute_em_reward(pred, qa["answer"])
      4. Return total_reward / len(qa_pairs) if qa_pairs else 0.0

    LEARNING NOTE: This is expensive! Each MM reward requires running the
    frozen AA for inference. For efficiency:
      - Use small batch sizes for MM training
      - Consider caching AA responses for repeated states
      - Start with a small number of QA pairs per MM example

    YOUR CODE HERE:
    """
    # No QA pairs = no reward signal; return neutral reward
    if not qa_pairs:
        return 0.0

    # Step 1: Parse the MM's free-form generation into structured operations
    ops = parse_mm_output(generated_operations)

    # Step 2: Apply operations to get the updated memory bank
    # This is the "environment step" -- the MM's action changes the state
    updated_bank = apply_memory_operations(memory_bank, ops)

    # Step 3: Format the updated memory bank into text for the AA prompt.
    # This mirrors how prepare_r1_data.py formats memories in format_aa_chatml():
    # group by speaker, then list each memory with its session number.
    by_speaker = defaultdict(list)
    for mem in updated_bank:
        by_speaker[mem.get("speaker", "Unknown")].append(mem)

    memory_lines = []
    for speaker in sorted(by_speaker):
        memory_lines.append(f"\n### {speaker}")
        for i, mem in enumerate(by_speaker[speaker], 1):
            session = mem.get("session_num", "?")
            memory_lines.append(f"{i}. [Session {session}] {mem['text']}")
    memories_text = "\n".join(memory_lines)

    # Step 4: Batch all QA prompts and run frozen AA in one generate call.
    # This is much faster than sequential generation per QA pair.
    prompts = [
        ANSWER_AGENT_PROMPT.format(memories=memories_text, question=qa["question"])
        for qa in qa_pairs
    ]

    # Tokenize with left-padding for batched generation
    frozen_aa_tokenizer.padding_side = "left"
    if frozen_aa_tokenizer.pad_token is None:
        frozen_aa_tokenizer.pad_token = frozen_aa_tokenizer.eos_token
    inputs = frozen_aa_tokenizer(prompts, return_tensors="pt", padding=True).to(
        frozen_aa_model.device
    )

    with torch.no_grad():
        outputs = frozen_aa_model.generate(
            **inputs, max_new_tokens=256, do_sample=False
        )

    # Decode each output and compute reward
    total_reward = 0.0
    for i, qa in enumerate(qa_pairs):
        response = frozen_aa_tokenizer.decode(outputs[i], skip_special_tokens=True)
        pred = extract_answer_from_completion(response)
        total_reward += compute_combined_reward(pred, qa["answer"])

    # Average across all associated QA pairs
    return total_reward / len(qa_pairs)


def train_mm_grpo(args: RLTrainingArgs):
    """Train Memory Manager using GRPO.

    Paper Section 3.1 + Eq. 3:
    For each state s = (x, M_old):
      1. Sample G candidate operations from the policy
      2. For each candidate: apply to memory bank, run frozen AA, compute EM
      3. Compute group-relative advantage: A_i = (r_i - mean(r)) / std(r)
      4. Update policy

    TODO: Implement this function.
    This follows the same pattern as train_aa_grpo() but with a more complex
    reward function that involves:
      - Parsing MM output as JSON operations
      - Applying operations to the memory bank
      - Running a frozen Answer Agent
      - Computing EM reward

    Steps:
      1. Load frozen AA model:
           frozen_aa, frozen_aa_tok = load_frozen_answer_agent(aa_adapter_path, args.model_name)
      2. Load MM RL dataset:
           train_ds = load_rl_dataset_mm(train_path, tokenizer, args.max_seq_length)
      3. Set up MM model for GRPO:
           model, tokenizer, device = setup_model_for_grpo(args)
      4. Define reward function:
           def reward_fn(completions, memory_bank_state, qa_pairs, **kwargs):
               rewards = []
               for comp, bank_json, qa_json in zip(completions, memory_bank_state, qa_pairs):
                   bank = json.loads(bank_json)
                   qas = json.loads(qa_json)
                   r = compute_mm_reward(comp, bank, qas, frozen_aa, frozen_aa_tok)
                   rewards.append(r)
               return rewards
      5. Configure GRPOConfig (same as AA but maybe smaller batch due to cost)
      6. Create GRPOTrainer and call trainer.train()
      7. Save adapter
    """
    frozen_aa, frozen_aa_tok = load_frozen_answer_agent(args.aa_adapter_path, args.model_name)
    model, tokenizer, device = setup_model_for_grpo(args)
    train_ds = load_rl_dataset_mm(DATA_DIR / "memory_manager_train.jsonl", tokenizer, args.max_seq_length)
    def reward_fn(completions, memory_bank_state, qa_pairs, **kwargs):
        rewards = []
        for comp, bank_json, qa_json in zip(completions, memory_bank_state, qa_pairs):
          bank = json.loads(bank_json)
          qas = json.loads(qa_json)
          r = compute_mm_reward(comp, bank, qas, frozen_aa, frozen_aa_tok)
          rewards.append(r)
        return rewards
    
    output_path = resolve_rl_output_path(args, "mm")

    grpo_config = GRPOConfig(
      output_dir=str(output_path),
      num_generations=GRPO_GROUP_SIZE,
      max_completion_length=GENERATION_MAX_NEW_TOKENS,
      temperature=GENERATION_TEMPERATURE,
      beta=GRPO_KL_COEFF,
      learning_rate=args.learning_rate,
      per_device_train_batch_size=GRPO_BATCH_SIZE,
      num_train_epochs=args.num_episodes,
      max_steps=args.max_steps if args.max_steps > 0 else -1,
      logging_steps=1,
      report_to="none",
      remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
      model=model,
      args=grpo_config,
      train_dataset=train_ds,
      reward_funcs=reward_fn,
      processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

def train_mm_ppo(args: RLTrainingArgs):
    """Train Memory Manager using PPO.

    Same as train_mm_grpo() but with PPO algorithm.
    Paper Section 3.1 + Eq. 2.

    TODO: Implement this function following the same pattern as train_aa_ppo()
    but with the indirect reward from compute_mm_reward().

    Steps:
      1. Load frozen AA (same as train_mm_grpo step 1)
      2. Load MM dataset, set up model with value head
      3. Manual training loop (same structure as train_aa_ppo):
           for batch in dataloader:
               query_tensors = tokenize prompts
               response_tensors = generate completions
               rewards = [compute_mm_reward(resp, bank, qas, frozen_aa, frozen_aa_tok)
                          for resp, bank, qas in zip(...)]
               rewards = [torch.tensor(r) for r in rewards]
               stats = trainer.step(query_tensors, response_tensors, rewards)
      4. Save adapter

    YOUR CODE HERE:
    """
    raise NotImplementedError(
        "Implement train_mm_ppo().\n"
        "Same structure as train_aa_ppo() but with indirect rewards\n"
        "from compute_mm_reward()."
    )


# ============================================================
# SECTION 9: Evaluation
# ============================================================
# LEARNING NOTE: After RL training, you want to compare against the SFT baseline.
# The paper uses three metrics:
#   - F1 (token-level overlap)
#   - BLEU-1 (unigram precision)
#   - LLM-as-Judge (semantic correctness -- uses a separate LLM)
#
# For quick iteration, EM and F1 are sufficient. LLM-as-Judge is expensive
# but gives a better picture of actual quality.


def evaluate_agent(
    model,
    tokenizer,
    eval_dataset: Dataset,
    agent_type: str,
    device: str,
) -> dict:
    """Evaluate a trained agent on the validation set.

    TODO: Implement this function.
    Steps:
      1. model.eval()
      2. results = {"em": [], "f1": []}
      3. For each example in eval_dataset:
         a. prompt = example["prompt"]
         b. inputs = tokenizer(prompt, return_tensors="pt").to(device)
         c. with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                                         temperature=0.0, do_sample=False)
            (temperature=0 = greedy decoding for deterministic evaluation)
         d. response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True)
            (Note: slice off the input tokens -- we only want the generated part)
         e. If agent_type == "aa":
                pred = extract_answer_from_completion(response)
                gold = example["gold_answer"]
                results["em"].append(compute_em_reward(pred, gold))
                results["f1"].append(compute_f1_reward(pred, gold))
      4. Return {
             "em": sum(results["em"]) / len(results["em"]),
             "f1": sum(results["f1"]) / len(results["f1"]),
             "n": len(results["em"]),
         }

    LEARNING NOTE: For a fair comparison with SFT, use greedy decoding
    (temperature=0) during evaluation, even though you used sampling during
    training. This gives deterministic, reproducible results.

    YOUR CODE HERE:
    """
    model.eval()
    results = {"em": [], "f1": []}

    for example in eval_dataset:
      prompt = example["prompt"]
      inputs = tokenizer(prompt, return_tensors="pt").to(device)
      with torch.no_grad():
        # Generate response
        outputs = model.generate(**inputs, max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                                 do_sample=False)
        
        # Decode response and remove input tokens
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        if agent_type == "aa":
          pred = extract_answer_from_completion(response)
          gold = example["gold_answer"]
          results["em"].append(compute_em_reward(pred, gold))
          results["f1"].append(compute_f1_reward(pred, gold))
    
    return {
      "em": sum(results["em"]) / len(results["em"]),
      "f1": sum(results["f1"]) / len(results["f1"]),
      "n": len(results["em"])
    }

# ============================================================
# SECTION 10: Main + CLI
# ============================================================


def parse_args() -> RLTrainingArgs:
    parser = argparse.ArgumentParser(
        description="RL training for Memory-R1 agents (PPO or GRPO)"
    )
    parser.add_argument(
        "--agent", choices=["mm", "aa"], default="aa",
        help="Which agent to train: mm (Memory Manager) or aa (Answer Agent)",
    )
    parser.add_argument(
        "--method", choices=["ppo", "grpo"], default="grpo",
        help="RL method: ppo or grpo (default: grpo)",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Base model name/path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--sft-adapter", default="",
        help="Path to SFT adapter checkpoint to initialize from",
    )
    parser.add_argument(
        "--aa-adapter", default="",
        help="Path to frozen Answer Agent adapter (for MM reward computation)",
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help=f"Output directory for RL adapters (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Minimal smoke test (tiny model, few steps)",
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--episodes", type=int, default=3, help="Number of RL episodes/epochs")
    parser.add_argument("--lr", type=float, default=RL_LEARNING_RATE)
    parser.add_argument("--no-4bit", action="store_true")

    cli = parser.parse_args()

    args = RLTrainingArgs(
        agent=cli.agent,
        method=cli.method,
        output_dir=cli.output_dir,
        sft_adapter_path=cli.sft_adapter,
        aa_adapter_path=cli.aa_adapter,
        num_episodes=cli.episodes,
        learning_rate=cli.lr,
        use_4bit=not cli.no_4bit,
        max_steps=cli.max_steps,
    )

    if cli.smoke_test:
        args.smoke_test = True
        args.model_name = cli.model or SMOKE_TEST_MODEL
        args.max_steps = 3
        args.lora_r = 8
        args.lora_alpha = 8
        args.use_4bit = False
    else:
        args.model_name = cli.model or DEFAULT_MODEL

    # Auto-detect frozen AA adapter for MM training (needed for reward computation)
    if args.agent == "mm" and not args.aa_adapter_path:
        candidate = Path(args.output_dir) / "adapter_answer_agent"
        if candidate.exists():
            args.aa_adapter_path = str(candidate)
            print(f"  Auto-detected AA adapter: {candidate}")
        else:
            print(f"  WARNING: No AA adapter found at {candidate}")
            print("  MM training requires a frozen AA for reward. Train AA first.")

    # Auto-detect SFT adapter path if not provided
    if not args.sft_adapter_path:
        adapter_name = "adapter_memory_manager" if args.agent == "mm" else "adapter_answer_agent"
        candidate = Path(args.output_dir) / adapter_name
        if candidate.exists():
            args.sft_adapter_path = str(candidate)
            print(f"  Auto-detected SFT adapter: {candidate}")
        else:
            print(f"  WARNING: No SFT adapter found at {candidate}")
            print("  RL training without SFT warmup is not recommended.")
            print("  Run train_memory_r1.py first, or pass --sft-adapter explicitly.")

    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("Memory-R1 RL Training")
    print("=" * 60)
    print(f"  Agent: {args.agent}")
    print(f"  Method: {args.method}")
    print(f"  Model: {args.model_name}")
    print(f"  SFT adapter: {args.sft_adapter_path or 'None (training from base)'}")
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  LR: {args.learning_rate}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Smoke test: {args.smoke_test}")

    # Verify data exists
    for fname in ["memory_manager_train.jsonl", "answer_agent_train.jsonl"]:
        p = DATA_DIR / fname
        if not p.exists():
            print(f"\nError: {p} not found.")
            print("Run prepare_r1_data.py first:")
            print("  cd agents-memory && uv run python scripts/prepare_r1_data.py")
            sys.exit(1)

    # Dispatch to the appropriate training function
    # LEARNING NOTE: The dispatch is simple -- 2 agents x 2 methods = 4 functions.
    # Start with train_aa_grpo (simplest) and work your way up.
    dispatch = {
        ("aa", "grpo"): train_aa_grpo,
        ("aa", "ppo"): train_aa_ppo,
        ("mm", "grpo"): train_mm_grpo,
        ("mm", "ppo"): train_mm_ppo,
    }

    train_fn = dispatch[(args.agent, args.method)]
    print(f"\n  Dispatching to: {train_fn.__name__}")
    train_fn(args)

    print("\n" + "=" * 60)
    print("RL TRAINING COMPLETE")
    print("=" * 60)

# TIPS:
# - Read the trl docs AND examples before starting. The API has changed
#   between versions, so check the docs for YOUR version.
# - Start with --smoke-test to verify the loop works before full training.
# - Print intermediate values liberally (prompts, completions, rewards).
# - If all rewards are 0, the generation quality might be too low.
#   Try: lower temperature, use SFT adapter as init, increase max_new_tokens.
# - If training is unstable, lower the learning rate or increase KL penalty.
# - GRPO is generally easier to get working than PPO. Start there.
#
# DEBUGGING CHECKLIST:
# [ ] Does load_rl_dataset_aa() return prompts that look correct?
# [ ] Does the model generate reasonable completions from those prompts?
# [ ] Does extract_answer_from_completion() correctly find the answer?
# [ ] Does compute_em_reward() return 1.0 for matching answers?
# [ ] Does GRPOTrainer accept your reward function without errors?
# [ ] Does training loss decrease over time?
# [ ] Does eval performance improve vs SFT baseline?


if __name__ == "__main__":
    main()
