"""Validation evaluation functions for Memory-R1 RL training.

Provides periodic evaluation during GRPO training:
- evaluate_aa: Greedy decode on val QA pairs, returns EM/F1
- evaluate_mm: Run MM on val turns, apply ops, evaluate via frozen AA

Both functions are rank0-only: on other ranks they return {} immediately so
that 4-GPU DDP training does not run validation 4 times in parallel.

Dataset prompts (`example["prompt"]`) are ALREADY chat-templated by
`load_rl_dataset_aa` / `load_rl_dataset_mm` — they must be fed to the model
as-is, without applying the chat template a second time.
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

from agents_memory.rl_rewards import (
    apply_memory_operations,
    build_aa_prompt,
    compute_em,
    compute_f1,
    extract_answer_from_completion,
    parse_mm_output,
    vllm_generate_batch,
)


def _is_main_process() -> bool:
    return not (torch.distributed.is_available() and torch.distributed.is_initialized()) \
        or torch.distributed.get_rank() == 0


def _hf_generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    """Greedy-generate a completion for an already chat-templated prompt string."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,  # Trainer disables cache when gradient checkpointing is on
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_aa(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_dataset: Dataset | list[dict],
    device: str = "cuda",
    max_new_tokens: int = 256,
    max_eval_samples: int = 40,
    **kwargs,
) -> dict:
    """Evaluate Answer Agent on validation set with greedy decoding (rank0 only).

    Args:
        model: The AA policy model under training.
        tokenizer: Tokenizer for the model.
        val_dataset: List of dicts with "prompt" (chat-templated) and "gold_answer".
        device: Device for inference.
        max_new_tokens: Maximum generation length (answers are short).
        max_eval_samples: Subsample cap to keep validation from stalling training.

    Returns:
        Dict with val_em, val_f1, val_em_by_category, n. Empty dict off rank0.
    """
    if not _is_main_process():
        return {}

    model.eval()
    examples = list(val_dataset)
    if max_eval_samples and len(examples) > max_eval_samples:
        examples = random.sample(examples, max_eval_samples)

    em_scores = []
    f1_scores = []
    category_ems: dict[str, list[float]] = {}

    try:
        for example in examples:
            prompt = example["prompt"]  # already chat-templated
            gold = example["gold_answer"]
            category = example.get("category", "unknown")

            completion = _hf_generate(model, tokenizer, prompt, device, max_new_tokens)
            predicted = extract_answer_from_completion(completion)

            em = compute_em(predicted, gold)
            f1 = compute_f1(predicted, gold)
            em_scores.append(em)
            f1_scores.append(f1)
            category_ems.setdefault(category, []).append(em)
    finally:
        model.train()

    n = len(em_scores)
    val_em = sum(em_scores) / n if n > 0 else 0.0
    val_f1 = sum(f1_scores) / n if n > 0 else 0.0
    val_em_by_category = {
        cat: sum(scores) / len(scores) for cat, scores in category_ems.items()
    }

    return {
        "val_em": val_em,
        "val_f1": val_f1,
        "val_em_by_category": val_em_by_category,
        "n": n,
    }


def evaluate_mm(
    model=None,
    frozen_aa: Any = None,
    tokenizer: AutoTokenizer = None,
    val_dataset=None,
    device: str = "cuda",
    max_new_tokens_mm: int = 1024,
    max_new_tokens_aa: int = 512,
    max_eval_samples: int = 20,
    frozen_aa_is_vllm: bool = True,
    **kwargs,
) -> dict:
    """Evaluate Memory Manager on validation set (rank0 only).

    Runs MM (policy under training, HF generate) on val turns, applies ops to
    the memory bank, then scores all resulting QA pairs in one batched call on
    the frozen AA.

    Args:
        model: The MM policy model under training.
        frozen_aa: Frozen Answer Agent — a vllm.LLM when frozen_aa_is_vllm else
            an HF model.
        tokenizer: Tokenizer shared by both models.
        val_dataset: List of dicts with "prompt", "memory_bank_state", "qa_pairs".
        device: Device for inference.
        max_new_tokens_mm: Maximum generation length for MM JSON output.
        max_new_tokens_aa: Maximum generation length for frozen AA answers.
        max_eval_samples: Max examples to evaluate (subsampling for speed).
        frozen_aa_is_vllm: Whether frozen_aa is a self-managed vllm.LLM.

    Returns:
        Dict with val_em, val_f1, n. Empty dict off rank0.
    """
    if not _is_main_process():
        return {}

    mm_model = model
    mm_model.eval()

    examples = list(val_dataset)
    if len(examples) > max_eval_samples:
        examples = random.sample(examples, max_eval_samples)

    flat_prompts: list[str] = []
    flat_golds: list[str] = []

    try:
        for example in examples:
            prompt = example["prompt"]  # already chat-templated
            bank_json = example.get("memory_bank_state", "[]")
            qa_json = example.get("qa_pairs", "[]")

            mm_completion = _hf_generate(
                mm_model, tokenizer, prompt, device, max_new_tokens_mm
            )

            operations = parse_mm_output(mm_completion)
            try:
                bank = json.loads(bank_json) if isinstance(bank_json, str) else bank_json
            except json.JSONDecodeError:
                bank = []
            updated_bank = apply_memory_operations(bank, operations)

            try:
                qa_list = json.loads(qa_json) if isinstance(qa_json, str) else qa_json
            except json.JSONDecodeError:
                qa_list = []

            for qa in qa_list:
                question = qa.get("question", "")
                gold = qa.get("answer", qa.get("gold_answer", ""))
                flat_prompts.append(build_aa_prompt(updated_bank, question, tokenizer))
                flat_golds.append(gold)
    finally:
        mm_model.train()

    # Batched frozen-AA scoring
    if frozen_aa_is_vllm:
        texts = vllm_generate_batch(
            frozen_aa, flat_prompts, max_tokens=max_new_tokens_aa, temperature=0.0
        )
    else:
        frozen_aa.eval()
        texts = [
            _hf_generate(frozen_aa, tokenizer, p, device, max_new_tokens_aa)
            for p in flat_prompts
        ]

    all_em_scores = []
    all_f1_scores = []
    for text, gold in zip(texts, flat_golds):
        predicted = extract_answer_from_completion(text)
        all_em_scores.append(compute_em(predicted, gold))
        all_f1_scores.append(compute_f1(predicted, gold))

    n = len(all_em_scores)
    return {
        "val_em": sum(all_em_scores) / n if n > 0 else 0.0,
        "val_f1": sum(all_f1_scores) / n if n > 0 else 0.0,
        "n": n,
    }
