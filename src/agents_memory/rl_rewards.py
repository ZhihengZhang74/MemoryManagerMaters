"""Reward functions for Memory-R1 GRPO training.

Implements reward functions matching TRL GRPOTrainer's `reward_funcs` interface:
    fn(completions, **kwargs) -> list[float]

Two reward modes:
1. Answer Agent (AA): Direct EM/F1 against gold answer (Paper Eq. 4)
2. Memory Manager (MM): Indirect EM via frozen AA after applying memory operations
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import torch

from agents_memory.prompts_r1 import ANSWER_AGENT_PROMPT

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Normalize answer text for EM/F1 comparison (SQuAD-style)."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text.strip()


def extract_answer_from_completion(completion: str) -> str:
    """Extract answer text after **Answer:** marker from model completion."""
    # Try bold markdown format: **Answer:** <text>
    match = re.search(r'\*\*Answer:\*\*\s*(.*)', completion, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: plain "Answer:" without bold markers
    match = re.search(r'Answer:\s*(.*)', completion, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # Last resort: return the last non-empty line
    lines = [l.strip() for l in completion.splitlines() if l.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Core EM / F1 metrics
# ---------------------------------------------------------------------------

def compute_em(predicted: str, gold: str) -> float:
    """Binary exact match after normalization. Returns 1.0 or 0.0."""
    return 1.0 if normalize_answer(predicted) == normalize_answer(gold) else 0.0


def compute_f1(predicted: str, gold: str) -> float:
    """Token-level F1 score after normalization."""
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Shared AA prompt construction (single source of truth, paper Figure 11)
# ---------------------------------------------------------------------------

def format_memories_for_prompt(memories: list[dict]) -> str:
    """Format memory bank entries into the memories block used by the AA prompt.

    Mirrors the training-data layout: entries grouped per speaker, prefixed with
    timestamp when available. Entries use the bank schema fields `id` / `text`.
    """
    by_speaker: dict[str, list[dict]] = {}
    no_speaker: list[dict] = []
    for m in memories:
        speaker = m.get("speaker")
        if speaker:
            by_speaker.setdefault(speaker, []).append(m)
        else:
            no_speaker.append(m)

    def fmt(m: dict) -> str:
        ts = m.get("timestamp")
        text = m.get("text", "")
        return f"- {ts}: {text}" if ts else f"- {text}"

    sections = []
    for speaker, mems in by_speaker.items():
        body = "\n".join(fmt(m) for m in mems)
        sections.append(f"Memories for user {speaker}:\n{body}")
    if no_speaker:
        body = "\n".join(fmt(m) for m in no_speaker)
        sections.append(f"Memories:\n{body}")

    return "\n\n".join(sections) if sections else "No memories available."


def build_aa_prompt(memories: list[dict], question: str, tokenizer: AutoTokenizer) -> str:
    """Build a fully chat-templated Answer Agent prompt from paper Figure 11.

    Single source of truth shared by MM reward computation and validation eval.
    Returns a plain string ready for LLM.generate / model.generate (no further
    chat templating must be applied).
    """
    content = ANSWER_AGENT_PROMPT.format(
        memories=format_memories_for_prompt(memories),
        question=question,
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# vLLM batch generation helper (shared by reward + eval)
# ---------------------------------------------------------------------------

def vllm_generate_batch(
    llm: Any,
    prompts: list[str],
    max_tokens: int = 512,
    temperature: float = 0.0,
    manage_sleep: bool = True,
) -> list[str]:
    """Greedy batch generation on a self-managed vLLM engine.

    Wakes the engine before generating and puts it back to level-1 sleep after
    (weights offloaded to CPU, KV cache freed) so it holds ~0 GPU memory
    between reward calls.
    """
    if not prompts:
        return []
    from vllm import SamplingParams

    if manage_sleep:
        # Release the training process's cached-but-free CUDA blocks back to
        # the driver first: the frozen AA engine lives in a separate process
        # and needs physical memory for its weights when waking up.
        torch.cuda.empty_cache()
        llm.wake_up()
    try:
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=temperature, max_tokens=max_tokens),
            use_tqdm=False,
        )
        return [out.outputs[0].text for out in outputs]
    finally:
        if manage_sleep:
            llm.sleep(level=1)


# ---------------------------------------------------------------------------
# Memory Manager reward helpers
# ---------------------------------------------------------------------------

def parse_mm_output(completion: str) -> list[dict]:
    """Parse Memory Manager JSON output into a list of operations.

    Expected: {"memory": [...]} or just [...]. Handles markdown fences and partial JSON.
    """
    text = completion.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try parsing as {"memory": [...]}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "memory" in parsed:
            return parsed["memory"]
        if isinstance(parsed, list):
            return parsed
        return [parsed] if isinstance(parsed, dict) else []
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object or array
    for pattern in [r'\{.*\}', r'\[.*\]']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict) and "memory" in parsed:
                    return parsed["memory"]
                if isinstance(parsed, list):
                    return parsed
                return [parsed] if isinstance(parsed, dict) else []
            except json.JSONDecodeError:
                continue
    return []


def apply_memory_operations(memory_bank: list[dict], operations: list[dict]) -> list[dict]:
    """Apply parsed MM operations to a memory bank state."""
    bank = [m.copy() for m in memory_bank]

    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    next_id = max((_safe_int(m.get("id", 0)) for m in bank), default=-1) + 1

    for op in operations:
        if not isinstance(op, dict):
            continue
        event = str(op.get("event", "NONE")).upper()

        if event == "ADD":
            # Ignore model-provided ids: they can collide with existing ones.
            bank.append({
                "id": str(next_id),
                "text": op.get("text", ""),
            })
            next_id += 1
        elif event == "UPDATE":
            op_id = op.get("id", "")
            for mem in bank:
                if mem.get("id") == op_id:
                    if "text" in op:
                        mem["text"] = op["text"]
                    break
        elif event == "DELETE":
            op_id = op.get("id", "")
            bank = [m for m in bank if m.get("id") != op_id]

    return bank


# ---------------------------------------------------------------------------
# Memory Manager reward computer (callable class)
# ---------------------------------------------------------------------------

class MMRewardComputer:
    """Callable reward function for Memory Manager training.

    For each MM completion:
    1. Parse JSON ops -> apply to memory bank -> build AA prompts for its QA pairs
    2. Batch all (completion, qa) prompts into ONE frozen-AA generate call
    3. Reduce: average EM per completion = reward

    Supports two frozen-AA backends:
    - `frozen_aa_llm`: a self-managed vllm.LLM (batched, fast path)
    - `frozen_aa_model` + HF generate (serial, debug fallback)
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        frozen_aa_llm: Any | None = None,
        frozen_aa_model: AutoModelForCausalLM | None = None,
        max_new_tokens: int = 512,
        device: str | None = None,
        manage_sleep: bool = True,
    ):
        if frozen_aa_llm is None and frozen_aa_model is None:
            raise ValueError("Provide either frozen_aa_llm (vLLM) or frozen_aa_model (HF)")
        self.frozen_aa_llm = frozen_aa_llm
        self.frozen_aa = frozen_aa_model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.manage_sleep = manage_sleep
        if self.frozen_aa is not None:
            self.frozen_aa.eval()
        self.__name__ = "mm_reward"
        # Diagnostics from the most recent call (picked up by metrics callback)
        self.last_stats: dict = {}

    def _generate_hf(self, prompts: list[str]) -> list[str]:
        """Serial HF generate fallback. Prompts are already chat-templated."""
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.frozen_aa.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            results.append(self.tokenizer.decode(generated, skip_special_tokens=True))
        return results

    def __call__(
        self,
        completions: list[str],
        memory_bank_state: list[str],
        qa_pairs: list[str],
        **kwargs,
    ) -> list[float]:
        """Compute MM reward for a batch of completions.

        Args:
            completions: Plain string completions from GRPOTrainer.
            memory_bank_state: JSON-serialized memory banks (one per prompt).
            qa_pairs: JSON-serialized QA pair lists (one per prompt).
        """
        # ---- Stage 1: flatten (completion, qa) pairs into one prompt list ----
        flat_prompts: list[str] = []
        flat_meta: list[tuple[int, str]] = []  # (completion_idx, gold_answer)
        parse_failures = 0
        op_counts: Counter = Counter()

        for idx, (completion, bank_json, qa_json) in enumerate(
            zip(completions, memory_bank_state, qa_pairs)
        ):
            operations = parse_mm_output(completion)
            if not operations:
                parse_failures += 1
            for op in operations:
                if isinstance(op, dict):
                    op_counts[str(op.get("event", "NONE")).upper()] += 1

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
                flat_prompts.append(build_aa_prompt(updated_bank, question, self.tokenizer))
                flat_meta.append((idx, gold))

        # ---- Stage 2: single batched frozen-AA generation ----
        if self.frozen_aa_llm is not None:
            texts = vllm_generate_batch(
                self.frozen_aa_llm,
                flat_prompts,
                max_tokens=self.max_new_tokens,
                temperature=0.0,
                manage_sleep=self.manage_sleep,
            )
        else:
            texts = self._generate_hf(flat_prompts)

        # ---- Stage 3: reduce per-completion average EM ----
        em_by_completion: dict[int, list[float]] = defaultdict(list)
        for (idx, gold), text in zip(flat_meta, texts):
            predicted = extract_answer_from_completion(text)
            em_by_completion[idx].append(compute_em(predicted, gold))

        rewards = []
        for idx in range(len(completions)):
            scores = em_by_completion.get(idx)
            rewards.append(sum(scores) / len(scores) if scores else 0.0)

        n = len(completions)
        self.last_stats = {
            "mm_json_parse_failure_rate": parse_failures / n if n else 0.0,
            "mm_ops_total": sum(op_counts.values()),
            **{f"mm_ops_{k.lower()}": v for k, v in op_counts.items()},
            "mm_aa_prompts": len(flat_prompts),
        }

        return rewards
