"""Reward functions for Memory-R1 GRPO training.

Implements reward functions matching TRL GRPOTrainer's `reward_funcs` interface:
    fn(completions, **kwargs) -> list[float]

Two reward modes:
1. Answer Agent (AA): Direct EM/F1 against gold answer (Paper Eq. 4)
2. Memory Manager (MM): Indirect EM via frozen AA after applying memory operations
"""

from __future__ import annotations

import json
import math
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


def parse_mm_atomic(completion: str) -> dict | None:
    """Parse a single atomic operation JSON from a typed MM completion.

    Expected formats (one per completion):
        {"op": "ADD", "text": "..."}
        {"op": "UPDATE", "id": "3", "text": "..."}
        {"op": "DELETE", "id": "3"}
        {"op": "NONE"}

    Returns the parsed dict (with "event" key normalised from "op") or None
    on parse failure. Reuses the same fence-stripping / regex-fallback logic
    as ``parse_mm_output``.
    """
    text = completion.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    def _normalise(parsed: dict) -> dict | None:
        if not isinstance(parsed, dict):
            return None
        op = str(parsed.get("op", "")).upper()
        if op not in ("ADD", "UPDATE", "DELETE", "NONE"):
            return None
        result: dict[str, Any] = {"event": op}
        if "text" in parsed:
            result["text"] = parsed["text"]
        if "id" in parsed:
            result["id"] = str(parsed["id"])
        return result

    try:
        parsed = json.loads(text)
        return _normalise(parsed)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return _normalise(parsed)
        except json.JSONDecodeError:
            pass
    return None


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
    3. Reduce: per-qa correctness -> average per completion = reward

    Correctness scoring (per --reward-mode):
    - EM (default, paper Eq. 4): binary exact match against gold
    - LLM judge (`judge` provided): 6-level graded score from a frozen judge;
      EM is still computed and exposed in `last_stats` as an un-gameable
      monitoring metric.

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
        judge: Any | None = None,
        use_delta: bool = False,
        w_after: float = 0.7,
        w_delta: float = 0.3,
        correct_threshold: float = 0.5,
        delta_keep_correct: float = 0.3,
        use_tanh_delta: bool = False,
        tanh_tau: float = 0.25,
    ):
        if frozen_aa_llm is None and frozen_aa_model is None:
            raise ValueError("Provide either frozen_aa_llm (vLLM) or frozen_aa_model (HF)")
        if use_delta and judge is None:
            raise ValueError("use_delta requires a judge (graded scoring for before/after)")
        if use_tanh_delta and not use_delta:
            raise ValueError("use_tanh_delta requires use_delta (needs before/after scores)")
        self.frozen_aa_llm = frozen_aa_llm
        self.frozen_aa = frozen_aa_model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.manage_sleep = manage_sleep
        self.judge = judge  # LLMJudge instance or None (pure EM)
        # Delta reward: R = w_after*R_after + w_delta*R_delta, where R_delta maps
        # the before/after correctness transition (baseline = pre-op bank).
        self.use_delta = use_delta
        self.w_after = w_after
        self.w_delta = w_delta
        self.correct_threshold = correct_threshold
        self.delta_keep_correct = delta_keep_correct
        # Smooth delta (--reward-mode mm_smooth): R_delta = tanh((s_after -
        # s_before) / tau) on the CONTINUOUS judge scores, no hard threshold.
        self.use_tanh_delta = use_tanh_delta
        self.tanh_tau = tanh_tau
        if self.frozen_aa is not None:
            self.frozen_aa.eval()
        self.__name__ = "mm_reward"
        # Diagnostics from the most recent call (picked up by metrics callback)
        self.last_stats: dict = {}

    def _delta_value(self, before_correct: bool, after_correct: bool) -> float:
        """Map a before/after correctness transition to the delta reward.

        wrong->right: +1.0 | right->right: +delta_keep_correct | wrong->wrong: 0
        | right->wrong: -1.0
        """
        if before_correct and after_correct:
            return self.delta_keep_correct
        if not before_correct and after_correct:
            return 1.0
        if before_correct and not after_correct:
            return -1.0
        return 0.0

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
            op_type: (typed mode only) list of requested operation types, one
                per completion. When present, completions are parsed with
                ``parse_mm_atomic`` and a type-match rate is tracked.
        """
        op_types = kwargs.get("op_type")
        typed_mode = op_types is not None

        # ---- Stage 1: flatten (completion, qa) pairs into prompt lists ----
        after_prompts: list[str] = []
        before_prompts: list[str] = []
        flat_meta: list[tuple[int, str, str]] = []  # (completion_idx, question, gold)
        parse_failures = 0
        op_counts: Counter = Counter()
        type_matches = 0

        for idx, (completion, bank_json, qa_json) in enumerate(
            zip(completions, memory_bank_state, qa_pairs)
        ):
            if typed_mode:
                atomic = parse_mm_atomic(completion)
                operations = [atomic] if atomic else []
                if not operations:
                    parse_failures += 1
                if atomic and op_types[idx] is not None:
                    actual = atomic.get("event", "")
                    if actual == op_types[idx]:
                        type_matches += 1
            else:
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
                after_prompts.append(build_aa_prompt(updated_bank, question, self.tokenizer))
                if self.use_delta:
                    before_prompts.append(build_aa_prompt(bank, question, self.tokenizer))
                flat_meta.append((idx, question, gold))

        # ---- Stage 2+3: batched frozen-AA generation, then scoring ----
        # When this computer manages sleep (colocated layout), hold the engine
        # awake across ALL generations and judge calls, so no call runs against
        # a slept engine.
        own_sleep = self.manage_sleep and self.frozen_aa_llm is not None
        if own_sleep:
            torch.cuda.empty_cache()
            self.frozen_aa_llm.wake_up()
        try:
            def _gen(prompts: list[str]) -> list[str]:
                if self.frozen_aa_llm is not None:
                    return vllm_generate_batch(
                        self.frozen_aa_llm, prompts,
                        max_tokens=self.max_new_tokens, temperature=0.0,
                        manage_sleep=False,
                    )
                return self._generate_hf(prompts)

            def _score(meta: list[tuple[int, str, str]], preds: list[str]) -> tuple[list[float], list[float]]:
                em = [compute_em(p, g) for (_, _, g), p in zip(meta, preds)]
                if self.judge is not None:
                    sc = self.judge.score([(q, g, p) for (_, q, g), p in zip(meta, preds)])
                else:
                    sc = em
                return sc, em

            after_texts = _gen(after_prompts)
            after_preds = [extract_answer_from_completion(t) for t in after_texts]
            after_scores, em_values = _score(flat_meta, after_preds)

            delta_values: list[float] = []
            before_scores: list[float] = []
            if self.use_delta:
                before_texts = _gen(before_prompts)
                before_preds = [extract_answer_from_completion(t) for t in before_texts]
                before_scores, _ = _score(flat_meta, before_preds)
                if self.use_tanh_delta:
                    # Smooth bounded trend signal on continuous scores: no
                    # threshold, small noise is damped, large moves saturate.
                    delta_values = [
                        math.tanh((a - b) / self.tanh_tau)
                        for b, a in zip(before_scores, after_scores)
                    ]
                else:
                    thr = self.correct_threshold
                    delta_values = [
                        self._delta_value(b >= thr, a >= thr)
                        for b, a in zip(before_scores, after_scores)
                    ]
        finally:
            if own_sleep:
                self.frozen_aa_llm.sleep(level=1)

        # ---- Reduce per completion ----
        reward_by_completion: dict[int, list[float]] = defaultdict(list)
        em_by_completion: dict[int, list[float]] = defaultdict(list)
        for i, (idx, _, _) in enumerate(flat_meta):
            if self.use_delta:
                r = self.w_after * after_scores[i] + self.w_delta * delta_values[i]
            else:
                r = after_scores[i]
            reward_by_completion[idx].append(r)
            em_by_completion[idx].append(em_values[i])

        rewards = []
        em_rewards = []
        for idx in range(len(completions)):
            s = reward_by_completion.get(idx)
            e = em_by_completion.get(idx)
            rewards.append(sum(s) / len(s) if s else 0.0)
            em_rewards.append(sum(e) / len(e) if e else 0.0)

        n = len(completions)
        self.last_stats = {
            "mm_json_parse_failure_rate": parse_failures / n if n else 0.0,
            "mm_ops_total": sum(op_counts.values()),
            **{f"mm_ops_{k.lower()}": v for k, v in op_counts.items()},
            "mm_aa_prompts": len(after_prompts),
            # EM is always tracked as an un-gameable monitor, even in llm mode
            "mm_em_reward_mean": sum(em_rewards) / n if n else 0.0,
        }
        if typed_mode:
            self.last_stats["mm_type_match_rate"] = type_matches / n if n else 0.0
            self.last_stats["mm_adv_mode"] = "typed"
        if self.use_delta:
            trans = Counter()
            for b, a in zip(before_scores, after_scores):
                key = ("R" if b >= self.correct_threshold else "W") + "2" + (
                    "R" if a >= self.correct_threshold else "W")
                trans[key] += 1
            self.last_stats.update({
                "mm_after_score_mean": (
                    sum(after_scores) / len(after_scores) if after_scores else 0.0),
                "mm_delta_mean": (
                    sum(delta_values) / len(delta_values) if delta_values else 0.0),
                **{f"mm_trans_{k}": v for k, v in sorted(trans.items())},
            })
            if self.use_tanh_delta:
                self.last_stats["mm_tanh_tau"] = self.tanh_tau
        if self.judge is not None:
            self.last_stats.update(self.judge.last_stats)

        return rewards


# ---------------------------------------------------------------------------
# Answer Agent LLM-judge reward (callable class, --reward-mode llm)
# ---------------------------------------------------------------------------

class AAJudgeReward:
    """Callable AA reward: 6-level graded correctness from a frozen LLM judge.

    Requires the dataset to carry `question` and `gold_answer` columns.
    EM is computed alongside and exposed via `last_stats` for monitoring.
    """

    def __init__(self, judge: Any):
        self.judge = judge
        self.__name__ = "aa_llm_judge_reward"
        self.last_stats: dict = {}

    def __call__(
        self,
        completions: list[str],
        gold_answer: list[str],
        question: list[str],
        **kwargs,
    ) -> list[float]:
        predictions = [extract_answer_from_completion(c) for c in completions]
        rewards = self.judge.score(list(zip(question, gold_answer, predictions)))

        n = len(completions)
        em_mean = (
            sum(compute_em(p, g) for p, g in zip(predictions, gold_answer)) / n
            if n else 0.0
        )
        self.last_stats = {"aa_em_reward_mean": em_mean, **self.judge.last_stats}
        return rewards
