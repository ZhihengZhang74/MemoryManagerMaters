"""LLM-as-Judge reward scoring for Memory-R1 GRPO training.

Provides a 6-level graded correctness score {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
judged by a FROZEN base model on a dedicated GPU (never the policy under
training — a moving judge would make the reward non-stationary and invite
self-reward hacking).

Determinism: greedy decoding (temperature=0) + vLLM structured outputs with a
6-way choice constraint, so the judge can only emit one of the valid scores.

NOTE: this reward deviates from the Memory-R1 paper (Eq. 4 is pure binary EM).
It exists to combat sparse-reward groups (measured frac_reward_zero_std was
0.85-0.93 with EM), and is selected via --reward-mode llm.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import AutoTokenizer

JUDGE_SCORES = ["0.0", "0.5", "1.0"]

JUDGE_PROMPT = """You are an impartial judge. Given a question, a gold (reference) answer, and a predicted answer, rate how correct the predicted answer is.

Scoring rubric (respond with exactly one value):
- 1.0: fully correct — the predicted answer conveys the same key information as the gold answer (paraphrases, or different formats of the same date/number, still count as fully correct)
- 0.5: partially correct — some of the key information matches, but part is missing, imprecise, or slightly wrong
- 0.0: wrong — contradicts the gold answer, misses the key information entirely, is unrelated, or is empty

Question: {question}
Gold answer: {gold_answer}
Predicted answer: {predicted_answer}

Score:"""


def build_judge_prompt(
    question: str, gold_answer: str, predicted_answer: str, tokenizer: AutoTokenizer
) -> str:
    """Build a fully chat-templated judge prompt (ready for LLM.generate)."""
    content = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer if predicted_answer.strip() else "(empty)",
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


class LLMJudge:
    """Batched 6-level correctness judge on a frozen vLLM engine.

    The engine is shared with the frozen AA (same base model, same GPU); the
    judge only issues additional generate calls with different prompts.
    """

    def __init__(
        self,
        llm: Any,
        tokenizer: AutoTokenizer,
        max_tokens: int = 8,
        manage_sleep: bool = False,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        # False when the engine lives on a dedicated aux GPU (stays awake).
        self.manage_sleep = manage_sleep
        # Diagnostics from the most recent call
        self.last_stats: dict = {}

    def score(self, triples: list[tuple[str, str, str]]) -> list[float]:
        """Score a batch of (question, gold_answer, predicted_answer) triples.

        Returns one float in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} per triple.
        """
        if not triples:
            return []
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        prompts = [
            build_judge_prompt(q, gold, pred, self.tokenizer)
            for q, gold, pred in triples
        ]
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
            structured_outputs=StructuredOutputsParams(choice=JUDGE_SCORES),
        )

        if self.manage_sleep:
            import torch
            torch.cuda.empty_cache()
            self.llm.wake_up()
        try:
            outputs = self.llm.generate(prompts, sampling, use_tqdm=False)
        finally:
            if self.manage_sleep:
                self.llm.sleep(level=1)

        scores = []
        parse_failures = 0
        dist: Counter = Counter()
        for out in outputs:
            text = out.outputs[0].text.strip()
            try:
                value = float(text)
                if text not in JUDGE_SCORES:
                    raise ValueError(text)
            except ValueError:
                parse_failures += 1
                value = 0.0
            dist[f"{value:.1f}"] += 1
            scores.append(value)

        n = len(scores)
        self.last_stats = {
            "judge_n": n,
            "judge_parse_failure_rate": parse_failures / n if n else 0.0,
            "judge_score_mean": sum(scores) / n if n else 0.0,
            **{f"judge_dist_{k}": v for k, v in sorted(dist.items())},
        }
        return scores
