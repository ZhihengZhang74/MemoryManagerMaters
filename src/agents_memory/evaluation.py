"""LLM-as-judge evaluation for memory retrieval systems."""

import json
import os

from openai import OpenAI

JUDGE_PROMPT = """You are evaluating a memory retrieval system.

Query: {query}
Expected: {expected}
Retrieved: {retrieved}

For each dimension, answer PASS (1) or FAIL (0):
1. Relevant: Are retrieved items relevant to the query?
2. Complete: Does retrieval cover the key expected information?
3. Accurate: Are the retrieved facts correct?

Return JSON:
{{"relevant": 0 or 1, "complete": 0 or 1, "accurate": 0 or 1, "explanation": "..."}}
"""


def evaluate_retrieval(query: str, expected: str, retrieved: str) -> dict:
    """Use LLM to judge retrieval quality.

    Args:
        query: The user query or context
        expected: What information should be retrieved
        retrieved: What was actually retrieved

    Returns:
        dict with relevant, complete, accurate scores (0 or 1) and explanation
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=os.environ.get("JUDGE_MODEL", "gpt-4.1"),
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    query=query, expected=expected, retrieved=retrieved
                ),
            }
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def evaluate_batch(results: list[dict]) -> dict:
    """Evaluate multiple results and return summary.

    Args:
        results: List of dicts with query, expected, retrieved keys

    Returns:
        dict with pass rates for each dimension (0.0 to 1.0)
    """
    scores = {"relevant": [], "complete": [], "accurate": []}

    for r in results:
        eval_result = evaluate_retrieval(
            query=r["query"], expected=r["expected"], retrieved=r["retrieved"]
        )
        for key in scores:
            scores[key].append(eval_result[key])

    n = len(results)
    return {
        "relevant_rate": sum(scores["relevant"]) / n,
        "complete_rate": sum(scores["complete"]) / n,
        "accurate_rate": sum(scores["accurate"]) / n,
        "pass_rate": sum(sum(v) for v in scores.values()) / (n * 3),
    }
