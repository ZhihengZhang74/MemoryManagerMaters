#!/usr/bin/env python3
"""Full-context baseline for LoCoMo benchmark.

Passes the entire conversation to the LLM without any memory/retrieval system.
This establishes an **upper bound** for what memory systems can achieve.

Why this matters:
- If FullContext >> Memory systems → Retrieval/extraction is the bottleneck
- If FullContext ≈ SimpleMem → SimpleMem's multi-round retrieval approximates full context
- Shows how much Mem0/MemU lose by extracting memories instead of storing raw text

Usage:
    uv run python scripts/benchmark_fullcontext.py
    uv run python scripts/benchmark_fullcontext.py --num-samples 1 --skip-judge
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# Configuration - SAME AS OTHER BENCHMARKS
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.2")

print("=" * 60)
print("Full Context Baseline Benchmark")
print("=" * 60)
print(f"  LLM Model: {LLM_MODEL}")
print(f"  Judge Model: {JUDGE_MODEL}")
print("=" * 60)

# ============================================================
# Data Loading
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
LOCOMO_PATH = DATA_DIR / "locomo10.json"

CATEGORY_NAMES = {
    1: "Factual",
    2: "Temporal",
    3: "Inferential",
    4: "Multi-hop",
    5: "Adversarial",
}


def load_locomo() -> list:
    """Load LoCoMo dataset."""
    print(f"Loading LoCoMo from {LOCOMO_PATH}")
    with open(LOCOMO_PATH) as f:
        return json.load(f)


def extract_dialogues(conversation: dict) -> list:
    """Extract all dialogue turns from a LoCoMo conversation."""
    dialogues = []
    conv_data = conversation.get("conversation", {})

    session_nums = []
    for key in conv_data.keys():
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                num = int(key.split("_")[1])
                session_nums.append(num)
            except (ValueError, IndexError):
                pass

    for num in sorted(session_nums):
        session_key = f"session_{num}"
        datetime_key = f"session_{num}_date_time"
        session_time = conv_data.get(datetime_key, "")
        session_turns = conv_data.get(session_key, [])

        if not isinstance(session_turns, list):
            continue

        for turn in session_turns:
            if isinstance(turn, dict):
                dialogues.append({
                    "speaker": turn.get("speaker", "Unknown"),
                    "text": turn.get("text", ""),
                    "timestamp": session_time,
                })

    return dialogues


# ============================================================
# Metrics
# ============================================================
def compute_f1(predicted: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    predicted = str(predicted) if not isinstance(predicted, str) else predicted
    ground_truth = str(ground_truth) if not isinstance(ground_truth, str) else ground_truth

    pred_tokens = set(re.findall(r"\w+", predicted.lower()))
    truth_tokens = set(re.findall(r"\w+", ground_truth.lower()))

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = pred_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


REFUSAL_MARKERS = [
    "no info", "not specified", "not mentioned", "no direct", "not available",
    "no evidence", "none", "not found", "no relevant", "no data",
    "cannot be determined", "not provided", "no specific", "unknown",
    "i don't", "no memory", "no record", "not addressed",
]


def _is_refusal(text: str) -> bool:
    """Check if the predicted answer is a refusal to answer."""
    return any(marker in text.lower() for marker in REFUSAL_MARKERS)


def evaluate_with_judge(question: str, expected: str, predicted: str) -> dict:
    """LLM-as-judge evaluation."""
    # Adversarial questions have empty ground truth — a refusal is correct
    if not str(expected).strip() and _is_refusal(str(predicted)):
        return {
            "judge_accurate": 1,
            "judge_explanation": "Correctly refused to answer unanswerable question",
        }

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = f"""You are evaluating a QA system's answer.

Question: {question}
Expected Answer: {expected}
Predicted Answer: {predicted}

Evaluate if the predicted answer is semantically correct:
- Different date formats ("May 7" = "7 May") are equivalent
- Extra words are OK if the core answer is correct
- "I don't know" and "None" are equivalent for unanswerable questions
- Paraphrases with same meaning are correct

Return JSON: {{"accurate": 0 or 1, "explanation": "brief reason"}}"""

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "judge_accurate": result.get("accurate", 0),
            "judge_explanation": result.get("explanation", ""),
        }
    except Exception as e:
        print(f"  Judge error: {e}")
        return {"judge_accurate": 0, "judge_explanation": str(e)}


# ============================================================
# Full Context Evaluation
# ============================================================
FULL_CONTEXT_PROMPT = """You are answering questions about a conversation between two people.
The conversation history is provided below. Answer based ONLY on information in the conversation.

Rules:
1. Give the SHORTEST answer possible - just the key fact (1-5 words max)
2. Use EXACT words from the conversation when possible
3. NO full sentences, NO explanations
4. For dates, use the format from the conversation
5. If the answer is truly not in the conversation, say "None"

CONVERSATION:
{conversation}

Now answer this question: {question}"""


def format_conversation_text(dialogues: list) -> str:
    """Format all dialogues as plain text with timestamps."""
    lines = []
    current_timestamp = None
    for d in dialogues:
        if d["timestamp"] != current_timestamp:
            current_timestamp = d["timestamp"]
            lines.append(f"\n--- {current_timestamp} ---\n")
        lines.append(f"[{d['speaker']}]: {d['text']}")
    return "\n".join(lines)


def evaluate_fullcontext(conversations: list, run_judge: bool = True) -> dict:
    """Evaluate full-context LLM on LoCoMo conversations."""
    results = []
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for conv in tqdm(conversations, desc="FullContext"):
        sample_id = conv.get("sample_id", "unknown")
        dialogues = extract_dialogues(conv)
        qa_pairs = conv.get("qa", [])

        # Format entire conversation as text
        conversation_text = format_conversation_text(dialogues)
        token_estimate = len(conversation_text) // 4  # rough estimate

        print(f"\n[FullContext] Conv {sample_id}: {len(dialogues)} turns, ~{token_estimate} tokens")
        print(f"  Evaluating {len(qa_pairs)} QA pairs...")

        # Evaluate each QA pair
        for i, qa in enumerate(qa_pairs, 1):
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            category = qa.get("category", 0)

            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{
                        "role": "user",
                        "content": FULL_CONTEXT_PROMPT.format(
                            conversation=conversation_text,
                            question=question
                        )
                    }],
                    max_tokens=50,
                    temperature=0,
                )
                predicted = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"  Error on Q{i}: {e}")
                predicted = ""

            f1 = compute_f1(predicted, ground_truth)

            result = {
                "sample_id": sample_id,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, "Unknown"),
                "f1": f1,
            }

            if run_judge:
                judge_result = evaluate_with_judge(question, ground_truth, predicted)
                result.update(judge_result)

            results.append(result)

            if i % 20 == 0:
                judge_acc = result.get("judge_accurate", "N/A")
                print(f"    QA {i}/{len(qa_pairs)} - F1: {f1:.3f}, Judge: {judge_acc}")

    return {"system": "fullcontext", "results": results}


# ============================================================
# Main
# ============================================================
def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description="Full-context baseline benchmark")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of conversations to evaluate (default: 1)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge evaluation (F1 only, faster)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: data/fullcontext_results.json)",
    )
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        return

    data = load_locomo()
    conversations = data if isinstance(data, list) else [data]
    conversations = conversations[: args.num_samples]

    print(f"\nEvaluating {len(conversations)} conversation(s)\n")

    results = evaluate_fullcontext(conversations, run_judge=not args.skip_judge)

    # Summary
    df = pd.DataFrame(results["results"])

    print("\n" + "=" * 70)
    print(f"FULL CONTEXT RESULTS (LLM={LLM_MODEL}, Judge={JUDGE_MODEL})")
    print("=" * 70)

    print("\nOverall Metrics:")
    print(f"  Total Questions: {len(df)}")
    print(f"  Avg F1: {df['f1'].mean():.3f}")
    print(f"  Median F1: {df['f1'].median():.3f}")

    if "judge_accurate" in df.columns:
        print(f"  Judge Accuracy: {df['judge_accurate'].mean():.1%}")

    print("\nBy Category:")
    print(f"  {'Category':12} {'F1':>8} {'Judge':>8} {'Count':>6}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 6}")
    for cat, name in CATEGORY_NAMES.items():
        cat_df = df[df["category"] == cat]
        if len(cat_df) > 0:
            f1 = cat_df["f1"].mean()
            judge = cat_df["judge_accurate"].mean() if "judge_accurate" in cat_df.columns else 0
            print(f"  {name:12} {f1:>8.3f} {judge:>7.1%} {len(cat_df):>6}")

    # Save results
    output_path = Path(args.output) if args.output else DATA_DIR / "fullcontext_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "llm_model": LLM_MODEL,
                    "judge_model": JUDGE_MODEL,
                },
                "summary": {
                    "total_questions": len(df),
                    "avg_f1": float(df["f1"].mean()),
                    "median_f1": float(df["f1"].median()),
                    "judge_accuracy": float(df["judge_accurate"].mean())
                    if "judge_accurate" in df.columns
                    else None,
                    "by_category": {
                        name: {
                            "f1": float(df[df["category"] == cat]["f1"].mean()),
                            "judge_accuracy": float(
                                df[df["category"] == cat]["judge_accurate"].mean()
                            )
                            if "judge_accurate" in df.columns
                            else None,
                            "count": int(len(df[df["category"] == cat])),
                        }
                        for cat, name in CATEGORY_NAMES.items()
                        if len(df[df["category"] == cat]) > 0
                    },
                },
                "results": results["results"],
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
