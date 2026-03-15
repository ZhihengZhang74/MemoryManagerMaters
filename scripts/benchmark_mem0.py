#!/usr/bin/env python3
"""Run Mem0 benchmark with fair configuration matching SimpleMem and MemU.

Uses the same configuration:
- LLM: gpt-4.1
- Embeddings: text-embedding-3-small
- Judge: gpt-5.2

Usage:
    uv run python scripts/benchmark_mem0_fair.py
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# Configuration - SAME AS SIMPLEMEM AND MEMU
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.2")
EMBEDDING_MODEL = "text-embedding-3-small"

print("=" * 60)
print("Mem0 Fair Benchmark")
print("=" * 60)
print(f"  LLM Model: {LLM_MODEL}")
print(f"  Judge Model: {JUDGE_MODEL}")
print(f"  Embedding Model: {EMBEDDING_MODEL}")
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
    """LLM-as-judge evaluation using gpt-5.2."""
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
# Answer Generation from Memories
# ============================================================
ANSWER_PROMPT = """You are answering questions based on memories about a user.
The memories refer to "the user" but the question may use a specific name - treat them as the same person.

Rules:
1. Give the SHORTEST answer possible - just the key fact (1-5 words max)
2. Use EXACT words from memories when possible
3. NO full sentences, NO "The user...", NO explanations
4. For dates, use the format from memories
5. If the answer is truly not in the memories, say "None"

Examples:
- Memory: "The user likes pizza" + Q: "What food does John like?" -> A: "pizza"
- Memory: "The user visited Paris in June 2023" + Q: "When did Mary visit Paris?" -> A: "June 2023"
- Memory: "The user works at Google" + Q: "Where does Alice work?" -> A: "Google"
"""


def generate_answer_from_memories(question: str, memories: str, client: OpenAI) -> str:
    """Generate concise answer from retrieved memories."""
    if not memories.strip():
        return "None"

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": f"Memories:\n{memories}\n\nQuestion: {question}"},
            ],
            max_tokens=50,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()
        # Clean up common issues
        answer = answer.replace("Answer:", "").replace("A:", "").strip()
        return answer
    except Exception as e:
        print(f"  Answer generation error: {e}")
        return ""


# ============================================================
# Mem0 Evaluation
# ============================================================
def evaluate_mem0(conversations: list, run_judge: bool = True) -> dict:
    """Evaluate Mem0 on LoCoMo conversations."""
    results = []
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for conv in tqdm(conversations, desc="Mem0"):
        sample_id = conv.get("sample_id", "unknown")
        dialogues = extract_dialogues(conv)
        qa_pairs = conv.get("qa", [])
        user_id = f"locomo-{sample_id}"

        print(f"\n[Mem0] Conv {sample_id}: {len(dialogues)} turns, {len(qa_pairs)} QA pairs")

        # Initialize fresh Mem0 instance with OpenAI config
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": LLM_MODEL,
                    "temperature": 0,
                    "api_key": os.environ["OPENAI_API_KEY"],
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": EMBEDDING_MODEL,
                    "api_key": os.environ["OPENAI_API_KEY"],
                },
            },
            "version": "v1.1",
        }
        memory = Memory.from_config(config)

        # Memorize the conversation
        print("  Memorizing conversation...")
        memories_added = 0

        # Add dialogues in batches for efficiency
        batch_size = 10
        for i in range(0, len(dialogues), batch_size):
            batch = dialogues[i : i + batch_size]
            messages = []
            for d in batch:
                # Format as conversation messages
                messages.append({
                    "role": "user",
                    "content": f"[{d['speaker']}] ({d['timestamp']}): {d['text']}",
                })

            try:
                result = memory.add(messages, user_id=user_id)
                if result:
                    memories_added += len(result) if isinstance(result, list) else 1
            except Exception as e:
                print(f"  Error adding batch {i // batch_size}: {e}")

            if (i + batch_size) % 100 == 0:
                print(f"  Processed {min(i + batch_size, len(dialogues))}/{len(dialogues)} turns")

        print(f"  Added {memories_added} memories")

        # Get all memories for debugging
        try:
            all_memories = memory.get_all(user_id=user_id)
            print(f"  Total memories stored: {len(all_memories.get('results', []))}")
        except Exception as e:
            print(f"  Could not get memory count: {e}")

        # Evaluate each QA pair
        print(f"  Evaluating {len(qa_pairs)} QA pairs...")
        for i, qa in enumerate(qa_pairs, 1):
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            category = qa.get("category", 0)

            try:
                # Retrieve relevant memories
                search_results = memory.search(query=question, user_id=user_id, limit=20)
                memories_list = search_results.get("results", []) if isinstance(search_results, dict) else search_results

                # Format memories for answer generation
                memory_text = "\n".join([
                    m.get("memory", "") if isinstance(m, dict) else str(m)
                    for m in memories_list[:20]
                ])

                # Generate answer
                predicted = generate_answer_from_memories(question, memory_text, openai_client)

            except Exception as e:
                print(f"  Error on Q{i}: {e}")
                predicted = ""
                memories_list = []

            f1 = compute_f1(predicted, ground_truth)

            result = {
                "sample_id": sample_id,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, "Unknown"),
                "f1": f1,
                "num_memories_retrieved": len(memories_list) if memories_list else 0,
            }

            if run_judge:
                judge_result = evaluate_with_judge(question, ground_truth, predicted)
                result.update(judge_result)

            results.append(result)

            if i % 20 == 0:
                judge_acc = result.get("judge_accurate", "N/A")
                print(f"    QA {i}/{len(qa_pairs)} - F1: {f1:.3f}, Judge: {judge_acc}")

        # Clean up memories for this user
        try:
            memory.delete_all(user_id=user_id)
        except Exception:
            pass  # Ignore cleanup errors

    return {"system": "mem0", "results": results}


# ============================================================
# Main
# ============================================================
def main():
    import pandas as pd

    data = load_locomo()
    conversations = data if isinstance(data, list) else [data]
    conversations = conversations[:1]  # Just 1 conversation for fair comparison

    print(f"\nEvaluating {len(conversations)} conversation(s)\n")

    results = evaluate_mem0(conversations, run_judge=True)

    # Summary
    df = pd.DataFrame(results["results"])

    print("\n" + "=" * 70)
    print(f"MEM0 RESULTS (LLM={LLM_MODEL}, Judge={JUDGE_MODEL})")
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
    output_path = DATA_DIR / "mem0_fair_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "config": {
                    "llm_model": LLM_MODEL,
                    "judge_model": JUDGE_MODEL,
                    "embedding_model": EMBEDDING_MODEL,
                },
                "summary": {
                    "total_questions": len(df),
                    "avg_f1": float(df["f1"].mean()),
                    "median_f1": float(df["f1"].median()),
                    "judge_accuracy": float(df["judge_accurate"].mean())
                    if "judge_accurate" in df.columns
                    else None,
                },
                "results": results["results"],
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
