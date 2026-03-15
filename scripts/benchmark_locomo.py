#!/usr/bin/env python3
"""Benchmark MemU and SimpleMem on the LoCoMo dataset.

LoCoMo: Long-term Conversational Memory benchmark
- 10 conversations with annotated QA pairs
- Categories: Factual(1), Temporal(2), Inferential(3), Multi-hop(4), Adversarial(5)

Usage:
    uv run python scripts/benchmark_locomo.py
    uv run python scripts/benchmark_locomo.py --num-samples 2 --systems simplemem
    uv run python scripts/benchmark_locomo.py --skip-judge --output results.json
"""

import argparse
import asyncio
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA_DIR = Path(__file__).parent.parent / "data"
LOCOMO_PATH = DATA_DIR / "locomo10.json"

CATEGORY_NAMES = {
    1: "Factual",
    2: "Temporal",
    3: "Inferential",
    4: "Multi-hop",
    5: "Adversarial",
}


def download_locomo() -> dict:
    """Download LoCoMo dataset if not present."""
    if LOCOMO_PATH.exists():
        print(f"Loading LoCoMo from {LOCOMO_PATH}")
        with open(LOCOMO_PATH) as f:
            return json.load(f)

    print(f"Downloading LoCoMo dataset from {LOCOMO_URL}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    response = httpx.get(LOCOMO_URL, timeout=60)
    response.raise_for_status()
    data = response.json()

    with open(LOCOMO_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved to {LOCOMO_PATH}")
    return data


def compute_f1(predicted: str, ground_truth: str | int) -> float:
    """Compute token-level F1 score between predicted and ground truth answers."""
    # Handle non-string types (e.g., integer years in LoCoMo)
    predicted = str(predicted) if not isinstance(predicted, str) else predicted
    ground_truth = str(ground_truth) if not isinstance(ground_truth, str) else ground_truth

    pred_tokens = set(re.findall(r"\w+", predicted.lower()))
    truth_tokens = set(re.findall(r"\w+", ground_truth.lower()))

    # Adversarial questions have empty ground truth — a refusal is correct
    if not truth_tokens:
        return 1.0 if (not pred_tokens or _is_refusal(predicted)) else 0.0

    if not pred_tokens:
        return 0.0

    common = pred_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


def extract_dialogues(conversation: dict) -> list[dict]:
    """Extract all dialogue turns from a LoCoMo conversation.

    LoCoMo format has sessions stored as 'session_N' keys with
    'session_N_date_time' for timestamps.
    """
    dialogues = []
    conv_data = conversation.get("conversation", {})

    # Find all session keys (session_1, session_2, etc.)
    session_nums = []
    for key in conv_data.keys():
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                num = int(key.split("_")[1])
                session_nums.append(num)
            except (ValueError, IndexError):
                pass

    # Process sessions in order
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
                    "dia_id": turn.get("dia_id", ""),
                    "timestamp": session_time,
                })

    return dialogues


def format_conversation_for_file(dialogues: list[dict], user_id: str) -> dict:
    """Format dialogues for MemU file-based memorize."""
    content = []
    for d in dialogues:
        content.append({
            "role": "user" if d["speaker"] == dialogues[0]["speaker"] else "assistant",
            "content": f"[{d['speaker']}]: {d['text']}"
        })

    return {
        "metadata": {"user_id": user_id},
        "content": content
    }


async def evaluate_simplemem(
    conversations: list[dict],
    run_judge: bool = True,
) -> dict:
    """Evaluate SimpleMem on LoCoMo conversations."""
    from simplemem import SimpleMemConfig, SimpleMemSystem, set_config

    results = []
    llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    config = SimpleMemConfig(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        llm_model=llm_model,
    )
    set_config(config)

    for conv in tqdm(conversations, desc="SimpleMem"):
        sample_id = conv.get("sample_id", "unknown")
        dialogues = extract_dialogues(conv)
        qa_pairs = conv.get("qa", [])

        print(f"\n[SimpleMem] Processing conversation {sample_id}", flush=True)
        print(f"  Dialogues: {len(dialogues)}, QA pairs: {len(qa_pairs)}", flush=True)

        # Initialize fresh memory for each conversation
        memory = SimpleMemSystem(clear_db=True)

        # Add all dialogues
        print(f"  Adding dialogues...", flush=True)
        for i, d in enumerate(dialogues):
            memory.add_dialogue(
                speaker=d["speaker"],
                content=d["text"],
                timestamp=d["timestamp"] or datetime.now().isoformat(),
            )
            if (i + 1) % 50 == 0:
                print(f"    Added {i + 1}/{len(dialogues)} dialogues", flush=True)

        # Finalize (compress and index)
        print(f"  Finalizing memory (compression + indexing)...", flush=True)
        memory.finalize()
        print(f"  Memory finalized.", flush=True)

        # Evaluate each QA pair
        print(f"  Evaluating {len(qa_pairs)} QA pairs...", flush=True)
        for i, qa in enumerate(qa_pairs, 1):
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            category = qa.get("category", 0)

            try:
                predicted = memory.ask(question)
            except Exception as e:
                print(f"  Error on Q: {question[:50]}... - {e}", flush=True)
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

            if i % 10 == 0 or i == len(qa_pairs):
                print(f"    QA {i}/{len(qa_pairs)} - F1: {f1:.3f}", flush=True)

    return {"system": "simplemem", "results": results}


async def evaluate_memu(
    conversations: list[dict],
    run_judge: bool = True,
) -> dict:
    """Evaluate MemU on LoCoMo conversations."""
    from memu.app.service import MemoryService

    results = []
    llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    embed_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    for conv in tqdm(conversations, desc="MemU"):
        sample_id = conv.get("sample_id", "unknown")
        dialogues = extract_dialogues(conv)
        qa_pairs = conv.get("qa", [])
        user_id = f"locomo-{sample_id}"

        # Initialize fresh service for each conversation
        service = MemoryService(
            llm_profiles={
                "default": {
                    "api_key": os.environ["OPENAI_API_KEY"],
                    "chat_model": llm_model,
                    "embed_model": embed_model,
                }
            },
            database_config={"metadata_store": {"provider": "inmemory"}},
            retrieve_config={"route_intention": False},
        )

        # Write conversation to temp file for memorize
        conv_data = format_conversation_for_file(dialogues, user_id)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(conv_data, f)
            temp_path = f.name

        try:
            # Memorize the conversation
            await service.memorize(
                resource_url=temp_path,
                modality="conversation",
                user={"user_id": user_id},
            )

            # Evaluate each QA pair
            for qa in qa_pairs:
                question = qa.get("question", "")
                ground_truth = qa.get("answer", "")
                category = qa.get("category", 0)

                try:
                    # Retrieve relevant memories
                    retrieval = await service.retrieve(
                        queries=[{"role": "user", "content": {"text": question}}],
                        where={"user_id": user_id},
                    )

                    memories = retrieval.get("items", [])
                    memory_text = "\n".join([m.get("summary", "") for m in memories[:10]])

                    # Generate answer from memories using LLM
                    predicted = await generate_answer_from_memories(question, memory_text)

                except Exception as e:
                    print(f"  Error on Q: {question[:50]}... - {e}")
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

        finally:
            Path(temp_path).unlink(missing_ok=True)

    return {"system": "memu", "results": results}


async def generate_answer_from_memories(question: str, memories: str) -> str:
    """Use LLM to generate an answer from retrieved memories."""
    if not memories.strip():
        return ""

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": "Answer the question concisely based only on the provided memories. "
                "If the memories don't contain relevant information, say 'I don't know'.",
            },
            {
                "role": "user",
                "content": f"Memories:\n{memories}\n\nQuestion: {question}",
            },
        ],
        max_tokens=200,
    )

    return response.choices[0].message.content.strip()


REFUSAL_MARKERS = [
    "no info", "not specified", "not mentioned", "no direct", "not available",
    "no evidence", "none", "not found", "no relevant", "no data",
    "cannot be determined", "not provided", "no specific", "unknown",
    "i don't", "no memory", "no record", "not addressed",
]


def _is_refusal(text: str) -> bool:
    """Check if the predicted answer is a refusal to answer."""
    return any(marker in text.lower() for marker in REFUSAL_MARKERS)


def evaluate_with_judge(query: str, expected: str, retrieved: str) -> dict:
    """Use LLM-as-judge to evaluate answer quality."""
    from agents_memory.evaluation import evaluate_retrieval

    # Adversarial questions have empty ground truth — a refusal is correct
    if not str(expected).strip() and _is_refusal(str(retrieved)):
        return {
            "relevant": 1, "complete": 1, "accurate": 1,
            "explanation": "Correctly refused to answer unanswerable question",
        }

    try:
        result = evaluate_retrieval(query, expected, retrieved)
        return {
            "relevant": result.get("relevant", 0),
            "complete": result.get("complete", 0),
            "accurate": result.get("accurate", 0),
            "explanation": result.get("explanation", ""),
        }
    except Exception as e:
        print(f"  Judge error: {e}")
        return {"relevant": 0, "complete": 0, "accurate": 0, "explanation": str(e)}


def aggregate_results(eval_results: list[dict]) -> dict:
    """Aggregate results by system and category."""
    summary = {}

    for system_result in eval_results:
        system = system_result["system"]
        results = system_result["results"]

        if not results:
            continue

        df = pd.DataFrame(results)

        # Overall metrics
        overall = {
            "total_questions": len(df),
            "avg_f1": df["f1"].mean(),
            "median_f1": df["f1"].median(),
        }

        if "relevant" in df.columns:
            overall["relevant_rate"] = df["relevant"].mean()
            overall["complete_rate"] = df["complete"].mean()
            overall["accurate_rate"] = df["accurate"].mean()

        # Per-category metrics
        by_category = {}
        for cat, name in CATEGORY_NAMES.items():
            cat_df = df[df["category"] == cat]
            if len(cat_df) > 0:
                cat_metrics = {
                    "count": len(cat_df),
                    "avg_f1": cat_df["f1"].mean(),
                }
                if "relevant" in cat_df.columns:
                    cat_metrics["relevant_rate"] = cat_df["relevant"].mean()
                    cat_metrics["complete_rate"] = cat_df["complete"].mean()
                    cat_metrics["accurate_rate"] = cat_df["accurate"].mean()
                by_category[name] = cat_metrics

        summary[system] = {
            "overall": overall,
            "by_category": by_category,
        }

    return summary


def print_comparison(summary: dict):
    """Print side-by-side comparison of systems."""
    print("\n" + "=" * 70)
    print("LOCOMO BENCHMARK RESULTS")
    print("=" * 70)

    systems = list(summary.keys())

    # Overall comparison
    print("\n## Overall Metrics")
    print("-" * 70)
    headers = ["Metric"] + systems
    print(f"{headers[0]:<20}", end="")
    for h in headers[1:]:
        print(f"{h:>15}", end="")
    print()

    metrics = ["total_questions", "avg_f1", "median_f1", "relevant_rate", "complete_rate", "accurate_rate"]
    for metric in metrics:
        print(f"{metric:<20}", end="")
        for sys in systems:
            val = summary[sys]["overall"].get(metric, "N/A")
            if isinstance(val, float):
                print(f"{val:>15.3f}", end="")
            else:
                print(f"{val:>15}", end="")
        print()

    # Per-category comparison
    print("\n## F1 by Category")
    print("-" * 70)
    print(f"{'Category':<20}", end="")
    for sys in systems:
        print(f"{sys:>15}", end="")
    print()

    for cat_name in CATEGORY_NAMES.values():
        print(f"{cat_name:<20}", end="")
        for sys in systems:
            cat_data = summary[sys]["by_category"].get(cat_name, {})
            f1 = cat_data.get("avg_f1", 0)
            count = cat_data.get("count", 0)
            print(f"{f1:>10.3f} ({count:>2})", end="")
        print()


async def main():
    parser = argparse.ArgumentParser(description="Benchmark memory systems on LoCoMo")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of conversations to evaluate (default: all 10)",
    )
    parser.add_argument(
        "--systems",
        choices=["both", "simplemem", "memu"],
        default="both",
        help="Which systems to benchmark",
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
        help="Output JSON file for detailed results",
    )
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        return

    # Load dataset
    data = download_locomo()
    conversations = data if isinstance(data, list) else [data]

    if args.num_samples:
        conversations = conversations[: args.num_samples]

    print(f"\nEvaluating {len(conversations)} conversation(s)")
    print(f"Systems: {args.systems}")
    print(f"LLM-as-judge: {'disabled' if args.skip_judge else 'enabled'}")

    # Run evaluations
    eval_results = []
    run_judge = not args.skip_judge

    if args.systems in ("both", "simplemem"):
        print("\n--- Evaluating SimpleMem ---")
        simplemem_results = await evaluate_simplemem(conversations, run_judge)
        eval_results.append(simplemem_results)

    if args.systems in ("both", "memu"):
        print("\n--- Evaluating MemU ---")
        memu_results = await evaluate_memu(conversations, run_judge)
        eval_results.append(memu_results)

    # Aggregate and display
    summary = aggregate_results(eval_results)
    print_comparison(summary)

    # Save detailed results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(conversations),
            "summary": summary,
            "detailed_results": {r["system"]: r["results"] for r in eval_results},
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
