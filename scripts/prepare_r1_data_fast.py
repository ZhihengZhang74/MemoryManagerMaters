#!/usr/bin/env python3
"""Fast parallel data preparation for Memory-R1.

Same as prepare_r1_data.py but parallelizes GPT-4o-mini teacher calls
using asyncio + httpx. ~5 min instead of 2 hours.
"""

import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prepare_r1_data import (
    ADVERSARIAL_CATEGORY,
    RETRIEVAL_K_UPDATE,
    RETRIEVAL_K_ANSWER,
    TRAIN_CONV,
    VAL_CONV,
    load_locomo,
    get_conversation,
    extract_dialogue_turns,
    extract_observations,
    extract_qa_pairs,
    extract_session_timestamps,
    build_full_memory_bank,
    generate_aa_examples,
    format_aa_chatml,
    format_mm_chatml,
    EmbeddingCache,
    retrieve_top_k,
)
from agents_memory.prompts_r1 import MEMORY_MANAGER_PROMPT

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "r1_training"
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "gpt-4o-mini")
MAX_CONCURRENT = 30


async def teacher_call(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Single teacher API call with concurrency limit."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=1024,
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("memory", [])
        except Exception as e:
            print(f"  Teacher error: {e}")
            return []


async def generate_mm_labels_parallel(
    turns: list[dict],
    observations: list[dict],
    embeddings: EmbeddingCache,
) -> list[dict]:
    """Generate MM labels using parallel teacher calls."""
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Build observation map
    obs_by_turn = defaultdict(list)
    for obs in observations:
        obs_by_turn[obs["evidence_ref"]].append(obs)

    # First pass: build memory bank incrementally and prepare prompts
    memory_bank = []
    next_id = 0
    turn_data = []

    for turn in turns:
        dia_id = turn["dia_id"]
        text = turn["text"].strip()
        turn_obs = obs_by_turn.get(dia_id, [])

        # Get related memories
        if memory_bank and text and len(text) >= 10:
            turn_emb = embeddings.get(text)
            related = retrieve_top_k(turn_emb, memory_bank, k=RETRIEVAL_K_UPDATE)
        else:
            related = []

        related_for_prompt = [
            {"id": m["id"], "text": m["text"], "speaker": m.get("speaker", "")}
            for m in related
        ]

        # Build prompt
        new_facts = f"[{turn['speaker']}] ({turn['timestamp']}) {text}"
        prompt = MEMORY_MANAGER_PROMPT.format(
            related_memories=json.dumps(related_for_prompt, indent=2) if related_for_prompt
            else "No existing memories yet.",
            new_facts=f"- {new_facts}",
        )

        turn_data.append({
            "turn": turn,
            "related": related_for_prompt,
            "prompt": prompt if text and len(text) >= 10 else None,
            "fallback_obs": turn_obs,
        })

        # Advance memory bank with observations (ground truth)
        for obs in turn_obs:
            emb = embeddings.get(obs["text"])
            memory_bank.append({
                "id": str(next_id),
                "text": obs["text"],
                "speaker": obs["speaker"],
                "embedding": emb,
            })
            next_id += 1

    # Second pass: parallel teacher calls
    prompt_indices = [i for i, td in enumerate(turn_data) if td["prompt"]]
    print(f"  Making {len(prompt_indices)} teacher API calls (max {MAX_CONCURRENT} concurrent)...")
    prompt_tasks = [teacher_call(client, turn_data[i]["prompt"], semaphore) for i in prompt_indices]
    results = await asyncio.gather(*prompt_tasks)
    print(f"  Done. Got {len(results)} responses.")

    result_map = {}
    for idx, ops in zip(prompt_indices, results):
        result_map[idx] = ops

    # Build examples
    examples = []
    for i, td in enumerate(turn_data):
        operations = result_map.get(i, [])

        # Fallback to observations if teacher returned nothing
        if not operations and td["fallback_obs"]:
            for obs in td["fallback_obs"]:
                operations.append({
                    "id": str(len(examples)),
                    "text": obs["text"],
                    "event": "ADD",
                })

        if not operations:
            # NOOP
            if td["related"]:
                operations = [{
                    "id": td["related"][0]["id"],
                    "text": td["related"][0]["text"],
                    "event": "NONE",
                }]

        examples.append({
            "turn": {
                "speaker": td["turn"]["speaker"],
                "text": td["turn"]["text"],
                "timestamp": td["turn"]["timestamp"],
                "dia_id": td["turn"]["dia_id"],
                "session_num": td["turn"]["session_num"],
            },
            "related_memories": td["related"],
            "operations": operations,
            "memory_bank_size": 0,
        })

    return examples


async def main_async():
    data = load_locomo()
    print(f"Loaded {len(data)} conversations")

    train_conv = get_conversation(data, TRAIN_CONV)
    val_conv = get_conversation(data, VAL_CONV)

    print(f"\n--- Train: {TRAIN_CONV} ---")
    train_turns = extract_dialogue_turns(train_conv)
    train_obs = extract_observations(train_conv)
    train_qa = extract_qa_pairs(train_conv, exclude_adversarial=True)
    train_timestamps = extract_session_timestamps(train_conv)
    print(f"  Turns: {len(train_turns)}, Observations: {len(train_obs)}, QAs: {len(train_qa)}")

    print(f"\n--- Val: {VAL_CONV} ---")
    val_turns = extract_dialogue_turns(val_conv)
    val_obs = extract_observations(val_conv)
    val_qa = extract_qa_pairs(val_conv, exclude_adversarial=True)
    val_timestamps = extract_session_timestamps(val_conv)
    print(f"  Turns: {len(val_turns)}, Observations: {len(val_obs)}, QAs: {len(val_qa)}")

    assert len(train_qa) == 152
    assert len(val_qa) == 81

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "memory_banks").mkdir(exist_ok=True)
    cache_path = OUTPUT_DIR / "embeddings_cache.json"
    embeddings = EmbeddingCache(cache_path=cache_path)

    # ---- Memory Manager Data (parallel teacher calls) ----
    print("\n=== Memory Manager Data (parallel teacher) ===")
    mm_train = await generate_mm_labels_parallel(train_turns, train_obs, embeddings)
    mm_val = await generate_mm_labels_parallel(val_turns, val_obs, embeddings)

    for name, examples in [("train", mm_train), ("val", mm_val)]:
        ops_count = defaultdict(int)
        for ex in examples:
            for op in ex["operations"]:
                ops_count[op.get("event", "NONE").upper()] += 1
        print(f"  MM {name}: {len(examples)} examples, ops: {dict(ops_count)}")

    mm_train_chatml = [format_mm_chatml(ex) for ex in mm_train if ex["operations"]]
    mm_val_chatml = [format_mm_chatml(ex) for ex in mm_val if ex["operations"]]

    for name, chatml, raw in [("train", mm_train_chatml, mm_train), ("val", mm_val_chatml, mm_val)]:
        path = OUTPUT_DIR / f"memory_manager_{name}.jsonl"
        with open(path, "w") as f:
            for ex in chatml:
                f.write(json.dumps(ex) + "\n")
        print(f"  Saved: {path} ({len(chatml)} examples)")

        raw_path = OUTPUT_DIR / f"memory_manager_{name}_raw.jsonl"
        with open(raw_path, "w") as f:
            for ex in raw:
                if not ex["operations"]:
                    continue
                f.write(json.dumps({
                    "turn": ex["turn"],
                    "operations": ex["operations"],
                    "related_memories": ex["related_memories"],
                }) + "\n")
        print(f"  Saved raw: {raw_path}")

    # ---- Answer Agent Data ----
    print("\n=== Answer Agent Data ===")
    train_memory_bank = build_full_memory_bank(train_obs, embeddings, train_timestamps)
    val_memory_bank = build_full_memory_bank(val_obs, embeddings, val_timestamps)

    for name, bank in [("train", train_memory_bank), ("val", val_memory_bank)]:
        bank_path = OUTPUT_DIR / "memory_banks" / f"{name}_memory_bank.json"
        bank_no_emb = [{k: v for k, v in m.items() if k != "embedding"} for m in bank]
        with open(bank_path, "w") as f:
            json.dump(bank_no_emb, f, indent=2)
        print(f"  Memory bank ({name}): {len(bank)} entries")

    aa_train = generate_aa_examples(train_qa, train_memory_bank, train_obs, embeddings)
    aa_val = generate_aa_examples(val_qa, val_memory_bank, val_obs, embeddings)

    aa_train_chatml = [format_aa_chatml(ex) for ex in aa_train]
    aa_val_chatml = [format_aa_chatml(ex) for ex in aa_val]

    for name, chatml, raw in [("train", aa_train_chatml, aa_train), ("val", aa_val_chatml, aa_val)]:
        path = OUTPUT_DIR / f"answer_agent_{name}.jsonl"
        with open(path, "w") as f:
            for ex in chatml:
                f.write(json.dumps(ex) + "\n")
        print(f"  Saved: {path} ({len(chatml)} examples)")

        raw_path = OUTPUT_DIR / f"answer_agent_{name}_raw.jsonl"
        with open(raw_path, "w") as f:
            for ex in raw:
                f.write(json.dumps({
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "category": ex["category"],
                    "evidence_refs": ex["evidence_refs"],
                }) + "\n")

    embeddings.save()

    # ---- Test set ----
    print("\n=== Test Set ===")
    test_conv_ids = ["conv-41", "conv-42", "conv-43", "conv-44", "conv-47", "conv-48", "conv-49", "conv-50"]
    all_test = []
    for cid in test_conv_ids:
        conv = get_conversation(data, cid)
        obs = extract_observations(conv)
        qa = extract_qa_pairs(conv, exclude_adversarial=True)
        ts = extract_session_timestamps(conv)
        bank = build_full_memory_bank(obs, embeddings, ts)
        examples = generate_aa_examples(qa, bank, obs, embeddings)
        chatml = [format_aa_chatml(ex) for ex in examples]
        all_test.extend(chatml)
        print(f"  {cid}: {len(chatml)} examples")

    test_path = OUTPUT_DIR / "answer_agent_test.jsonl"
    with open(test_path, "w") as f:
        for ex in all_test:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved: {test_path} ({len(all_test)} examples)")

    embeddings.save()

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  MM train: {len(mm_train_chatml)}")
    print(f"  MM val:   {len(mm_val_chatml)}")
    print(f"  AA train: {len(aa_train_chatml)}")
    print(f"  AA val:   {len(aa_val_chatml)}")
    print(f"  AA test:  {len(all_test)}")


if __name__ == "__main__":
    asyncio.run(main_async())
