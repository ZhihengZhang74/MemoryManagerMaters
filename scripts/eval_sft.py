#!/usr/bin/env python3
"""Evaluate SFT adapters for Memory-R1 (Memory Manager and Answer Agent).

Loads base model + LoRA adapter, runs greedy decoding on val set, reports EM/F1.

Usage:
    python scripts/eval_sft.py --agent mm --adapter-path models/adapter_memory_manager
    python scripts/eval_sft.py --agent aa --adapter-path models/adapter_answer_agent
    python scripts/eval_sft.py --agent both --mm-adapter models/adapter_memory_manager --aa-adapter models/adapter_answer_agent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents_memory.rl_rewards import (
    compute_em,
    compute_f1,
    extract_answer_from_completion,
    normalize_answer,
    parse_mm_output,
)


def compute_bleu1(predicted: str, gold: str) -> float:
    """Compute unigram BLEU (BLEU-1) after normalization."""
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens or not pred_tokens:
        return 1.0 if (not gold_tokens and not pred_tokens) else 0.0
    matches = sum(1 for t in pred_tokens if t in gold_tokens)
    precision = matches / len(pred_tokens)
    # BLEU-1 with brevity penalty
    bp = min(1.0, len(pred_tokens) / len(gold_tokens)) if gold_tokens else 1.0
    return bp * precision

DATA_DIR = Path(__file__).parent.parent / "data" / "r1_training"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_model(base_model: str, model_path: str = ""):
    """Load model in bf16. If model_path provided, loads full fine-tuned model."""
    load_from = model_path if model_path and Path(model_path).exists() else base_model
    print(f"Loading model: {load_from}")
    model = AutoModelForCausalLM.from_pretrained(
        load_from,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(load_from, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def eval_aa(model, tokenizer, val_path: Path, max_new_tokens: int = 512):
    """Evaluate Answer Agent on val set."""
    print(f"\n=== Evaluating Answer Agent ===")
    print(f"Val data: {val_path}")

    examples = []
    with open(val_path) as f:
        for line in f:
            examples.append(json.loads(line))

    em_scores = []
    f1_scores = []
    b1_scores = []
    results = []

    for i, ex in enumerate(examples):
        user_msg = ex["messages"][0]["content"]
        gold_full = ex["messages"][1]["content"]
        gold_answer = gold_full.split("**Answer:**")[-1].strip()

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(generated, skip_special_tokens=True)
        predicted = extract_answer_from_completion(completion)

        em = compute_em(predicted, gold_answer)
        f1 = compute_f1(predicted, gold_answer)
        b1 = compute_bleu1(predicted, gold_answer)
        em_scores.append(em)
        f1_scores.append(f1)
        b1_scores.append(b1)

        results.append({
            "question": user_msg[:100] + "...",
            "gold": gold_answer,
            "predicted": predicted,
            "em": em,
            "f1": f1,
            "b1": b1,
        })

        if i < 5 or em == 1.0:
            print(f"  [{i}] EM={em:.0f} F1={f1:.3f} B1={b1:.3f} | gold='{gold_answer}' pred='{predicted}'")

    n = len(em_scores)
    avg_em = sum(em_scores) / n
    avg_f1 = sum(f1_scores) / n
    avg_b1 = sum(b1_scores) / n
    print(f"\n  AA Results (n={n}):")
    print(f"    EM:   {avg_em:.4f}")
    print(f"    F1:   {avg_f1:.4f}")
    print(f"    B1:   {avg_b1:.4f}")
    return {"agent": "aa", "n": n, "em": avg_em, "f1": avg_f1, "b1": avg_b1, "results": results}


def eval_mm(model, tokenizer, val_path: Path, max_new_tokens: int = 1024):
    """Evaluate Memory Manager on val set (format accuracy + operation distribution)."""
    print(f"\n=== Evaluating Memory Manager ===")
    print(f"Val data: {val_path}")

    examples = []
    with open(val_path) as f:
        for line in f:
            examples.append(json.loads(line))

    valid_json = 0
    op_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0, "UNKNOWN": 0}
    gold_op_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
    total = 0

    for i, ex in enumerate(examples):
        user_msg = ex["messages"][0]["content"]
        gold_output = ex["messages"][1]["content"]

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(generated, skip_special_tokens=True)

        # Check if output is valid JSON
        ops = parse_mm_output(completion)
        if ops:
            valid_json += 1
            for op in ops:
                event = op.get("event", "UNKNOWN").upper()
                op_counts[event] = op_counts.get(event, 0) + 1

        # Count gold operations
        gold_ops = parse_mm_output(gold_output)
        for op in gold_ops:
            event = op.get("event", "NONE").upper()
            gold_op_counts[event] = gold_op_counts.get(event, 0) + 1

        total += 1

        if i < 3:
            print(f"  [{i}] Valid JSON: {bool(ops)} | Ops: {[o.get('event') for o in ops]}")

    json_rate = valid_json / total if total > 0 else 0
    print(f"\n  MM Results (n={total}):")
    print(f"    Valid JSON rate: {json_rate:.2%}")
    print(f"    Predicted ops:  {dict(op_counts)}")
    print(f"    Gold ops:       {dict(gold_op_counts)}")
    return {
        "agent": "mm",
        "n": total,
        "valid_json_rate": json_rate,
        "predicted_ops": dict(op_counts),
        "gold_ops": dict(gold_op_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT adapters")
    parser.add_argument("--agent", choices=["mm", "aa", "both"], required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default="", help="Adapter path (for single agent)")
    parser.add_argument("--mm-adapter", default="", help="MM adapter path (for --agent both)")
    parser.add_argument("--aa-adapter", default="", help="AA adapter path (for --agent both)")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Eval split")
    parser.add_argument("--output", default="", help="Save results JSON to this path")
    args = parser.parse_args()

    results = []

    if args.agent in ("mm", "both"):
        adapter = args.mm_adapter or args.adapter_path
        if not adapter:
            for p in ["models/adapter_memory_manager", "models/memory-r1-adapters/adapter_memory_manager"]:
                if Path(p).exists():
                    adapter = p
                    break
        model, tokenizer = load_model(args.base_model, adapter)
        mm_file = f"memory_manager_{args.split}.jsonl"
        mm_result = eval_mm(model, tokenizer, DATA_DIR / mm_file)
        results.append(mm_result)
        del model, tokenizer
        torch.cuda.empty_cache()

    if args.agent in ("aa", "both"):
        adapter = args.aa_adapter or args.adapter_path
        if not adapter:
            for p in ["models/adapter_answer_agent", "models/memory-r1-adapters/adapter_answer_agent"]:
                if Path(p).exists():
                    adapter = p
                    break
        model, tokenizer = load_model(args.base_model, adapter)
        aa_file = f"answer_agent_{args.split}.jsonl"
        aa_result = eval_aa(model, tokenizer, DATA_DIR / aa_file)
        results.append(aa_result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\n=== EVAL COMPLETE ===")


if __name__ == "__main__":
    main()
