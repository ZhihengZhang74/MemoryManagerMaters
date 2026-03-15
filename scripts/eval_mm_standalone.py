#!/usr/bin/env python3
"""Standalone MM validation evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from agents_memory.rl_eval import evaluate_mm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_memory_r1_rl_tracked import load_rl_dataset_mm, load_frozen_aa, DATA_DIR, OUTPUT_DIR, MAX_SEQ_LENGTH

def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda"
    dtype = torch.bfloat16

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load frozen AA (SFT merged + RL adapter)
    sft_aa_path = str(OUTPUT_DIR / "memory-r1-adapters" / "adapter_answer_agent")
    rl_aa_path = str(OUTPUT_DIR / "memory-r1-rl" / "adapter_answer_agent_rl" / "best")
    print(f"Loading frozen AA from: {rl_aa_path}")
    frozen_aa, _ = load_frozen_aa(model_name, rl_aa_path, sft_adapter_path=sft_aa_path, use_4bit=True)
    print("  Frozen AA loaded")

    # Load MM model (SFT merged + RL adapter)
    print("Loading MM model...")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    mm_base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, trust_remote_code=True, quantization_config=quant_config)

    sft_mm_path = str(OUTPUT_DIR / "memory-r1-adapters" / "adapter_memory_manager")
    if Path(sft_mm_path).exists():
        mm_base = PeftModel.from_pretrained(mm_base, sft_mm_path)
        mm_base = mm_base.merge_and_unload()
        print("  SFT MM adapter merged")

    rl_mm_path = str(OUTPUT_DIR / "memory-r1-rl" / "adapter_memory_manager_rl" / "final")
    if Path(rl_mm_path).exists():
        mm_model = PeftModel.from_pretrained(mm_base, rl_mm_path)
        mm_model = mm_model.merge_and_unload()
        print(f"  RL MM adapter merged from: {rl_mm_path}")
    else:
        mm_model = mm_base
        print(f"  Warning: RL MM adapter not found at {rl_mm_path}")

    # Load val data
    val_path = DATA_DIR / "memory_manager_val.jsonl"
    val_data = list(load_rl_dataset_mm(val_path, tokenizer, MAX_SEQ_LENGTH))
    print(f"  Validation examples: {len(val_data)}")

    # Run evaluation
    print("\nRunning MM validation (20 sample subset)...")
    result = evaluate_mm(
        model=mm_model,
        frozen_aa=frozen_aa,
        tokenizer=tokenizer,
        val_dataset=val_data,
        device=device,
        max_eval_samples=20,
    )
    print(f"\nResults:")
    print(f"  val_em  = {result['val_em']:.4f}")
    print(f"  val_f1  = {result['val_f1']:.4f}")
    print(f"  n       = {result['n']}")

if __name__ == "__main__":
    main()
