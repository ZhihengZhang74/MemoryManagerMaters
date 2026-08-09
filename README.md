# Agents Memory

Agent memory experiments implementing [Memory-R1](https://arxiv.org/abs/2508.19828): two-phase GRPO reinforcement learning on Qwen2.5-7B-Instruct, plus LoCoMo benchmark comparisons (SimpleMem / Mem0 / MemU).

## Setup

```bash
uv sync --all-extras          # core + dev + training extras
```

Training extras include: accelerate, datasets, peft, transformers, trl (pinned 1.9.1), bitsandbytes.

## Data Preparation

Training data is generated from the [LoCoMo](https://arxiv.org/abs/2402.17753) benchmark dataset (`data/locomo10.json`).

```bash
uv run python scripts/prepare_r1_data_fast.py       # async parallel (~5 min, preferred)
uv run python scripts/prepare_r1_data.py             # sequential (~2h)
uv run python scripts/prepare_r1_data.py --skip-teacher   # observations only, no API
```

Requires `OPENAI_API_KEY` in `.env` (for `text-embedding-3-small` embeddings + `gpt-4o-mini` teacher model).

**Conversation-level split:**

| Split | Conversation | AA QAs | MM Turns |
|-------|-------------|--------|----------|
| Train | conv-26 | 152 | 419 |
| Val | conv-30 | 81 | 365 |
| Test | conv-41..50 | 1307 | - |

Output goes to `data/r1_training/` — ChatML `.jsonl` files paired with `*_raw.jsonl` files (carrying `question`, `evidence_refs`, `turn`, `operations` fields needed by reward functions).

## Training

Two-phase GRPO reinforcement learning following the paper's order:

1. **Phase 1 — Memory Manager (MM):** trains memory operations (ADD / UPDATE / DELETE / NONE). Reward = indirect EM via a frozen base-model Answer Agent.
2. **Phase 2 — Answer Agent (AA):** trains answer generation. Reward = direct EM against the gold answer.

### Hyperparameters (Paper Appendix D)

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-7B-Instruct |
| RL algorithm | GRPO (Group Relative Policy Optimization) |
| Group size (G) | 8 |
| KL coefficient (β) | 0.01 |
| Learning rate | 1e-6 |
| Max prompt length | 4096 |
| Max completion length | 2048 |
| Generation temperature | 1.0 |
| Per-device batch size | 2 |
| Gradient accumulation | 16 (4-GPU) / 32 (3-GPU) |
| Effective batch size | 128 |
| Fine-tuning mode | Full (not LoRA) |

### Local H200 Training

```bash
source scripts/env_h200.sh   # activates conda env r1vllm, sets offline HF + PYTHONPATH + BASE_MODEL

# 4-GPU layout (colocated vLLM + frozen AA)
bash scripts/run_mm_4gpu.sh
bash scripts/run_aa_4gpu.sh

# 3-GPU layout (2 DDP ranks + 1 aux GPU for frozen engine)
bash scripts/run_mm_3gpu.sh em       # reward mode: em | llm | mm_delta | mm_smooth
bash scripts/run_aa_3gpu.sh em

# Quick smoke test (1 GPU, 2 steps)
bash scripts/run_smoke.sh
```

Run MM and AA as **separate launches** — `--phase both` keeps MM-phase vLLM engines resident during AA.

### Reward Modes

| Mode | Description |
|------|-------------|
| `em` | Binary exact match (paper Eq. 4, default) |
| `llm` | 3-level graded score {0.0, 0.5, 1.0} from a frozen LLM judge |
| `mm_delta` | `w_after * R_after + w_delta * R_delta` — rewards correctness improvement (MM only) |
| `mm_smooth` | Continuous tanh-smoothed delta on judge scores (MM only) |

### Additional Options

- `--mm-sft-epochs N` — SFT cold-start for MM before GRPO (teaches the output format from gold ops)
- `--mm-gen-mode typed` — typed 4×2 sampling with per-operation-type prompts (ADD/UPDATE/DELETE/NONE)
- `--advantage-mode twolayer` — two-level advantage (global group + local pair-level), requires typed mode
- `--use-zero2` — DeepSpeed ZeRO-2 sharding (memory fallback)

## Evaluation

```bash
# Answer Agent on val (81 QAs) or test (1307 QAs)
python scripts/eval_sft.py --agent aa --aa-adapter <path> --split val
python scripts/eval_sft.py --agent aa --aa-adapter <path> --split test --output results.json

# Memory Manager standalone (loads frozen AA + MM, runs full pipeline)
python scripts/eval_mm_standalone.py
```

Metrics: Exact Match (EM), token-level F1, BLEU-1. Answer normalization is SQuAD-style (lowercase, strip articles/punctuation, NFD unicode). Gold answers are extracted from the `**Answer:**` marker.

## Deployment

- **SageMaker:** `sagemaker/launch.py` — two-step workflow (prep on cheap CPU → train on GPU). Uses DeepSpeed ZeRO-3 + CPU offload. Phases: `rl_mm`, `rl_aa`, `eval`, `all`.
- **Databricks:** `databricks.yml` bundle + `scripts/sync_to_databricks.sh`. Deploy with `databricks bundle deploy` / `databricks bundle run`.
- **Local:** `scripts/run_*gpu.sh` on H200 machine with conda env `r1vllm`.

## LoCoMo Benchmark

The [LoCoMo](https://arxiv.org/abs/2402.17753) benchmark evaluates long-term memory in conversational agents using realistic multi-session dialogues. LLM-generated conversations, human-verified for consistency, with adversarial (unanswerable) questions to test hallucination.

### QA Categories

| Category | Description | Example Q | Example A |
|----------|-------------|-----------|-----------|
| **Factual** (36%) | Single-hop, basic recall | "What activities does Melanie partake in?" | "pottery, camping, painting, swimming" |
| **Temporal** (20.6%) | Time-related reasoning | "When did Melanie paint a sunrise?" | "2022" |
| **Inferential** (3.9%) | Requires world knowledge | "Would Melanie prefer a national park or theme park?" | "National park; she likes outdoors" |
| **Multi-hop** (14.6%) | Cross-session synthesis | "What does Caroline's necklace symbolize?" | "love, faith, and strength" |

### Benchmark Results

Fair comparison on conversation `conv-26` (199 QA pairs) using identical configuration:
- **LLM**: gpt-4.1
- **Embeddings**: OpenAI text-embedding-3-small
- **Judge**: gpt-5.2 (LLM-as-judge for semantic evaluation)

#### Token-level F1 Scores (excluding adversarial)

| System | Overall | Factual | Temporal | Inferential | Multi-hop |
|--------|---------|---------|----------|-------------|-----------|
| **SimpleMem** | **0.512** | 0.532 | 0.612 | 0.215 | 0.505 |
| **Mem0** | 0.260 | 0.221 | 0.257 | 0.138 | 0.303 |
| **MemU** | 0.155 | 0.197 | 0.106 | 0.026 | 0.186 |

#### LLM-as-Judge Accuracy (excluding adversarial)

| System | Overall | Factual | Temporal | Inferential | Multi-hop |
|--------|---------|---------|----------|-------------|-----------|
| **SimpleMem** | **78.3%** | 71.9% | 75.7% | 84.6% | 81.4% |
| **Mem0** | 32.2% | 21.9% | 21.6% | 38.5% | 41.4% |
| **MemU** | 21.7% | 31.2% | 10.8% | 7.7% | 25.7% |

*Adversarial questions (47) excluded — they test hallucination, not retrieval. Question counts: Factual (32), Temporal (37), Inferential (13), Multi-hop (70).*

**Key Findings:**
- SimpleMem leads on retrieval (F1: 0.512, Judge: 78.3%) — stores raw conversation, preserves exact facts
- Mem0 ranks second (F1: 0.260, Judge: 32.2%) — extracts structured summaries (lossy)
- MemU ranks third (F1: 0.155, Judge: 21.7%)
- Adversarial (hallucination test): MemU/Mem0 refuse correctly (79%/66%), SimpleMem hallucinates more (51%)

Note: LoCoMo is a QA-oriented benchmark that rewards exact fact retrieval. Real-world agent performance may differ — systems like Mem0 and MemU that extract structured memories can be more useful for preference tracking and proactive personalization, even if they score lower on exact recall.

## Repository Structure

```
src/agents_memory/          Core library
  prompts_r1.py             Prompt templates (AA prompt, typed MM prompt)
  rl_rewards.py             EM/F1 metrics, answer extraction, MM reward computer
  rl_grpo.py                TypedTwoLevelGRPOTrainer (typed 4x2 sampling + two-level advantage)
  rl_eval.py                evaluate_aa / evaluate_mm validation functions
  rl_callbacks.py           Metrics logging + checkpointing callback
  rl_judge.py               LLMJudge (frozen-model graded scoring)
scripts/                    All runnable scripts (data prep, training, eval, benchmarks)
configs/                    DeepSpeed ZeRO-2 config
sagemaker/                  SageMaker deployment (launch.py, entrypoint.sh, ZeRO-3 config)
data/                       LoCoMo dataset + generated training data
```

## References

- **Memory-R1** — [arXiv:2508.19828](https://arxiv.org/abs/2508.19828)
- **LoCoMo** — [arXiv:2402.17753](https://arxiv.org/abs/2402.17753)
- **GRPO** — Shao et al. 2024 (DeepSeek-Math)
- **Qwen2.5** — [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
