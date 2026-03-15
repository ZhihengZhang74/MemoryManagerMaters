# Agents Memory

Experiments with agent memory systems.

## Setup

```bash
uv sync --all-extras
uv run jupyter lab
```

## LoCoMo Benchmark

The [LoCoMo (Long Context Models)](https://arxiv.org/abs/2402.17753) benchmark evaluates long-term memory in conversational agents using realistic multi-session dialogues.

### About the Dataset

LLM-generated conversations, human-verified for consistency. QA answers are taken directly from the conversations and annotated with evidence turn IDs. Includes adversarial (unanswerable) questions to test hallucination.

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

*F1 is a token-overlap metric and is undefined for adversarial questions (empty ground truth), so they are excluded.*

#### LLM-as-Judge Accuracy (excluding adversarial)

| System | Overall | Factual | Temporal | Inferential | Multi-hop |
|--------|---------|---------|----------|-------------|-----------|
| **SimpleMem** | **78.3%** | 71.9% | 75.7% | 84.6% | 81.4% |
| **Mem0** | 32.2% | 21.9% | 21.6% | 38.5% | 41.4% |
| **MemU** | 21.7% | 31.2% | 10.8% | 7.7% | 25.7% |

*Adversarial questions (47) excluded from both tables — they test hallucination, not retrieval. Question counts: Factual (32), Temporal (37), Inferential (13), Multi-hop (70).*

#### Full Context Baseline (Upper Bound)

A FullContext baseline passes the entire conversation to the LLM without any memory/retrieval system, establishing an upper bound for what memory systems can achieve. In practice this is prohibitively slow — cost scales as O(n^2) with conversation length (every question sends the full history), and it's bounded by the LLM's context window.

**Key Findings:**
- SimpleMem leads on retrieval (F1: 0.512, Judge: 78.3%)
- Mem0 ranks second (F1: 0.260, Judge: 32.2%)
- MemU ranks third (F1: 0.155, Judge: 21.7%)
- Adversarial (hallucination test): MemU/Mem0 refuse correctly (79%/66%), SimpleMem hallucinates more (51%)
- SimpleMem stores raw conversation; Mem0/MemU extract summaries (lossy)

Benchmark logs: `logs/`

### Examples from Logs

**SimpleMem wins on factual/temporal** — stores raw conversation, preserves exact dates:

```
Q: "When did Melanie paint a sunrise?"  GT: "2022"
  SimpleMem: "2022"                      (raw text retained)
  Mem0:      "None"                      (date lost during extraction)
  MemU:      "None"

Q: "Where has Melanie camped?"  GT: "beach, mountains, forest"
  SimpleMem: "mountains, beach, forest"  (all locations preserved)
  Mem0:      "forest"                    (incomplete extraction)
  MemU:      "None"
```

Note: LoCoMo is a QA-oriented benchmark that rewards exact fact retrieval. Real-world agent performance may differ — systems like Mem0 and MemU that extract structured memories can be more useful for preference tracking and proactive personalization, even if they score lower on exact recall.

## TODO: Frameworks to Explore

| Framework | Paper | Code |
|-----------|-------|------|
| **Mem0** | [arXiv:2504.19413](https://arxiv.org/abs/2504.19413) | [mem0ai/mem0](https://github.com/mem0ai/mem0) |
| **SimpleMem** | [arXiv:2601.02553](https://arxiv.org/abs/2601.02553) | [aiming-lab/SimpleMem](https://github.com/aiming-lab/SimpleMem) |
| **OpenClaw** | [Docs](https://docs.openclaw.ai/concepts/memory) | [openclaw/openclaw](https://github.com/openclaw/openclaw) |
| **Memory-SFT** | [arXiv:2508.19828](https://arxiv.org/abs/2508.19828) | Not released |
| **Memory-R1** | [arXiv:2508.19828](https://arxiv.org/abs/2508.19828) | Not released |

## Memory-R1 RL Training Results

Two-phase GRPO reinforcement learning following [arXiv:2508.19828](https://arxiv.org/abs/2508.19828), training on LoCoMo conversation data with Qwen2.5-7B-Instruct as the backbone.

### Training Setup

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-7B-Instruct |
| RL algorithm | GRPO (Group Relative Policy Optimization) |
| Group size (G) | 8 |
| KL coefficient (β) | 0.01 |
| Learning rate | 5e-5 |
| LoRA | r=64, α=64, dropout=0.05 |
| Batch size | 1 × 8 grad accumulation |
| Hardware | NVIDIA H100 (Snellius) |

### Phase 1: Answer Agent (AA) — 100 steps, 67 min

Reward: direct Exact Match against gold answer (Paper Eq. 4).

| Step | val_em | val_f1 |
|------|--------|--------|
| 20 | 0.0741 | 0.2247 |
| 40 | 0.0000 | 0.0739 |
| 60 | 0.1111 | 0.2692 |
| 80 | 0.0988 | 0.2455 |
| 100 | **0.1235** | **0.2687** |

Evaluated on 81 validation QA pairs with greedy decoding.

### Phase 2: Memory Manager (MM) — 100 steps, 127 min

Reward: indirect EM via frozen AA (MM generates memory operations → applied to bank → frozen AA answers → EM score).

| Metric | Value |
|--------|-------|
| val_em | 0.0000 |
| val_f1 | 0.0371 |
| n (QA pairs) | 31 |

Evaluated on 20 sampled MM examples (74 total available).

### Notes

- AA shows clear learning signal with EM improving from 0.02 → 0.12 over 100 steps
- MM reward is sparse and indirect (credit assignment through frozen AA), resulting in near-zero EM after 100 steps — likely needs significantly more training steps
- Loss trajectory for MM was stable (0.001–0.004) suggesting the model is updating but the reward signal is too noisy/sparse at this scale
- Adapters saved at 
- Per-step metrics logged to  in each adapter directory

## Memory-R1 RL Training Results

Two-phase GRPO reinforcement learning following [arXiv:2508.19828](https://arxiv.org/abs/2508.19828), training on LoCoMo conversation data with Qwen2.5-7B-Instruct as the backbone.

### Training Setup

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-7B-Instruct |
| RL algorithm | GRPO (Group Relative Policy Optimization) |
| Group size (G) | 8 |
| KL coefficient (beta) | 0.01 |
| Learning rate | 5e-5 |
| LoRA | r=64, alpha=64, dropout=0.05 |
| Batch size | 1 x 8 grad accumulation |
| Hardware | NVIDIA H100 (Snellius) |

### Phase 1: Answer Agent (AA) -- 100 steps, 67 min

Reward: direct Exact Match against gold answer (Paper Eq. 4).

| Step | val_em | val_f1 |
|------|--------|--------|
| 20 | 0.0741 | 0.2247 |
| 40 | 0.0000 | 0.0739 |
| 60 | 0.1111 | 0.2692 |
| 80 | 0.0988 | 0.2455 |
| 100 | **0.1235** | **0.2687** |

Evaluated on 81 validation QA pairs with greedy decoding.

### Phase 2: Memory Manager (MM) -- 100 steps, 127 min

Reward: indirect EM via frozen AA (MM generates memory operations, applied to bank, frozen AA answers, EM score).

| Metric | Value |
|--------|-------|
| val_em | 0.0000 |
| val_f1 | 0.0371 |
| n (QA pairs) | 31 |

Evaluated on 20 sampled MM examples (74 total available).

### Notes

- AA shows clear learning signal with EM improving from 0.02 to 0.12 over 100 steps
- MM reward is sparse and indirect (credit assignment through frozen AA), resulting in near-zero EM after 100 steps -- likely needs significantly more training steps
- Loss trajectory for MM was stable (0.001-0.004) suggesting the model is updating but the reward signal is too noisy/sparse at this scale
- Adapters saved at `models/memory-r1-rl/` under `adapter_answer_agent_rl/` and `adapter_memory_manager_rl/`
- Per-step metrics logged to `metrics.jsonl` in each adapter directory

### Sample AA Outputs (RL-trained, val set)

10 random validation examples with greedy decoding from the best RL checkpoint (step 100):

| # | Question | Gold | Predicted | EM | F1 |
|---|----------|------|-----------|----|----|
| 1 | What do the dancers represent? | performing at the festival | Passion, dedication, and support | 0 | 0.000 |
| 2 | Gina's favorite dancing memory? | Winning first place at regionals | None | 0 | 0.000 |
| 3 | Where is Gina's fashion internship? | fashion dept of international company | international company | 0 | 0.571 |
| 4 | Jon's plan for grand opening? | savor all the good vibes | Work on opening dance studio | 0 | 0.000 |
| 5 | When is Jon opening his studio? | 20 June, 2023 | Tomorrow | 0 | 0.000 |
| 6 | Why did Gina choose the furniture? | personal style and customer comfort | matches her style, makes customers comfortable | 0 | 0.308 |
| 7 | When did Gina mention Shia Labeouf? | 23 July, 2023 | None | 0 | 0.000 |
| 8 | What did Gina design for her store? | the space, furniture, and decor | furniture and decor | 0 | 0.857 |
| 9 | When did Jon expand social media? | April, 2023 | Session 8 | 0 | 0.000 |
| 10 | Gina's favorite dance style? | Contemporary | contemporary | 1 | 1.000 |

**Sample avg: EM=0.100, F1=0.274**

Observations:
- Temporal questions (dates) are weakest -- model outputs relative references instead of exact dates
- Short, distinctive factual answers work well (e.g. "contemporary")
- Partial credit on several answers where the model captures the gist but misses exact wording
- "None" outputs on some questions suggest memory retrieval/selection failures
