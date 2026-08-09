# 训练体系完整统计

> 基于 `src/agents_memory/` 全部源码 + `scripts/train_memory_r1_rl_tracked.py` 的完整梳理。
> 训练入口：`scripts/train_memory_r1_rl_tracked.py`，Trainer：`TypedTwoLevelGRPOTrainer`（继承 TRL 1.9.1 `GRPOTrainer`）。

---

## 一、Answer Agent (AA) 训练

### 1.1 训练流程

入口函数：`train_aa()`（`train_memory_r1_rl_tracked.py:692`）

- 无 SFT 冷启动，直接从 base model 开始 RL（符合论文）
- 使用标准 `GRPOTrainer`（不走 typed 扩展）
- Reward 函数通过 `--reward-mode` 选择

### 1.2 AA Reward 模式（2 种）

| 模式 | CLI 参数 | Reward 函数 | 评分来源 | 分值范围 |
|------|---------|------------|---------|---------|
| **EM** | `--reward-mode em`（默认） | `aa_em_reward()` | 本地计算，二值精确匹配（Paper Eq. 4） | {0.0, 1.0} |
| **LLM Judge** | `--reward-mode llm` | `AAJudgeReward` 类 | 冻结 base model vLLM 评分 | {0.0, 0.5, 1.0}（离散）或 [0, 1] 两位小数（连续） |

#### EM 模式细节

- 函数：`aa_em_reward(completions, gold_answer, **kwargs) -> list[float]`（`rl_rewards.py:624`）
- 从 completion 中用正则提取 `**Answer:**` 后的文本
- SQuAD 式归一化：小写化 → NFD unicode → 去冠词(a/an/the) → 去标点 → 合并空格
- 完全匹配返回 1.0，否则 0.0

#### LLM Judge 模式细节

- 类：`AAJudgeReward(judge)`（`rl_rewards.py:587`）
- 提取 predicted answer 后，将 `(question, gold_answer, predicted_answer)` 三元组送入 `LLMJudge.score()`
- Judge 使用与 frozen AA 相同的冻结 base model vLLM 引擎（共享引擎，不同 prompt）
- **离散 judge**（默认）：3 级 rubric {0.0=错误, 0.5=部分正确, 1.0=完全正确}，vLLM `StructuredOutputsParams(choice=["0.0", "0.5", "1.0"])` 约束输出
- **连续 judge**（`--judge-continuous`）：5 级 rubric，输出两位小数 [0.00, 1.00]，vLLM regex 约束 `0\.[0-9]{2}|1\.00`
- EM 始终并行计算，作为不可作弊的监控指标（`aa_em_reward_mean`），不参与 reward

#### 辅助 reward（不参与训练）

- `aa_f1_reward()`：token 级 F1，定义在 `rl_rewards.py:633`，weight=0，仅做信息记录

### 1.3 AA 评估

- 函数：`evaluate_aa()`（`rl_eval.py:58`），rank0-only
- 贪心解码（temperature=0），最多评估 40 个样本
- 指标：EM、F1、按 QA 类别分组的 EM

---

## 二、Memory Manager (MM) 训练

### 2.1 训练流程

入口函数：`train_mm()`（`train_memory_r1_rl_tracked.py:783`）

完整流程：
1. （可选）SFT 冷启动 → 2. 加载 policy model → 3. 加载 frozen AA + 可选 judge → 4. 加载数据 → 5. 构建 `MMRewardComputer` → 6. 用 `TypedTwoLevelGRPOTrainer` 训练

### 2.2 MM 生成模式（`--mm-gen-mode`，2 种）

| 模式 | Prompt | 每次输出 | Group 构成 | 解析函数 |
|------|--------|---------|-----------|---------|
| **standard**（默认） | `MEMORY_MANAGER_PROMPT`，整库输出 | `{"memory": [...]}` 全量操作 JSON | 8 个相同 prompt，各自独立生成 | `parse_mm_output()` |
| **typed** | `MEMORY_MANAGER_TYPED_PROMPT`，指定操作类型 | 单个原子操作 `{"op": "ADD", ...}` | 4 类型(ADD/UPDATE/DELETE/NONE) × 2 采样 = 8 | `parse_mm_atomic()` |

#### typed 模式的 group 布局

```
位置:  [0, 1, 2, 3, 4, 5, 6, 7]
类型:  ADD ADD UPD UPD DEL DEL NON NON
        └─pair─┘ └─pair─┘ └─pair─┘ └─pair─┘
```

- 每个类型 2 个采样，构成 4 个 pair
- 生成时临时将 `num_generations` 设为 1（每个 typed prompt 各生成 1 次），生成后恢复为 8
- pair (2k, 2k+1) 用于 two-level advantage 的局部归一化

### 2.3 MM Advantage 模式（`--advantage-mode`，2 种）

| 模式 | 公式 | 说明 |
|------|------|------|
| **group**（默认） | `A = (r - μ_group) / (σ_group + ε)` | 标准 GRPO group advantage，与 TRL 原版 bit-identical |
| **twolayer** | `A = w_global · A_global + w_local · A_local` | 全局+局部双层混合 advantage |

#### twolayer 细节

- `A_global = (r - μ_group) / (σ_group + ε)`：与 group 模式相同
- `A_local = (r - μ_pair) / (σ_pair + ε)`：pair 内归一化，population std (ddof=0)
  - pair 为 (2k, 2k+1)，即同一 typed prompt 的 2 个采样
  - 两个 reward 不同时 |A_local| = 1；相同时 A_local = 0
  - NaN 成员归零（`nan_to_num`）
- 权重：`--adv-global-weight 0.5` / `--adv-local-weight 0.5`（默认）
- **要求** `mm_gen_mode="typed"`（pair 需要 typed prompt）
- 实现位置：`compute_two_level_advantages()`（`rl_grpo.py:98`）

### 2.4 MM SFT 冷启动（可选）

CLI：`--mm-sft-epochs N`（默认 0=关闭）

- 目的：在 GRPO 前让 policy 学会 ADD/UPDATE/DELETE 的输出格式
- 只影响 MM policy 的初始权重；frozen AA 和 judge 始终用 base model

| mm-gen-mode | SFT 数据来源 | 样本构造 |
|-------------|------------|---------|
| standard | `memory_manager_train.jsonl`（ChatML） | prompt = user message, completion = assistant message（gold ops JSON） |
| typed | `memory_manager_train_raw.jsonl`（在线构造） | 每个 gold op (event=E) → 样本(prompt_E, 原子JSON)；每个缺失类型 T → 样本(prompt_T, `{"op":"NONE"}`)，NONE 样本按类型 ≤ 2× 正样本数平衡（seed=42） |

- SFT 使用 `SFTTrainer` + `completion_only_loss=True`（仅监督 assistant token）
- 学习率：`--mm-sft-lr 1e-5`（默认）
- 保存到 `output_dir/sft_init/`，GRPO 从该路径加载

### 2.5 MM Reward 模式（`--reward-mode`，4 种）

所有 MM reward 统一由 `MMRewardComputer` 类计算（`rl_rewards.py:314`）。

| 模式 | CLI 参数 | Judge | Delta | 评分方式 | 分值范围 |
|------|---------|-------|-------|---------|---------|
| **EM** | `--reward-mode em`（默认） | 无 | 否 | frozen AA 答题 → EM 二值 | {0.0, 1.0} |
| **LLM Judge** | `--reward-mode llm` | 离散3级（可选连续） | 否 | frozen AA 答题 → judge 评分 | {0.0, 0.5, 1.0} 或 [0, 1] |
| **MM Delta** | `--reward-mode mm_delta` | 离散3级 | 是（阈值） | `w_after·R_after + w_delta·R_delta` | 混合 |
| **MM Smooth** | `--reward-mode mm_smooth` | 连续2位小数（自动） | 是（tanh） | `w_after·s_after + w_delta·tanh((s_after-s_before)/τ)` | 混合 |

#### Reward 计算管线（`MMRewardComputer.__call__`）

```
MM completion
  │
  ├─ standard: parse_mm_output() → 操作列表
  ├─ typed:    parse_mm_atomic() → 单个原子操作
  │
  ▼
apply_memory_operations(bank, ops) → updated_bank
  │
  ├─ 对每个 QA pair:
  │   ├─ after_prompt = build_aa_prompt(updated_bank, question)
  │   └─ (delta模式) before_prompt = build_aa_prompt(original_bank, question)
  │
  ▼
frozen AA vLLM 批量生成 → 提取 predicted answer
  │
  ├─ EM 模式:  compute_em(predicted, gold)
  ├─ Judge模式: LLMJudge.score((question, gold, predicted))
  │
  ▼ (delta模式额外步骤)
frozen AA 生成 before_prompts → 评分 before_scores
  │
  ├─ mm_delta:  R_delta = _delta_value(before_correct, after_correct)
  └─ mm_smooth: R_delta = tanh((s_after - s_before) / tau)
  │
  ▼
每个 completion 的 reward = mean(所有关联 QA 的分数)
```

#### 各模式详细说明

**1. EM 模式（Paper Eq. 4）**
- `judge=None`, `use_delta=False`
- frozen AA 用贪心解码答题 → `compute_em()` 二值匹配
- Reward = 该 completion 关联的所有 QA 的平均 EM
- EM 始终作为监控指标

**2. LLM Judge 模式**
- `judge=LLMJudge(...)`, `use_delta=False`
- frozen AA 答题后，judge 对 `(question, gold, predicted)` 评分
- 离散（默认）：3 级 {0.0, 0.5, 1.0}
- 连续（`--judge-continuous`）：5 级 rubric，两位小数 [0, 1]
- Reward = 平均 judge 分数
- EM 仍并行计算作为监控（`mm_em_reward_mean`）

**3. MM Delta 模式**
- `judge=LLMJudge(离散)`, `use_delta=True`, `use_tanh_delta=False`
- `R = w_after · R_after + w_delta · R_delta`
- `R_after` = 操作后 bank 的 judge 分数
- `R_delta` = 正确性转移奖励（阈值 `correct_threshold=0.5`）：

| Before | After | R_delta |
|--------|-------|---------|
| 错 | 对 | +1.0 |
| 对 | 对 | +0.3（`--mm-delta-keep`） |
| 错 | 错 | 0.0 |
| 对 | 错 | -1.0 |

- 权重：`--mm-w-after 0.7` / `--mm-w-delta 0.3`

**4. MM Smooth 模式**
- `judge=LLMJudge(连续, 自动启用)`, `use_delta=True`, `use_tanh_delta=True`
- `R = w_after · s_after + w_delta · tanh((s_after - s_before) / τ)`
- 连续 judge 分数（两位小数），无硬阈值
- 小噪声被 tanh 阻尼，大变化饱和
- `τ = 0.25`（`--mm-tanh-tau`）：一个 judge 级别的改进 ≈ tanh(0.8) = 0.66

### 2.6 MM 评估

- 函数：`evaluate_mm()`（`rl_eval.py:124`），rank0-only
- 最多评估 20 个样本
- standard 模式：MM 生成 → 解析操作 → 应用到 bank → frozen AA 批量答题 → EM/F1
- typed 模式：对每个 turn 运行 4 个 typed prompt → 解析原子操作 → 合并所有有效操作（union-of-4）→ 应用到 bank → frozen AA 答题

---

## 三、Frozen AA 引擎

### 3.1 模型
- 始终使用 base model（Qwen2.5-7B-Instruct，未训练），符合论文
- MM phase 1：frozen AA = base model
- AA phase 2：不需要 frozen AA（EM 本地计算；llm 模式的 judge 也用 base model）

### 3.2 后端

| 后端 | CLI | 说明 |
|------|-----|------|
| vLLM（默认） | — | `frozen_aa_llm`，批量生成，自动 sleep/wake 管理 |
| HF generate | `--no-vllm` | `frozen_aa_model`，串行生成，仅用于调试 |

### 3.3 GPU 放置模式

| 模式 | CLI 参数 | GPU 布局 | Sleep 行为 |
|------|---------|---------|-----------|
| **Colocated**（4-GPU） | `--aux-gpu` 不设 | 每个 rank 的引擎绑定到自己的 GPU | 调用间 sleep(level=1)，释放显存给训练 |
| **Dedicated Aux**（3-GPU） | `--aux-gpu 2` | 所有 rank 共享 GPU 2 上的一个引擎 | `keep_awake=True`，永不 sleep（无 wake/sleep 延迟） |

### 3.4 环境变量处理
- `load_frozen_aa_vllm()` 在 spawn EngineCore 前清除 torchelastic 环境变量（`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR` 等）
- 原因：EngineCore 子进程继承 `TORCHELASTIC_USE_AGENT_STORE` 会导致 TCPStore 无 host 而 600s 超时
- 清除后恢复原值，保证 DDP 不受影响

---

## 四、LLM Judge 详解

### 4.1 引擎共享
- Judge 与 frozen AA 共享同一个 vLLM 引擎（同一 base model，同一 GPU）
- Judge 只额外发起不同 prompt 的 generate 调用

### 4.2 两种评分模式

| 模式 | 启用方式 | Rubric | 输出约束 | 分值 |
|------|---------|--------|---------|------|
| **离散** | 默认 | 3 级（错误/部分正确/完全正确） | vLLM `choice=["0.0", "0.5", "1.0"]` | {0.0, 0.5, 1.0} |
| **连续** | `--judge-continuous` 或 `--reward-mode mm_smooth` | 5 级（0.00-0.19 / 0.20-0.39 / 0.40-0.59 / 0.60-0.79 / 0.80-1.00） | vLLM `regex="0\.[0-9]{2}|1\.00"` | [0.00, 1.00] 两位小数 |

### 4.3 解析容错
- 优先 `float(text)` 解析
- 失败则正则提取第一个合法数字
- 仍失败则记为 parse failure，分值 0.0
- 监控指标：`judge_parse_failure_rate`, `judge_score_mean`, 分值分布

---

## 五、GRPO 超参数（Paper Appendix D）

| 参数 | 值 | 来源 |
|------|-----|------|
| Group size (G) | 8 | `GRPO_GROUP_SIZE` |
| KL coefficient (β) | 0.01 | `GRPO_KL_COEFF` |
| Learning rate | 1e-6 | `RL_LEARNING_RATE` |
| Max prompt length | 4096 | `MAX_SEQ_LENGTH` |
| Max completion length | 2048 | `MAX_COMPLETION_TOKENS_AA/MM` |
| Generation temperature | 1.0 | `GENERATION_TEMPERATURE` |
| Per-device batch size | 2 | `PER_DEVICE_BATCH_SIZE` |
| Gradient accumulation | 16（4-GPU）/ 32（3-GPU） | `GRADIENT_ACCUMULATION` / CLI |
| Effective batch size | 128 | 2 × 4 × 16 或 2 × 2 × 32 |
| loss_type | `"grpo"` | 显式设置（TRL 1.9.1 默认 `"dapo"`） |
| scale_rewards | `"group"` | group 内归一化 |
| lr_scheduler | `"constant"` | 无 warmup |
| Fine-tuning mode | Full（非 LoRA） | `setup_model_for_grpo()` |

---

## 六、训练变体组合矩阵

### AA 可用组合

| reward-mode | judge-continuous | 说明 |
|-------------|-----------------|------|
| `em` | — | 纯二值 EM（论文基线） |
| `llm` | 否 | 离散 3 级 judge |
| `llm` | 是 | 连续 2 位小数 judge |

### MM 可用组合

| reward-mode | mm-gen-mode | advantage-mode | judge-continuous | 说明 |
|-------------|-------------|----------------|-----------------|------|
| `em` | standard | group | — | 论文基线 |
| `em` | standard | group | — | typed 无 advantage 改进 |
| `em` | typed | group | — | typed 4×2 采样 |
| `em` | typed | twolayer | — | typed + 双层 advantage |
| `llm` | standard/typed | group/twolayer | 否/是 | judge 评分 |
| `mm_delta` | standard/typed | group/twolayer | 否 | after + delta 转移 |
| `mm_smooth` | standard/typed | group/twolayer | 自动 | after + tanh 平滑 delta |

### MM SFT 冷启动可选叠加

- `--mm-sft-epochs N`（N>0 时启用，可与任意上述组合叠加）
- `--mm-sft-lr`（SFT 学习率，默认 1e-5）

---

## 七、源码文件索引

| 文件 | 职责 |
|------|------|
| `src/agents_memory/prompts_r1.py` | AA prompt（Figure 11）、MM standard prompt（Figure 9-10）、MM typed prompt、fact extraction prompt |
| `src/agents_memory/rl_rewards.py` | EM/F1 指标、answer 提取/归一化、MM 操作解析（standard/atomic）、`apply_memory_operations`、`MMRewardComputer`、`AAJudgeReward`、vLLM 批量生成工具 |
| `src/agents_memory/rl_judge.py` | `LLMJudge` 类：离散/连续两种 judge、prompt 模板、vLLM 结构化输出约束、解析容错 |
| `src/agents_memory/rl_grpo.py` | `TypedTwoLevelGRPOTrainer`：typed prompt 扩展（site 1）、n=1 生成、双层 advantage（site 2）、`compute_group_advantages`、`compute_two_level_advantages` |
| `src/agents_memory/rl_eval.py` | `evaluate_aa`（贪心解码 EM/F1）、`evaluate_mm`（MM 生成→应用操作→frozen AA 答题→EM/F1） |
| `src/agents_memory/rl_callbacks.py` | `MemoryR1MetricsCallback`：JSON-lines 指标记录、周期性验证、best checkpoint 保存、周期性 checkpoint |
| `scripts/train_memory_r1_rl_tracked.py` | 训练入口：CLI 解析、数据加载（`load_rl_dataset_aa/mm`）、模型加载、frozen AA 引擎加载、SFT 冷启动、AA/MM 训练函数、GRPO config 构建 |

---

## 八、关键 CLI 参数速查

### 训练阶段
- `--phase aa|mm|both`（required；`both` 不推荐）
- `--base-model`（默认 `$BASE_MODEL` 环境变量）
- `--max-steps 100`
- `--eval-every 10`
- `--checkpoint-every 100`

### Reward
- `--reward-mode em|llm|mm_delta|mm_smooth`（默认 `em`）
- `--judge-continuous`（离散→连续 judge）
- `--mm-w-after 0.7` / `--mm-w-delta 0.3`
- `--mm-correct-threshold 0.5`
- `--mm-delta-keep 0.3`
- `--mm-tanh-tau 0.25`
- `--judge-max-tokens 8`
- `--frozen-aa-max-tokens 512`

### MM 生成 & Advantage
- `--mm-gen-mode standard|typed`（默认 `standard`）
- `--advantage-mode group|twolayer`（默认 `group`）
- `--adv-global-weight 0.5` / `--adv-local-weight 0.5`
- `--mm-sft-epochs 0` / `--mm-sft-lr 1e-5`

### GPU & vLLM
- `--aux-gpu`（不设=colocated 4-GPU；设=dedicated aux 3-GPU）
- `--vllm-util 0.14`（policy vLLM 显存占比）
- `--vllm-aa-util 0.14`（frozen AA vLLM 显存占比）
- `--vllm-sleep`（默认关；开启后 TRL 1.9.1 每步从磁盘重载权重）
- `--no-vllm`（HF fallback，调试用）
- `--optim adamw_torch`（3-GPU 用 `adamw_bnb_8bit`）
- `--use-zero2`（DeepSpeed ZeRO-2 内存兜底）
- `--per-device-batch-size 2` / `--grad-accum 16`
