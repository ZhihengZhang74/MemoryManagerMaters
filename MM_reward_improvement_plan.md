# MM Reward 改进计划：根治「存原话 + 偏向 ADD」

- **状态**：待评审 / 待实施
- **范围**：Memory Manager（MM）的 RL reward 与训练数据流
- **不改动**：Answer Agent（AA）训练逻辑、评测协议（`eval/` 保持不变，作为对齐目标）
- **遵循约定**：reward 一律做成可选开关、不默认启用；不引入格式奖励；实验输出目录 / 日志名体现 reward 类型

---

## 1. 背景

在 `mm_delta` 实验的端到端评测中观察到 MM 建库的两个系统性缺陷：

1. **存原话**：MM ADD 的记忆几乎都是原始对话句的复制（conv-41 平均每条 133 字符），而非凝练的关键信息。
2. **偏向 ADD**：`mm_stats` 显示 `ADD=5001 / UPDATE=57 / DELETE=6`，几乎每个 turn 都 ADD，几乎不 UPDATE/DELETE。最终库膨胀到 ~600 条/对话，本质是「全量对话切片」而非「记忆」。

评测结果对照（同一 MM 建库，只换回答端）：

| 场景 | 记忆库 | AA-llm 回答 | MM 回答 |
|------|--------|-----------|--------|
| baseline（gold memory） | 结构化 | EM 0.168 / F1 0.411 | EM 0.135 / F1 0.393 |
| pipeline（MM 建库） | 原始句库 | EM 0.152 / F1 0.362 | EM 0.157 / F1 0.386 |

gold 数据本身是「精选 ~5% ADD」（train: NONE 94.5% / ADD 4.4% / UPDATE 0.9% / DELETE 0.1%），而 RL 把模型推向了「~100% 全 ADD」。说明 **reward 在对抗 gold 的精选先验**。

---

## 2. 问题诊断：四处 train-eval 不一致

| # | 维度 | MM 训练时 | 评测时 | 后果 |
|---|------|----------|--------|------|
| 1 | **AA 看到的记忆** | 全量 bank，**无 topK**（`rl_rewards.py:396 build_aa_prompt(updated_bank,…)`） | top-60 检索（`pipeline_lib.py:421 retrieve_for_question`） | 训练里「全存」无代价，评测里稀释检索 |
| 2 | **bank 的来源** | gold 操作重放的精选库（`train_memory_r1_rl_tracked.py:182-198`） | 模型从零自建、误差累积 | 模型从不承受自己建库的膨胀后果 |
| 3 | **决策时序** | 单 turn 单步决策，下一样本 bank 重置回 gold | 跨 turn 累积建库 | 模型不为长期库质量买单 |
| 4 | **库规模** | gold 小库（train 全程 ADD 仅 186 条） | 自建大库（~600 条/对话） | 训练分布与评测分布严重错位 |

**结论**：训练目标（全量库 + gold 小库 + 单步）在主动奖励「存原话 + 只 ADD」——这是稳赚策略（AA 开卷易答对、无膨胀代价、无去重奖励）。修复核心是让训练目标与评测协议对齐。

---

## 3. 改进目标与验收指标

**目标**：让 MM 学会「选择性存储 + 凝练表达 + 合理 UPDATE/DELETE」，使自建库更小、更精、可检索，最终提升端到端 pipeline 指标。

**验收指标**（评测后读 `eval/results/<tag>/mm_stats.json` 与 `summary.json`）：

| 指标 | 当前（mm_delta） | 目标 |
|------|----------------|------|
| 平均每对话 `final_bank_size` | ~620 | **< 200**（降 >65%） |
| `UPDATE + DELETE` 总数 | 63 | **> 0 且显著上升**（打破只 ADD） |
| ADD 的文本平均长度 | 133 | **下降**（更凝练） |
| pipeline F1 / LLM-J | 0.362 / 0.331 | **不低于基线，理想超过** |

**训练期监控**（写入 `metrics.jsonl`，经 `last_stats`）：
- 已有：`mm_ops_add/update/delete/none`、`mm_json_parse_failure_rate`、`mm_em_reward_mean`
- 新增：`mm_retrieval_k`、`mm_eff_reward_mean`、`mm_bank_size_mean`

---

## 4. 方案总览与优先级

| 方案 | 内容 | 优先级 | 治哪个不一致 | 改动量 |
|------|------|--------|-------------|--------|
| **A** 检索感知 reward | reward 里 AA 改用 top-K 检索，不用全量 | **P0** | #1 | 中 |
| **B** 效率/库规模惩罚 | reward 里惩罚本 turn 的 ADD 数量 | **P0** | #3/#4 | 小 |
| **E** 自累积建库 | bank_state 用模型自建库替代 gold 重放 | P1 | #2/#3/#4 | 中-大 |
| C 反复制/凝练约束 | 惩罚 ADD 与原 turn 高重叠 | P2（可选） | 存原话 | 中 |
| D 数据增强 UPDATE/DELETE | 构造更多需 UPDATE/DELETE 的场景 | P2（可选） | gold 偏斜 | 大 |

**本轮先落地 P0（A + B），打包为新 reward mode `mm_v2`；跑通见效后再评估 E。**

---

## 5. 详细设计

### 5.1 方案 A：检索感知 reward（P0，治本）

**动机**：把训练 reward 的 AA 输入从「全量库」改为「top-K 检索」，与评测协议对齐。这样「塞一堆原话」会稀释 top-K、降低答对率，模型被迫学会存「凝练且可被检索到的关键信息」，并用 UPDATE/DELETE 去重降噪。

**设计**：
- `MMRewardComputer` 新增可选参数 `embedder` 与 `retrieval_k`（`retrieval_k=0` 表示关闭、走现状全量，保证向后兼容）。
- 对每个 QA：`q_emb = embedder.encode(question)`；对 `updated_bank` 的每条记忆计算 embedding（ADD/UPDATE 的新文本即时 encode 并缓存）；按 **speaker 均分 top-K**（与 `pipeline_lib.retrieve_for_question` 一致）得到 `retrieved`；用 `retrieved` 而非全量去 `build_aa_prompt`。
- `before_prompts`（delta 分支）同样走检索，保持 before/after 协议一致；bank 为空时检索返回全部。
- 检索函数单独抽成 `_retrieve_topk_by_speaker(bank, bank_embs, q_emb, k)`，逻辑镜像 `pipeline_lib.retrieve_for_question`（`per_speaker = max(1, k // n_speakers)`）。

**超参**：`--mm-retrieval-k 20`（评测用 60；训练用更小值加大「精炼压力」，每 speaker 10）。

**伪代码**：
```python
# MMRewardComputer.__call__，updated_bank 计算后：
if self.retrieval_k > 0:
    bank_embs = self._embed_bank(updated_bank)          # (N, D)，ADD/UPDATE 文本即时 encode
    for qa in qa_list:
        q_emb = self.embedder.encode([question])[0]
        retrieved = self._retrieve_topk_by_speaker(updated_bank, bank_embs, q_emb, self.retrieval_k)
        after_prompts.append(build_aa_prompt(retrieved, question, self.tokenizer))
        if self.use_delta:
            before_retr = self._retrieve_topk_by_speaker(bank, bank_embs_before, q_emb, self.retrieval_k)
            before_prompts.append(build_aa_prompt(before_retr, question, self.tokenizer))
else:
    ...  # 现状：build_aa_prompt(updated_bank, ...)
```

**Embedder 复用**：新建 `src/agents_memory/embedding.py`，从 `eval/pipeline_lib.py` 的 `Embedder` 移植 MiniLM mean-pooling 实现（`/home/zhangzhiheng/models/all-MiniLM-L6-v2`），避免训练代码依赖 `eval/`。`train_mm` 在 `--mm-retrieval-k > 0` 时构造该 embedder 并传入 `MMRewardComputer`。

### 5.2 方案 B：效率/库规模惩罚（P0，直接对抗膨胀）

**动机**：方案 A 对「当前 turn QA」的膨胀惩罚有限（逐 turn 短视），B 在每一步直接为「多存」定价。

**设计**：在 reward 里加效率项，惩罚本 completion 的 ADD 条数：
```
R_total = w_after·R_after + w_delta·R_delta + R_eff
R_eff   = − w_eff · n_add_this_turn
```
- `n_add_this_turn` = 该 completion 解析出的 ADD 操作条数（已在 `op_counts` 统计）。
- 不惩罚 UPDATE/DELETE/NONE（它们不增大库）。
- 与 `R_after` 天然制衡：完全不存会让 AA 答错、`R_after` 掉得更多，因此模型会收敛到「存够用的最少信息」。

**超参**：`--mm-eff-weight 0.05`（reward 主项在 0~1，ADD 通常 1~3 条/turn；ADD 1 条扣 0.05、3 条扣 0.15，形成压力但不压倒主项）。`w_eff=0` 表示关闭。

**监控**：`last_stats` 增加 `mm_eff_reward_mean`、`mm_bank_size_mean`（`updated_bank` 平均大小）。

### 5.3 方案 E：自累积建库（P1，修复 gold 库 vs 自建库）

**动机**：训练时 bank 是 gold 重放的，模型从不体验自建库的误差累积与膨胀。要根治需让模型见到「自己风格的、会累积的库」。

**现实约束**：TRL `GRPOTrainer` 的 rollout 是单 completion，不支持跨 turn 的多步环境交互；完全 on-policy 序列决策需自定义 GRPO loop，改动大。

**采用折中（E-lite，离线自举）**：
- 新增脚本 `scripts/gen_self_banks.py`：用当前 MM 策略对训练对话**逐 turn 推理建库**（复用评测的建库逻辑：top-K 检索上下文 + apply 操作），把每个 turn 的「自建库快照」存下来。
- `load_rl_dataset_mm` 增加开关 `--mm-self-bank`：为真时用自举生成的 bank_state 替代 gold 重放的 bank_state。
- 可迭代：训练若干步 → 用新策略重新自举 bank → 继续训练（self-play 式）。

**完全 on-policy 版本**列为 future work，本轮不做。

### 5.4 方案 C：反复制/凝练约束（P2，可选）

惩罚 ADD/UPDATE 文本与当前 turn 原文的高 n-gram 重叠（超过阈值视为「直接复制」则扣分），强制改写凝练。**风险**：可能诱发「无意义改写以降重叠度但丢信息」的 hacking，必须与 `R_after` 强绑定。仅当 A+B 后凝练度仍不足时再上。

### 5.5 方案 D：数据增强 UPDATE/DELETE（P2，可选）

gold 里 UPDATE/DELETE <1%，先验太弱。用 teacher 模型构造更多「矛盾信息→该 DELETE」「重复/细化→该 UPDATE」的训练场景。见效慢，作为 A/B/E 之外的中长期补充。

---

## 6. 代码改动清单

| 文件 | 改动 |
|------|------|
| `src/agents_memory/embedding.py` | **新增**：MiniLM mean-pooling embedder（移植自 `pipeline_lib.Embedder`） |
| `src/agents_memory/rl_rewards.py` | `MMRewardComputer` 增加 `embedder/retrieval_k/w_eff` 参数；实现 `_embed_bank`、`_retrieve_topk_by_speaker`、效率项；`last_stats` 新增诊断字段 |
| `scripts/train_memory_r1_rl_tracked.py` | `train_mm` 接线 embedder 与新参数；新增 CLI（见 §7）；`--reward-mode` 增加 `mm_v2` |
| `scripts/gen_self_banks.py` | **新增**（方案 E-lite，P1 再做） |
| `jobs/train_mm_3gpu.job` / `scripts/run_mm_3gpu.sh` | 透传新 CLI 参数 |

**不改动**：`eval/`（评测协议是对齐目标）、`prompts_r1.py`（prompt 不动，靠 reward 驱动）、AA 训练。

---

## 7. CLI 与实验标识

**新增 CLI（全部可选、默认关闭/向后兼容）**：
```
--mm-retrieval-k   INT   # >0 启用检索感知 reward（建议 20）；0=关闭走全量
--mm-eff-weight    FLOAT # >0 启用效率惩罚（建议 0.05）；0=关闭
--mm-self-bank           # 开关，用自举自建库（E-lite，P1）
```

**reward mode**：新增 `mm_v2` = `mm_delta`（0.7·R_after + 0.3·R_delta）+ 检索感知 + 效率惩罚。`--reward-mode` choices 变为 `{em, llm, mm_delta, mm_v2}`。

**实验标识**（遵循命名规范）：
- 输出目录：`models/memory-r1-rl/memory_manager_rl_mm_v2/`
- 日志：`logs/mm_mm_v2_<时间戳>.log`
- 评测 TAG：`mm_v2_x_llm`（搭配现有 `answer_agent_rl_llm/best`）

---

## 8. 超参与训练配置

沿用现有 3-GPU 布局（`scripts/run_mm_3gpu.sh`：2 DDP + 1 aux）：

| 项 | 值 | 说明 |
|----|----|------|
| reward-mode | `mm_v2` | 新组合 |
| `--mm-w-after` / `--mm-w-delta` | 0.7 / 0.3 | 沿用 mm_delta |
| `--mm-retrieval-k` | 20 | 每 speaker 10 |
| `--mm-eff-weight` | 0.05 | 起步值 |
| max-steps / eval-every | 100 / 10 | 沿用 |
| group / lr / batch | 8 / 1e-6 / 128 | 沿用论文 |
| SFT 冷启动 | 可选（`--mm-sft-epochs`） | 沿用现有开关 |

**消融建议**（资源允许时）：
1. `mm_v2`（A+B 全开）— 主实验
2. 仅 A（`--mm-eff-weight 0`）
3. 仅 B（`--mm-retrieval-k 0`）
4. 基线 `mm_delta`（已有）

---

## 9. 实施顺序与里程碑

1. **M1（P0-A）**：`embedding.py` + `MMRewardComputer` 检索分支；smoke 跑 3 步确认双路生成 + 检索不报错。
2. **M2（P0-B）**：效率项 + 诊断字段；smoke 确认 `mm_eff_reward_mean`、`mm_ops_*` 正常记录。
3. **M3**：`mm_v2` 全量训练（100 步）；看 `metrics.jsonl` 里 `mm_ops_update/delete` 是否上升、`mm_ops_add` 是否下降。
4. **M4**：评测 `mm_v2_x_llm`，对照 §3 验收指标；达标则固化，未达标则调 `retrieval_k`/`eff_weight` 或进入 E。
5. **M5（P1-E，视 M4 结果）**：`gen_self_banks.py` + `--mm-self-bank`，重训对照。

---

## 10. 风险与回退

| 风险 | 缓解 |
|------|------|
| `w_eff` 过大 → MM 变懒（什么都不存） | 监控 `mm_ops_add` 与 `final_bank_size`；掉到 0 附近则降 `w_eff` |
| `retrieval_k` 过小 → 检索漏掉关键记忆、reward 失真 | 从 20 起，必要时上调到 30 |
| 检索引入 embedder 增加显存/耗时 | MiniLM 很小；bank embedding 缓存；必要时降 `vllm-aa-util` |
| reward 幅度变化影响 GRPO 归一化 | 保持 `R_eff` 量级远小于主项；观察 `frac_reward_zero_std` 是否恶化 |
| 新逻辑 bug | 所有新项默认关闭；`mm_delta`/`llm`/`em` 路径完全不受影响，可随时回退 |

---

## 11. 明确不做的事

- **不加格式奖励**（遵循 MM 分层 reward 设计原则）。
- **不改 AA 训练**与评测协议。
- **不动 `MEMORY_MANAGER_PROMPT`**——靠 reward 信号引导行为，而非改指令。
- **不做完全 on-policy 多步建库**（改动过大，列为 future work，用 E-lite 折中）。
