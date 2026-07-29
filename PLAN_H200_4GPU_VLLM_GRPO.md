# Memory-R1 在 4×H200 上的 DDP + vLLM GRPO 训练改造方案

> 目标：把仓库现有的「单卡 / DeepSpeed ZeRO-3 CPU offload / HF generate」生产脚本，
> 改造成「4×H200 纯 DDP + 每卡独立 vLLM」的高吞吐 GRPO 训练，忠实复现
> Memory-R1 (arXiv:2508.19828) 的两阶段 RL（MM → AA）。
>
> 本文档是**实施计划**，不含代码。经确认后再落地。

---

## 0. 已对齐的需求（决策记录）

| 项目 | 决策 | 备注 |
|---|---|---|
| GPU | **4×H200 (141GB/卡)**，计算节点 | 登录节点 `nvidia-smi` 不可见，脚本内不做 GPU 探测硬断言 |
| Conda 环境 | **`r1vllm`** (`/home/zhangzhiheng/miniconda3/envs/r1vllm`) | 直接在该环境安装 vLLM，接受 torch 降级 |
| 基座模型 | **Qwen2.5-7B-Instruct** | 本地 `/home/zhangzhiheng/models/Qwen2.5-7B-Instruct` |
| 微调方式 | **全参微调（无 LoRA）** | 忠实论文；不用 DeepSpeed，改纯 DDP |
| 并行方式 | **DDP，4 rank** | `accelerate launch --num_processes 4` |
| Rollout 推理 | **每卡一个 vLLM**（TRL `vllm_mode="colocate"`） | 权重每步自动同步 |
| MM 的 frozen AA | **每卡再起一个独立 vLLM 实例** | 用户明确要求；显存够 |
| 阶段与顺序 | **MM → AA**（沿用生产脚本顺序） | frozen AA = 未训练基座，符合论文 |
| 步数 | **每阶段 200 步**（脚本默认值） | 落地前先跑 smoke test |
| 网络 | **完全离线** | `HF_HUB_OFFLINE=1`，全部本地路径 |
| Embedding / nltk | **训练链路不需要** | 见 §11 附录说明 |

---

## 1. 现状盘点

### 1.1 环境实测（`r1vllm`）

```
python        3.11.15
torch         2.13.0+cu126
transformers  5.14.1          <-- 注意是 v5 大版本
trl           1.9.1
accelerate    1.14.0
deepspeed     0.19.3
peft          0.19.1
datasets      5.0.0
bitsandbytes  0.50.0
numpy         2.4.6
vllm          未安装           <-- 需要安装
flash-attn    未安装
ray           未安装
```

`trl 1.9.1` 的 vLLM extra 约束：`vllm>=0.17.0,<=0.25.1`。

### 1.2 代码结构（与本次改造相关）

| 文件 | 作用 | 本次是否改动 |
|---|---|---|
| `scripts/train_memory_r1_rl_tracked.py` | **生产 RL 主脚本**，两阶段 GRPO | ✅ 主要改动 |
| `src/agents_memory/rl_rewards.py` | EM/F1、MM 输出解析、`MMRewardComputer` | ✅ 改动（vLLM 打分 + 批量化） |
| `src/agents_memory/rl_eval.py` | `evaluate_aa` / `evaluate_mm` 周期验证 | ✅ 改动（修 bug + vLLM 化） |
| `src/agents_memory/rl_callbacks.py` | `MemoryR1MetricsCallback` 指标 & checkpoint | ✅ 改动（多进程安全） |
| `src/agents_memory/prompts_r1.py` | 论文原始 prompt（Figure 9/10/11） | ⬜ 只读复用 |
| `sagemaker/ds_zero3_offload.json` | ZeRO-3 CPU offload 配置 | ⬜ 保留但不再使用 |
| `jobs/*.job` | SLURM 脚本（单卡、`uv run`） | ⬜ 新写独立启动脚本，不动原文件 |
| `entrypoint.sh` | SageMaker 入口 | ⬜ 不动 |
| `data/r1_training/*.jsonl` | 已备好的 RL 数据 | ⬜ 只读 |

### 1.3 数据现状（已就绪，无需重新生成）

```
data/r1_training/
├── answer_agent_train.jsonl        (ChatML, AA 训练)
├── answer_agent_train_raw.jsonl    {question, answer, category, evidence_refs}
├── answer_agent_val.jsonl / _raw
├── answer_agent_test.jsonl         (1307 条，测试用)
├── memory_manager_train.jsonl      (ChatML, MM 训练)
├── memory_manager_train_raw.jsonl  {turn:{dia_id,speaker}, operations:[...]}
├── memory_manager_val.jsonl / _raw
└── memory_banks/{train,val}_memory_bank.json   [{id,text,speaker,evidence_ref,session_num,timestamp}]
```

关键 schema：memory 条目字段是 **`id` / `text`**（不是 `key` / `content`）——这是 §3 中一个 bug 的根源。

### 1.4 论文超参（Appendix D，脚本内已实现）

```
GRPO_GROUP_SIZE        = 8       (G=8)
GRPO_KL_COEFF          = 0.01    (β=0.01)
RL_LEARNING_RATE       = 1e-6
MAX_COMPLETION_TOKENS  = 2048
GENERATION_TEMPERATURE = 1.0
MAX_SEQ_LENGTH         = 4096
PER_DEVICE_BATCH_SIZE  = 2
GRADIENT_ACCUMULATION  = 16      -> 2 × 4卡 × 16 = 128 effective batch
lr_scheduler_type      = constant, warmup_steps=0
max_steps              = 200
```

---

## 2. 显存预算分析（每卡 141GB）

### 2.1 训练侧（纯 DDP，7.6B 参数）

| 项 | 计算 | 占用 |
|---|---|---|
| 模型权重 bf16 | 7.6B × 2B | 15.2 GB |
| 梯度 bf16 | 7.6B × 2B | 15.2 GB |
| AdamW 状态 fp32 (m + v) | 7.6B × 8B | **60.8 GB** |
| 激活（gradient checkpointing, micro-bs 2, 6144 tok） | — | ~4–8 GB |
| CUDA context / NCCL buffer / 碎片 | — | ~3 GB |
| **训练小计** | | **≈ 98–102 GB** |

### 2.2 推理侧（vLLM，两个引擎）

| 引擎 | 权重 | KV cache | 小计 |
|---|---|---|---|
| Policy colocate vLLM（TRL 管理） | 15.2 GB | 可配 | ~20 GB @ `util=0.14` |
| Frozen AA vLLM（自管，仅 MM 阶段） | 15.2 GB | 可配 | ~20 GB @ `util=0.14` |

### 2.3 结论

- **AA 阶段**（1 个 vLLM）：102 + 20 = **122 GB** → 安全。
- **MM 阶段**（2 个 vLLM）：102 + 20 + 20 = **142 GB** → **超了 141GB，会 OOM**。

### 2.4 应对：三层降压策略（按顺序启用）

**L1 —— vLLM sleep mode（默认开启，首选）**

- Policy 引擎：`vllm_enable_sleep_mode=True`，TRL 在 rollout 结束后自动 `sleep()`，释放 KV cache 与权重显存。
- Frozen AA 引擎：我们自己管理，`enable_sleep_mode=True`，仅在计算 reward 时 `wake_up()`，之后 `sleep()`。
- 效果：两个引擎的常驻显存降到 ~1–2 GB，峰值只在生成阶段出现，而生成阶段梯度/优化器状态尚未产生峰值。**MM 阶段峰值 ≈ 105–115 GB，可行。**

**L2 —— 8-bit 优化器（若 L1 仍紧张）**

- `optim="adamw_bnb_8bit"`（bitsandbytes 0.50.0 已装）。
- AdamW 状态 60.8 GB → **15.2 GB**，直接省 45 GB。
- 代价：与论文的 fp32 AdamW 有微小数值差异；1e-6 的极小学习率下影响可忽略。

**L3 —— DeepSpeed ZeRO-2（若仍不够 / 想留更多 KV cache）**

- 优化器状态 + 梯度跨 4 卡分片：60.8/4 + 15.2/4 = **19 GB**，训练侧降到 ~40 GB。
- 需要 `ds3_gather_for_generation` 相关配合（ZeRO-2 无需 gather，比 ZeRO-3 简单）。
- 代价：偏离「纯 DDP」的要求，但显存最宽裕。

**L4 —— micro batch 降到 1 + `gradient_accumulation=32`**

- 保持 effective batch 128 不变，降激活峰值。

> **实施顺序**：默认 L1；smoke test OOM 就加 L2；再 OOM 就上 L3。
> 每一层都做成命令行开关（`--optim`、`--use-zero2`、`--vllm-util`），不写死。

---

## 3. 代码问题清单（改造中必须一并修复）

以下问题是逐行审阅 `train_memory_r1_rl_tracked.py`、`rl_rewards.py`、`rl_eval.py`、`rl_callbacks.py` 后确认的。

### 3.1 阻塞性问题（不修则跑不通或结果错误）

| # | 位置 | 问题 | 修复方案 |
|---|---|---|---|
| B1 | `GRPOConfig` 未设 `loss_type` | **trl 1.9.1 默认 `loss_type="dapo"`**，不是论文的 GRPO 目标函数 | 显式 `loss_type="grpo"`；`scale_rewards="group"` 保持默认（对应论文 `A=(r-mean)/std`） |
| B2 | `rl_callbacks.py:136,162` | `model.save_pretrained()` **在所有 4 个 rank 上执行**，并发写同一目录 → 文件损坏 / race | 加 `if state.is_world_process_zero:` 守卫；DDP 下用 `trainer.accelerator.unwrap_model()` 或 `state`+`kwargs["model"]` 确认拿到未包装模型 |
| B3 | `rl_callbacks.py` 全部写文件处 | `metrics.jsonl` 4 个 rank 同时 append → 交错乱行 | 只在 rank0 写；或文件名加 rank 后缀（推荐 rank0 单写） |
| B4 | `train_tracked.py:236` `load_frozen_aa` 用 `device_map="auto"` | 4 个进程各自 `auto` 会把 frozen AA 摊到全部 4 卡，互相踩踏 + 显存爆 | 改为 vLLM 实例，`CUDA_VISIBLE_DEVICES` 已由 accelerate 按 rank 隔离；显式 pin 到 `local_rank` |
| B5 | `rl_eval.py:59-62`（`evaluate_aa`） | 数据集 `prompt` 在 `load_rl_dataset_aa:97` **已经套过 chat template**，这里又套一次 → **双重模板**，验证 EM 严重偏低 | 直接用已模板化的 `prompt` 字符串，不再 `apply_chat_template` |
| B6 | `rl_eval.py:148-151`（`evaluate_mm`） | 同上，MM prompt 双重模板 | 同上 |
| B7 | `rl_eval.py:183-184` | 用 `m.get('key')` / `m.get('content')`，但 memory 实际字段是 **`id` / `text`** → 拼出的 memory 全是 `- [?]: `，**MM 验证指标恒为噪声** | 改为 `m.get('id')` / `m.get('text')` |
| B8 | `transformers 5.14.1` | v5 中 `from_pretrained(torch_dtype=...)` 已改名为 `dtype=`（`torch_dtype` 可能仅告警也可能报错） | 统一改用 `dtype=`；落地时先跑一次 import 级验证 |
| B9 | `pyproject.toml` 要求 `python>=3.13`，env 是 3.11 | 无法 `pip install -e .` | 不安装包，依赖脚本已有的 `sys.path.insert(..., "src")`（`train_tracked.py:31`）；启动脚本额外导出 `PYTHONPATH` |

### 3.2 一致性 / 保真度问题

| # | 位置 | 问题 | 修复方案 |
|---|---|---|---|
| C1 | `rl_rewards.py:185-191` vs `rl_eval.py:186-191` vs `prompts_r1.py:ANSWER_AGENT_PROMPT` | **三处 AA prompt 各不相同**，且都不是论文 Figure 11 的原 prompt。训练 reward 和验证指标不可比 | 统一改为从 `prompts_r1.ANSWER_AGENT_PROMPT` 构造，单一函数 `build_aa_prompt(memories, question)` 供 reward / eval / 推理共用 |
| C2 | `train_tracked.py:366` 输出目录 `memory_manager_rl` | `jobs/train_mm_rl.job` 和 `eval_mm_standalone.py` 找的是 `adapter_memory_manager_rl` | 统一为 `memory_manager_rl`，同步修 job / eval 脚本的路径常量 |
| C3 | `train_tracked.py:371-377` | `tokenizer` 被 `setup_model_for_grpo` 的返回值覆盖 | 显式命名区分，避免误读（同模型所以当前无实际 bug） |
| D1 | `rl_rewards.py:170` frozen AA `max_new_tokens=2048` | 答案通常 <20 token，greedy 跑满 2048 极其浪费；这是 MM 阶段慢的主因 | vLLM 化后设 `max_tokens=256` + `stop=["\n\n"]`；纯性能优化，不改 reward 语义（EM 只看第一段答案） |
| D2 | `rl_rewards.py:__call__` | 逐条串行调用 frozen AA（`completions × qa_pairs` 次单条 generate） | **批量化**：把整批所有 (completion, qa) 展平成一个 prompt list，一次 `llm.generate()` 提交给 vLLM，再按索引归约。这是本次最大的加速点（预计 20–50×） |
| D3 | `rl_rewards.py:136-140` ADD 分支 | `op.get("id", str(next_id))` 采用模型给的 id，可能与已有 id 冲突 | ADD 一律忽略模型 id，强制用 `next_id` 递增分配 |
| D4 | `train_tracked.py:281,363` 打印 | AA 打印 "PHASE 1"，MM 也打印 "PHASE 1" | MM=Phase 1，AA=Phase 2，与实际执行顺序对齐 |

### 3.3 有意保留（不改）

- `aa_f1_reward` 不接入 `reward_funcs`（论文 Eq.4 是纯 EM），仅在验证时报告 F1。
- `num_iterations=1`（严格 on-policy），符合论文。
- `lr_scheduler_type="constant"`, `warmup_steps=0`，符合 Appendix D。

---

## 4. 目标架构

### 4.1 AA 阶段（单模型）

```
accelerate launch --num_processes 4
  ├── rank0 (GPU0): [DDP policy 7B] + [colocate vLLM (sleep mode)]
  ├── rank1 (GPU1): 同上
  ├── rank2 (GPU2): 同上
  └── rank3 (GPU3): 同上

每步：
  1. TRL 把最新权重同步进本 rank 的 colocate vLLM
  2. vLLM 生成 G=8 组 completion（temperature=1.0, max_tokens=2048）
  3. vLLM sleep()
  4. aa_em_reward: 纯 CPU 正则 + EM 计算（Paper Eq.4），零 GPU 开销
  5. GRPO loss（loss_type="grpo", β=0.01）→ backward → DDP allreduce → AdamW step
```

### 4.2 MM 阶段（两模型）

```
每 rank (GPU_i) 上驻留三份东西：
  ├── [DDP policy MM 7B]           (训练)
  ├── [colocate vLLM]              (MM rollout, TRL 管理, sleep mode)
  └── [frozen AA vLLM]             (reward 打分, 自管, sleep mode, 权重=基座 Qwen)

每步：
  1. colocate vLLM 生成 G=8 组 MM JSON 操作 → sleep()
  2. MMRewardComputer:
       a. parse_mm_output 解析每条 completion 的 ops
       b. apply_memory_operations 得到更新后的 memory bank
       c. 【批量】把 (bank, question) 展平成 N 个 AA prompt
       d. frozen AA vLLM.wake_up() → 一次 generate(N 个 prompt) → sleep()
       e. 按 (completion, qa) 索引归约 → 每条 completion 的平均 EM = reward
  3. GRPO loss → backward → allreduce → step
```

**关键设计点**：frozen AA 是**冻结基座**，权重永不变 → vLLM 实例可以全程复用，不需要权重同步，只需 sleep/wake 管显存。

### 4.3 为什么不用 vLLM server 模式

- server 模式要独占 GPU（4 卡里拿 1 张出来），训练只剩 3 卡；
- colocate 模式下 TRL 自动处理权重同步（server 模式也支持，但多一层 HTTP 开销和进程管理）；
- 用户显存充裕，colocate 更简单、无跨进程通信开销。

---

## 5. 详细实施步骤

### Step 1 — 环境准备（含回滚点）

1. **备份当前环境包清单**
   `pip freeze > ~/r1vllm_backup_requirements.txt`（回滚依据）
2. **安装 vLLM**
   `pip install "vllm==0.25.1"`
   - 副作用（已 dry-run 确认）：`torch 2.13.0+cu126` → **`torch 2.11.0`**，并拉入 cu13 系列 nvidia wheels（nccl-cu13 2.28.9、cudnn-cu13 9.19 等），同时装 `torchvision 2.11`/`torchaudio`/`flashinfer 0.6.13`/`xgrammar`/`numba` 等约 150 个包。
   - **风险**：若 `r1vllm` 还被其它项目使用，torch 降级会影响它们。→ 已确认可接受。
3. **验证矩阵**（逐条通过才进下一步）
   - `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"` → 期望 `2.11.0 True 4`（**必须在计算节点上跑**）
   - `python -c "import vllm; print(vllm.__version__)"` → `0.25.1`
   - `python -c "import trl, transformers, accelerate; print(trl.__version__, transformers.__version__, accelerate.__version__)"`
   - `python -c "from trl import GRPOConfig; GRPOConfig(output_dir='/tmp/x', use_vllm=True, vllm_mode='colocate')"` → 无异常
   - vLLM 单卡冒烟：用本地 Qwen 路径起一个 `LLM(...)` 生成一句话
4. **可选**：`pip install flash-attn --no-build-isolation`（加速 attention；失败不阻塞，脚本用 sdpa 兜底）

### Step 2 — 离线与路径配置

新建 `scripts/env_h200.sh`（被两个启动脚本 source）：

- `source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1vllm`
- `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`
- `export HF_HOME=/home/zhangzhiheng/.cache/huggingface`
- `export PYTHONPATH=/home/zhangzhiheng/memory-r1/src:$PYTHONPATH`
- `export BASE_MODEL=/home/zhangzhiheng/models/Qwen2.5-7B-Instruct`
- `export TOKENIZERS_PARALLELISM=false`
- `export VLLM_WORKER_MULTIPROC_METHOD=spawn`
- `export NCCL_P2P_DISABLE=0`（H200 有 NVLink，保持开启）
- `export OMP_NUM_THREADS=8`
- 不设 `CUDA_VISIBLE_DEVICES`，交给 accelerate 按 rank 分配

### Step 3 — 改造 `scripts/train_memory_r1_rl_tracked.py`

**3a. 常量与 CLI**
- `DEFAULT_BASE_MODEL` → 读环境变量 `BASE_MODEL`，兜底本地路径
- 删除 `DEEPSPEED_CONFIG` 的默认使用；新增 `--use-zero2` 开关（L3 降压手段）
- 新增 CLI：`--vllm-util`（默认 0.14）、`--vllm-aa-util`（默认 0.14）、`--optim`（默认 `adamw_torch`）、`--no-vllm`（退回 HF generate，调试用）、`--frozen-aa-max-tokens`（默认 256）

**3b. `GRPOConfig` 改造（AA 和 MM 两处）**

新增/修改字段：
```
use_vllm                    = True
vllm_mode                   = "colocate"
vllm_gpu_memory_utilization = args.vllm_util
vllm_enable_sleep_mode      = True
vllm_tensor_parallel_size   = 1          # 每 rank 一个独立引擎
vllm_max_model_length       = MAX_SEQ_LENGTH + MAX_COMPLETION_TOKENS  # 6144
loss_type                   = "grpo"     # 修 B1
scale_rewards               = "group"    # 论文 A=(r-mean)/std
optim                       = args.optim
deepspeed                   = None（或 ZeRO-2 配置，取决于 --use-zero2）
ddp_find_unused_parameters  = False
dataloader_num_workers      = 2
```
保持不变：`num_generations=8`、`beta=0.01`、`learning_rate=1e-6`、`per_device_train_batch_size=2`、`gradient_accumulation_steps=16`、`max_steps`、`bf16=True`、`gradient_checkpointing=True`、`remove_unused_columns=False`、`lr_scheduler_type="constant"`、`warmup_steps=0`、`logging_steps=1`。

**3c. `setup_model_for_grpo`**
- `torch_dtype=` → `dtype=`（修 B8）
- 不设 `device_map`（DDP 下 accelerate 负责放卡）
- 更新 docstring：说明现在是 DDP 而非 ZeRO-3

**3d. `load_frozen_aa` → 重写为 `load_frozen_aa_vllm`**
- 返回一个自管的 `vllm.LLM` 实例（`enable_sleep_mode=True`、`gpu_memory_utilization=args.vllm_aa_util`、`max_model_len=6144`、`dtype="bfloat16"`、`tensor_parallel_size=1`、`enforce_eager=False`）
- 初始化后立即 `sleep(level=2)`，等 reward 阶段再唤醒
- 保留 `--no-vllm` 分支走原来的 HF 路径（便于对照调试）

**3e. `train_mm` / `train_aa`**
- 修 D4（阶段编号打印）
- MM 阶段把 vLLM frozen AA 传给 `MMRewardComputer`
- `eval_kwargs` 传入 vLLM frozen AA
- 结束时用 `trainer.save_model()`（rank0 安全）而非裸 `save_pretrained`
- 新增：训练开始前 rank0 打印一次显存快照，便于事后定位 OOM

**3f. `main`**
- 阶段顺序保持 MM → AA
- `--phase both` 时，AA 阶段前显式释放 MM 阶段的所有 vLLM 引擎与模型（`del` + `torch.cuda.empty_cache()`），否则显存不会回收 → **建议实际操作分两次单独启动，更干净**

### Step 4 — 改造 `src/agents_memory/rl_rewards.py`

1. 新增 `build_aa_prompt(memories, question, tokenizer)`：基于 `prompts_r1.ANSWER_AGENT_PROMPT`，返回**已套 chat template** 的字符串。供 reward / eval / 推理三处共用（修 C1）。
2. `MMRewardComputer` 重写：
   - `__init__(frozen_aa_llm, tokenizer, max_tokens=256, use_vllm=True)`
   - `__call__` 流程改为**三段式**：
     - **展平**：遍历 `(completion, bank, qa_list)`，解析 ops、应用 ops、为每个 qa 构造 prompt，记录 `(completion_idx, gold)` 索引表
     - **批量生成**：`llm.wake_up()` → `llm.generate(all_prompts, SamplingParams(temperature=0, max_tokens=256, stop=["\n\n"]))` → `llm.sleep(level=2)`
     - **归约**：按 `completion_idx` 分组算平均 EM → rewards
   - 空 `qa_list` 的 completion 直接给 0.0（保持现行为）
   - 保留 `use_vllm=False` 的串行 HF 分支
3. `apply_memory_operations` 修 D3（ADD 强制自增 id）
4. 新增可选诊断计数（JSON 解析失败率、ops 类型分布），写进 metrics 便于分析 MM 为何 reward 稀疏

### Step 5 — 改造 `src/agents_memory/rl_eval.py`

1. `evaluate_aa`：
   - 删掉重复的 `apply_chat_template`（修 B5）
   - 改为 vLLM 批量生成（一次提交全部 val prompt，greedy `temperature=0`）
   - **注意**：验证时策略权重在训练中，需要复用 trainer 的 colocate 引擎（TRL 已同步过权重）。若接口不便，退化为 HF `generate` 但**限制 `max_new_tokens=256`** 并加 `--eval-subsample` 上限，避免 81 条 × 2048 token 拖慢训练
2. `evaluate_mm`：
   - 删掉重复模板（修 B6）
   - `m.get('key')/m.get('content')` → `m.get('id')/m.get('text')`（修 B7）
   - AA prompt 改用 `build_aa_prompt`（修 C1）
   - frozen AA 部分改批量 vLLM
   - 保留 `max_eval_samples=20` 子采样
3. 两个函数都加 `if not is_main_process: return {}` 守卫（只在 rank0 做验证），避免 4 卡重复算 + 指标重复写

### Step 6 — 改造 `src/agents_memory/rl_callbacks.py`

1. `on_train_begin` / `on_log` / `on_step_end` / `on_train_end` 全部加 `state.is_world_process_zero` 守卫（修 B2/B3）
2. checkpoint 保存改用传入的 `trainer`/`accelerator` 引用做 `unwrap_model` 后再 `save_pretrained`；或改为触发 `control.should_save = True` 交给 HF Trainer 处理（更稳，推荐）
3. `run_start` 快照补充记录：`loss_type`、`scale_rewards`、`num_generations`、`beta`、`use_vllm`、`vllm_gpu_memory_utilization`、`world_size`、`optim`
4. 新增每步显存峰值记录 `torch.cuda.max_memory_allocated()`，便于调参

### Step 7 — 新建启动脚本

| 文件 | 用途 |
|---|---|
| `scripts/env_h200.sh` | 公共环境变量（见 Step 2） |
| `scripts/run_smoke.sh` | **单卡 + 10 步**冒烟，快速验证 import / vLLM 起得来 / 数据能加载 |
| `scripts/run_mm_4gpu.sh` | MM 阶段，`accelerate launch --num_processes 4 ... --phase mm --max-steps 200` |
| `scripts/run_aa_4gpu.sh` | AA 阶段，`--phase aa --max-steps 200`，`--frozen-aa-path` 不需要 |

启动脚本统一行为：
- `set -euo pipefail`
- `source scripts/env_h200.sh`
- 日志重定向到 `logs/{phase}_$(date +%Y%m%d_%H%M%S).log`
- 训练前打印 `nvidia-smi` 与关键包版本，写入日志头部
- `accelerate launch --num_processes 4 --mixed_precision bf16 --machine_rank 0 --num_machines 1`（不使用 accelerate config 文件，全部走命令行参数，避免隐式配置）

### Step 8 — 同步修复外围路径不一致（C2）

- `jobs/train_mm_rl.job`、`scripts/eval_mm_standalone.py` 中的 `adapter_memory_manager_rl` → `memory_manager_rl`
- `scripts/eval_mm_standalone.py` 的 `load_frozen_aa(...)` 调用签名已过期（传了 `sft_adapter_path`/`use_4bit`），一并修正
- 这一步是低风险清理，可以放在训练跑起来之后做

---

## 6. 验证与放行门（Gate）

按顺序执行，每个 Gate 通过才进下一步。

| Gate | 内容 | 通过标准 |
|---|---|---|
| G0 | 计算节点上 `torch.cuda.device_count()` | `== 4` |
| G1 | vLLM 单卡加载本地 Qwen2.5-7B 并生成 | 有正常输出，无 OOM |
| G2 | `--phase aa --max-steps 2` 单卡 | 跑完 2 步，`metrics.jsonl` 有 `step_metrics` |
| G3 | `--phase aa --max-steps 10` **4 卡** | 无 NCCL hang / 无 OOM；`reward` 字段非全 0 或至少有方差 |
| G4 | 显存检查 | `nvidia-smi` 峰值 < 135 GB/卡 |
| G5 | `--phase mm --max-steps 5` 4 卡 | 两个 vLLM 共存不 OOM；MM reward 能算出非 NaN 值 |
| G6 | 验证路径 | `eval_every` 触发一次 `validation` 记录，`val_em` 数值合理（修 bug 后应显著高于修复前） |
| G7 | checkpoint | rank0 单独产出 `checkpoint-N/`，文件完整（config.json + safetensors + tokenizer） |
| G8 | 正式跑 MM 200 步 | 完成，`final/` 落盘 |
| G9 | 正式跑 AA 200 步 | 完成，`best/` + `final/` 落盘 |

**中途 OOM 的处置顺序**：降 `--vllm-util` → 换 `--optim adamw_bnb_8bit`(L2) → `--use-zero2`(L3) → micro-bs 1 + accum 32(L4)。

---

## 7. 时间与产出预估

### 7.1 单步耗时（粗估，4 卡并行）

- **AA 阶段**：每 optimizer step 需要 128 条 completion（16 prompt × G=8），4 卡各 32 条。
  vLLM 上 32 条 × ≤2048 token ≈ 20–40 s；训练 forward/backward（16 micro-step）≈ 30–50 s。
  → **约 60–100 s/step**，200 步 ≈ **3.5–5.5 小时**。
- **MM 阶段**：额外要跑 frozen AA。每 step 每卡 32 条 completion × 平均 ~2–4 个 qa ≈ 64–128 个 AA prompt，批量 vLLM ≈ 10–20 s（256 token 上限）。
  → **约 90–140 s/step**，200 步 ≈ **5–8 小时**。

> 对照：改造前用 HF `generate` 逐条跑、`max_new_tokens=2048`，MM 单步可能要 10+ 分钟 → 200 步不可行。这是本次改造的核心价值。

### 7.2 产出目录

```
models/memory-r1-rl/
├── memory_manager_rl/
│   ├── metrics.jsonl
│   ├── checkpoint-100/  checkpoint-200/
│   ├── best/            (按 val_em)
│   └── final/
└── adapter_answer_agent_rl/
    ├── metrics.jsonl
    ├── checkpoint-100/  checkpoint-200/
    ├── best/
    └── final/
logs/
├── mm_YYYYmmdd_HHMMSS.log
└── aa_YYYYmmdd_HHMMSS.log
```

注意：全参微调每个 checkpoint ≈ **15 GB**。200 步 + `checkpoint_every=100` → 每阶段约 4 个目录 ≈ **60 GB**，两阶段 **120 GB**。**落地前需确认磁盘余量**，或把 `--checkpoint-every` 提到 200（只留 best + final）。

---

## 8. 风险登记表

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| torch 降级 2.13→2.11 破坏 `r1vllm` 里其它项目 | 中 | 中 | 已备份 pip freeze；必要时 `conda create --clone` 回滚 |
| MM 阶段两 vLLM + 全参训练 OOM | **高** | 高 | sleep mode(L1) + 8bit optim(L2) + ZeRO-2(L3) 三层预案 |
| `transformers 5.x` 与 `trl 1.9.1` API 不兼容（`dtype`/callback 签名/`processing_class`） | 中 | 中 | G2 冒烟提前暴露；必要时 pin `transformers` 到 4.56–4.57 |
| colocate vLLM 与 DDP 的 NCCL 通信抢端口 / hang | 中 | 高 | 设 `vllm_group_port` 唯一、`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`enforce_eager` 备选 |
| MM reward 恒为 0（论文级难点，非 bug） | **高** | 中 | 修 B7/C1 后应改善；额外记录 JSON 解析成功率、ops 分布做归因；必要时讨论加 format reward（**会偏离论文，需另行确认**） |
| 磁盘不足（120GB checkpoint） | 中 | 中 | Step 7 前 `df -h` 检查；调大 `--checkpoint-every` |
| 数据量小导致过拟合（AA train 仅 152 条，200 步 ≈ 21 epoch） | 中 | 中 | 依赖 `best/`（按 val_em）选点，而非 `final/`；metrics 曲线监控 |
| vLLM 0.25.1 与 Qwen2.5 chat template 处理差异 | 低 | 中 | 我们传入的是**已模板化的纯字符串**，走 `LLM.generate` 而非 `chat`，规避模板二次处理 |

---

## 9. 明确不做的事（Out of Scope）

- ❌ 不重新生成 RL 数据（`prepare_r1_data.py` 需要 OpenAI API + 联网，且数据已就绪）
- ❌ 不改论文超参（G=8、β=0.01、lr=1e-6、batch 128、200 步 保持原样）
- ❌ 不引入 LoRA / QLoRA
- ❌ 不改 `scripts/train_memory_r1_rl.py`（教学版，非生产路径）
- ❌ 不改 `entrypoint.sh` / `sagemaker/` / `databricks.yml`
- ❌ 不加 format reward / length penalty 等论文没有的 reward 项（除非另行确认）
- ❌ 不做最终 test set（1307 条）评测——那是训练完成后的独立任务

---

## 10. 改动文件清单汇总

### 修改
| 文件 | 改动摘要 |
|---|---|
| `scripts/train_memory_r1_rl_tracked.py` | 去 ZeRO-3 → DDP；接入 colocate vLLM；frozen AA 改 vLLM；修 B1/B4/B8/C2/C3/D4；新增 CLI 开关 |
| `src/agents_memory/rl_rewards.py` | 新增 `build_aa_prompt`；`MMRewardComputer` 批量 vLLM 化；修 C1/D1/D2/D3 |
| `src/agents_memory/rl_eval.py` | 修 B5/B6/B7/C1；批量化；rank0 守卫 |
| `src/agents_memory/rl_callbacks.py` | 修 B2/B3；补充 config 快照与显存记录 |
| `jobs/train_mm_rl.job`（可选，低优先） | 路径 `adapter_memory_manager_rl` → `memory_manager_rl` |
| `scripts/eval_mm_standalone.py`（可选，低优先） | 修路径 + `load_frozen_aa` 过期签名 |

### 新增
| 文件 | 用途 |
|---|---|
| `scripts/env_h200.sh` | 公共环境变量 / conda 激活 / 离线开关 |
| `scripts/run_smoke.sh` | 单卡 10 步冒烟 |
| `scripts/run_mm_4gpu.sh` | MM 阶段 4 卡启动 |
| `scripts/run_aa_4gpu.sh` | AA 阶段 4 卡启动 |
| `configs/ds_zero2.json`（仅当启用 L3） | ZeRO-2 降压配置 |

### 不动
`src/agents_memory/prompts_r1.py`、`data/**`、`entrypoint.sh`、`sagemaker/**`、`databricks.yml`、`scripts/train_memory_r1_rl.py`、`scripts/prepare_r1_data*.py`、`scripts/benchmark_*.py`

---

## 11. 附录

### 11.1 关于 embedding 模型与 nltk（结论：本次不需要）

已全仓库检索确认：

- **nltk**：整个仓库 **零引用**。`scripts/eval_sft.py:33` 的 BLEU-1 是手写实现（纯 Python，无外部依赖）。
- **embedding 模型**：只有 `scripts/prepare_r1_data*.py` 用到 `text-embedding-3-small`，且是 **OpenAI HTTP API**，不是本地模型。该脚本的产物（`data/r1_training/**`）已经在仓库里，无需重跑。
- **sentence-transformers / rouge**：零引用。

因此 **RL 训练链路（MM + AA）只需要本地的 Qwen2.5-7B-Instruct**，无需从 ModelScope 下载任何额外模型。

**如果**后续要做以下事情，才需要额外下载（届时另开任务）：
| 任务 | 需要的资源 | 获取方式 |
|---|---|---|
| 重新生成 RL 数据 | OpenAI API（`gpt-4o-mini` + `text-embedding-3-small`） | 需联网/代理，或替换为本地方案 |
| `benchmark_mem0.py` / `benchmark_locomo.py` | OpenAI API（`gpt-4.1` / `gpt-5.2` judge） | 同上 |
| 若要把上述 embedding 换成本地模型 | 如 `BAAI/bge-m3` 或 `Qwen3-Embedding` | `modelscope download --model <id> --local_dir ~/models/<name>` |

### 11.2 关键行号索引（便于实施时定位）

```
scripts/train_memory_r1_rl_tracked.py
  :31          sys.path.insert -> src/         (保留，B9 依赖)
  :49          DEFAULT_BASE_MODEL              (改为读 BASE_MODEL 环境变量)
  :52-60       论文超参常量                     (不动)
  :63          DEEPSPEED_CONFIG                (改为可选)
  :81-111      load_rl_dataset_aa              (prompt 已套模板 -> B5 根因)
  :114-180     load_rl_dataset_mm              (prompt 已套模板 -> B6 根因)
  :196-218     setup_model_for_grpo            (dtype=, 去 ZeRO-3 注释)
  :221-249     load_frozen_aa                  (重写为 vLLM, 修 B4)
  :256-262     aa_em_reward                    (不动, 论文 Eq.4)
  :303-322     AA GRPOConfig                   (加 vLLM + loss_type)
  :366         MM output_dir                   (C2)
  :392-411     MM GRPOConfig                   (加 vLLM + loss_type)
  :413-418     MMRewardComputer 构造            (改传 vLLM 引擎)
  :508-514     main 阶段顺序 MM -> AA           (不动)

src/agents_memory/rl_rewards.py
  :127-152     apply_memory_operations         (D3)
  :159-250     MMRewardComputer                (批量 vLLM 化, C1/D1/D2)
  :180-206     _run_frozen_aa                  (ad-hoc prompt -> C1)

src/agents_memory/rl_eval.py
  :59-62       evaluate_aa 双重模板             (B5)
  :148-151     evaluate_mm 双重模板             (B6)
  :183-184     key/content 字段错误             (B7)
  :186-191     第三套 AA prompt                 (C1)

src/agents_memory/rl_callbacks.py
  :49-53       _write_record                   (B3)
  :132-139     best checkpoint 保存             (B2)
  :152-165     周期 checkpoint 保存             (B2)
```

### 11.3 待确认事项（实施前请回复）

1. **磁盘余量**：两阶段 checkpoint 约 120 GB，`/home/zhangzhiheng` 是否够？还是把 `--checkpoint-every` 设为 200（只留 best + final，约 60 GB）？
2. **降压策略默认值**：是先按「纯 DDP + sleep mode」跑（最忠实，但 MM 阶段有 OOM 风险），还是**默认直接开 8-bit 优化器**（稳妥，数值差异极小）？
3. **`--phase both` vs 分两次跑**：建议分两次单独启动（显存干净、失败可单独重跑）。是否同意？
4. **MM reward 若仍接近 0**：是否允许加一个「JSON 格式合法性」的辅助 reward（会偏离论文原设定，但能提供梯度信号）？还是严格保持论文原样、如实记录负结果？
5. **是否需要在 SLURM 下提交**（写 `.job` 文件），还是直接在计算节点交互式 `bash scripts/run_mm_4gpu.sh`？

---

*文档生成于 2026-07-29。基于 commit `04eefea`。*
