## 4. 实验方案

本节把整套实验的执行路径写清楚，包括每一步的命令、产出文件、以及几个会影响结果解读的设计选择。后续各节的分析全部由这里产出的中间文件重算，不需要重新训练。

### 4.1 整体流程

```
① 构造数据集      cli.py data      → data/{tag}/{dataset}/train.jsonl
② LoRA 微调       cli.py train     → runs/{tag}/{dataset}/metrics/*.jsonl
③ 汇总逐样本指标  cli.py analyze --kind features
                                   → results/{tag}/per_sample_metrics.csv
④ 各项分析        cli.py analyze --kind {unsupervised,cross_type,...}
                                   → results/{tag}/{kind}.csv
⑤ 闭环清洗        cli.py clean     → data/{tag}/cleaning_loop/{name}/train_{targeted,random}.jsonl
⑥ 清洗后重训+评测 cli.py train / evaluate（回到步骤②）
```

步骤 ①-④ 是"能不能检测"，⑤-⑥ 是"清洗是否真的有用"。**只有 ⑤-⑥ 会实际动训练数据**，前面全是在既有轨迹上重算，因此改分析方法的成本极低，改数据或训练配置的成本是一整轮 GPU 时间。

### 4.2 步骤①：数据集构造与噪音注入

```bash
python3 cli.py data --source dolly --tag ratio10 --ratio 0.10 \
  --datasets clean,garbled,template,duplicate,unrelated,truncation,near_duplicate,keyword,mixed
```

以 dolly-15k 为基座，先切出 400 条留出集（`data/{tag}/heldout.jsonl`，9 个数据集共用同一份；`ref_samples=200` 条做参考梯度、`heldout_samples=200` 条做 held-out loss 监控），剩余 14,611 条作为训练集。噪音注入逻辑在 `data.py::apply`：用 `np.random.default_rng(seed=42)` 抽取 `int(len(rows) * ratio)` 条样本索引，对命中的样本做对应变换，未命中的原样保留。

两个细节会影响后续读数：

**`duplicate` 是唯一改变数据集行数的类型。** 其余 6 类都是原地替换（`[fn(r) if i in ids else r for ...]`），行数不变；`duplicate` 走的是 `rows + [_duplicate_copy(...)]`，即追加副本，因此 `duplicate` 数据集有 16,072 行而非 14,611 行，噪音占比也因此是 1461/16072 ≈ 9.1% 而非 10%。

**`mixed` 的每类实际数量有随机波动。** `data.py:175` 对 7 个子类型各按 `ratio/len(types)` = 0.10/7 ≈ 1.43% 独立注入，每次用 `seed + offset` 换种子。理论每类应为 14611 × 0.0143 ≈ 209 条，实测 191-211 条（near_duplicate 211、template 206、truncation 204、keyword 197、unrelated 197、garbled 194、duplicate 191），偏差来自独立抽样可能命中同一条样本时后一次覆盖前一次。总噪音 1400 条，占 14819 行的 9.4%。第 6.12 节的逐类型切片就建立在这批标签上。

9 个数据集：`clean`（不注入，作为下游对比基线）、7 个单类型、`mixed`。两个噪音比例 `ratio10`（10%）与 `ratio5`（5%），后者用于验证结论不是特定比例下的偶然（第 6.4 节）。

### 4.3 步骤②：LoRA 微调与逐样本指标采集

```bash
python3 cli.py train --tag ratio10 --dataset garbled --model hf-lora
```

Qwen2.5-3B-Instruct + LoRA（r=32, alpha=64, dropout=0.05），5 epochs，`micro_batch=1` + `grad_accum=16`（等效 batch 16），lr=2e-4，`max_len=1024`。单卡 NVIDIA RTX PRO 6000 Blackwell（~98GB），9 个数据集串行排队，每个约 1.5 小时（第 8.5 节有实测分解）。

`micro_batch=1` 不是性能选择，而是**逐样本梯度追踪的前提**——只有一个样本单独构成一个前向-反向时，才能把梯度范数、与参考方向的余弦相似度归属到这个样本。这是整套方法的数据基础，也是它比常规微调慢的原因。

每个样本每个 epoch 落盘一条记录（`runs/{tag}/{dataset}/metrics/per_sample.jsonl`），字段包括：

| 字段 | 含义 |
|---|---|
| `loss` | 该样本本 epoch 的训练 loss |
| `grad_norm` | 该样本梯度的 L2 范数 |
| `cos_sim_ref` | 该样本梯度与"参考梯度"（200 条留出样本的平均梯度方向）的余弦相似度 |
| `cos_sim_global` | 该样本梯度与同一优化器 step 内其他样本平均梯度的余弦相似度 |
| `update_contrib` | 该样本对本次参数更新的贡献占比 |

另有两类**子采样**产出：`diag_epoch*.jsonl`（诊断层聚合量）和 `token_diag_epoch*.jsonl`（每样本 loss 最高的 top-32 token 的 `[位置, token_id, loss]`）。这两类按 `diag_subsample=8` 采集，即每 8 条样本取 1 条做纯前向诊断推理，**覆盖率约 12.5%**。

这个覆盖率差异贯穿全报告，是读数时最容易踩的坑：**37 个特征里有 13 个只有 12.5% 覆盖率**，而 `dropna` 要求全部非空，所以任何用全部特征的分析实际只跑在约 900-1200 行上。第 6.10 节开头专门解释了这个口径差异，第 6.12 节则同时报 `full_diag`（37 特征、约 919 行）和 `full_coverage`（19 特征、全量 14819 行）两个口径。

### 4.4 步骤③：汇总逐样本指标表

```bash
python3 cli.py analyze --tag ratio10 --kind features
```

`analyze.py::build_table` 把四个来源拼成一张宽表 `results/{tag}/per_sample_metrics.csv`：

1. **轨迹聚合**（`_load_run_metrics`）：把 `per_sample.jsonl` 按 `sample_id × epoch` 透视，对 loss/grad_norm/cos_ref/cos_global 四组各算 `_mean`/`_last`/`_std`/`_slope`（末值减首值）。loss 额外算 `loss_min`、`converge_epoch`（首次 loss<2.0 的 epoch，从未达到则记为 5）、`loss_curvature`（对 epoch 做二次多项式拟合的常数项）、`loss_rank`（各 epoch 内百分位排名的均值）。
2. **诊断层聚合**（`diag_epoch*.jsonl` 按样本取均值）。
3. **token 级统计**（`_load_token_features`）：从 top-32 难 token 三元组算 `hard_loss_mean`/`hard_loss_max`/`hard_pos_peak`/`hard_id_uniq`/`hard_pos_jaccard` 等。
4. **静态文本特征**（`textsim.text_nn_sim`）：不依赖训练，直接从 `train.jsonl` 算每条样本与其最近邻的文本相似度。这是唯一一个 100% 覆盖且不需要 GPU 的特征，第 6.11 节的消融和第 6.12 节的免标定基线都用它。

真实标签 `noise_type` 从 `train.jsonl` 读进来**只用于评估**，任何打分器都不使用它——这是"免标签"的准确含义：标签存在于评估侧，不存在于检测侧。

### 4.5 步骤④：各项分析的口径

| 分析 | 命令 `--kind` | 做法要点 |
|---|---|---|
| 域内检测难度（第 6.2 节） | `unsupervised` | 每个数据集**独立**拟合 IsolationForest 与 MAD z-score，不跨数据集合并（否则 garbled 的极端量级会抬高 template 的离群基线） |
| 跨类型迁移（第 6.3 节） | `cross_type` | 源域训 LR + RF，`StandardScaler` 只在源域 `fit`，目标域只 `transform`；取两者 AUC 较高值。对角线复用 5-fold CV 结果，保证与非对角线同尺 |
| 跨比例迁移（第 6.4 节） | `cross_ratio` | ratio10 ↔ ratio5 双向 |
| 清洗精度 lift（第 6.5 节） | `precision_lift` | P@10% ÷ 随机基线 |
| 方向反转（第 6.6 节） | `memorization` | 固定符号规则，**从不按数据集重拟合方向** |
| 早期检测（第 6.7 节） | `early_unsupervised` / `early_memorization` | `build_table(max_epoch=k)` 截断轨迹，模拟"只训了 k 个 epoch"，无需真的提前停止 |
| 特征归因（第 6.8 节） | `feature_attribution` | permutation importance，`n_repeats=20` |

**跨类型迁移与 `mixed` 的关系**：`cross_type` 在代码层面显式跳过 `mixed`（`if ds in ('clean','mixed'): continue`），所以它回答不了"混合流"的问题。第 6.12 节用一个独立脚本 `scripts/transfer_to_mixed.py` 补上这一环。

### 4.6 步骤⑤-⑥：闭环清洗与重训

```bash
python3 cli.py clean --tag ratio10 --dataset template --method memo_signed --budget 0.10
python3 cli.py train    --tag ratio10 --dataset cleaning_loop_targeted_template_signed \
  --train-file data/ratio10/cleaning_loop/template_memo_signed/train_targeted.jsonl --model hf-lora
python3 cli.py evaluate --tag ratio10 --dataset cleaning_loop_targeted_template_signed --model hf-lora
```

`cleaning_loop.py::build` 的关键设计：

**只用全覆盖特征。** 打分对象是**全量训练集**而非诊断子样本——否则真实噪音大部分根本不在候选范围内，剔除就没有意义。代价是只能用在该数据集上取值全部非空的 19 个特征，排除全部 token 级诊断和 `cos_global_*`。这就是第 6.10 节所说的"报告 AUC 与生产精度不总是一致"的来源。

**三个打分器，对应三种噪音假设**（详见第 6.6 节和 6.12.5 节）：`iforest`（噪音 = 离群，无方向）、`memo_signed`（噪音 = 异常地容易学，固定负号）、`pooled`（三条腿标准化后取逐样本最大值，用于成分未知的场景）。

**必须有等量随机剔除对照。** 剔除 10% 样本本身就会减少 10% 训练数据，如果只跟"未清洗基线"比，就无法区分"去噪收益"和"数据量损失"。所以每次 `clean` 同时产出 `train_targeted.jsonl`（按分数剔除）和 `train_random.jsonl`（同一 `n_drop`、同种子随机剔除），重训时两个都跑。第 6.13 节的四方对比（干净基线 / 未清洗 / 定向剔除 / 随机剔除）就是这么来的。

**剔除精度是事后统计，不参与打分。** `targeted_precision` 是剔除集合里真实 `noise_type != 'none'` 的占比，写进 `metadata.json` 供分析，打分过程从未看到它。

### 4.7 已知的方案性局限

以下几点是设计层面的，不是执行偏差，读结论时需要一并考虑：

- **子采样覆盖率 12.5%** 让 token 级特征在全量场景下不可用，而第 6.11 节的消融显示这批特征恰好是 near_duplicate 唯一有效的信号来源——这类噪音的报告 AUC 比生产可达值偏乐观。
- **单一基座模型与单一规模**（Qwen2.5-3B + LoRA r=32，5 epochs）。噪音的下游危害本身很小（第 6.9 节：7 项 benchmark 平均值集中在 0.42-0.44），不能排除更大模型或更长训练会放大差异。
- **7 类噪音都是程序化注入的**，扰动方式已知且一致。第 6.12 节的"未见过的类型"仍属这 7 类之一（只是对检测器池未见过），真正的野生噪音不在验证范围内。
- **闭环重训只做了 garbled 与 template 两类**，`mixed` 的闭环尚未跑（第 7.3 节列为下一步）。
---
