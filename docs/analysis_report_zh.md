# 噪音样本对 LLM 微调的影响: 逐样本指标追踪与检测分析报告

> 实验日期: 2026-08-12 ~ 2026-08-14
> 基座模型: Qwen2.5-3B-Instruct (LoRA r=32) · 训练数据: databricks-dolly-15k
> 噪音比例: 10% · 5 epochs · 每 run 14611 训练样本

---

## 1. 实验设计

### 1.1 数据集 (6 个, 样本顺序一致, 固定 seed)

| 数据集 | 噪音构造 | 训练行数 |
|---|---|---|
| `clean` | 原始数据 (基准线) | 14611 |
| `garbled` | 10% 样本注入乱码字符 (混合 Unicode 替换/插入/字符交换) | 14611 |
| `duplicate` | 追加 10% 完全重复的副本行 | 16072 |
| `unrelated` | 10% 样本的 instruction 与 response 来自不同类别 (response 本身语义正确、通顺) | 14611 |
| `keyword` | 10% 样本仅替换关键实体/数字/专有名词, 语法保持通顺 | 14611 |
| `mixed` | 上述 4 类各 2.5%, 共 10% | 14976 |

另有 400 条干净保留样本 (`heldout.jsonl`) 用于: 训练前计算参考梯度方向 (LESS 式影响力基准) + 训练中 held-out 泛化损失评估。每个样本带 `noise_label` / `noise_type` / `category` 标签。

### 1.2 训练配置

- 微批大小 1 + 梯度累积 16 → 每样本梯度可精确捕获
- lr 2e-4, cosine 衰减 + 3% warmup, AdamW, bf16 + flash-attention
- 每 run 5 epochs (4570~5025 优化器步), RTX 5090 单卡, 每 run 3.3~3.9 小时

### 1.3 记录的指标 (3 个层级)

**样本级 (每样本 × 每 epoch, 共 ~73K 行/run):**
`loss`、`grad_norm` (LoRA 梯度 L2 范数)、`cos_sim_ref` (与干净参考方向的余弦相似度)、`cos_sim_global` (与累积窗口梯度方向的余弦)、`update_contrib` (B 参数上样本梯度相对 Adam-RMS 的比值)、`tokens`

**诊断级 (每 epoch 末, 每 8 个样本取 1, ~1827 条/run):**
`max_token_loss`、`frac_hard` (loss>4 的 token 占比)、`user_loss` (prompt 部分平均损失)、`entropy` (next-token 熵)、`token_loss_skew/kurt` (逐 token 损失分布形状)、top-k 最难 token 明细 (位置/token id/损失)

**派生特征 (分析期):**
`loss_std`、`loss_slope`、`converge_epoch`、`loss_rank`、`loss_curvature`、`grad_norm_cv`、`cos_ref_trend`、`text_nn_sim` (TF-IDF 最近邻相似度)

**token 级 (离线, 每个数据集 60 噪音 + 60 正常样本):** 对每样本 top-24 最难 label token 逐一反向传播, 得到逐 token 的精确 LoRA 梯度范数与余弦相似度。

---

## 2. 训练动态: 噪音如何影响训练过程

### 2.1 训练集 loss 轨迹 (各 epoch 均值)

| run | epoch 0 | epoch 1 | epoch 2 | epoch 3 | epoch 4 |
|---|---|---|---|---|---|
| clean | 1.366 | 1.127 | 0.861 | 0.642 | 0.514 |
| garbled | **1.669** | 1.386 | 1.093 | 0.848 | 0.702 |
| unrelated | 1.494 | 1.248 | 0.896 | 0.641 | 0.498 |
| keyword | 1.427 | 1.164 | 0.894 | 0.665 | 0.533 |
| mixed | 1.496 | 1.207 | 0.904 | 0.662 | 0.525 |
| duplicate | 1.349 | 1.077 | 0.794 | 0.557 | **0.425** |

**发现:**
1. **garbled 全程损失最高** (epoch 0 高出 clean 22%, epoch 4 仍高 37%) — 乱码样本无法被模型"学会", 持续抬升整体损失;
2. **duplicate 收敛到最低** (0.425, 比 clean 低 17%) — 重复样本被快速记忆, 反而拉低训练损失 (过拟合信号);
3. unrelated / keyword / mixed 的轨迹与 clean 接近, 说明这些噪音在损失均值层面伪装得很好。

### 2.2 Held-out 干净样本损失 (泛化损伤, 每 200 步评估)

| run | 初始 (step 200) | 最终 | 增幅 |
|---|---|---|---|
| clean | 1.628 | 2.051 | +0.423 |
| keyword | 1.627 | **2.044** | **+0.417 (最小)** |
| garbled | 1.629 | 2.059 | +0.430 |
| mixed | 1.624 | 2.081 | +0.457 |
| unrelated | 1.626 | 2.090 | +0.465 |
| duplicate | 1.626 | **2.143** | **+0.517 (最大)** |

**发现:** 所有 run (含 clean) 的 held-out 损失都随训练上升 — dolly-15k 上 5 epoch 的 LoRA 微调本身就在过拟合。**duplicate 的过拟合最严重** (重复样本强化记忆、损害泛化), **keyword 反而最轻**; 这与验证集结果互相印证 (见第 5 节)。

---

## 3. 样本级噪音检测

### 3.1 单指标 AUC (噪音 vs 同 run 正常样本)

| 噪音类型 | 最强指标 | AUC | 次强指标 |
|---|---|---|---|
| garbled | loss_curvature | **0.985** | user_loss 0.979 / entropy 0.971 |
| duplicate | text_nn_sim | **0.939** | cos_global_mean 0.610 |
| unrelated | loss_curvature | **0.830** | loss_std 0.827 / grad_norm_mean 0.764 |
| mixed | text_nn_sim | **0.716** | loss_std 0.695 / loss_curvature 0.691 |
| keyword | loss_curvature | **0.669** | loss_std 0.649 / grad_norm_mean 0.639 |

### 3.2 多指标分类器 (LR / 随机森林, 19 维特征, 70/30 划分)

| 噪音类型 | LR AUC | RF AUC | 准确率 | 结论 |
|---|---|---|---|---|
| garbled | **0.9996** | 0.9996 | 99.3% | 近乎完美可分 |
| duplicate | 0.974 | 0.973 | 95.3% | 强可分 |
| unrelated | 0.923 | 0.887 | 94.1% | 强可分 |
| mixed | 0.850 | 0.827 | 92.1% | 中等可分 |
| keyword | **0.531** | 0.551 | (全判正常) | **不可分** |

### 3.3 各类噪音的"指标指纹"

- **garbled**: 提示词与回复都被污染 → `user_loss` 与 `entropy` 极高、loss 轨迹曲率异常 (学不动) → 检测最可靠;
- **duplicate**: 文本重复是本质 → `text_nn_sim ≈ 1.0` 一击命中; 训练侧特征 (loss 低、被记忆) 反而与"难样本"方向相反;
- **unrelated**: 整段回复与上下文错配 → `frac_hard` / `loss_slope` 偏高 (全程高损失但伪装成正常难样本);
- **keyword**: 只改几个实体词, 文本与语义基本完整 → 所有训练侧指标接近正常; **样本级检测的盲区**, 单靠训练动态无法分离;
- **mixed**: 四类混合后各特征互相稀释, 但 text_nn_sim 仍捕获其中的 duplicate 子集。

### 3.4 跨任务类型迁移性 (按 dolly 的 8 个 category 分层)

| category | RF AUC |
|---|---|
| closed_qa | 0.987 |
| creative_writing | 0.979 |
| information_extraction | 0.977 |
| brainstorming | 0.977 |
| general_qa | 0.943 |
| summarization | 0.942 |
| open_qa | 0.919 |
| classification | **0.870 (最难)** |

**发现:** 检测方法在全部 8 种任务类型上有效 (AUC 0.87~0.99), 其中 **classification 类最难** — 短结构化回复使 token 级信号 (entropy / user_loss) 变弱。garbled 检测在所有类别均接近 1.0, 是最普适的检测目标。

---

## 4. Token 级检测 (精确逐 token 梯度归因)

对每个数据集抽样 60 噪音 + 60 正常样本, 对每样本 top-24 最难 label token 逐一 `autograd.grad` 得到逐 token 梯度, 提取 `hard_loss_mean` / `hard_gradnorm_mean` / `hard_cos_ref_mean` / `pos_std` 等特征:

| 特征 | garbled | duplicate | unrelated | keyword |
|---|---|---|---|---|
| hard_loss_mean | **0.767** | 0.414 | 0.582 | 0.486 |
| hard_gradnorm_mean | **0.767** | 0.414 | 0.601 | 0.502 |
| hard_cos_ref_mean | 0.624 | 0.588 | 0.553 | 0.478 |
| pos_std | 0.571 | 0.461 | 0.533 | 0.411 |

**发现:**
1. **garbled 在 token 级仍最强可分** (0.77) — 乱码位置产生局部极端损失的 token;
2. **duplicate 的 token 级 AUC 低于 0.5** — 重复样本的 token 被完美记忆 (低损失), 与正常样本不可分; 它的可检测性完全来自**数据侧**特征 (文本相似度), 而非训练动态;
3. 整体而言 token 级 AUC 低于样本级 — 单个 hard token 的信号噪声比有限, 样本级聚合 (跨 token 与跨 epoch) 更稳。

**已知局限:** 乱码定位验证 (`loc_mismatch_frac`) 结果为 0 — 位置对齐法在字符级污染改变 tokenization 边界时失效, 需要序列对齐算法 (如编辑距离对齐) 才能正确定位被污染的 token, 留作后续工作。

---

## 5. 噪音对模型最终能力的影响 (7 模型 × 7 验证集)

### 5.1 总体对比

| 模型 | MMLU | GSM8K | HellaSwag | ARC | BBH | TruthfulQA | Winogrande |
|---|---|---|---|---|---|---|---|
| clean | 0.6295 | 0.5413 | 0.2715 | 0.7995 | 0.0741 | 0.1922 | 0.5383 |
| garbled | 0.6354 | 0.5269 | 0.2664 | 0.8080 | 0.0944 | 0.1873 | 0.5359 |
| duplicate | 0.6309 | 0.5125 | 0.2732 | 0.7918 | 0.0778 | 0.1983 | 0.5525 |
| unrelated | 0.6241 | 0.4981 | 0.2705 | 0.7901 | 0.0833 | 0.1824 | 0.5335 |
| keyword | 0.6333 | 0.5231 | 0.2750 | 0.7986 | 0.0759 | 0.1848 | 0.5241 |
| mixed | 0.6315 | **0.5732** | 0.2673 | 0.7952 | 0.0907 | 0.1836 | 0.5375 |
| **base (未微调)** | **0.6637** | **0.7460** | 0.2745 | **0.8311** | 0.0611 | **0.1934** | **0.5856** |

### 5.2 核心发现

1. **噪音的伤害远小于微调本身的伤害**: 6 个微调模型的成绩互相接近 (MMLU 0.624~0.635), 而**基座模型在 4/7 个验证集上全面优于所有微调模型** (尤其 GSM8K: 0.746 vs ~0.52, ARC: 0.831 vs ~0.79)。dolly-15k 上的 SFT 损害了通用能力, 且这一损伤淹没了 10% 噪音的差异;
2. **unrelated 整体最差** (MMLU -0.005、GSM8K -0.043、ARC -0.009 vs clean) — 语义通顺但上下文错配的噪音对模型的误导最深;
3. **duplicate 在 Winogrande/TruthfulQA 上反而略高** — 重复带来的记忆效应对少数任务有微弱正收益;
4. **BBH 例外**: 微调模型全部优于基座 (0.074~0.094 vs 0.061) — dolly 的 instruction 风格有帮助;
5. **garbled 几乎没有损害 MMLU** (0.635 > clean 0.630) — 10% 的乱码样本对模型伤害可忽略, 与它"检测最容易"形成鲜明对比: **最易检测的噪音最无害**。

### 5.3 MMLU 57 学科明细

- 基座最强学科: marketing (0.88)、high_school_world_history (0.87)、high_school_government (0.87); 最弱: college_mathematics (0.35)、global_facts (0.36);
- **噪音 vs clean 的学科级差异均值仅 ±0.005** — 各噪音类型的学科影响不显著 (最大单学科偏差: mixed 的 anatomy -0.082);
- 有趣: 所有噪音 run 在 college_mathematics 上都**高于** clean (+0.05~+0.11), 可能源于噪音对过拟合的轻微正则化。

### 5.4 置信度与生成行为 (逐题原始记录)

- **MC 置信度 (margin, 最优与次优 nll 差)**: base 3.143 最高; 微调模型中 unrelated 3.071 > mixed 2.938 > keyword 2.947 > garbled 2.759 > duplicate 2.629 > clean 2.474 — clean 模型"最犹豫";
- **生成长度**: base 平均 109 token/题 vs 微调模型 ~54 token — **dolly 微调使回答显著变简洁** (dolly 回复本身短)。

---

## 6. 结论与讨论

### 6.1 结论

1. **样本级检测可行性排序**: garbled (0.9996) > duplicate (0.974) > unrelated (0.923) > mixed (0.850) > **keyword (0.531, 不可行)**;
2. **最有效的特征组合**: 训练动态特征 (loss 曲率/方差、user_loss、entropy、梯度变异) 捕获 garbled/unrelated; **数据侧特征 (文本最近邻) 是 duplicate 的唯一有效手段**; keyword 需要更强的特征 (如实体级比对、反事实扰动);
3. **检测难度与危害性反相关**: 最易检测的 garbled 对模型能力几乎无害, 最难检测的 keyword/unrelated 反而是潜在危害最大的噪音 — 提示真实数据清洗应优先投入在"难以检测"的语义级噪音上;
4. **10% 数据污染的绝对影响很小** (相对 SFT 本身的伤害可忽略), 但 duplicate 的过拟合效应 (held-out +0.517) 是最明确的负面信号;
5. 方法跨任务类型稳健 (8 类别 AUC 0.87~0.99)。

### 6.2 局限与后续工作

1. **keyword 检测盲区** — 需要实体感知的检测手段;
2. **乱码定位** — 位置对齐法失效, 需序列对齐;
3. **噪音比例外推** — 10% 结果未必适用于 5%/20% (5% 数据已就绪, `run_experiment.sh --ratio 0.05 --tag ratio05 --reuse-clean` 可一键启动, 复用 clean run 省 3 小时);
4. **单一数据集/模型** — 结论基于 dolly-15k + Qwen2.5-3B, 换用分类型数据集或更大模型需再验证 (类别分层分析已给出初步迁移性证据);
5. **eval 协议** — HellaSwag/TruthfulQA 的绝对分偏低 (5-shot/0-shot 与 chat 模板的交互), 模型间比较仍有效但绝对值需谨慎引用。

### 6.3 复现

```bash
# 数据 + 训练 + 评估 + 分析 (10% 为默认 tag)
python scripts/make_noise.py
bash run_all.sh
bash run_all_eval.sh
python scripts/analyze_detection.py
python scripts/analyze_token_level.py
# 其他比例: bash run_experiment.sh --ratio 0.05 --tag ratio05 --reuse-clean
```

---

*本报告由实验流水线自动生成的产物汇总而成; 全部原始数据见 `results/` (评测明细与逐题记录) 与 `<data_root>/runs/ratio10/` (逐样本指标)。*
