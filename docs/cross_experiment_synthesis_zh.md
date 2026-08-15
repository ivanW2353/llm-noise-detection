# 跨实验综合: 噪音检测三连实验的结论汇总

> 综合三个实验: **qa-noise-experiment** (2026-07, SQuAD v1 抽取式 QA, 87K 样本, 1.5B) ·
> **dynanoise** (2026-08, dolly-15k 生成式, 1.5B/3B, loss 动力学信号) ·
> **llm-noise-detection** (本仓库, dolly-15k, 3B, 逐样本梯度 + 19 特征)

---

## 1. 三个实验的对应关系

| 噪音类型 | qa-noise | dynanoise | 本实验 | 检测难度 (本实验) | 危害性 |
|---|---|---|---|---|---|
| 表面损坏 | random_word | A (BPE 乱码) | garbled | **0.9996 (最易)** | **最轻** |
| 一致模式/快捷键 | fixed_wrong | E (shortcut "42") | — (缺失!) | 数据侧/IFD 可检 | **灾难级** |
| 重复冗余 | — | C (redundant) | duplicate | 0.974 (仅数据侧) | 过拟合损伤 |
| 语义错配 | random_replacement | B (LLM 流畅错答) | unrelated | 0.923 | 中等 (生成式) |
| 精致篡改 | — | D (改一个事实) | keyword | **0.531 (不可检)** | 随比例显现 |

---

## 2. 跨实验一致的 6 条结论

### 2.1 "噪音 = 高 loss" 的方向直觉是错的, 且是普遍陷阱

- dynanoise: unlearnable 噪音的 loss_cv 反而**低于** clean (0.013 vs 0.041); 修正方向 (`-loss_cv`) 后 A 类命中率从 3.8% 跃升至 86.5%;
- 本实验: duplicate 的 loss 也**低于**正常样本 (AUC 0.37, 方向反转);
- **结论**: 可被记忆/学习的噪音 (duplicate/冗余) 在 loss 侧呈反向信号。任何检测器必须经验性验证方向, 采用双向 (zscore) 联合方案。

### 2.2 Token 级信号是跨实验最稳健的信号族

- dynanoise: `token_loss_top20` 跨 1.5B/3B 完全稳定 (AUROC 0.947 ± 0.001), 也是唯一跨模型尺度稳定的信号;
- 本实验: `entropy` / `frac_hard` / `max_token_loss` / `user_loss` 是 garbled 检测的主力 (0.95~0.98);
- 本实验 token 级逐 token 梯度归因 (hard_loss AUC 0.77) 也确认 token 级信息有效;
- **结论**: 检测信号应优先考虑 token 级特征; 样本级聚合 (跨 token/epoch) 是可靠尺度。

### 2.3 噪音危害性由"任务类型"决定 — 跨实验最强的对比

| 任务 | 噪音 | 比例 | 影响 |
|---|---|---|---|
| SQuAD 抽取式 (qa-noise) | random_replacement (答案互换) | **50%** | EM 仅 -0.6% (几乎无害) |
| SQuAD 抽取式 | fixed_wrong (统一错答) | 50% | EM **-41.8%** (灾难) |
| dolly 生成式 (本实验) | unrelated (回复互换) | 10% | GSM8K -0.043 (最有害) |

- 抽取式任务中 context 自带答案 → 模型靠上下文兜底, label 噪音几乎无效;
- 生成式任务中 response 即知识 → 语义错配直接污染知识;
- **结论**: 噪音影响分析不能脱离任务类型; 同一噪音在抽取式任务无害、在生成式任务有害。

### 2.4 一致模式噪音 (shortcut) 是最危险且最需检测的类型

- qa-noise: fixed_wrong 近线性灾难 (-8.4 分/10%, R²≈0.99);
- dynanoise Phase 5: shortcut "42" 噪音 loss_cv 检测弱 (0.67), 但 **IFD 有效 (0.90)** — 需要指令感知的信号;
- 本实验**恰好缺少这一类噪音** (4 类中无一致模式类) — 这是当前实验设计的已知缺口;
- **结论**: 一致模式噪音 = "高危害 + 可检测 (用对信号)" 象限, 是数据清洗的最高价值目标。

### 2.5 检测力与清洗收益解耦 — 过滤提升存在天花板

- dynanoise Phase 4: 三组独立实验中, 精准过滤 (RHO 命中 99.8%) 的 MT-Bench 提升 (+0.48) 仅略优于**随机丢弃 10%** (+0.41);
- 本实验: 10% 噪音对验证集几乎无影响 (6 模型 MMLU 极差 0.011, 且全部劣于基座);
- **结论**: 低比例语义噪音的"过滤收益"极低 — 检测信号的价值应定位在**质量监控/审计/数据治理**, 而非"清洗后模型变强"; 若追求清洗收益, 应针对高危害类型 (shortcut/高比例 label 噪音)。

### 2.6 受控实验的信号方向在自然数据上成立

- dynanoise Phase 6 (lmsys-chat-1m, 50K): token_loss_top20 与 loss_mu 的 Spearman ρ = -0.78, 方向与受控实验一致 (AUROC 0.946);
- **结论**: loss 动力学信号可迁移到无标签的真实数据, 用于数据质量监控。

---

## 3. 检测难度光谱 (三实验合并)

```
可检测 ◄─────────────────────────────────────────► 不可检测
一致模式(duplicate/fixed_wrong)   表面损坏(garbled)   语义错配(unrelated)   精致篡改(keyword)
数据侧/IFD 可检                   训练侧可检           部分可检             几乎不可检
(过拟合/灾难性危害)               (最轻危害)           (中等危害)           (低比例时无害)
```

**关键洞察**: 检测难度与危害性**不单调相关** — 最易检的 (garbled) 最无害, 最难检的 (keyword) 低比例无害; 真正的"检测价值区"是**一致模式噪音** (可检且灾难) 与**高比例语义噪音** (难检但开始有害)。

---

## 4. 对当前实验的可借鉴改进 (按价值排序)

### 4.1 补第 5 类噪音: 一致模式 shortcut (最高价值)
- 构造: 噪音样本统一回复如 "The answer is 42." (dynanoise E) 或统一错误模式 (qa-noise fixed_wrong);
- 预期: 检测需 IFD 或数据侧一致性信号; 危害性应最大 — 补齐当前"高危害+可检测"象限空缺;
- 成本: 1 个数据集 + 1 次训练 (~3.5h), 复用现有流水线 (`make_noise.py` 加类型即可)。

### 4.2 增加 token_loss_top20 集中度信号
- dynanoise 跨模型最稳信号, 本实验未直接实现 (现有 frac_hard/max_token_loss/entropy 近似互补);
- 实现: `train.py` 的 diagnostic_pass 已算逐 token CE, 增加 top-20% 损失占比输出 (~零成本);
- 分析侧可从已存的 top-32 token 明细近似计算。

### 4.3 增加 IFD 信号 (Instruction-Following Difficulty)
- 公式: IFD = L(A|Q) / L(A) (需一次额外前向, 对 prompt-only 输入);
- dynanoise 中唯一能检 shortcut (0.90) 且对 D 类 (0.60) 有信号的指标;
- 实现: diagnostic_pass 中额外一次无 prompt 的前向, 每 epoch 抽样即可 (~1 分钟/epoch)。

### 4.4 双向信号显式化
- 本实验 RF 已隐式学到 duplicate 的"低 loss"方向, 但文档/阈值应显式说明双向性;
- 建议: 对每个指标同时取高/低两个方向的 zscore 特征 (dynanoise 教训)。

### 4.5 清洗收益对照实验 (验证 dynanoise 的"天花板"结论)
- 跑 1 个 run: 训练前移除 garbled 样本 (或随机移除 10%) → 对比验证集;
- 预期: 提升极小, 复现 dynanoise 的 random_drop ≈ 精准过滤;
- 成本: 2 个 run (~7h)。

### 4.6 自然数据信号验证 (复现 Phase 6)
- lmsys-chat-1m 已在本地缓存 (50K);
- 用训练好的 clean 模型计算 token 级信号, 与 loss_mu 做 Spearman 相关;
- 成本: ~1h, 纯验证性, 增强结论的部署说服力。

### 4.7 RHO 参照对比
- dynanoise 用 holdout 模型作 RHO 基准; 本实验用 cos_sim_ref (LESS 式) 作参考方向;
- 可在分析中加一列 RHO 式信号 (需训练 1 个 holdout 模型, ~3.5h) 作为对照基准。

---

## 5. 一句话总结

> 三个实验共同说明: **loss 动力学信号真实有效但方向常反转、token 级最稳、危害由任务类型决定、一致模式噪音最危险、清洗收益有天花板** — 检测技术的价值在于数据治理与质量监控, 而最高价值目标是一致模式/高比例语义噪音, 而非最容易检测的表面乱码。
