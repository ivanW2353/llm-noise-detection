# 相关文献综述: 训练数据噪音检测与数据质量

> 检索时间: 2026-08-15 · 来源: arXiv API · 与当前实验 (llm-noise-detection, 10% 噪音 × 6/7 类 × Qwen2.5-3B LoRA) 逐条标注关联
> 阅读优先级: ★★★ 必读 (直接相关) · ★★ 建议 (方法互补) · ★ 参考

---

## A. 训练动力学检测噪音 (最直接相关)

| 论文 | 年份/venue | 关联点 | 优先级 |
|---|---|---|---|
| **Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics** (Swayamdipta et al.) — [arXiv:2009.10795](https://arxiv.org/abs/2009.10795) | EMNLP 2020 | 奠基作: 用置信度+变异性区分 hard-to-learn / ambiguous / easy-to-learn。**与我们的 loss_mean/loss_std/loss_curvature 指纹直接对应**; 我们的 19 维特征是其"置信度-变异性"二维投影的推广 | ★★★ |
| **An Empirical Study of Example Forgetting during Deep Neural Network Learning** (Toneva et al.) — [arXiv:1812.05159](https://arxiv.org/abs/1812.05159) | ICLR 2019 | 遗忘事件 (forgetting events) 检测噪音标签 — 与我们 converge_epoch / 跨 epoch loss 轨迹类特征同族; 我们是"首次收敛"视角, 他们是"遗忘次数"视角, 可互补 | ★★★ |
| **Early-Learning Regularization Prevents Memorization of Noisy Labels** (Liu et al.) — [arXiv:2007.00151](https://arxiv.org/abs/2007.00151) | NeurIPS 2020 | 早期学习现象: 模型先学干净样本、后记忆噪音 — **直接解释我们的关键发现"检测 AUC 随 epoch 单调衰减" (epoch 0-1 检测窗口最佳)** | ★★★ |
| **A Closer Look at Memorization in Deep Networks** (Arpit et al.) — [arXiv:1706.05394](https://arxiv.org/abs/1706.05394) | ICML 2017 | 记忆化理论: 网络先记简单模式再记噪音 — 解释 duplicate 的"瞬间收敛+低 loss"现象 (我们的 loss AUC 0.37 反转) | ★★ |
| **Learning Discriminative Dynamics with Label Corruption for Noisy Label Detection** (Kim et al.) — [arXiv:2405.19902](https://arxiv.org/abs/2405.19902) | 2024 | 用动力学区分噪音 — 方法与 DynClean 同族, 与我们的特征设计思路一致 | ★★ |
| **DynClean: Training Dynamics-based Label Cleaning** (Zhang et al.) — [arXiv:2504.04616](https://arxiv.org/abs/2504.04616) | 2025 | NER 场景的动力学标签清洗 — 我们方法在结构化任务上的对照 | ★ |
| **Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt** (Mindermann et al.) — [arXiv:2206.07137](https://arxiv.org/abs/2206.07137) | ICML 2022 | **RHO-Loss** (dynanoise 实验的 gold standard 基准): "可学+值得学+未学" 三原则 — 与我们的 update_contrib / cos_ref 概念对照 | ★★ |
| **Rho-1: Not All Tokens Are What You Need** (Lin et al.) — [arXiv:2404.07965](https://arxiv.org/abs/2404.07965) | 2024 | token 级选择性语言建模 — 与我们的 token 级归因/frac_hard 同向 | ★★ |

## B. 小损失原则与噪音标签学习

| 论文 | 年份/venue | 关联点 | 优先级 |
|---|---|---|---|
| **Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels** (Han et al.) — [arXiv:1804.06872](https://arxiv.org/abs/1804.06872) | NeurIPS 2018 | 小损失原则 (small-loss = clean) 的经典应用 — **我们的 duplicate 结果 (低 loss 反而是噪音) 构成对"小损失=干净"假设的反例补充** | ★★★ |
| **Towards Understanding Deep Learning from Noisy Labels with Small-Loss Criterion** (Gui et al.) — [arXiv:2106.09291](https://arxiv.org/abs/2106.09291) | 2021 | 小损失准则的理论分析 — 界定其适用边界 (可学习噪音 vs 不可学习噪音) | ★★ |
| **A Survey on Deep Learning with Noisy Labels** (Cordeiro & Carneiro) — [arXiv:2012.03061](https://arxiv.org/abs/2012.03061) | 2020 | 噪音标签方法全景 (loss 修正/加权/选择) | ★ |

## C. 数据选择/剪枝 (LLM 时代)

| 论文 | 年份/venue | 关联点 | 优先级 |
|---|---|---|---|
| **LESS: Selecting Influential Data for Targeted Instruction Tuning** (Xia et al.) — [arXiv:2402.04333](https://arxiv.org/abs/2402.04333) | ICML 2024 | **我们的 cos_sim_ref 即 LESS 风格** (参考方向梯度相似度) — 方法血缘的直接对照 | ★★★ |
| **Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models** (Ankner et al.) — [arXiv:2405.20541](https://arxiv.org/abs/2405.20541) | 2024 | 困惑度剪枝 — 与我们的 loss_mean/entropy 特征同族; 我们验证了其在噪音检测上的边界 | ★★ |
| **When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale** (Marion et al.) — [arXiv:2309.04564](https://arxiv.org/abs/2309.04564) | 2023 | 大规模预训练剪枝 (dolly 同款数据) — 剪枝能提升质量; 与我们的"过滤收益天花板"发现形成张力 | ★★ |
| **The FineWeb Datasets: Decanting the Web** (Penedo et al.) — [arXiv:2406.17557](https://arxiv.org/abs/2406.17557) | 2024 | 工业级质量过滤管线 (去重是其核心!) — 印证 duplicate 检测 (text_nn_sim) 的实际价值 | ★★ |
| **SlimPajama-DC** (Shen et al.) — [arXiv:2309.10818](https://arxiv.org/abs/2309.10818) | 2023 | 数据组合与质量配置对训练的影响 | ★ |
| **A Survey on Data Selection for LLM Instruction Tuning** (Zhang et al.) — [arXiv:2402.05123](https://arxiv.org/abs/2402.05123) | 2024 | 指令微调数据选择全景 (包含 RHO/LESS/IFD 等) — 定位本工作的最佳综述入口 | ★★★ |
| **Token Cleaning: Fine-Grained Data Selection for LLM SFT** (Pang et al.) — [arXiv:2502.01968](https://arxiv.org/abs/2502.01968) | 2025 | token 级数据选择 — 与我们 token 级归因分析直接对照 | ★★ |
| **D3: Diversity, Difficulty, and Dependability-Aware Data Selection** (Zhang et al.) — [arXiv:2503.11441](https://arxiv.org/abs/2503.11441) | 2025 | 难度感知选择 — 与"难样本 vs 噪音"的边界讨论相关 | ★ |

## D. 影响力函数

| 论文 | 年份/venue | 关联点 | 优先级 |
|---|---|---|---|
| **Understanding Black-box Predictions via Influence Functions** (Koh & Liang) — [arXiv:1703.04730](https://arxiv.org/abs/1703.04730) | ICML 2017 | 影响力函数奠基 — 我们的 cos_sim_ref 是其高效近似; 用其检测标签噪音的原始动机 | ★★★ |
| **Detecting labeling bias using influence functions** (Jørgensen et al.) — [arXiv:2602.19130](https://arxiv.org/abs/2602.19130) | 2026 | 影响力函数检测标注偏差 — 与我们的噪音检测目标一致, 最新对照 | ★★ |
| **Scaling Up Influence Functions** (Schioppa et al.) — [arXiv:2112.03052](https://arxiv.org/abs/2112.03052) | 2022 | 大规模影响力函数加速 — 对比我们扁平梯度向量的效率设计 | ★ |

---

## 关联图谱: 我们的发现 vs 文献

```
我们的发现                          ↔  文献支撑
────────────────────────────────────────────────────────────
检测 AUC 随 epoch 衰减              ↔  Early-Learning (Liu 2020); Arpit 2017
duplicate 低 loss (方向反转)        ↔  Co-teaching 小损失假设 (Han 2018) 的反例
loss 轨迹曲率/方差指纹              ↔  Dataset Cartography (Swayamdipta 2020)
cos_sim_ref 参考方向               ↔  LESS (Xia 2024); Koh & Liang 2017
converge_epoch 收敛速度            ↔  Toneva forgetting (2019); RHO (2022)
token 级信号最稳                   ↔  RHO-1 (Lin 2024); Token Cleaning (Pang 2025)
过滤收益天花板                     ↔  When Less is More (Marion 2023) 的张力
keyword 盲区 (实体级)              ↔  无直接文献 — 潜在贡献点 (可引 NER/反事实)
truncation/near_duplicate 类型     ↔  FineWeb 去重管线 (Penedo 2024) 的实操背景
```

## 写作建议 (论文引用组合)

1. **方法部分**: Swayamdipta (特征指纹) + Toneva (轨迹) + Xia/LESS (参考方向) + Koh&Liang (影响力动机)
2. **检测窗口发现**: Liu/ELR + Arpit (早期学习→记忆的机理解释)
3. **duplicate 反转发现**: Han/Co-teaching 小损失假设的反例讨论 (创新点)
4. **过滤收益**: Marion "When Less is More" 的对照 + 我们的天花板证据
5. **综述定位**: Zhang 2024 (指令微调数据选择综述) 作为相关工作的组织框架

---

*arXiv 链接均为 API 实测返回; 引用前建议在 Semantic Scholar/Google Scholar 复核完整元数据。*
