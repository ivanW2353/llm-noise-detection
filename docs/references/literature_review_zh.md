# 相关文献综述: 训练数据噪音检测与数据质量

> 初版检索: 2026-08-15 · 更新: 2026-09-14（结合 `docs/analysis_report_zh.md` 全部 12 节的最终实验结果重新检索并标注）
> 来源: arXiv API · 与当前实验 (llm-noise-detection, ratio10/ratio5 两个噪音比例 × 7 类噪音+clean+mixed × Qwen2.5-3B LoRA) 逐条标注关联
> 阅读优先级: ★★★ 必读 (直接相关) · ★★ 建议 (方法互补) · ★ 参考

---

## A. 训练动力学检测噪音 (最直接相关)

| 论文 | 年份/venue | 关联点 | 优先级 |
|---|---|---|---|
| **Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics** (Swayamdipta et al.) — [arXiv:2009.10795](https://arxiv.org/abs/2009.10795) | EMNLP 2020 | 奠基作: 用置信度+变异性区分 hard-to-learn / ambiguous / easy-to-learn。**与我们的 loss_mean/loss_std/loss_curvature 指纹直接对应**; 我们的 19 维特征是其"置信度-变异性"二维投影的推广 | ★★★ |
| **An Empirical Study of Example Forgetting during Deep Neural Network Learning** (Toneva et al.) — [arXiv:1812.05159](https://arxiv.org/abs/1812.05159) | ICLR 2019 | 遗忘事件 (forgetting events) 检测噪音标签 — 与我们 converge_epoch / 跨 epoch loss 轨迹类特征同族; 我们是"首次收敛"视角, 他们是"遗忘次数"视角, 可互补 | ★★★ |
| **Identifying Mislabeled Data using the Area Under the Margin Ranking** (Pleiss et al.) — [arXiv:2001.10528](https://arxiv.org/abs/2001.10528) | NeurIPS 2020 | AUM 统计量：干净/错标样本的分类 margin 轨迹形状不同，用一个额外的"故意错标阈值类"标定判别边界。**与第 6 节"方向反转陷阱"直接对话**——AUM 本身也是有符号的margin统计量，但论文默认假设"错标=margin持续走低"，未处理我们发现的"模板化/完全重复=margin/loss 异常快速走优"这种反向记忆现象，说明这类"超典型噪音"是 AUM 框架未覆盖的一个盲区 | ★★★ |
| **Characterizing Datapoints via Second-Split Forgetting** (Maini et al.) — [arXiv:2210.15031](https://arxiv.org/abs/2210.15031) | NeurIPS 2022 | 提出 SSFT：在保留集上二次微调，观察原训练样本"被遗忘"的时间点。核心结论——**错标样本被迅速遗忘，而稀有但干净的难样本被缓慢遗忘**——与第 6 节核心矛盾（"离群"≠"噪音"，超典型样本反而最不像离群点）是同一问题的另一个观测视角，为区分"真噪音"与"真难例"提供了额外的时间维度工具 | ★★★ |
| **Early-Learning Regularization Prevents Memorization of Noisy Labels** (Liu et al.) — [arXiv:2007.00151](https://arxiv.org/abs/2007.00151) | NeurIPS 2020 | 早期学习现象: 模型先学干净样本、后记忆噪音 — **直接解释我们的关键发现"检测 AUC 随 epoch 单调衰减"(第 7 节早期检测)** | ★★★ |
| **A Closer Look at Memorization in Deep Networks** (Arpit et al.) — [arXiv:1706.05394](https://arxiv.org/abs/1706.05394) | ICML 2017 | 记忆化理论: 网络先记简单模式再记噪音 — 解释 duplicate/template 的"瞬间收敛+低 loss"现象(第 6 节方向反转) | ★★ |
| **Learning Discriminative Dynamics with Label Corruption for Noisy Label Detection** (Kim et al.) — [arXiv:2405.19902](https://arxiv.org/abs/2405.19902) | 2024 | 用动力学区分噪音 — 方法与 DynClean 同族, 与我们的特征设计思路一致 | ★★ |
| **DynClean: Training Dynamics-based Label Cleaning** (Zhang et al.) — [arXiv:2504.04616](https://arxiv.org/abs/2504.04616) | 2025 | NER 场景的动力学标签清洗 — 我们方法在结构化任务上的对照 | ★ |
| **Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt** (Mindermann et al.) — [arXiv:2206.07137](https://arxiv.org/abs/2206.07137) | ICML 2022 | **RHO-Loss**: "可学+值得学+未学"三原则 — 与我们的 update_contrib / cos_ref 概念对照 | ★★ |
| **Rho-1: Not All Tokens Are What You Need** (Lin et al.) — [arXiv:2404.07965](https://arxiv.org/abs/2404.07965) | 2024 | token 级选择性语言建模 — 与我们的 token 级归因/frac_hard 同向，也是第 12.2 节"token 级信号被子采样稀释"问题的一个高覆盖率对照实现（该文对全部 token 都算分，未做子采样） | ★★ |

## B. 小损失原则与无监督噪音标签学习

| 论文 | 年份/venue | 关联点 | 优先级 |
|---|---|---|---|
| **Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels** (Han et al.) — [arXiv:1804.06872](https://arxiv.org/abs/1804.06872) | NeurIPS 2018 | 小损失原则 (small-loss = clean) 的经典应用 — **我们的 duplicate/template 结果(低 loss 反而是噪音)构成对"小损失=干净"假设的反例补充** | ★★★ |
| **Confident Learning: Estimating Uncertainty in Dataset Labels** (Northcutt et al.) — [arXiv:1911.00068](https://arxiv.org/abs/1911.00068) | JAIR 2021 | 经典无监督噪音标签检测框架（剪枝+计数+排序三原则，开源为 `cleanlab`）— 与我们第 6 节 IsolationForest/robust-z 属于同一大类"不看标签、靠统计量排序"的方法，但 CL 依赖类别条件混淆矩阵，天然不适用于我们这种"无离散类别标签、只有连续 loss 轨迹"的开放式生成任务，说明现有无监督框架向指令微调场景迁移时需要重新设计统计量（这正是本项目在做的事） | ★★ |
| **Towards Understanding Deep Learning from Noisy Labels with Small-Loss Criterion** (Gui et al.) — [arXiv:2106.09291](https://arxiv.org/abs/2106.09291) | 2021 | 小损失准则的理论分析 — 界定其适用边界(可学习噪音 vs 不可学习噪音) | ★★ |
| **A Survey on Deep Learning with Noisy Labels** (Cordeiro & Carneiro) — [arXiv:2012.03061](https://arxiv.org/abs/2012.03061) | 2020 | 噪音标签方法全景(loss 修正/加权/选择) | ★ |

## C. 数据选择/剪枝 (LLM 时代)

| 论文 | 年份/venue | 关联点 | 优先级 |
|---|---|---|---|
| **LESS: Selecting Influential Data for Targeted Instruction Tuning** (Xia et al.) — [arXiv:2402.04333](https://arxiv.org/abs/2402.04333) | ICML 2024 | **我们的 cos_sim_ref 即 LESS 风格**(参考方向梯度相似度) — 方法血缘的直接对照 | ★★★ |
| **Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models** (Ankner et al.) — [arXiv:2405.20541](https://arxiv.org/abs/2405.20541) | 2024 | 困惑度剪枝 — 与我们的 loss_mean/entropy 特征同族; 我们验证了其在噪音检测上的边界(第 2 节关键词替换/近似重复上的失效) | ★★ |
| **When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale** (Marion et al.) — [arXiv:2309.04564](https://arxiv.org/abs/2309.04564) | 2023 | 大规模预训练剪枝(dolly 同款数据) — 剪枝能提升质量; 与我们第 9 节"7 项 benchmark 区分度很小"的发现形成张力，提示预训练规模 vs. 我们的 LoRA 微调规模下噪音的边际影响机制可能不同 | ★★ |
| **The FineWeb Datasets: Decanting the Web** (Penedo et al.) — [arXiv:2406.17557](https://arxiv.org/abs/2406.17557) | 2024 | 工业级质量过滤管线(去重是其核心!) — 印证 duplicate 检测(text_nn_sim)的实际价值 | ★★ |
| **SlimPajama-DC** (Shen et al.) — [arXiv:2309.10818](https://arxiv.org/abs/2309.10818) | 2023 | 数据组合与质量配置对训练的影响 | ★ |
| **A Survey on Data Selection for LLM Instruction Tuning** (Zhang et al.) — [arXiv:2402.05123](https://arxiv.org/abs/2402.05123) | 2024 | 指令微调数据选择全景(包含 RHO/LESS/IFD 等) — 定位本工作的最佳综述入口 | ★★★ |
| **Token Cleaning: Fine-Grained Data Selection for LLM SFT** (Pang et al.) — [arXiv:2502.01968](https://arxiv.org/abs/2502.01968) | 2025 | token 级数据选择 — 与我们 token 级归因分析(第 12.2 节)直接对照 | ★★ |
| **D3: Diversity, Difficulty, and Dependability-Aware Data Selection** (Zhang et al.) — [arXiv:2503.11441](https://arxiv.org/abs/2503.11441) | 2025 | 难度感知选择 — 与"难样本 vs 噪音"的边界讨论相关 | ★ |
| **DEITA: What Makes Good Data for Alignment?** (Liu et al.) — [arXiv:2312.15685](https://arxiv.org/abs/2312.15685) | ICLR 2024 | 复杂度(EVOL-COMPLEXITY)+质量(EVOL-QUALITY)+多样性(embedding 距离)三维打分，仅用 6K 样本即达到 SOTA 对齐效果 — 为第 10 节"免标签闭环清洗"提供多信号扩展思路：当前我们只用单一无监督异常分数(IsolationForest)做剔除，DEITA 式的多维度打分可能进一步提升 P@10% lift | ★★ |

## D. 影响力函数

| 论文 | 年份/venue | 关联点 | 优先级 |
|---|---|---|---|
| **Understanding Black-box Predictions via Influence Functions** (Koh & Liang) — [arXiv:1703.04730](https://arxiv.org/abs/1703.04730) | ICML 2017 | 影响力函数奠基 — 我们的 cos_sim_ref 是其高效近似; 用其检测标签噪音的原始动机 | ★★★ |
| **Detecting labeling bias using influence functions** (Jørgensen et al.) — [arXiv:2602.19130](https://arxiv.org/abs/2602.19130) | 2026 | 影响力函数检测标注偏差 — 与我们的噪音检测目标一致, 最新对照 | ★★ |
| **Scaling Up Influence Functions** (Schioppa et al.) — [arXiv:2112.03052](https://arxiv.org/abs/2112.03052) | 2022 | 大规模影响力函数加速 — 对比我们扁平梯度向量的效率设计 | ★ |

## E. LoRA/参数高效微调场景下的噪音检测(直接方法对照)

这一类是本次更新新增的重点方向——此前的综述完全没有覆盖"在 LoRA 而非全参数微调框架下做噪音检测"这一与本项目最匹配的技术设定。

| 论文 | 年份/venue | 关联点 | 优先级 |
|---|---|---|---|
| **Weed Out, Then Harvest: Dual Low-Rank Adaptation is an Effective Noisy Label Detector for Noise-Robust Learning** (Yuan et al.) — [arXiv:2510.10208](https://arxiv.org/abs/2510.10208) | ACL 2025 Findings | 与本项目**设定最接近**的直接对照方法：用一对 LoRA（"clean LoRA"被鼓励记忆干净样本、"noisy LoRA"被约束只记忆错标样本，二者形成可学习阈值）做免标签样本选择。我们的路线是"单 LoRA + 逐样本梯度/loss 轨迹+多维统计特征"，Delora 的路线是"双 LoRA 结构性分工"——同一问题(LoRA 微调过程中免标签识别噪音样本)的两条正交技术路径，可作为未来的直接 baseline 对比对象 | ★★★ |
| **RobustFT: Robust Supervised Fine-tuning for Large Language Models under Noisy Response** (Luo et al.) — [arXiv:2412.14922](https://arxiv.org/abs/2412.14922) | 2024 | LLM SFT 噪音处理全流程框架：多专家协同噪音检测 → 上下文增强去噪 → 基于响应熵的数据选择。与我们第 10 节"免标签闭环清洗"目标一致，但策略不同——RobustFT 选择"检测后去噪重标注"，我们目前是"检测后直接剔除"；重标注策略可能是我们闭环清洗流程的一个可选增强分支(尤其对 P@10% lift < 1 的 duplicate 类型，直接剔除可能有害，去噪重标注或许是更安全的替代) | ★★ |

## F. 数据去重与文本相似度(对照第 8 节特征归因发现)

第 8 节最重要的发现之一是：完全重复(duplicate)与话题不相关(unrelated)的检测信号 90% 以上来自静态文本特征 `text_nn_sim`，而非训练动态。以下文献为这一发现提供背景与延伸方向。

| 论文 | 年份/venue | 关联点 | 优先级 |
|---|---|---|---|
| **Deduplicating Training Data Makes Language Models Better** (Lee et al.) — [arXiv:2107.06499](https://arxiv.org/abs/2107.06499) | ACL 2022 | 证明训练集中的(近)重复片段会被模型逐字记忆、拖累生成质量与评测有效性，提出的去重工具本质上是纯文本层面的算法(后缀数组等)，**不依赖任何训练动态信号**——与我们"duplicate 检测的高 AUC 本质上是文本相似度的功劳、不能算训练动态方法论的贡献"这一发现的立场完全一致，是该发现最直接的外部佐证 | ★★★ |
| **SemDeDup: Data-efficient learning at web-scale through semantic deduplication** (Abbas et al.) — [arXiv:2303.09540](https://arxiv.org/abs/2303.09540) | 2023 | 用预训练模型 embedding(而非 TF-IDF 词汇特征)做语义去重，可去除表面用词不同但语义重复的样本 — 是我们 `text_nn_sim`(纯词汇层面 TF-IDF)的语义升级版；第 2 节中近似重复(near_duplicate)域内 AUC 仅 0.674，明显低于完全重复的 0.986，一个可能原因正是 TF-IDF 对"语义相同、措辞不同"的改写不够敏感——换成语义 embedding 相似度是一个值得验证的改进方向 | ★★ |

## G. 关键词替换/实体级对抗扰动(对照 keyword 检测盲区)

第 2/3/8 节反复印证关键词替换(keyword)是全部 7 类噪音中最难检测的类型(域内 AUC 仅 0.577，特征归因也找不到稳定的主导特征)。这类"只换 1-2 个词、句子整体结构不变"的噪音在 NLP 鲁棒性/对抗样本文献中有对应的研究传统。

| 论文 | 年份/venue | 关联点 | 优先级 |
|---|---|---|---|
| **Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment (TextFooler)** (Jin et al.) — [arXiv:1907.11932](https://arxiv.org/abs/1907.11932) | AAAI 2020 | 提出的攻击方法与我们的 keyword 噪音构造方式同构：保持句子整体语义/结构不变，只替换少量关键词以达成扰动目的。该文献证明这类局部同义替换对模型的表层统计特征(含 TF-IDF 式相似度)天然不敏感——为我们"keyword 类型 `text_nn_sim` 与训练动态特征双双失效"提供了独立的理论解释，而不只是"这次实验凑巧没找到好特征" | ★★ |

## H. 指令微调数据质量与下游影响的实证研究(对照第 9/10 节)

| 论文 | 年份/venue | 关联点 | 优先级 |
|---|---|---|---|
| **Fine-Tuning on Noisy Instructions: Effects on Generalization and Performance** (Alajrami et al.) — [arXiv:2510.03528](https://arxiv.org/abs/2510.03528) | AACL 2025 | 与本项目**实验设计高度同构**的独立研究：同样基于 Dolly 等指令数据注入扰动(去停用词/打乱词序等)，同样在 MMLU/BBH/GSM8K 等通用 benchmark 上评测下游影响。可直接用于交叉验证第 9 节"7 项 benchmark 区分度小、噪音下游危害有限"这一结论是否具有跨扰动类型/跨评测集的普遍性，而不是本项目特有的巧合 | ★★★ |
| **AlpaGasus: Training A Better Alpaca with Fewer Data** (Chen et al.) — [arXiv:2307.08701](https://arxiv.org/abs/2307.08701) | ICLR 2024 | 用强 LLM 打分过滤 Alpaca 52k→9k 数据，重训后指令遵循效果反而更好、训练提速 5.7 倍 — 是"打分剔除 → 重训 → 下游评测验证提升"这一完整闭环范式最直接的先例，与我们第 10 节免标签闭环清洗结构完全对应(区别仅在于 AlpaGasus 用 LLM 打分做质量筛选，我们用训练动态无监督打分做噪音剔除) | ★★★ |
| **LIMA: Less Is More for Alignment** (Zhou et al.) — [arXiv:2305.11206](https://arxiv.org/abs/2305.11206) | NeurIPS 2023 | 1000 条精心挑选样本即可达到强对齐效果，指令微调阶段的知识主要来自预训练——为第 9 节"7 项通用能力 benchmark 区分度小"提供了机理解释：这些 benchmark 主要考察预训练知识，SFT 阶段无论是清洗噪音还是引入少量噪音，边际影响本就有限 | ★★ |
| **RobustFT** — 见 E 节 | 2024 | 同时也是第 9/10 节的相关工作 | ★★ |

---

## 关联图谱: 我们的发现 vs 文献

```
我们的发现(报告章节)                    ↔  文献支撑
──────────────────────────────────────────────────────────────────
检测 AUC 随 epoch 衰减 (§7)             ↔  Early-Learning (Liu 2020); Arpit 2017
duplicate/template 低 loss 反转 (§6)    ↔  Co-teaching 小损失假设 (Han 2018) 的反例
                                           AUM (Pleiss 2020); Second-Split Forgetting (Maini 2022)
loss 轨迹曲率/方差指纹 (§12.1)          ↔  Dataset Cartography (Swayamdipta 2020)
cos_sim_ref 参考方向 (§12.1)           ↔  LESS (Xia 2024); Koh & Liang 2017
converge_epoch 收敛速度 (§12.1)        ↔  Toneva forgetting (2019); RHO (2022)
token 级信号最稳但被子采样稀释 (§12.2)  ↔  RHO-1 (Lin 2024); Token Cleaning (Pang 2025)
跨类型迁移损失~17-20点 (§3)             ↔  暂无直接文献对照 — 大多数噪音检测论文只在单一噪音类型/单一corruption
                                           process 下验证，很少系统评测"跨噪音类型迁移矩阵"，是本项目的潜在方法论贡献
跨比例迁移几乎无损 (§4)                 ↔  暂无直接文献对照 — 同上，是潜在贡献点
P@10% lift 与 AUC 结论相悖 (§5)         ↔  暂无直接文献系统讨论 — 多数论文只报告 AUC/F1，precision@budget
                                           在预算受限清洗场景下的重要性未被充分强调，是潜在方法论批评点
duplicate/unrelated 靠 text_nn_sim (§8) ↔  Dedup (Lee 2022); SemDeDup (Abbas 2023); FineWeb (Penedo 2024)
keyword 盲区 (实体/局部替换) (§2/8)      ↔  TextFooler (Jin 2019) 的"局部替换绕过表层特征"机制
LoRA 框架下噪音检测的技术路线对照        ↔  Delora/双LoRA (Yuan 2025)
下游任务影响有限，仅模板化确有危害 (§9)  ↔  Fine-Tuning on Noisy Instructions (Alajrami 2025, 同构实验设计)
                                           LIMA (Zhou 2023)
免标签闭环清洗 (§10)                    ↔  AlpaGasus (Chen 2023); RobustFT (Luo 2024); DEITA (Liu 2023, 多维度打分扩展)
```

## 写作建议 (论文引用组合)

1. **方法部分**: Swayamdipta (特征指纹) + Toneva (轨迹) + Xia/LESS (参考方向) + Koh&Liang (影响力动机) + Delora (LoRA 框架下的直接对照 baseline)
2. **检测窗口发现**: Liu/ELR + Arpit (早期学习→记忆的机理解释)
3. **duplicate/template 反转发现**: Han/Co-teaching 小损失假设的反例讨论 + AUM/Second-Split Forgetting (对"错标=异常统计量"假设的进一步挑战) — 这是本工作最具原创性的讨论点之一
4. **跨类型/跨比例迁移矩阵 + P@10% lift vs AUC**: 目前没有找到系统对照的文献，建议作为本工作**独立的方法论贡献**明确写出("多数噪音检测研究只报告单一类型/单一指标下的 AUC，我们系统评测了跨类型迁移与预算受限精度，揭示了两个此前未被充分讨论的失效模式")
5. **text_nn_sim 主导 duplicate/unrelated 检测**: Lee 2022 (去重) + SemDeDup 2023 (语义去重扩展方向) 作为外部佐证与改进方向
6. **keyword 盲区**: TextFooler (Jin 2019) 提供"局部替换绕过表层特征"的独立理论支撑
7. **下游影响有限**: Alajrami 2025 (同构实验设计的交叉验证) + LIMA (机理解释) + Marion "When Less is More" (张力对照)
8. **免标签闭环清洗**: AlpaGasus (最直接先例) + RobustFT (去噪重标注的替代策略) + DEITA (多维度打分扩展)
9. **综述定位**: Zhang 2024 (指令微调数据选择综述) 作为相关工作的组织框架

---

*arXiv 链接均为 API 实测返回; 引用前建议在 Semantic Scholar/Google Scholar 复核完整元数据。*
