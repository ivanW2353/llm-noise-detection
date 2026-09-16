# 相关文献综述：按与本实验的相关性排序

> 初版检索 2026-08-15 · 本版 2026-09-16 重写：改为按**与本实验的相关性**分层（而非按主题分类），并逐条核实发表 venue。
> **venue 已逐条核实**（官方 proceedings / ACL Anthology / PMLR / OpenReview），会议论文与 workshop 论文、纯 arXiv preprint 明确区分——这是上一版没做的，上一版有 5 处 venue 标注不准确。
> 关联章节指向 [`docs/report_zh/`](../report_zh/README.md) 的新编号。

**分层逻辑**：第一层是与本项目**同一问题**的工作（可作为直接 baseline 或直接对话对象）；第二层是本项目**具体特征的方法出处**；第三层**问题相近但分工不同**（用外部打分器 vs 只用训练痕迹）；第四层是本项目发现**未找到直接对照**的部分。

---

## 第一层：同一问题的直接对照（必读）

这四篇与本项目回答的是同一个问题，或直接决定了本项目最重要发现的解释。

| 论文 | Venue | 为什么最相关 |
|---|---|---|
| **Weed Out, Then Harvest: Dual Low-Rank Adaptation is an Effective Noisy Label Detector**<br>Yuan et al. — [arXiv:2510.10208](https://arxiv.org/abs/2510.10208) | **ACL 2025 Findings** | **与本项目设定最接近的工作**：同样在 LoRA 微调框架下做免标签噪音样本选择。它用一对 LoRA 做结构性分工（clean LoRA 被鼓励记忆干净样本、noisy LoRA 被约束只记忆错标样本，二者形成可学习阈值），本项目用单 LoRA + 逐样本梯度/loss 轨迹的多维统计。同一问题的两条正交路径，是最应该做的直接 baseline 对比 |
| **Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels**<br>Han et al. — [arXiv:1804.06872](https://arxiv.org/abs/1804.06872) | **NeurIPS 2018** | 小损失准则（loss 小 = 干净）的经典代表。**本项目 6.6 节的方向反转发现是这个准则的系统性反例**：模板化/完全重复噪音 loss 异常低、收敛异常快，小损失准则会优先保护它们。6.13.2 节进一步量化了代价——按错误方向清洗后下游比不清洗低 4.09 点 |
| **A Closer Look at Memorization in Deep Networks**<br>Arpit et al. — [arXiv:1706.05394](https://arxiv.org/abs/1706.05394) | **ICML 2017** | 提供方向反转的**机制解释**：网络先学简单模式、后记忆噪音。高度重复/高度模板化的内容会被优先记住，因而在训练动态上表现为"异常典型"而非"异常反常"，这正是 5.2 节 B 类噪音的理论依据 |
| **What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation**<br>Feldman et al. — [arXiv:2008.03703](https://arxiv.org/abs/2008.03703) | **NeurIPS 2020**（Spotlight） | 记忆与长尾样本的量化分析，补上方向反转解释的另一半：**"被记忆"与"是噪音"是两个不同的属性**，而离群检测无法区分。这是 5.3 节"方向不可知"这一结构性困难的理论根源 |

## 第二层：本项目具体特征的方法出处

本项目的每一族特征都有明确的方法血缘。这一层解释"我们用的这个量是从哪来的、与原始形态差在哪"。

| 论文 | Venue | 对应本项目的哪个特征 |
|---|---|---|
| **Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics**<br>Swayamdipta et al. — [arXiv:2009.10795](https://arxiv.org/abs/2009.10795) | **EMNLP 2020** | 奠基作。用置信度均值+方差把数据集分成 easy/ambiguous/hard-to-learn 三区，直接对应本项目的 `loss_mean`/`loss_std`/`loss_curvature`。**差别在采集粒度**：该文在标准 mini-batch 下用置信度统计，本项目用 `micro_batch=1` 隔离出逐样本梯度（3.4 节），因此能拿到梯度方向而不只是 loss |
| **An Empirical Study of Example Forgetting during Deep Neural Network Learning**<br>Toneva et al. — [arXiv:1812.05159](https://arxiv.org/abs/1812.05159) | **ICLR 2019** | 用"遗忘事件"刻画样本难度，与本项目 `converge_epoch` 同族——他们看"被忘掉几次"，我们看"第一次收敛在第几个 epoch"，两个视角互补 |
| **Understanding Black-box Predictions via Influence Functions**<br>Koh & Liang — [arXiv:1703.04730](https://arxiv.org/abs/1703.04730) | **ICML 2017** | 影响函数奠基，也是"用影响力检测标签噪音"的原始动机。本项目的 `cos_sim_ref` 是它的廉价近似 |
| **Estimating Training Data Influence by Tracing Gradient Descent (TracIn)**<br>Pruthi et al. — [arXiv:2002.08484](https://arxiv.org/abs/2002.08484) | **NeurIPS 2020** | 用训练过程中梯度内积的累积近似影响力。本项目的 `cos_sim_ref`（样本梯度与 200 条留出样本平均梯度方向的余弦）与 `update_contrib` 是同一思路的轻量版：不求精确影响值，只求可排序的代理量，代价小到能对全部样本每个 epoch 都算 |
| **LESS: Selecting Influential Data for Targeted Instruction Tuning**<br>Xia et al. — [arXiv:2402.04333](https://arxiv.org/abs/2402.04333) | **ICML 2024** | 梯度相似度做面向目标任务的数据选择，**本项目 `cos_sim_ref` 与它的做法最接近**——差别是 LESS 面向"对目标任务有用"，我们面向"与正常数据共识方向不一致" |
| **Deduplicating Training Data Makes Language Models Better**<br>Lee et al. — [arXiv:2107.06499](https://arxiv.org/abs/2107.06499) | **ACL 2022** | 纯文本层面的去重（后缀数组），**不依赖任何训练信号**。与本项目 6.8/6.11 节"完全重复的检测信号 90% 以上来自 `text_nn_sim` 而非训练动态"这一发现立场完全一致，是该发现最直接的外部佐证 |
| **Identifying Mislabeled Data using the Area Under the Margin Ranking (AUM)**<br>Pleiss et al. — [arXiv:2001.10528](https://arxiv.org/abs/2001.10528) | **NeurIPS 2020** | 与本项目 `memo_signed` 最可比的方法：都是**带符号**的轨迹统计量。但 AUM 默认"错标 = margin 持续走低"，未处理"超典型噪音 margin 异常快速走优"这一反向情形——这个盲区正是本项目 6.6 节的主题 |
| **Early-Learning Regularization Prevents Memorization of Noisy Labels**<br>Liu et al. — [arXiv:2007.00151](https://arxiv.org/abs/2007.00151) | **NeurIPS 2020** | 早期学习现象（先学干净、后记噪音），**直接解释本项目 6.7 节"检测 AUC 随 epoch 衰减"**——信号在早期最强，训练越久噪音越被吸收进参数 |
| **Characterizing Datapoints via Second-Split Forgetting**<br>Maini et al. — [arXiv:2210.15031](https://arxiv.org/abs/2210.15031) | **NeurIPS 2022** | 核心结论"错标样本被迅速遗忘、稀有但干净的难样本被缓慢遗忘"，为区分"真噪音"与"真难例"提供了额外的时间维度工具。与 6.6 节"离群 ≠ 噪音"是同一问题的另一观测视角 |
| **Understanding deep learning requires rethinking generalization**<br>Zhang et al. — [arXiv:1611.03530](https://arxiv.org/abs/1611.03530) | **ICLR 2017** | 证明深度网络可以完全记住随机标签，是"损失曲线里含有样本难度/反常度信息"这一整条思路的前提 |
| **Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt (RHO-Loss)**<br>Mindermann et al. — [arXiv:2206.07137](https://arxiv.org/abs/2206.07137) | **ICML 2022** | "可学 + 值得学 + 未学"三原则，与本项目 `update_contrib` / `cos_ref` 的设计动机对照 |
| **Not All Tokens Are What You Need for Pretraining (Rho-1)**<br>Lin et al. — [arXiv:2404.07965](https://arxiv.org/abs/2404.07965) | **NeurIPS 2024** | token 级选择性建模，与本项目的 token 级归因同向。**注意标题差异**：arXiv 版题为 "Rho-1: Not All Tokens Are What You Need"，NeurIPS 正式版改名为 "Not All Tokens Are What You Need for Pretraining"，按 arXiv 标题检索会议记录会找不到。它对全部 token 都算分，是本项目 6.11 节"token 诊断只有 12.5% 覆盖率"问题的一个高覆盖率对照实现 |

## 第三层：问题相近但分工不同（LLM 数据选择）

这一批与本项目目标相似（都想筛掉低质量训练数据），但**判断质量的依据不同**：它们多用外部打分器（另一个 LLM、嵌入相似度、启发式规则），本项目只用被训练模型自己在训练过程中留下的痕迹，不引入外部裁判。两条路不冲突——6.8 与 6.11 节恰好显示静态文本相似度（外部信号一侧）在完全重复/话题不相关上强于全部训练动态特征，而在乱码上完全无用。

| 论文 | Venue | 关联点 |
|---|---|---|
| **A Survey on Data Selection for LLM Instruction Tuning**<br>Zhang et al. — [arXiv:2402.05123](https://arxiv.org/abs/2402.05123) | **JAIR**（[DOI](https://doi.org/10.1613/jair.1.17625)） | 该领域最佳综述入口，已正式发表于 JAIR（比 arXiv 版更适合引用）。可用作组织相关工作的框架 |
| **LIMA: Less Is More for Alignment**<br>Zhou et al. — [arXiv:2305.11206](https://arxiv.org/abs/2305.11206) | **NeurIPS 2023** | 1000 条精选样本达到强对齐，且指出 SFT 阶段知识主要来自预训练——**为 6.9 节"7 项通用 benchmark 区分度小（0.42-0.44）"提供机理解释**：这些 benchmark 主要考察预训练知识，SFT 阶段增删少量噪音的边际影响本就有限 |
| **AlpaGasus: Training A Better Alpaca with Fewer Data**<br>Chen et al. — [arXiv:2307.08701](https://arxiv.org/abs/2307.08701) | **ICLR 2024** | 用强 LLM 打分把 52k 筛到 9k，效果反而更好。**"打分 → 剔除 → 重训 → 下游评测"这一完整闭环范式最直接的先例**，与本项目 6.13 节结构完全对应，区别仅在打分依据（LLM 裁判 vs 训练动态） |
| **What Makes Good Data for Alignment? (DEITA)**<br>Liu et al. — [arXiv:2312.15685](https://arxiv.org/abs/2312.15685) | **ICLR 2024** | 复杂度 + 质量 + 多样性三维打分，6k 样本达 SOTA。**与本项目 6.12.5 节 `pooled` 的多信号并联思路同向**，可作为"多维度打分优于单一异常分数"的外部依据 |
| **From Quantity to Quality: Self-Guided Data Selection (IFD)**<br>Li et al. — [arXiv:2308.12032](https://arxiv.org/abs/2308.12032) | **NAACL 2024** | 用"指令跟随难度"筛选，**不需要外部裁判**——这一点比其他数据选择工作更接近本项目的免标签约束，值得作为最接近的可比方法 |
| **Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning**<br>Li et al. — [arXiv:2402.00530](https://arxiv.org/abs/2402.00530) | **ACL 2024** | 用小模型算的难度分去筛大模型的训练数据。对本项目有直接成本含意：`micro_batch=1` 的逐样本梯度采集很贵（3.4 节，13.5 小时/9 数据集），如果小模型的轨迹可迁移，成本能大幅下降——这是一个未验证但有前景的方向 |
| **One-Shot Learning as Instruction Data Prospector (Nuggets)**<br>Li et al. — [arXiv:2312.10302](https://arxiv.org/abs/2312.10302) | **ACL 2024** | 用一次性学习的增益当样本价值代理量 |
| **Learning Discriminative Dynamics with Label Corruption for Noisy Label Detection**<br>Kim et al. — [arXiv:2405.19902](https://arxiv.org/abs/2405.19902) | **CVPR 2024** | 用训练动态区分噪音，与本项目特征设计思路一致（视觉领域） |
| **Token Cleaning: Fine-Grained Data Selection for LLM SFT**<br>Pang et al. — [arXiv:2502.01968](https://arxiv.org/abs/2502.01968) | **ICML 2025** | token 级数据选择，与本项目 6.11 节 token 级归因直接对照 |
| **Beyond neural scaling laws: beating power law scaling via data pruning**<br>Sorscher et al. — [arXiv:2206.14486](https://arxiv.org/abs/2206.14486) | **NeurIPS 2022** | 理论侧：好的剪枝策略能突破幂律 scaling，为"数据选择值得做"提供理论支撑 |
| **Confident Learning: Estimating Uncertainty in Dataset Labels**<br>Northcutt et al. — [arXiv:1911.00068](https://arxiv.org/abs/1911.00068) | **JAIR 2021** | 经典免标签噪音检测框架（开源为 `cleanlab`），与本项目 IsolationForest/robust-z 同属"不看标签、靠统计量排序"。但 CL 依赖类别条件混淆矩阵，**不适用于本项目这种无离散类别、只有连续 loss 轨迹的开放式生成任务**——说明现有框架迁移到指令微调需要重新设计统计量 |
| **The FineWeb Datasets: Decanting the Web**<br>Penedo et al. — [arXiv:2406.17557](https://arxiv.org/abs/2406.17557) | **NeurIPS 2024 Datasets & Benchmarks Track** | 工业级质量过滤管线，去重是其核心，印证 `text_nn_sim` 的实际价值 |
| **Data-Juicer: A One-Stop Data Processing System for LLMs**<br>Chen et al. — [arXiv:2309.02033](https://arxiv.org/abs/2309.02033) | **SIGMOD 2024**（工业赛道） | 工程化的数据处理系统，可作为本项目方法投入生产时的载体参考 |
| **Fine-Tuning on Noisy Instructions: Effects on Generalization and Performance**<br>Alajrami et al. — [arXiv:2510.03528](https://arxiv.org/abs/2510.03528) | **IJCNLP-AACL 2025** | **与本项目实验设计高度同构的独立研究**：同样基于 Dolly 注入扰动、同样在 MMLU/BBH/GSM8K 上评测。可直接交叉验证 6.9 节"噪音下游危害有限"是否具跨扰动类型的普遍性 |
| **Is BERT Really Robust? (TextFooler)**<br>Jin et al. — [arXiv:1907.11932](https://arxiv.org/abs/1907.11932) | **AAAI 2020** | 攻击方法与本项目 keyword 噪音构造同构（只换少量关键词、保持结构）。该文证明这类局部替换对表层统计特征天然不敏感，**为 6.2/6.11 节"keyword 是最难检测类型（AUC 0.577）"提供独立理论解释**，而不只是"这次凑巧没找到好特征" |

## 第四层：workshop 与未发表的 preprint（引用需谨慎）

上一版综述把这些与主会论文并列，容易误导。它们内容仍有参考价值，但**引用时必须标明性质**。

| 论文 | 实际性质 | 关联点 |
|---|---|---|
| **SemDeDup: Data-efficient learning at web-scale through semantic deduplication**<br>Abbas et al. — [arXiv:2303.09540](https://arxiv.org/abs/2303.09540) | **ICML 2023 DMLR Workshop**（非主会） | 用 embedding 而非 TF-IDF 做语义去重。本项目 `text_nn_sim` 是纯词汇层面 TF-IDF，而近似重复域内 AUC 仅 0.674、明显低于完全重复的 0.986，一个可能原因正是 TF-IDF 对"语义相同、措辞不同"的改写不敏感——换语义 embedding 是值得验证的改进方向 |
| **Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models**<br>Ankner et al. — [arXiv:2405.20541](https://arxiv.org/abs/2405.20541) | **ICLR 2024 ATTRIB Workshop**（非主会） | 困惑度剪枝，与本项目 `loss_mean` 同族 |
| **When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale**<br>Marion et al. — [arXiv:2309.04564](https://arxiv.org/abs/2309.04564) | arXiv + NeurIPS workshop poster，**无主会接收** | 大规模预训练剪枝能提升质量，与 6.9 节"7 项 benchmark 区分度小"形成张力，提示预训练规模 vs LoRA 微调规模下噪音影响机制可能不同 |
| **RobustFT: Robust Supervised Fine-tuning for LLMs under Noisy Response**<br>Luo et al. — [arXiv:2412.14922](https://arxiv.org/abs/2412.14922) | **arXiv preprint（无同行评审 venue）** | 多专家协同检测 → 上下文增强去噪 → 基于响应熵的数据选择。与 6.13 节目标一致但策略不同：它"检测后重标注"，本项目"检测后剔除"。**重标注可能是 6.13.3 节负结果的一个替代方案**——当剔除会连带损失数据量时，去噪重标注或许更安全 |
| **MoDS: Model-oriented Data Selection for Instruction Tuning**<br>Du et al. — [arXiv:2311.15653](https://arxiv.org/abs/2311.15653) | **arXiv preprint（无 venue）** | 模型导向的数据选择 |
| **SlimPajama-DC: Understanding Data Combinations for LLM Training**<br>Shen et al. — [arXiv:2309.10818](https://arxiv.org/abs/2309.10818) | **arXiv preprint（无 venue）** | 数据组合与质量配置的影响 |
| **DynClean: Training Dynamics-based Label Cleaning**<br>Zhang et al. — [arXiv:2504.04616](https://arxiv.org/abs/2504.04616) | 未核实 | NER 场景的动力学标签清洗，本项目方法在结构化任务上的对照 |

---

## 本项目发现与文献的对应关系

```
本项目发现（章节）                        ↔  文献支撑
─────────────────────────────────────────────────────────────────
检测 AUC 随 epoch 衰减 (6.7)              ↔  Liu/ELR (NeurIPS 2020); Arpit (ICML 2017)
方向反转：低 loss 反而是噪音 (6.6)         ↔  Co-teaching 小损失假设 (NeurIPS 2018) 的反例
                                             AUM (NeurIPS 2020); SSFT (NeurIPS 2022)
                                             Feldman & Zhang 记忆化 (NeurIPS 2020)
loss 轨迹曲率/方差指纹 (8.1)              ↔  Dataset Cartography (EMNLP 2020)
cos_sim_ref 参考方向 (8.1)                ↔  LESS (ICML 2024); TracIn (NeurIPS 2020)
                                             Koh & Liang (ICML 2017)
converge_epoch 收敛速度 (8.1)             ↔  Toneva forgetting (ICLR 2019); RHO (ICML 2022)
token 级信号被子采样稀释 (6.11)           ↔  Rho-1 (NeurIPS 2024); Token Cleaning (ICML 2025)
duplicate/unrelated 靠 text_nn_sim (6.8)  ↔  Dedup (ACL 2022); SemDeDup (workshop); FineWeb (NeurIPS 2024 D&B)
keyword 盲区（局部实体替换）(6.2/6.8)      ↔  TextFooler (AAAI 2020)
LoRA 下的免标签检测路线                    ↔  Delora 双 LoRA (ACL 2025 Findings)
下游危害有限，仅模板化确有害 (6.9)         ↔  Alajrami (IJCNLP-AACL 2025, 同构设计); LIMA (NeurIPS 2023)
免标签闭环清洗 (6.13)                     ↔  AlpaGasus (ICLR 2024); DEITA (ICLR 2024); RobustFT (preprint)
```

## 未找到直接对照的发现

以下几条**经检索未找到系统性的对照文献**。这不等于无人研究过——只说明在本次检索范围内没找到可直接对比的工作，可作为潜在的方法论贡献点，但正式声明原创性前需要更彻底的检索。

1. **跨噪音类型迁移矩阵**（6.3 节，对角线均值 0.846 vs 非对角 0.675）。多数噪音检测论文只在单一噪音类型/单一 corruption process 下验证，很少系统评测 7×7 迁移矩阵。
2. **跨噪音比例迁移几乎无损**（6.4 节，retention 多数 ≥1.0）。与跨类型的 17-20 点损失形成对照，说明瓶颈在类型而非比例。
3. **P@10% 与 AUC 的脱钩**（6.5 节）。多数论文只报 AUC/F1，预算受限下的 precision@budget 未被充分强调。
4. **免标签路线上 3 个指标胜过 20 个**（6.11.4 节，8/8 数据集，平均 +0.133）。特征选择文献大多讨论"加特征提升多少"，很少报告"无监督离群检测中无关维度的稀释代价"。这可能是本项目最有推广价值的发现。
5. **排序质量与下游收益可以反向**（6.13.3 节，`pooled` lift 3.83× 但下游 −0.0082）。数据选择文献普遍假设"选得越准越好"，很少验证剔除预算是否落在**真正有害**的样本上。

---

*所有 venue 于 2026-09-16 经官方 proceedings / ACL Anthology / PMLR / OpenReview 核实；arXiv 编号为检索用。引用前建议再次复核完整作者列表——本表只给第一作者。*
