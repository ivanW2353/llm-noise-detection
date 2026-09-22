## 6. 实验过程与结果：索引页

本章按三条贯穿全项目的研究主线组织为三个文件；本文件承担导语、共享基础设施说明和导航的角色，正文见下列三个文件。

机制性发现（如方向反转陷阱、`text_nn_sim` 主导）归入回答"指标特征是什么"的主线2；由此发现驱动的检测器设计与落地决策归入回答"能否分离"的主线3。闭环清洗实验（06c 第 6c.5/6c.6 节）的完整正文置于主线3（实验设计目的是验证清洗可行性），主线1仅引用其中"下游危害是否存在/多大"的数字。

### 三条主线

1. **[06a. 下游危害排序](06a-harm-ranking.md)（主线1）**：什么样的噪音类型对训练效果的影响最大？——从"能不能检测"退一步，先问"值不值得检测"。
2. **[06b. 各噪音类型的指标特征](06b-feature-signatures.md)（主线2）**：各噪音类型都有什么样的指标特征，训练动态/文本特征如何随类型变化？——本章篇幅最大的部分，包含检测难度全景、跨类型/跨比例迁移、方向反转陷阱、特征归因、外部基线对比、天然噪音的长度混淆等核心机制性发现。
3. **[06c. 能否免标签分离噪音样本](06c-detectability.md)（主线3）**：依据这些指标特征，能不能免标签把噪音样本分离出来，精度如何，清洗后训练是否真的变好？——从排序质量（AUC/lift）推进到真实清洗动作（闭环重训+下游评测）。

阅读时有三个贯穿三个文件的口径提醒（详见第 3.5 节），在三个文件中都会反复出现：

- **有监督 vs 免标签**：06b 第 6b.1、6b.2 节用随机森林 + 真实标签，衡量的是**信号上限**；其余各节的 `iforest` / `memo_signed` / `pooled` 完全不用标签，衡量的是**生产可达**。同一类噪音在两个口径下的差距可能极大（模板化 0.999 vs 0.522），这个差距本身就是 06b 第 6b.4 节的主题。
- **全量 vs 诊断子采样**：多数 AUC 在约 900-1200 行的诊断子采样上计算（37 特征），而闭环清洗在全量 14,611+ 行上进行（19 特征）。前者偏乐观，尤其对 near_duplicate。
- **排序质量 vs 下游收益**：06b/06c 大部分小节测的都是"能不能把噪音排到前面"，只有 06c 第 6c.5/6c.6 节真的剔除样本并重新训练。两者可以脱钩，这是 06c 最后几节的核心发现。

### 6.1 实验设置与产出文件（三条主线共享的基础设施）

第 4 章已经写清了流程与命令，这里只归纳读结果时需要随手查的三件事。

**数据规模。** 每个数据集 14,611 条训练样本（`duplicate` 因追加副本为 16,072 条），留出集 400 条（200 条算参考梯度、200 条监控 held-out loss，9 个数据集共用）。噪音比例 `dolly-ratio10` = 10%、`dolly-ratio5` = 5%，两个比例各 9 个数据集全部跑完。

**每节用的数据来源。**

| 产出文件 | 由哪条命令生成 | 被哪些节引用 |
|---|---|---|
| `results/{tag}/per_sample_metrics.csv` | `analyze --kind features` | 全部后续分析的输入 |
| `results/{tag}/cross_type.csv` | `analyze --kind cross_type` | 06b 6b.1（对角线）、6b.2（非对角线） |
| `results/{tag}/unsupervised.csv` | `analyze --kind unsupervised` | 06c 6c.1、06b 6b.4 |
| `results/{tag}/precision_lift.csv` | `analyze --kind precision_lift` | 06c 6c.1 |
| `results/{tag}/memorization.csv` | `analyze --kind memorization` | 06b 6b.4 |
| `results/{tag}/early_*.csv` | `analyze --kind early_*` | 06b 6b.5 |
| `results/{tag}/feature_attribution.csv` | `analyze --kind feature_attribution` | 06b 6b.6 |
| `results/eval/eval_{tag}_{dataset}.json` | `evaluate` | 06a 6a.1、06c 6c.5 |
| `results/{tag}/feature_ablation.csv` | `analyze.py::feature_group_ablation()` | 06c 6c.3.1-6c.3.2 |
| `results/{tag}/feature_correlation.csv` | `analyze.py::feature_correlation()` | 06b 6b.7 |
| `results/{tag}/single_feature_ablation.csv` | `analyze.py::single_feature_ablation()` | 06b 6b.8 |
| `results/{tag}/minimal_feature_set.csv` | `analyze.py::minimal_feature_set()` | 06c 6c.3.3 |
| `results/{tag}/transfer_to_mixed.csv` | `analyze.py::transfer_to_mixed()` | 06c 6c.4 |
| `results/{tag}/pooled_scorer_compare.csv` | `analyze.py::pooled_scorer_compare()` | 06c 6c.4.5 |
| `datasets/{tag}/cleaning_loop/{name}/metadata.json` | `clean` | 06a/06c 6c.5 |
| `results/{tag}/external_baselines.csv` | `baselines.py`（不纳入 Git） | 06b 6b.9 |
| `results/oasst-wild/length_confound.csv` | `analyze.py::length_confound()` | 06b 6b.10 |

**下游评测口径。** 7 项 benchmark：MMLU（n=14042）、GSM8K（1319）、HellaSwag（10042）、ARC（1172）、BBH（540）、TruthfulQA（817）、WinoGrande（1267）。报告中的"7 项平均"是这 7 个准确率的**无权重算术平均**，不按样本量加权——所以 n=540 的 BBH 与 n=14042 的 MMLU 权重相同，单项波动会被放大，06a 第 6a.1 节与 06c 第 6c.5 节因此都同时给出分项表而非只给平均值。

除 06c 第 6c.5/6c.6 节外，三个文件里所有分析都是在既有轨迹上重算，不需要重新训练；6c.5/6c.6 节是全报告唯一实际改动训练数据并重新训练的实验。

### 导航表

| 文件 | 主线 | 核心问题 | 主要小节 |
|---|---|---|---|
| [06a-harm-ranking.md](06a-harm-ranking.md) | 主线1：危害排序 | 什么噪音对训练效果影响最大 | 6a.1、6a.2.1-6a.2.4 |
| [06b-feature-signatures.md](06b-feature-signatures.md) | 主线2：指标特征 | 各噪音类型的训练动态/文本特征长什么样 | 6b.1（含 6b.1.1）、6b.2、6b.3、6b.4、6b.5、6b.6、6b.7、6b.8、6b.9、6b.10 |
| [06c-detectability.md](06c-detectability.md) | 主线3：可分离性 | 能否免标签分离，精度如何，清洗是否有实际收益 | 6c.1、6c.2、6c.3.1-6c.3.4、6c.4.1-6c.4.5、6c.5.1-6c.5.3、6c.6 |
