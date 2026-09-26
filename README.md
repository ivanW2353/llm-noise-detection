# NoiseDetect

一个按领域组织的 LLM 噪声实验项目。实验数据位于 `datasets/`，结果位于 `results/`，报告位于 `docs/`；所有运行代码都在根目录，避免多层命令/工作流目录。

研究的核心问题：**在完全不使用噪音标签的前提下，能否仅凭 LoRA 微调过程中的训练动态（loss 轨迹、
梯度范数、余弦相似度等）识别出被注入的低质量训练样本？**

## 结论与报告

完整分析报告按章节分文件：[`docs/report_zh/`](docs/report_zh/README.md)（英文版
[`docs/report_en/`](docs/report_en/README.md)，两版结构与内容同步）。报告结构为
相关工作 → 实验前提 → 实验方案 → 理论分析 → 实验过程与结果 → 结论与 future work → 附录。

另有一套**按数据集分述的独立报告**：[`docs/reports_by_dataset/`](docs/reports_by_dataset/README.md)
（英文版 [`docs/reports_by_dataset_en/`](docs/reports_by_dataset_en/README.md)）。四个 tag
各一份，每份都可以单独读完，说明该组实验的设计、观测，以及它单独能支撑与不能支撑的结论。
与上面的报告同源、只是切面不同：上面按研究问题横向比较，这里按实验纵向分述。

想快速了解做了什么、怎么做的，看第 4 章（实验方案）；想知道为什么这么设计，看第 5 章（理论分析）；
想直接看结论，看第 7 章。原始数据展示（噪音文本对照、单样本特征值、原始 loss 曲线）都在第 8 章附录。
几个值得先知道的结果：

- **检测难度因噪音类型差异极大**，且排序稳定：乱码/模板化/完全重复 AUC > 0.98，关键词替换/近似重复
  只有 0.57-0.76。
- **方向反转是真实的方法论坑**：模板化这类"被记忆"的噪音 loss 更低、收敛更快，通用离群检测会把它们
  排到最干净的一端。用错方向的清洗精度（4.0%）比随机剔除（9.2%）还差。
- **检测器是"窄"，不是"垮"**：单类型检测器拉到混合噪音流上整体 AUC 掉到 0.563-0.730，但限定
  "自己那一类 vs 干净"后保持率 0.96-1.23——是覆盖缺口，不是分布失效。
- **清洗有收益需要三个条件同时成立**：噪音确有下游危害、打分器方向正确、且剔除预算落在有害的那部分
  噪音上。模板化上三条齐备，回收了 GSM8K 缺口的 70%；garbled 上精度达随机 5.7 倍却零收益（本身无害）；
  混合噪音上排序最好的 `pooled` 下游反而为负，因为约四分之一预算花在了无害的 duplicate/garbled 上。
- **免标签路线上 3 个指标胜过全部 19 个**，8 个数据集无一例外，平均高 0.133（template 0.537→0.875）。
  当前把 19 个特征全喂给 IsolationForest 在系统性自我拖累——"方向反转"有一部分其实是维度稀释。
- **两类噪音的高 AUC 名不副实**：完全重复、话题不相关的检测信号 90% 以上来自静态文本相似度，
  不是训练动态。

## 项目组织

代码分两层：三个子包放具体实现，根目录放**入口 + facade**——`cli.py`/`textsim.py`/`cleaning_loop.py`
等不方便挪的调用方直接 import 根目录文件，所以 facade 只做转发（`from xxx import *`），这些 import
路径从未因重构而改变。

```
.
├── settings.py              配置读取和路径（load()、data_dir()/runs_dir()/results_dir()）
├── data.py                  facade → datalib/
├── model.py                 facade：Mock 后端 + create() 工厂在此；真正的 hf-lora 后端在 lora/lora_train.py
├── train.py                 围绕 model.create() 的训练编排（Trainer）
├── evaluate.py              下游 benchmark 评测（mmlu/gsm8k/hellaswag/arc/bbh/truthfulqa/winogrande），逐任务可续跑
├── textsim.py               text_nn_sim()：TF-IDF 最近邻相似度，数据层特征而非训练动态特征
├── analyze.py               facade → analysis/
├── cleaning_loop.py         build()：免标签闭环清洗（三种打分器 + 等量随机剔除对照，全量训练集）
├── wild_data.py             从 OASST2 构建天然噪音数据集，未接入 cli.py，需直接运行
├── cli.py                   data/train/evaluate/analyze/clean 五个子命令的唯一入口
├── run.py                   CLI 启动器，只调用 cli.main()
│
├── datalib/                 sample.py（Sample/Provider）、data_io.py（Jsonl 读写、load_rows() 含 hf://、validate()）、
│                             data_split.py（split_holdout()/reindex()/split_fractions()）、noise.py（噪声注入 apply()/TRANSFORMS/NOISE_TYPES）
├── lora/                    lora_train.py（真正的 hf-lora 后端，LoRA 类；fit() 拆成 _setup_run/_run_epoch/_finalize_epoch 等阶段函数，
│                             按 epoch 边界写 checkpoint/ 并自动续训）、lora_internals.py（逐样本梯度/loss/cos-sim 辅助函数）
├── analysis/                 metrics_common.py（auc()/precision_at_k() 等底层原语、FULL_COVERAGE_FEATS/MEMO_FEATS）、
│                             metrics_table.py（build_table() 组装 per_sample_metrics.csv）、
│                             detect_unsupervised.py（unsupervised_metrics()/memorization_score()/early_detection_sweep()/pooled_scorer_compare()）、
│                             detect_transfer.py（cross_type_transfer()/cross_ratio_transfer()/transfer_to_mixed()）、
│                             feature_diagnostics.py（feature_attribution()/minimal_feature_set()/single_feature_ablation()/feature_group_ablation()/length_confound()）
│
└── scripts/                 编排脚本（完整流程、评测、分析），不纳入 git 跟踪，见下方说明
```

**放置原则**：`scripts/`（已 gitignore）只放**编排**代码——跑什么、按什么顺序跑、排队等待 GPU 的轮询逻辑。
检测器/评分规则/特征这类实验方法本身一律放根目录三个子包并纳入 git 跟踪，通过 `cli.py` 的子命令/`--kind`
暴露。这条原则已经因为图省事在 `scripts/` 里堆方法代码而破坏过两次（`cleaning_loop.py`/`early_detection_sweep()`，
之后是 `feature_ablation.py`/`feature_correlation.py`/`minimal_feature_set.py`/`pooled_scorer_compare.py`/
`single_feature_ablation.py`/`transfer_to_mixed.py`），最终都搬进了 `analyze.py`（对应 `feature_group_ablation()`/
`feature_correlation()`/`minimal_feature_set()`/`pooled_scorer_compare()`/`single_feature_ablation()`/
`transfer_to_mixed()`），每次都先跟旧脚本输出逐字节比对再删旧文件。`make_report_charts.py`
是唯一故意留在 `scripts/` 里不跟踪的脚本（只画图，不是方法，产出的 PNG 本身入库）。

`scripts/` 目录：

一次性分析脚本（产出纳入 git 跟踪的 CSV/PNG，脚本本身不跟踪）：

- `transfer_to_mixed.py`：把 7 个单类型检测器全部拉到 `mixed` 上评估，并做检测器并联与留一法（`cross_type` 在代码层面跳过 `mixed`，回答不了混合流的问题）→ `results/dolly-ratio10/transfer_to_mixed.csv`，见报告 6.12 节。
- `pooled_scorer_compare.py`：对比 `cleaning_loop.py` 三个打分器在 `mixed` 上的整体与逐类型表现 → `results/dolly-ratio10/pooled_scorer_compare.csv`，见报告 6.12.5 节。
- `feature_ablation.py`：类别级特征消融（`text_nn_sim` 是否被稀释、token 级诊断值多少）→ `results/dolly-ratio10/feature_ablation.csv`，见报告 6.11.2-6.11.3 节。
- `single_feature_ablation.py`：单指标留一消融，对 19 个全覆盖特征逐个删除，同时测 RF（有监督，6.2 节口径）和 IsolationForest（免标签，6.6 节口径）两条路线 → `results/dolly-ratio10/single_feature_ablation.csv`，见报告 6.11.4 节。
- `minimal_feature_set.py`：最小指标集合，贪心前向选择（从零开始逐个加入最有增益的特征）回答"最少需要哪几个指标"，留一消融回答不了这个问题 → `results/dolly-ratio10/minimal_feature_set.csv`，见报告 6.11.5 节。**主要发现：免标签路线上 3 个指标胜过全部 19 个，8/8 数据集无例外。**
- `make_report_charts.py`：生成报告全部图表 → `results/charts/`（中文）与 `results/charts/en/`（英文）。

编排脚本：

- `run_full.sh <tag> [ratio]`：当前通用入口——数据生成、9 类数据集 LoRA 训练、全量分析表（features/training/unsupervised/cross_type/precision_lift/memorization）一次跑完。
- `run_full_dolly-ratio10.sh`：dolly-ratio10 实验的历史执行记录（已完成，保留用于复现；未包含后续新增的三项分析，新实验请用 `run_full.sh`）。
- `run_eval_dolly-ratio5.sh` / `run_analysis_dolly-ratio5.sh`：dolly-ratio5 训练完成后待执行的下游 benchmark 评测与后处理分析脚本。

## 快速开始

```bash
python cli.py --help
python cli.py data --source /path/to/train.jsonl --tag dolly-ratio10
python cli.py train --tag dolly-ratio10 --dataset clean --model mock
python cli.py evaluate --tag dolly-ratio10 --dataset clean
python cli.py analyze --tag dolly-ratio10 --input results/dolly-ratio10/per_sample_metrics.csv

# 训练过程指标（loss、梯度范数、cosine、update contribution）
python cli.py analyze --kind training --tag dolly-ratio10
# token 级 hard-token 统计
python cli.py analyze --kind token --tag dolly-ratio10              # 自动汇总该 tag 全部 token 文件
python cli.py analyze --kind token --tag dolly-ratio10 --dataset garbled  # 单一数据集
# 无标签 IsolationForest / robust-z 检测
python cli.py analyze --kind unsupervised --tag dolly-ratio10
# 带符号记忆性规则（duplicate/template 等"超典型"噪音）
python cli.py analyze --kind memorization --tag dolly-ratio10
# 跨噪音类型 / 跨噪音比例 检测器迁移矩阵
python cli.py analyze --kind cross_type --tag dolly-ratio10
python cli.py analyze --kind cross_ratio --tags dolly-ratio10,dolly-ratio5
# P@10% 清洗精度 lift（比 AUC 更贴近实际清洗预算下的可用性）
python cli.py analyze --kind precision_lift --tag dolly-ratio10
# 早期检测：按训练 epoch 截断重算检测 AUC，看多早能拿到可用信号
python cli.py analyze --kind early_unsupervised --tag dolly-ratio10
python cli.py analyze --kind early_memorization --tag dolly-ratio10
# 特征归因：每个噪音类型的 RF 检测器到底在用哪些特征（permutation importance）
python cli.py analyze --kind feature_attribution --tag dolly-ratio10
# 免标签闭环清洗：无监督打分剔除疑似噪音 + 等量随机剔除对照，供重训练对比
python cli.py clean --tag dolly-ratio10 --dataset garbled --budget 0.10                      # iforest（默认）
python cli.py clean --tag dolly-ratio10 --dataset template --budget 0.10 --method memo_signed # 记忆型噪音
python cli.py clean --tag dolly-ratio10 --dataset mixed --budget 0.10 --method pooled         # 成分未知/混合
# 读取已保存的迁移结果
python cli.py analyze --kind transfer --input results/transfer_cross_ratio.csv --tags dolly-ratio5,dolly-ratio10
```

完整流程：

```bash
bash scripts/run_full.sh dolly-ratio10        # ratio=0.10（默认）
bash scripts/run_full.sh dolly-ratio5 0.05    # 自定义噪音比例
```

大规模数据可直接从 Hugging Face 读取（需要 `datasets`）：

```bash
python cli.py data --tag ultra200k --source hf://HuggingFaceH4/ultrachat_200k --ratio 0.10
```

加载器会将 `messages` 或常见的 `prompt`/`response` 字段统一为项目格式，并受 `config.yaml` 中 `data.max_samples` 限制。

`mock` 后端用于接口和 CPU 检查；真实 LoRA 训练使用 `--model hf-lora`，按需安装 `torch`、`transformers`、`peft`。更换模型在 `model.py` 增加后端，更换数据集在 `data.py` 增加 `Provider.rows()` 实现，评测任务在 `evaluate.py` 扩展。

分析入口统一为 `analyze.py`/`cli.py`，不需要额外的分析脚本：

- `training`：按 epoch 聚合 loss、梯度范数、cosine、update contribution 和 token 数。
- `token`：按噪音类型聚合 hard-token 损失、梯度、位置稳定性等指标。
- `unsupervised`：按数据集单独 fit 的 robust-z 与 IsolationForest，输出 AUC、P@10% 和随机基线。
- `memorization`：带符号超典型性规则（方向由记忆化假设先验固定，不按数据集拟合），专门捕捉
  duplicate/template 这类"损失异常低而非异常高"的记忆型噪音，无监督离群检测对这类噪音会失效甚至反向。
- `cross_type` / `cross_ratio`：检测器跨噪音类型 / 跨噪音比例的迁移矩阵，对角线为域内 5-fold CV。
- `precision_lift`：P@10% 清洗精度相对随机基线的 lift，比单看 AUC 更能反映实际清洗预算下的可用性。
- `early_unsupervised` / `early_memorization`：把训练轨迹截断到前 k 个 epoch 重算检测 AUC，判断
  某类噪音是否不需要等训练跑完就能有效识别。
- `feature_attribution`：每个噪音类型 RF 检测器的 permutation importance——回答"检测器在看哪个特征"
  而非"能不能检测"；发现 duplicate/unrelated 主要靠 `text_nn_sim`（文本相似度，非训练动态特征）。
- `transfer`：读取跨比例/跨类型迁移结果，支持 `--tags` 筛选。

`clean`（`cli.py clean --tag <tag> --dataset <ds> --budget <frac> [--method ...]`）：免标签闭环清洗——
在**全量**训练集（而非诊断子样本）上打分，剔除疑似噪音最多的 `budget` 比例，另建等量随机剔除对照组，
产出 `datasets/{tag}/cleaning_loop/{name}/{train_targeted,train_random}.jsonl` 供重新训练+评测对比。
随机对照是必须的：剔除 10% 样本本身就减少 10% 训练数据，只跟未清洗基线比无法区分"去噪收益"和"数据量损失"。

三个 `--method` 对应三种噪音假设，选错方向的代价很大（同为 template 剔 1461 条，`iforest` 命中精度
4.0%、比随机的 9.2% 还差一半，`memo_signed` 则是 53.1%）：

| `--method` | 假设 | 适用 | 特征 |
|---|---|---|---|
| `iforest`（默认） | 噪音 = 离群，无方向 | garbled / unrelated 等真正异常的噪音 | 19 个全覆盖特征 |
| `memo_signed` | 噪音 = 异常地**容易学**（loss 低、收敛快），符号先验固定 | template / duplicate 等记忆型噪音 | 6 个带符号轨迹特征 |
| `pooled` | 以上两者 + `text_nn_sim` 的 \|z\|，各自标准化后取逐样本最大值 | **噪音成分未知或混合**时 | 27 项 |

`pooled` 是"用精度换覆盖面"而非免费改进：`mixed` 上它的 P@10% 0.323 高于 `iforest` 的 0.268，
但代价是 near_duplicate 略低于随机，且剔除预算会被最显著的类型（duplicate/garbled）占据。
噪音类型已知且已校准时，单方法精度更高。详见报告 6.12.5 节。

## 派生实验数据

闭环清洗实验的产物（报告 6.13 节）：

| 目录 | 内容 |
|---|---|
| `datasets/{tag}/cleaning_loop/{name}/` | 清洗后训练集 `train_targeted.jsonl` / `train_random.jsonl` + `metadata.json`（含剔除精度、用到的特征列表） |
| `runs/{tag}/cleaning_loop_*/` | 重训练的逐样本指标 |
| `results/eval/eval_{tag}_cleaning_loop_*.json` | 重训练后的 7 项 benchmark 结果 |

`{name}` 为数据集名（`iforest` 默认方法）或 `{dataset}_{method}`（其他打分器），已跑完的有
`garbled`、`template`、`template_memo_signed`、`mixed`、`mixed_memo_signed`、`mixed_pooled`。

清洗增益实验的原始数据仍归档在 `datasets/dolly-ratio10/cleaning_gain/unrelated/`（`train_random.jsonl`/`train_targeted.jsonl` 及对应的 `training_commands.sh`）。其训练产物目录 `runs/dolly-ratio10/cleaning_gain/` 及汇总表 `results/cleaning_gain_comparison.csv` 已在后续清理中删除；若需要该项对比结论，需重新执行 `datasets/dolly-ratio10/cleaning_gain/unrelated/training_commands.sh` 并重新汇总。

## 数据契约

每行 JSONL 必须包含 `sample_id`、`messages`、`noise_type`。详见 `datasets/README.md`。已有实验结果和报告不会被 CLI 覆盖。
