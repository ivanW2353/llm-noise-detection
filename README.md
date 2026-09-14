# NoiseDetect

一个按领域组织的 LLM 噪声实验项目。实验数据位于 `data/`，结果位于 `results/`，报告位于 `docs/`；所有运行代码都在根目录，避免多层命令/工作流目录。

## 目录

```
.
├── settings.py              配置读取和路径
├── data.py                  JSONL 数据读写与噪声变换
├── model.py                 mock 与 HuggingFace/LoRA 后端
├── train.py                 训练编排与逐样本指标
├── evaluate.py              评测结果持久化
├── textsim.py               文本层面的最近邻相似度特征（TF-IDF，非训练动态）
├── analyze.py               训练/token/无监督/迁移/早期检测/特征归因分析
├── cleaning_loop.py          免标签闭环清洗（打分剔除 + 随机剔除对照）
├── cli.py                   统一命令入口
├── run.py                   CLI 启动器
└── scripts/                 编排脚本（完整流程、评测、分析），不纳入 git 跟踪，见下方说明
```

`scripts/` 只放**编排**代码（跑什么、按什么顺序跑，含排队等待 GPU 的轮询逻辑）；检测器/评分规则/特征
这类实验方法本身一律放根目录并纳入 git 跟踪，通过 `cli.py` 的子命令/`--kind` 暴露。

`scripts/` 目录：

- `run_full.sh <tag> [ratio]`：当前通用入口——数据生成、9 类数据集 LoRA 训练、全量分析表（features/training/unsupervised/cross_type/precision_lift/memorization）一次跑完。
- `run_full_ratio10.sh`：ratio10 实验的历史执行记录（已完成，保留用于复现；未包含后续新增的三项分析，新实验请用 `run_full.sh`）。
- `run_eval_ratio5.sh` / `run_analysis_ratio5.sh`：ratio5 训练完成后待执行的下游 benchmark 评测与后处理分析脚本。

## 快速开始

```bash
python cli.py --help
python cli.py data --source /path/to/train.jsonl --tag ratio10
python cli.py train --tag ratio10 --dataset clean --model mock
python cli.py evaluate --tag ratio10 --dataset clean
python cli.py analyze --tag ratio10 --input results/ratio10/per_sample_metrics.csv

# 训练过程指标（loss、梯度范数、cosine、update contribution）
python cli.py analyze --kind training --tag ratio10
# token 级 hard-token 统计
python cli.py analyze --kind token --tag ratio10              # 自动汇总该 tag 全部 token 文件
python cli.py analyze --kind token --tag ratio10 --dataset garbled  # 单一数据集
# 无标签 IsolationForest / robust-z 检测
python cli.py analyze --kind unsupervised --tag ratio10
# 带符号记忆性规则（duplicate/template 等"超典型"噪音）
python cli.py analyze --kind memorization --tag ratio10
# 跨噪音类型 / 跨噪音比例 检测器迁移矩阵
python cli.py analyze --kind cross_type --tag ratio10
python cli.py analyze --kind cross_ratio --tags ratio10,ratio5
# P@10% 清洗精度 lift（比 AUC 更贴近实际清洗预算下的可用性）
python cli.py analyze --kind precision_lift --tag ratio10
# 早期检测：按训练 epoch 截断重算检测 AUC，看多早能拿到可用信号
python cli.py analyze --kind early_unsupervised --tag ratio10
python cli.py analyze --kind early_memorization --tag ratio10
# 特征归因：每个噪音类型的 RF 检测器到底在用哪些特征（permutation importance）
python cli.py analyze --kind feature_attribution --tag ratio10
# 免标签闭环清洗：无监督打分剔除疑似噪音 + 等量随机剔除对照，供重训练对比
python cli.py clean --tag ratio10 --dataset garbled --budget 0.10
# 读取已保存的迁移结果
python cli.py analyze --kind transfer --input results/transfer_cross_ratio.csv --tags ratio5,ratio10
```

完整流程：

```bash
bash scripts/run_full.sh ratio10        # ratio=0.10（默认）
bash scripts/run_full.sh ratio5 0.05    # 自定义噪音比例
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

`clean`（`cli.py clean --tag <tag> --dataset <ds> --budget <frac>`）：免标签闭环清洗——用
IsolationForest 在全量训练集上打分，剔除疑似噪音最多的 `budget` 比例，另建等量随机剔除对照组，
产出 `data/{tag}/cleaning_loop/{dataset}/{train_targeted,train_random}.jsonl` 供重新训练+评测对比。

## 派生实验数据

清洗增益实验的原始数据仍归档在 `data/ratio10/cleaning_gain/unrelated/`（`train_random.jsonl`/`train_targeted.jsonl` 及对应的 `training_commands.sh`）。其训练产物目录 `runs/ratio10/cleaning_gain/` 及汇总表 `results/cleaning_gain_comparison.csv` 已在后续清理中删除；若需要该项对比结论，需重新执行 `data/ratio10/cleaning_gain/unrelated/training_commands.sh` 并重新汇总。

## 数据契约

每行 JSONL 必须包含 `sample_id`、`messages`、`noise_type`。详见 `data/README.md`。已有实验结果和报告不会被 CLI 覆盖。
