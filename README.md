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
├── analyze.py               训练/token/无监督/迁移分析
├── cli.py                   统一命令入口
├── run.py                   CLI 启动器
└── run_experiment.sh        完整流程
```

## 快速开始

```bash
python cli.py --help
python cli.py data --source /path/to/train.jsonl --tag ratio05
python cli.py train --tag ratio05 --dataset clean --model mock
python cli.py evaluate --tag ratio05 --dataset clean
python cli.py analyze --tag ratio05 --input results/ratio05/per_sample_metrics.csv

# 训练过程指标（loss、梯度范数、cosine、update contribution）
python cli.py analyze --kind training --tag ratio10
# token 级 hard-token 统计
python cli.py analyze --kind token --tag ratio10              # 自动汇总该 tag 全部 token 文件
python cli.py analyze --kind token --tag ratio10 --dataset garbled  # 单一数据集
# 无标签 IsolationForest / robust-z 检测
python cli.py analyze --kind unsupervised --tag ratio10
# 跨比例或跨类型迁移
python cli.py analyze --kind transfer --input results/transfer_cross_ratio.csv --tags ratio05,ratio10
```

完整流程：

```bash
bash run_experiment.sh ratio05 clean,garbled,duplicate,unrelated,keyword,mixed /path/to/train.jsonl
```

`mock` 后端用于接口和 CPU 检查；真实 LoRA 训练使用 `--model hf-lora`，按需安装 `torch`、`transformers`、`peft`。更换模型在 `model.py` 增加后端，更换数据集在 `data.py` 增加 `Provider.rows()` 实现，评测任务在 `evaluate.py` 扩展。

分析入口统一为 `analyze.py`/`cli.py`，不需要额外的分析脚本：

- `training`：按 epoch 聚合 loss、梯度范数、cosine、update contribution 和 token 数。
- `token`：按噪音类型聚合 hard-token 损失、梯度、位置稳定性等指标。
- `unsupervised`：使用 robust-z 与 IsolationForest，输出 AUC、P@10% 和随机基线。
- `transfer`：读取跨比例/跨类型迁移结果，支持 `--tags` 筛选。

## 派生实验数据

清洗增益实验已归档到 `data/ratio10/cleaning_gain/unrelated/`，与主数据按实验标签统一管理；其汇总结果仍在 `results/cleaning_gain_comparison.csv`。

## 数据契约

每行 JSONL 必须包含 `sample_id`、`messages`、`noise_type`。详见 `data/README.md`。已有实验结果和报告不会被 CLI 覆盖。
