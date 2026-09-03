# NoiseDetect

一个按领域组织的 LLM 噪声实验项目。实验数据位于 `data/`，结果位于 `results/`，报告位于 `docs/`；代码不依赖旧的编号脚本结构。

## 目录

```
.
├── settings.py              配置读取和路径
├── datasets/                数据格式、噪声变换、校验
├── models/                  mock 与 HuggingFace/LoRA 后端
├── training/                训练编排
├── evaluation/              评测结果持久化
├── analysis/                检测分析
└── cli.py                   唯一命令入口
run.py              CLI 启动器
run_experiment.sh             完整流程
```

## 快速开始

```bash
python cli.py --help
python cli.py data --source /path/to/train.jsonl --tag ratio05
python cli.py train --tag ratio05 --dataset clean --model mock
python cli.py evaluate --tag ratio05 --dataset clean
python cli.py analyze --tag ratio05 --input results/ratio05/per_sample_metrics.csv
```

完整流程：

```bash
bash run_experiment.sh ratio05 clean,garbled,duplicate,unrelated,keyword,mixed /path/to/train.jsonl
```

`mock` 后端用于接口和 CPU 检查；真实 LoRA 训练使用 `--model hf-lora`，按需安装 `torch`、`transformers`、`peft`。更换模型只需新增 `models/` 后端，更换数据集只需实现 `Provider.rows()`，评测任务在 `evaluation/` 中扩展。

## 派生实验数据

清洗增益实验已归档到 `data/ratio10/cleaning_gain/unrelated/`，与主数据按实验标签统一管理；其汇总结果仍在 `results/cleaning_gain_comparison.csv`。

## 数据契约

每行 JSONL 必须包含 `sample_id`、`messages`、`noise_type`。详见 `data/README.md`。已有实验结果和报告不会被 CLI 覆盖。
