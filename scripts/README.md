# Scripts

脚本按实验生命周期分为四个阶段。所有命令都应从仓库根目录执行；大模型训练与评估请放在现有 `noisedetect` tmux 会话中运行。

## 阶段

| 目录 | 作用 | 入口 |
|---|---|---|
| `1_data/` | 构造带标签的噪音数据集 | `make_noise.py` |
| `2_train/` | LoRA 训练、benchmark 评估、旧诊断修复 | `train.py`, `evaluate.py`, `recompute_diag.py` |
| `3_analysis/` | 监督/无监督检测、迁移、token/IFD 分析 | `analyze_detection.py` 等 |
| `4_reports/` | 汇总跨实验结果并生成报告表 | `generate_report_tables.py`, `compare_ratios.py` |

## 常用命令

```bash
python scripts/1_data/make_noise.py --tag ratio05 --ratio 0.05
python scripts/2_train/train.py --tag ratio05 --dataset garbled
python scripts/2_train/evaluate.py --tag ratio05 --dataset garbled
python scripts/3_analysis/analyze_detection.py --tag ratio05
python scripts/4_reports/generate_report_tables.py > docs/report_tables.md
```

完整流程使用 `workflows/run_full_experiment.sh`；训练和评估阶段支持续跑，适合中断后继续：

```bash
bash workflows/run_full_experiment.sh ratio05 garbled,duplicate,unrelated,keyword
bash workflows/run_full_experiment.sh extra10 template,truncation,near_duplicate
```

## 分析脚本

- `analyze_detection.py`: 监督式 LR/RF、单变量 AUC、轨迹与评估汇总
- `analyze_unsupervised.py`: Isolation Forest、马氏距离、z-score
- `analyze_memorization.py`: 面向“过度典型”记忆噪音的带符号规则
- `analyze_transfer.py`: 跨比例/跨类型迁移
- `analyze_early_detection.py`: 按训练 epoch 的早期检测
- `analyze_token_concentration.py`: token loss 集中度
- `analyze_token_level.py`: 少量样本的 token 级归因（需要 GPU）
- `compute_ifd.py`: IFD 后处理（需要 GPU）
- `analyze_all_features.py`: 全量特征探索
- `natural_signal_validation.py`: 自然数据外部验证（需要 GPU）
