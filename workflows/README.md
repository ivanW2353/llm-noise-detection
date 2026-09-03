# 实验工作流

`run_full_experiment.sh` 是唯一的完整实验入口，按“数据生成 → 训练 → 评估 → 分析 → 报告”顺序执行。所有命令从仓库根目录运行。

## 完整流程

```bash
# 默认四类噪音
bash workflows/run_full_experiment.sh ratio10

# 指定噪音类型；包含扩展类型时自动启用 --with-extra
bash workflows/run_full_experiment.sh ratio05 garbled,duplicate,unrelated,keyword
bash workflows/run_full_experiment.sh extra10 template,truncation,near_duplicate
```

脚本会按 tag 写入 `data/<tag>/`、`runs/<tag>/` 和 `results/<tag>/`，并复用已经完成的阶段。第二个参数只控制需要训练和评估的类型；数据生成器仍会写出该实验所支持的完整数据集清单。

## 分阶段执行

```bash
python scripts/1_data/make_noise.py --tag ratio10
python scripts/2_train/train.py --tag ratio10 --dataset clean
python scripts/2_train/evaluate.py --tag ratio10 --dataset clean
python scripts/3_analysis/analyze_detection.py --tag ratio10
python scripts/4_reports/generate_report_tables.py > docs/report_tables.md
```

更多分析入口和用途见 [`scripts/README.md`](../scripts/README.md)。
