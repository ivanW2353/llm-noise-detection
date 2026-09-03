# Cleaning-gain experiments

`ratio10` 的派生数据清洗实验统一存放于此：

- `garbled/random/train.jsonl`：随机删除对照组
- `garbled/scored/train.jsonl`：按检测分数清洗组
- `garbled/heldout.jsonl`：garbled 清洗实验验证集
- `unrelated/train_random.jsonl`：随机删除对照组
- `unrelated/train_targeted.jsonl`：检测器定向清洗组
- `unrelated/sample_scores.csv`：逐样本检测分数
- `unrelated/detector_info.json`：检测器参数与特征
- `unrelated/metadata.json`：实验配置和样本统计

模型权重仍位于 `runs/ratio10/cleaning_gain/`，汇总结果位于 `results/cleaning_gain_comparison.csv`。
