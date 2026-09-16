# 存储布局

主盘 `/`（30 GB）空间紧张，大块二进制产物放在 `/root/autodl-tmp`（50 GB），在仓库里以软链接接入，**路径与搬移前一致**，因此 `evaluate.py`、`cli.py` 等不需要改动。

## 搬到大盘的内容

| 仓库路径 | 实际位置 | 大小 | 为什么可以搬 |
|---|---|---|---|
| `runs/{tag}/{dataset}/lora` | `/root/autodl-tmp/noisedetect_runs/runs/{tag}/{dataset}/lora` | 6.3 GB（25 个适配器 × 251 MB） | 已被 `.gitignore` 排除，从不入库；`evaluate.py:34` 按 `runs_dir()/dataset/lora` 读取，软链接照常解析 |
| `data/wild_oasst` | `/root/autodl-tmp/noisedetect_data/wild_oasst` | 126 MB | 天然噪音数据集，可由 `wild_data.py` 从 OASST 重建 |
| HuggingFace 缓存（OASST） | `/root/autodl-tmp/hf` | 365 MB | 通过 `HF_HOME=/root/autodl-tmp/hf` 指定 |

搬移用"先复制 → 校验 md5 → 再删除原件 → 建软链接"的顺序，不先删后拷。

## 留在主盘的内容

- `runs/{tag}/{dataset}/metrics/*.jsonl`（除 `layer_norms.jsonl`）——**git 跟踪**，是全部分析的输入，不能搬。
- `results/`、`docs/`、`data/{tag}/`——体积小且大部分入库。

## 保留在本地但不再入库

`runs/{tag}/{dataset}/tb/`（457 MB，32 个文件）原先有 30 个被 git 跟踪，现已从索引移除并加入 `.gitignore`，**本地文件全部保留**。移除理由：

- `model.py:223` 只写不读，全项目没有任何分析消费它；
- 内容是 `per_sample.jsonl` 同一批数据的降采样编码，要复现图表从后者重算更准；
- 文件名嵌入了产出机器的容器 ID 与进程号（`events.out.tfevents.<ts>.autodl-container-<id>.<pid>.0`），对其他人零价值。ratio10 多数数据集有 2 个 events 文件，因为在两台容器上各跑过一次。

## 注意事项

**`runs/**/lora` 在 `.gitignore` 里有两条规则。** 带斜杠的 `runs/**/lora/` 只匹配目录，软链接不是目录，所以额外加了不带斜杠的一条。少了它，25 个软链接会以未跟踪文件的形式出现，可能被误提交成指向本机绝对路径的死链。

**换机器要重建软链接。** 链接内容是本机绝对路径。适配器可以由对应数据集重新训练得到，或从备份复制；`data/wild_oasst` 用 `wild_data.py` 重建。

**`layer_norms.jsonl`（198 MB）** 已被 gitignore 且全项目无任何代码读取（见附录 8.4），可以直接删除以再腾空间，此处保留仅为人工在 TensorBoard 里查看。
