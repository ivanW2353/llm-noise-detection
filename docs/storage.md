# 存储布局

主盘 `/`（30 GB）空间紧张，大块二进制产物放在 `/root/autodl-tmp`（50 GB），在仓库里以软链接接入，**路径与搬移前一致**，因此 `evaluate.py`、`cli.py` 等不需要改动。

## 搬到大盘的内容

| 仓库路径 | 实际位置 | 大小 | 为什么可以搬 |
|---|---|---|---|
| `runs/{tag}/{dataset}/lora` | `/root/autodl-tmp/noisedetect_runs/runs/{tag}/{dataset}/lora` | 6.3 GB（25 个适配器 × 251 MB） | 已被 `.gitignore` 排除，从不入库；`evaluate.py:34` 按 `runs_dir()/dataset/lora` 读取，软链接照常解析 |
| `datasets/oasst-wild` | `/root/autodl-tmp/noisedetect_data/wild_all` | 85 MB | 天然噪音数据集（OASST2，68,762 样本 + 400 留出），可由 `wild_data.py` 重建；大盘目录名沿用改名前的 `wild_all`，尚未同步改名（仓库内 symlink 名已是 `oasst-wild`） |
| HuggingFace 缓存（OASST） | `/root/autodl-tmp/hf` | 365 MB | 通过 `HF_HOME=/root/autodl-tmp/hf` 指定 |

搬移用"先复制 → 校验 md5 → 再删除原件 → 建软链接"的顺序，不先删后拷。

## 留在主盘的内容

- `runs/{tag}/{dataset}/metrics/*.jsonl`（除 `layer_norms.jsonl`）——**git 跟踪**，是全部分析的输入，不能搬。
- `results/`、`docs/`、`datasets/{tag}/`——体积小且大部分入库。

## 大文件走 Git LFS，不在 git 历史里

这一点容易误判：仓库看起来很大，但**真正的 git 对象只有约 7.8 MB**（201 个文件：代码、报告、CSV、PNG）。另外 309 个文件由 `.gitattributes` 规则化进 **Git LFS**，在 git 历史里只是一行哈希指针。

| LFS 规则 | 文件数 | 为什么值得留 |
|---|---|---|
| `runs/**/metrics/*.jsonl` | 253 | `analyze.py` 的直接输入，报告每个数字的源头 |
| `datasets/**/*.jsonl` | 31 | 定义"实验用的到底是哪批数据"，缺了无法核对结论 |
| `results/*/per_sample_metrics.csv` | 2 | 全部分析的汇总表 |

含意：**清理仓库体积不需要重写历史**。历史本身就小，体积在 LFS 缓存里，`git lfs prune` 即可回收。克隆时也可以用 `GIT_LFS_SKIP_SMUDGE=1 git clone` 只拿代码与报告，按需再拉数据。

## 保留在本地但不再入库

两类：

**`runs/{tag}/{dataset}/tb/`**（457 MB，32 个文件）原先有 30 个被跟踪，已从索引移除并加入 `.gitignore`，本地文件保留。`model.py:223` 只写不读，全项目没有任何分析消费它；内容是 `per_sample.jsonl` 同一批数据的降采样编码，复现图表从后者重算更准；文件名还嵌入了产出机器的容器 ID 与进程号，对其他人零价值。

**`results/eval/eval_raw_*.jsonl`**（100 MB，23 个文件）同样已移除。`evaluate.py:368` 写入后没有任何代码或报告读取它——那是逐题作答记录（`qid`/`correct`/`margin`），而报告引用的聚合准确率在 `eval_{tag}_{dataset}.json` 里，总共 368 KB。

## 一次性清理的记录（2026-09-17）

- `git lfs prune` 回收 1.6 GB 无引用的 LFS 历史版本（`.git` 从 2.6 GB 降到 1.2 GB）。前置条件是本地与远端同一 commit、远端持有全部对象，清理后 `git lfs fsck` 通过。
- 删除未采用的 HuggingFace 缓存：`lmsys-chat-1m`（3.9 GB）与 `10k_prompts_ranked`（8.4 MB），两者都是选定 OASST 之前探索数据源时下载的，代码中零引用。
- 清除 `__pycache__` 与 pip 缓存。

清理前用 `git bundle create ... --all` 做了完整备份（120 MB，含全部 ref 与历史），`git bundle verify` 确认完整。用 bundle 而非 tar 打包 `.git`，因为训练正在写文件、tar 会读到不一致状态（实测确实报了 `file changed as we read it`）。

## 注意事项

**`runs/**/lora` 在 `.gitignore` 里有两条规则。** 带斜杠的 `runs/**/lora/` 只匹配目录，软链接不是目录，所以额外加了不带斜杠的一条。少了它，25 个软链接会以未跟踪文件的形式出现，可能被误提交成指向本机绝对路径的死链。

**换机器要重建软链接。** 链接内容是本机绝对路径。适配器可以由对应数据集重新训练得到，或从备份复制；`datasets/oasst-wild` 用 `wild_data.py` 重建。

**`layer_norms.jsonl`（198 MB）** 已被 gitignore 且全项目无任何代码读取（见附录 8.4），可以直接删除以再腾空间，此处保留仅为人工在 TensorBoard 里查看。
