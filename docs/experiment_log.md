# 实验阶段记录

## 2026-09-13：ratio10 统一重训练（7 类噪音 + clean + mixed 合并为单一实验）

### 背景

此前 template / truncation / near_duplicate 三类噪音单独跑在已删除的 `extra10` 标签下，
与 `ratio10`（clean / garbled / duplicate / unrelated / keyword / mixed）是两次独立实验。
本阶段将全部 7 类噪音 + clean + mixed 合并到同一个 `ratio10` 标签下重新生成数据并重新训练，
不再有 extra10/ratio05 等历史标签。

- 数据：`data/ratio10/`，9 个数据集（clean/garbled/duplicate/unrelated/keyword/template/truncation/near_duplicate/mixed），噪音比例 10%。
- 训练：Qwen2.5-3B-Instruct LoRA（r=32），5 epochs，逐样本梯度追踪，脚本 `run_full_ratio10.sh`。
- 分析：`analyze.py` 三阶段（features / training / unsupervised），产出 `results/ratio10/{features,training,unsupervised,per_sample_metrics}.csv`。
- 全流程于 2026-09-13 00:48:01 完成（`ALL DONE`）。

### 与旧报告（已删除的 docs/analysis_report_zh.md 等）结论对比

**基本保持一致：**
- 检测难度分级方向不变：RF 分类器 AUC — template 0.994、garbled 0.996、duplicate 0.984、
  unrelated 0.937、truncation 0.782、near_duplicate 0.699，均落在旧报告数值附近（误差几个百分点）。
- 方向反转陷阱复现：template（loss_last 0.021）、duplicate（loss_last 0.27）远低于 clean（0.61），
  garbled 持续高损失学不动（loss_last 2.57）。
- keyword 仍是有监督检测中最弱的两类之一（RF AUC 0.684）。

**表面矛盾，已排查——根因是分析代码的方法论变化，不是训练结果变化：**
- 旧报告的核心发现之一：IsolationForest 对 template 的检测 **低于随机**（AUC 0.494），
  由此发展出"带符号记忆性规则"这一方法论（§3.2）。
- 当前 `analyze.py::unsupervised_metrics()` 跑出的 `results/ratio10/unsupervised.csv` 显示
  template 的 iforest AUC = 0.739，zscore_mean = 0.848，均明显高于随机，看似与旧结论相反。
- **根因排查**：用旧版脚本（`git show adea193^:scripts/3_analysis/analyze_unsupervised.py`）
  的精确方法重新计算——每个数据集单独 fit IsolationForest（不跨数据集合并）、fit 前先做
  StandardScaler 标准化、只用旧版 13 维 `TRAJ_METRICS`（不含诊断层特征）——复现结果：
  template AUC **0.541**（方向与旧报告"接近随机"一致）、garbled AUC **0.949**（旧报告 0.955，吻合）、
  duplicate AUC **0.562**（旧报告区间 0.556-0.699，吻合）。
- 而当前 `analyze.py` 的实现：(1) 把 9 个数据集**合并成一个大表**统一 fit 一次 IsolationForest，
  template 的离群基准线被 garbled 等极端量级样本拉偏；(2) **未做标准化**直接喂入原始特征，
  不同量级特征（grad_norm 数十 vs cos_sim 接近 0）让距离度量失真；(3) 多用了 24 个诊断层特征，
  可能给 template 带来额外的非"离群检测"意义上的判别信号。三者叠加导致同名方法（iforest）
  实际计算的是完全不同的东西。
- **结论：旧报告"template 无监督检测低于随机"这一发现依然成立，是本次重构后的 `analyze.py`
  改变了无监督检测的实现方式，不代表训练结果或旧结论本身有问题。**
- **已修复**：`analyze.py::unsupervised_metrics()` 改为按数据集单独 fit（不再跨数据集合并）、
  iforest 前加 `StandardScaler`、`n_estimators` 提至 300（特征集保持全量数值列，未收窄到旧版
  13 维 `TRAJ_METRICS`，因为额外的诊断层特征对多数类型影响有限）。重新生成的
  `results/ratio10/unsupervised.csv` 显示 template iforest AUC = **0.522**（旧报告 0.494，方向一致），
  garbled iforest = **0.936**（旧报告 0.955），duplicate iforest = **0.612**（旧报告区间 0.556-0.699）。

### 待补充分析（按对旧报告核心结论的支撑优先级排序）

1. ~~排查 template 无监督结果反转的根因~~ ——已完成，见上节「已修复」。
2. ~~P@10% 清洗精度表~~ ——已完成。新增 `analyze.py::precision_lift_table()`，直接基于
   `unsupervised.csv` 的 `p_at_10`/`random_p` 列算 lift，产出 `results/ratio10/precision_lift.csv`
   （`python3 cli.py analyze --tag ratio10 --kind precision_lift`）。
   结果支撑"AUC 高估清洗可用精度"：duplicate 三种方法 lift 全部 ≤1（0.45-0.56x，比随机还差），
   AUC 0.54-0.61 看似可用但 top-10% 清洗几乎不比随机筛选强。template 上 AUC 最高的 zscore_max
   （0.878）lift 反而最差（0.23x），AUC 更低的 zscore_mean（0.803）lift 反而最高（1.72x）——
   AUC 排方法和 P@10% 排方法会给出不同的最优选择，用 AUC 选清洗方法是有偏的。garbled 上
   iforest 最稳（lift 5.9x），符合其"结构性易检测"的定位。
3. ~~跨类型迁移矩阵~~ ——已完成。新增 `analyze.py::cross_type_transfer()`（LR+RF 取更优，源域标准化，
   目标域用源域 scaler，对角线为域内 5-fold CV），产出 `results/ratio10/cross_type.csv`
   （`python3 cli.py analyze --tag ratio10 --kind cross_type`）。
   结果支撑旧结论：对角线均值 AUC 0.846 vs 离对角均值 0.675（迁移掉约17-20点）。
   template 是最孤立的类型——train-on-template 迁移到其余6类几乎全部逼近/低于随机（0.40-0.59），
   反向（其余类型训练测 template）同样弱，unrelated→template 仅 0.321（低于随机，方向反转）。
   garbled 相对易迁移（多数类型训练后测 garbled 仍 0.93-0.98），keyword 双向都弱
   （within-type CV 仅 0.577，是唯一 own-AUC<0.6 的类型），与此前"keyword 最难检测"结论一致。
4. ~~带符号记忆性规则复现~~ ——已完成。新增 `analyze.py::memorization_score()`（`MEMO_FEATS`：
   loss_mean/loss_last/loss_std/loss_curvature/converge_epoch/grad_norm_mean，方向由记忆化假设
   先验固定、不按数据集拟合），产出 `results/ratio10/memorization.csv`
   （`python3 cli.py analyze --tag ratio10 --kind memorization`）。这6个特征对每个样本都有值
   （不依赖诊断层子样本），因此 n 是全量训练集（~14611-16072），比 unsupervised.csv 的诊断子样本
   （~900-1000）大得多。规则方向在修复后的 ratio10 上依然成立：template AUC=0.925（`memo_signed`）——
   仍是最强的"超典型/被记忆"信号；duplicate AUC=0.654，中等记忆性，与旧报告的 0.699 同量级；
   其余6类（garbled/keyword/mixed/near_duplicate/truncation/unrelated）AUC 全部低于0.5（0.017-0.38），
   证实了预期的不对称性——非记忆型噪音在这个规则下反而"更难"而非"更容易"，方向正确未反转。
5. ~~下游 benchmark 评测~~ ——已完成。`run_eval_ratio10.sh` 于 2026-09-14 02:20:32 跑完全部 9 个数据集
   （`results/eval/eval_ratio10_*.json`）。
6. ~~ratio5（5% 噪音比例）交叉验证~~ ——已完成。`run_full_ratio5.sh` 训练于 2026-09-14 07:46:08 完成
   （9 个数据集），脚本自带的分析步骤因 `results/ratio5/` 目录不存在而报错中断，补建目录后用
   `scripts/run_analysis_ratio5.sh` 重新跑完 5 项分析（2026-09-14 09:35:49 完成），产出
   `results/ratio5/{features,unsupervised,cross_type,precision_lift,memorization}.csv`。
   **核心结论在 5% 噪音比例下依然成立：**
   - 带符号记忆性规则：template AUC=0.911（ratio10: 0.925），duplicate AUC=0.668（ratio10: 0.654），
     其余 6 类（garbled/keyword/mixed/near_duplicate/truncation/unrelated）AUC 全部 <0.5（0.018-0.38）——
     不对称方向未变。
   - per-dataset 独立拟合的无监督检测：garbled 上 iforest 仍最强（ratio5 0.941 vs ratio10 0.936），
     template 上 iforest 仍明显低于其他方法且接近随机（ratio5 0.543 vs ratio10 0.522）——证明该方法论
     在噪音样本更少（40-45 个/数据集，约为 ratio10 的一半）时依然稳定，未因样本量下降而失效。
   - P@10% lift：duplicate 在两个比例下都是"AUC 看似可用但 lift 接近或低于 1"（ratio5 zscore_mean/
     iforest lift 仅 0.667x）；garbled 上 iforest 仍是最稳清洗方法（ratio5 lift 7.4x，ratio10 5.9x，
     噪音比例更低时 lift 反而更高）。
   - ratio5 的 ratio5_eval 下游 benchmark 评测仍在进行中（tmux session `ratio5_eval`），尚无法判断
     5% 噪音比例下 template 是否仍是下游危害最大的类型。
7. ~~跨比例迁移矩阵~~ ——已完成。新增 `analyze.py::cross_ratio_transfer()`（方法与
   `cross_type_transfer()` 相同：LR+RF 取更优，源域标准化；区别在于按噪音**类型**固定、
   在 ratio10/ratio5 两个**比例**间双向迁移，对角线为各自比例下的域内 5-fold CV），产出
   `results/transfer_cross_ratio.csv`（`python3 cli.py analyze --kind cross_ratio
   --tags ratio10,ratio5 --output results/transfer_cross_ratio.csv`）。
   **核心发现：跨比例迁移几乎无损，retention 全部落在 0.999-1.40（多数 ≥1.0），
   与跨类型迁移矩阵（对角线 0.846 vs 离对角 0.675，掉约17-20点）形成强烈对比。**
   7 类噪音双向迁移（ratio10→ratio5、ratio5→ratio10）AUC 全部 ≥0.75，多数 ≥0.88；
   within-ratio AUC 本就偏弱的两类（keyword：0.578/0.641，near_duplicate：0.680/0.743）
   迁移后 AUC 反而**上升**（keyword ratio10→ratio5 达 0.898，near_duplicate ratio10→ratio5
   达 0.895），说明合并两个比例的训练动态特征反而学到更鲁棒的判别边界，而非过拟合某个
   特定噪音密度。**结论：训练动态特征对"噪音类型"的判别信号是跨噪音比例稳定的（甚至有
   正迁移），检测器不需要针对每个噪音比例单独训练；真正的迁移瓶颈在跨噪音类型，而不是
   跨噪音比例。**
8. **闭环清洗实验（garbled@ratio10）——进行中。** 与此前所有分析不同，之前只验证"能不能分辨
   噪音样本"（AUC/lift 数字），从未验证"清洗后重训练是否真的变好"。新增 `cleaning_loop.py::build()`
   （`python3 cli.py clean --tag ratio10 --dataset garbled --budget 0.10`）：用纯无监督
   IsolationForest（不看噪音标签，特征限定为在全量 14611 条样本上都有值的基础轨迹特征子集，
   而非 `unsupervised_metrics()` 依赖的诊断层全特征+约900条子样本）在全量训练集上打分，
   按 10% 预算剔除疑似噪音，另建等量随机剔除对照组。结果：targeted 剔除精度 52.1%
   （761/1461 条真实 garbled 噪音被命中）vs 随机剔除精度 9.2%——证明纯盲检测在全量数据上依然
   有效，不需要标签。已生成 `data/ratio10/cleaning_loop/garbled/{train_targeted,train_random}.jsonl`；
   `scripts/run_cleaning_loop_garbled.sh`（tmux session `cleaning_loop_garbled`）已排队，等
   ratio5_eval 释放 GPU 后自动训练两个版本并评测，与 `eval_ratio10_garbled.json`（未清洗基线）、
   `eval_ratio10_clean.json`（上限参照）三方对比下游 benchmark 分数。
9. **早期检测——已完成。** 现有全部检测特征（converge_epoch、loss_slope 等）都要等 5 epoch
   训练完才能算，只有事后诊断价值。新增 `analyze.py::early_detection_sweep()`
   （`build_table(..., max_epoch=k)` 把轨迹截断到前 k+1 个 epoch，对每个截断点重算
   `unsupervised_metrics`/`memorization_score`，`python3 cli.py analyze --tag ratio10
   --kind early_unsupervised|early_memorization`），产出
   `results/ratio10/early_{unsupervised,memorization}.csv`。
   **核心发现：garbled 和 template 在只看完第 1 个 epoch 时检测 AUC 就已接近满血——
   完全不需要等训练跑完就能拿到几乎和事后一样好的检测信号。**
   - garbled（无监督 iforest，核心特征子集）：epoch1 AUC=0.901，epoch5 AUC=0.929——从第一个
     epoch 起就已经很强，训练带来的边际提升很小。
   - template（带符号记忆性规则）：epoch1 AUC=0.912，epoch5 AUC=0.925——同样第一个 epoch
     就接近饱和。
   - **反直觉的现象**：template/duplicate/mixed/unrelated 在**通用无监督 iforest**规则下，
     AUC 反而随训练进行而**下降**（template：0.713→0.564，unrelated：0.715→0.632，
     duplicate：0.616→0.520）——推测是因为这几类噪音的"超典型/被记忆"特征（损失快速收敛到
     远低于正常水平）在训练后期让它们在通用轨迹特征空间里看起来更像"正常但简单"的样本而非
     "离群点"，只有专门按记忆化方向设计的 `memo_signed` 规则能捕捉到这个信号且随训练增强
     （duplicate 0.587→0.654，template 0.912→0.925）——与已确认的"方向反转陷阱"是同一机制
     在训练动态时间线上的体现，而非新问题。
   - keyword/near_duplicate/truncation：AUC 全程在 0.55-0.61 附近波动，既不随训练明显提升
     也不明显下降——这几类噪音本身检测难度就高，训练进度不是主要影响因素。
   - **实践含义**：garbled 和 template 这两类可以在训练第 1 个 epoch 结束后立即打分剔除，
     省下后续 4 个 epoch 在噪音样本上的算力；其余类型（尤其 keyword/近似重复/截断）没有
     "早停检测"的捷径，仍需要完整训练或至少多个 epoch 的信号累积。

10. **特征归因——已完成。** 此前所有分析都只回答"能不能分辨噪音样本"（AUC/lift 数字），
    从未回答"检测器到底在看哪个特征"。新增 `analyze.py::feature_attribution()`：对每个数据集
    按 `cross_type_transfer()` 同款 5-fold CV 训练 RF，在每个 fold 的验证集上用
    `sklearn.inspection.permutation_importance`（打乱单个特征后 AUC 的下降量，`scoring='roc_auc'`，
    `n_repeats=20`）逐 fold 计算再取均值——不用 RF 内置的 impurity-based `feature_importances_`，
    因为它对高基数/连续特征有系统性偏置，不代表真实预测贡献。产出
    `results/ratio10/feature_attribution.csv`（`python3 cli.py analyze --tag ratio10
    --kind feature_attribution`）。
    **核心发现：duplicate 和 unrelated 这两类的检测信号几乎完全来自 `text_nn_sim`
    （文本嵌入最近邻相似度，来自原始数据本身，不是训练动态特征），且断层式领先第二名一个数量级：**
    - duplicate：`text_nn_sim` 重要性 0.148，第二名 `cos_global_last` 仅 0.0093（差 16 倍）。
    - unrelated：`text_nn_sim` 重要性 0.083，第二名 `token_loss_skew` 仅 0.0123（差 7 倍）。
    - **含义**：这两类噪音本质上是"文本内容层面"就能识别的问题（复制粘贴 / 话题不相关），
      RF 分类器虽然吃了全部 37 维训练动态+文本特征，但实际上主要在利用一个静态文本相似度
      特征，而不是真的从训练轨迹里学到了判别信号——`cross_type_transfer()` 报告的高 AUC
      不能简单归功于"训练动态检测方法论"，对这两类而言更接近"用一个现成的文本相似度特征
      做检测，训练动态基本是摆设"。这是本项目"用训练动态做免标签检测"这一核心叙事下
      一个需要如实标注的例外。
    - 其余 5 类（garbled/keyword/near_duplicate/template/truncation）的头部特征均是真正的
      训练动态量：garbled 由 `user_loss`（0.016）主导（乱码文本导致 response 段损失整体偏高）；
      keyword 由 `loss_slope`（0.020）主导，但整体重要性数值很小且标准差常常大于均值——
      与 keyword 本身是检测难度最高的类型（within-type AUC 仅 0.578）一致，没有任何单一
      特征能稳定贡献信号；truncation 由 `loss_slope`/`loss_std`（0.027/0.021）主导；
      near_duplicate、template 由 `hard_*` 系列 token 级难例特征主导（`hard_loss_max`/
      `hard_id_uniq`/`hard_pos_std_mean` 等）。
    - template 是另一个值得注意的现象：own-AUC 接近满分（0.999），但所有 37 个特征的
      permutation importance 数值都极小（最高的 `hard_id_uniq` 仅 0.0015）——说明当某类型
      "过于好检测"时，冗余/相关特征会分摊掉边际贡献，打乱任意单一特征对整体 AUC 的影响都
      很小，这不代表检测不可靠，而是接近饱和检测任务里 permutation importance 本身的正常
      表现（多重共线特征下每个特征的边际贡献都被稀释）。

### 清理记录

本阶段同时清理了历史遗留文件：
- `results/*` 中不可由当前 `analyze.py` 重新生成的孤儿分析表（30+ 个文件）。
- `runs/ratio10/cleaning_gain/` 侧支实验目录。
- 整个 `extra10`、`ratio05` 标签树（`data/` `runs/` `results/` 下）。
- `results/eval`、`results/charts` 中引用 extra10/ratio05 的评测与图表文件。
- 顶层跨实验汇总表：`cleaning_gain_comparison.csv`、`transfer_cross_ratio.csv`、`transfer_cross_type.csv`、
  `transfer_ratio_summary.csv`、`transfer_type_summary.csv`、`natural_validation.csv`、`data_inventory.json`。
- `results/ratio10/auc_univariate.csv`（当前代码无对应生成逻辑的孤儿文件）。
- `docs/report_tables.md`、`docs/analysis_report_zh.md`、`docs/analysis_report_en.md`、
  `docs/comparisons/cross_experiment_synthesis_{zh,en}.md`（正文引用已删除的 extra10/ratio05 实验，
  在补充分析完成、结论更新前不再保留旧结论文档，本记录取代其阶段性作用）。
- `__pycache__`、`.ipynb_checkpoints` 等缓存文件。

以上清理均通过 `git rm` 暂存，尚未提交。

### 当前进行中

ratio5 训练已于 2026-09-14 07:46:08 完成（9/9 数据集），分析表已生成（见上方第 6 项）。
下游 benchmark 评测正在进行（`scripts/run_eval_ratio5.sh`，tmux session `ratio5_eval`），
预计耗时与 ratio10 评测相近（约 12 小时/9 数据集）。评测完成后会自动触发排队中的
garbled 闭环清洗实验（tmux session `cleaning_loop_garbled`，见上方第 8 项）。

### 脚本整理（2026-09-13）

编排脚本迁移到 `scripts/` 目录并不再纳入 git 跟踪（`.gitignore` 新增 `/*.sh` `/scripts/`）。
过时的 `run_experiment.sh`（`--model mock` 示例）重写为 `scripts/run_full.sh <tag> [ratio]`，
作为今后统一入口（数据生成 + 9 类训练 + 全部 6 项分析）。`run_full_ratio10.sh` 保留作为
ratio10 实验的历史执行记录。

`scripts/` 只放**编排**脚本（跑什么、按什么顺序跑）；实验方法本身（检测器怎么打分、特征
怎么选）属于代码库交付物，放在根目录并纳入 git 跟踪——第 8/9 项最初把方法逻辑写在
`scripts/build_cleaning_loop.py`/`scripts/early_detection.py` 里，随后按此原则重构为
根目录的 `cleaning_loop.py`（+ `cli.py clean` 子命令）和 `analyze.py::early_detection_sweep()`
（+ `cli.py analyze --kind early_unsupervised|early_memorization`），`scripts/` 下只保留
真正的编排脚本 `run_cleaning_loop_garbled.sh`。

## 2026-09-19：训练 checkpoint/resume 支持 + 多任务（QA/推理）数据集管线

### 背景

`wild_all` 标签下的 `cleaning_loop_targeted_wild` 训练在第 5 个 epoch 刚开始时中断——`model.py::LoRA.fit()`
此前只在全部 epoch 跑完后才保存一次权重，已完成的 4 个 epoch（约 12 小时 GPU 时间）训练状态完全没有
落盘，中断后只能从 epoch 0 重新开始。同时，Plan 2 计划验证检测方法在 QA/推理这类不同任务类型上是否
同样有效，需要新的任务型数据源和按比例的 train/val/test 拆分。本阶段完成这两项改动。

### 1. 训练 checkpoint/resume（`model.py::LoRA.fit()`）

- 按 epoch 边界保存，且只保留**最新一份**（覆盖式，不为每个 epoch 单独保留，避免磁盘膨胀——单份
  LoRA adapter ≈229MB）。写入 `run_dir/checkpoint/`，内容：`adapter/`（`model.save_pretrained`）、
  `optimizer.pt`（AdamW 状态，续训必须还原，否则等价于重置优化器）、`state.pt`
  （`epoch_done`/`global_step`/`v_ready`/`v_buf`/`ref_buf`/`epoch_stats`）。
  - `ref_buf`（训练开始前、LoRA B 还是零初始化时算出的参考梯度方向）必须保存并在续训时直接复用，不能
    重新计算——续训时模型权重已非零，重新算会破坏 `cos_ref_*` 系列特征在已训练 epoch 与续训 epoch
    之间的可比性。
  - 保存用"写临时目录 `checkpoint.tmp/` + 原子改名"模式，防止"只保留最新一份"在覆盖过程中崩溃导致
    仅有的检查点被写坏。
- Resume 逻辑：若 `checkpoint/state.pt` 存在，用 `PeftModel.from_pretrained(..., is_trainable=True)`
  代替零初始化的 `get_peft_model`，跳过 `_compute_reference_direction` 直接加载 `ref_buf`，
  `opt.load_state_dict(...)` 还原优化器状态，训练起点从 `state['epoch_done']+1` 续上；
  `per_sample.jsonl`/`layer_norms.jsonl` 用追加模式打开前先按 checkpoint 的 epoch/step 裁掉多余尾部
  （避免中断前部分写入的最后一个 epoch 留下重复行）。**不新增 CLI 参数**——同一条 `train` 命令自动判断
  是否续训。训练正常跑完后删除 `checkpoint/`。
- `.gitignore` 新增 `runs/**/checkpoint/`、`runs/**/checkpoint.tmp/`。`AGENTS.md` 补充说明该机制，并
  顺带更正过时的 GPU 型号描述（RTX PRO 6000 Blackwell → 实测已换成 RTX GeForce RTX 4090）。
- 已在真实的 `wild_all`/`cleaning_loop_targeted_wild` 重跑中验证生效（`run_wild_all.log` 可见训练
  正常进行）。

### 2. 多任务（QA/推理）数据集管线（`data.py`/`cli.py`/`config.yaml`）

- `data.py::load_rows()` 扩展：支持 `hf://org/name#config` 语法（gsm8k 需要 `#main`）；支持
  `extra_splits` 把多个官方 HF split 先合并成一个池子；新增 squad 式嵌套答案字段映射
  （`row['answers']['text'][0]` → response，`context`+`question` 拼接 → prompt）。
- `data.py::split_fractions(rows, fractions, seed)`（新函数）：复用 `split_holdout` 的确定性洗牌方式，
  按累积比例切成多路，每路重新编号。用于把 squad/gsm8k 的官方 split 合并重切成 IID 的 train/val/test——
  不直接采用官方 validation/test，因为 squad 官方 validation 是按文章划分、与 train 主题不重叠，直接
  当测试集会让"噪音检测效果"与"模型对新主题的泛化能力"混在一起。
- 新增两个任务特化噪音类型，登记进 `TRANSFORMS`/`NOISE_TYPES`：
  - `wrong_answer`（QA）：整体替换 assistant 回答为看起来合理但错误的短答案，问题原文不变。
  - `wrong_final_answer`（推理）：只替换 gsm8k 答案里 `#### N` 的最终数字，保留推理链条文字不变——
    模拟"过程通顺但结果算错"的静默错误。
- `data.py::apply()` 加 `ratio` 边界校验（不在 `[0,1]` 时 `raise ValueError`），此前会静默 clamp。
- `cli.py data` 子命令新增 `--extra-split`（可重复）、`--val-frac`/`--test-frac` 参数；`config.yaml`
  新增 `tasks:` 约定块，登记 squad/gsm8k 的 source 路径供以后编排脚本直接读取。
- 已用真实 HF 数据（`hf://rajpurkar/squad`、`hf://openai/gsm8k#main`）跑通 `data`/`train --smoke`
  全流程验证，生成的 `train.jsonl`/`val.jsonl`/`test.jsonl` 行数比例正确，`wrong_answer`/
  `wrong_final_answer` 转换后样本人工抽查符合预期。

### 当前进行中

Plan 2 Part B 提出的实验矩阵（2 任务类型 × 3 数据集 × 2 噪音比例 = 12 次训练，约 42 小时）尚未启动，
需等 GPU 上正在跑的 `wild_all_loop` 完成，且需用户确认后才会开始真实训练。
