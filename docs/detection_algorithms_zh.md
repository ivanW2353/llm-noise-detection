# 噪音检测算法规范 (实验验证版)

> 全部算法基于 10% 比例实验 (Qwen2.5-3B + LoRA, dolly-15k) 实测验证。
> 检测目标: 在训练过程中/训练后, 将噪音样本从正常样本中分离。
> 实测 AUC: garbled 0.9996 · duplicate 0.974 · unrelated 0.923 · mixed 0.850 · keyword 0.531(不可行)。

---

## 0. 符号与前提

- 样本 $x = (p, r)$: prompt $p$ 与 assistant 回复 $r$; 标签 token 集合 $L$, user token 集合 $U$
- 微批大小 = 1 (逐样本梯度捕获的前提); 每样本记录 loss, 梯度
- $v$: Adam 二阶矩 (exp_avg_sq), 每优化器步后快照
- 参考方向 $\mathbf{g}^*$: 训练前在 $N_{\text{ref}}=200$ 条干净保留样本上计算的平均 LoRA 梯度, 归一化为单位向量
- 阈值建议: 用正常样本分布的**分位数**自适应 (默认 95th 百分位), 绝对阈值随模型/数据变化

---

## 1. 逐样本特征计算

### 1.1 训练中实时计算 (每样本 × 每 epoch)

**输入**: 样本 $x$, 模型 LoRA 参数 $\theta$, 窗口内已累积梯度 $\mathbf{g}_{\text{acc}}$

```
1. before ← flat(θ.grad)                      # 反向前快照累积梯度 (扁平向量)
2. loss ← CE(model(p, r), L)                    # 仅标签 token 的交叉熵
3. θ.grad.backward()
4. after ← flat(θ.grad)
5. δ ← after − before                          # 该样本的精确梯度 (微批=1)
6. grad_norm ← ||δ||₂
7. cos_sim_ref ← ⟨δ, g*⟩ / (||δ||₂ · ||g*||₂)   # 与干净参考方向的对齐度
8. cos_sim_global ← ⟨δ, g_acc⟩ / (||δ||₂ · ||g_acc||₂)   # 窗口内梯度冲突度
9. update_contrib ← ||δ_B||₂ / (||√v_B||₂ + 1e-8)          # 仅 B 矩阵 (A 在 B=0 时梯度为零)
```

### 1.2 每 epoch 末诊断 (抽样 1/8, 前向-only)

**输入**: 样本 $x$, 模型前向 logits $\mathbf{Z} \in \mathbb{R}^{L \times V}$

```
1. ce[t] ← −log softmax(Z[t])[next_id[t]]      # 全序列 next-token CE (用真实 id, 非 -100)
2. user_loss ← mean{ ce[t] : t ∈ U }           # prompt 部分损失
3. entropy ← mean{ −Σ_v p(v|t)·log p(v|t) : t ∈ L }   # 标签 token 熵
4. frac_hard ← |{ t ∈ L : ce[t] > 4.0 }| / |L|
5. max_token_loss ← max{ ce[t] : t ∈ L }
6. skew/kurt ← 偏度/峰度 of {ce[t] : t ∈ L}     # 逐 token 损失分布形状
```

### 1.3 训练后派生 (轨迹特征)

设样本的逐 epoch 损失序列 $l_0, l_1, ..., l_{E-1}$ ($E$=epoch 数):

```
loss_mean    ← mean(l_e)
loss_last    ← l_{E-1}
loss_std     ← std(l_e)
loss_slope   ← l_{E-1} − l_0
converge_epoch ← min{ e : l_e < 2.0 }, 若无则为 E   # 收敛速度
loss_rank    ← mean_e( percentile_e(l_e) )           # epoch 内分位数
loss_curvature ← a, 其中 [l_e] 用二次多项式拟合:
                X = [1, e, e²] (E×3),  a = (X⁺)² 行 …   # 最小二乘系数, 向量化: coeffs = y @ pinv(X)ᵀ
grad_norm_cv ← std(grad_norm_e) / mean(grad_norm_e)
cos_ref_trend ← cos_ref_{E-1} − cos_ref_0
```

### 1.4 数据侧特征 (无需训练)

```
text_nn_sim(x) ← 1 − min_{x' ≠ x} cosine( TF-IDF(x), TF-IDF(x') )
TF-IDF: 1-2 gram, min_df=10, sublinear_tf=True, max_features=200,000
```

---

## 2. 单噪音类型检测算法

### 2.1 乱码 garbled (AUC 0.9996) — 训练动力学探针

**特征** (按判别力排序): `loss_curvature` (0.985) > `user_loss` (0.979) > `entropy` (0.971) > `frac_hard` (0.954)

**算法**:
```
输入: 每样本的 user_loss, entropy, loss_curvature
1. 计算正常样本各指标的第 95 百分位: q_ul, q_ent, q_curv
2. 标记 s = (user_loss > q_ul) 或 (entropy > q_ent) 或 (loss_curvature > q_curv)
   # 多指标 OR 合并; 单指标即可达 AUC>0.97
可选的强化判据 (5-epoch 设置): converge_epoch = E (永不收敛) 或 frac_hard 持续不降
```

### 2.2 重复 duplicate (AUC 0.974) — 文本相似度去重

**特征**: `text_nn_sim` (0.939) 一枝独秀; **训练侧指标无效 (loss AUC 0.37, 方向相反)**

**算法**:
```
输入: 全部样本文本
1. X ← TfidfVectorizer(ngram(1,2), min_df=10, sublinear_tf, max_features=200K)(texts)
2. dist, _ ← NearestNeighbors(k=2, metric="cosine").fit(X).kneighbors(X)
3. sim_i ← 1 − dist[i, 1]            # 排除自身后的最近邻
4. 标记 s = (sim_i > 0.9)             # 副本相似度 ≈1.0; 正常 ≈0.2-0.5
```

### 2.3 上下文错配 unrelated (AUC 0.923) — 损失波动

**特征**: `loss_curvature` (0.830) > `loss_std` (0.827) > `grad_norm_mean` (0.764)

**算法**:
```
输入: 每样本跨 epoch 的 loss 轨迹与 grad_norm 序列
1. 计算正常样本 loss_std 的第 95 百分位 q_std
2. 标记 s = (loss_std > q_std) 且 (loss_slope 不显著为负)
   # 正常难样本: 平滑下降 (斜率负、波动小)
   # 错配样本: 波动大、下降慢 → 曲率异常
```

### 2.4 混合噪音 (AUC 0.850) — 组合分类器

```
特征: 全部 19 维 → StandardScaler → 逻辑回归 (max_iter=2000) 或
      随机森林 (n_estimators=200), 70/30 划分评估
```

### 2.5 keyword (AUC 0.531) — **不可行**, 需模型外手段

训练侧 19 个指标与正常样本完全重叠。可选方向: NER 实体一致性校验 (实体频率异常)、反事实扰动 (替换实体后 loss 是否骤降)、外部知识库比对。

---

## 3. 通用检测流水线 (部署建议)

```
输入: 训练语料 D, 干净校验集 C (与 D 不相交, ~2.7% 规模)
阶段 A (训练前, 一次性):
  1. 在 C 上前向/反向计算参考方向 g*
阶段 B (训练中, epoch 0-1 内检测, 每样本实时):
  2. 微批=1 训练; 每样本计算 §1.1 的 6 个特征
  3. 每 epoch 末: 抽样计算 §1.2 的诊断特征
  4. 每 epoch 末: 用当前模型对全部样本做一次前向, 计算 loss_rank
阶段 C (分类与清洗):
  5. 按噪音类型选用 §2 的算法; 或 19 维特征 + LR 分类
  6. 阈值用正常样本 95th 百分位自适应
  7. 标记样本从训练集中移除 (或降权), 继续训练
注意事项:
  - 检测越早越好: 检测 AUC 随 epoch 单调衰减 (garbled: 0.985→0.865)
  - 每类型单独调阈值, 不要用统一阈值
  - 训练侧 + 数据侧特征互补: duplicate 只能靠数据侧
```

---

## 4. 复杂度

| 步骤 | 复杂度 |
|---|---|
| 逐样本梯度特征 | 每样本 1 次额外 backward 等价量 (fill/dot, ~1% 训练开销) |
| 诊断特征 | 每 epoch 1 次 1/8 抽样前向 (~30s/epoch) |
| text_nn_sim | TF-IDF + kNN: 对 15K 样本 ~2 分钟 (CPU) |
| 分类器 | LR/RF 秒级 |

---

*实现参考: `scripts/train.py` (特征捕获), `scripts/analyze_detection.py` (派生特征与分类), `scripts/analyze_token_level.py` (token 级)。验证数据: `results/auc_univariate_ratio10.csv`, `results/detection_multivariate_ratio10.csv`。*
