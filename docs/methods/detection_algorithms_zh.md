# 噪音检测算法规范 (实验验证版)

> 全部算法基于 10% 比例实验 (Qwen2.5-3B + LoRA, dolly-15k) 实测验证。
> 检测目标: 在训练过程中/训练后, 将噪音样本从正常样本中分离。
> 实测 AUC: garbled 0.9996 · duplicate 0.974 · unrelated 0.923 · mixed 0.850 · keyword 0.531(不可行)。

---

## 0. 符号与前提

- 样本 $x = (p, r)$: prompt $p$ 与 assistant 回复 $r$; 标签 token 集合 $L$, user token 集合 $U$
- 微批大小 = 1 (逐样本梯度捕获的前提); 每样本记录 loss, 梯度
- $\mathbf{v}$: Adam 二阶矩 ($\exp\\_avg\\_sq$), 每优化器步后快照
- 参考方向 $\mathbf{g}^\ast$: 训练前在 $N_{\text{ref}}=200$ 条干净保留样本上计算的平均 LoRA 梯度, 归一化为单位向量: $\\|\mathbf{g}^\ast\\|_2 = 1$
- 阈值建议: 用正常样本分布的**分位数**自适应 (默认 95th 百分位), 绝对阈值随模型/数据变化

---

## 1. 逐样本特征计算

### 1.1 训练中实时计算 (每样本 × 每 epoch)

**输入**: 样本 $x$, 模型 LoRA 参数 $\theta$, 窗口内已累积梯度 $\mathbf{g}_{\text{acc}}$

步骤 (微批=1):

1. 反向传播前快照累积梯度扁平向量: $\mathbf{b} \leftarrow \text{flat}(\theta.\text{grad})$
2. 计算标签 token 交叉熵: $\text{loss} = \mathrm{CE}\big(\text{model}(p, r),\\, L\big)$
3. 反向传播: $\theta.\text{grad}.\text{backward}()$
4. 快照新梯度: $\mathbf{a} \leftarrow \text{flat}(\theta.\text{grad})$
5. **该样本的精确梯度** (微批=1 的差分):

$$\delta = \mathbf{a} - \mathbf{b}$$

6. **梯度范数**:

$$\text{grad\\_norm} = \\|\delta\\|_2$$

7. **与干净参考方向的对齐度** (LESS 式影响力):

$$\text{cos\\_sim\\_ref} = \frac{\langle \delta,\\, \mathbf{g}^\ast \rangle}{\\|\delta\\|_2 \\, \\|\mathbf{g}^\ast\\|_2}$$

8. **窗口内梯度冲突度**:

$$\text{cos\\_sim\\_global} = \frac{\langle \delta,\\, \mathbf{g}_{\text{acc}} \rangle}{\\|\delta\\|_2 \\, \\|\mathbf{g}_{\text{acc}}\\|_2}$$

9. **Adam 归一化更新贡献** (仅 B 矩阵 — A 在 B=0 初始化时梯度为零, 逐元素归一化会爆炸):

$$\text{update\\_contrib} = \frac{\\|\delta_B\\|_2}{\big\\|\sqrt{\mathbf{v}_B}\big\\|_2 + 10^{-8}}$$

### 1.2 每 epoch 末诊断 (抽样 1/8, 前向-only)

**输入**: 样本 $x$, 模型前向 logits $\mathbf{Z} \in \mathbb{R}^{L \times V}$

1. **全序列 next-token CE** (目标必须用真实 token id, 不能用 $-100$ — 否则该位置 CE 被置 0):

$$\text{ce}[t] = -\log p\big(\text{next\\_id}[t] \mid x_{<t}\big)$$

2. **prompt 部分损失** (乱码同时污染输入 → 该值极高):

$$\text{user\\_loss} = \frac{1}{|U|} \sum_{t \in U} \text{ce}[t]$$

3. **标签 token 熵**:

$$\text{entropy} = \frac{1}{|L|} \sum_{t \in L} \Big( -\textstyle\sum_v p(v \mid t)\\,\log p(v \mid t) \Big)$$

4. **困难 token 占比** (阈值 $\tau = 4.0$):

$$\text{frac\\_hard} = \frac{\big|\\{t \in L : \text{ce}[t] > \tau\\}\big|}{|L|}$$

5. **最大 token 损失**: $\text{max\\_token\\_loss} = \max_{t \in L} \text{ce}[t]$
6. **逐 token 损失分布形状**: $\text{skew}$ / $\text{kurt}$ (偏度/峰度 of $\\{\text{ce}[t] : t \in L\\}$)

### 1.3 训练后派生 (轨迹特征)

设样本的逐 epoch 损失序列 $l_0, l_1, \ldots, l_{E-1}$ ($E$ = epoch 数):

$$\text{loss\\_mean} = \frac{1}{E}\sum_{e=0}^{E-1} l_e, \qquad \text{loss\\_last} = l_{E-1}, \qquad \text{loss\\_std} = \sqrt{\frac{1}{E}\sum_{e}(l_e - \bar{l})^2}, \qquad \text{loss\\_slope} = l_{E-1} - l_0$$

$$\text{converge\\_epoch} = \min\\{ e : l_e < 2.0 \\} \quad (\text{若不存在则为 } E)$$

$$\text{loss\\_rank} = \frac{1}{E}\sum_{e=0}^{E-1} \text{percentile}_e(l_e)$$

**loss 轨迹曲率** (二次最小二乘拟合的 $e^2$ 系数, 向量化实现 $\mathbf{c} = \mathbf{y}\\,\mathbf{X}^{+}$ 其中 $\mathbf{X} = [\mathbf{1},\\, \mathbf{e},\\, \mathbf{e}^2] \in \mathbb{R}^{E \times 3}$):

$$\text{loss\\_curvature} = c_0 \quad \text{其中} \quad [l_e] \approx c_2 e^2 + c_1 e + c_0$$

**梯度变异性与参考对齐趋势**:

$$\text{grad\\_norm\\_cv} = \frac{\sigma(\text{grad\\_norm}_e)}{\mu(\text{grad\\_norm}_e)}, \qquad \text{cos\\_ref\\_trend} = \text{cos\\_ref}_{E-1} - \text{cos\\_ref}_{0}$$

### 1.4 数据侧特征 (无需训练)

$$\text{text\\_nn\\_sim}(x) = 1 - \min_{x' \neq x}\; \cos\!\big(\text{TF-IDF}(x),\\, \text{TF-IDF}(x')\big)$$

TF-IDF 参数: 1-2 gram, $\min\\_df = 10$, $\text{sublinear\\_tf} = \text{True}$, $\max\\_features = 200{,}000$。

---

## 2. 单噪音类型检测算法

### 2.1 乱码 garbled (AUC 0.9996) — 训练动力学探针

**特征** (按判别力排序): `loss_curvature` (0.985) > `user_loss` (0.979) > `entropy` (0.971) > `frac_hard` (0.954)

**算法**:

1. 计算正常样本各指标的 95th 百分位: $q_{\text{ul}},\ q_{\text{ent}},\ q_{\text{curv}}$
2. 标记 (多指标 OR 合并; 单指标即可达 AUC > 0.97):

$$s = \big(\text{user\\_loss} > q_{\text{ul}}\big) \lor \big(\text{entropy} > q_{\text{ent}}\big) \lor \big(\text{loss\\_curvature} > q_{\text{curv}}\big)$$

可选的强化判据 (5-epoch 设置): $\text{converge\\_epoch} = E$ (永不收敛) 或 $\text{frac\\_hard}$ 持续不降。

### 2.2 重复 duplicate (AUC 0.974) — 文本相似度去重

**特征**: `text_nn_sim` (0.939) 一枝独秀; **训练侧指标无效 (loss AUC 0.37, 方向相反)**

**算法**:

1. $X \leftarrow \text{TfidfVectorizer}(1\text{-}2\text{gram},\\, \min\\_df{=}10,\\, \text{sublinear\\_tf},\\, \max\\_features{=}200\text{K})(\text{texts})$
2. $(\text{dist}, \\_) \leftarrow \text{NearestNeighbors}(k{=}2,\\, \text{metric}{=}\text{cosine}).\text{fit}(X).\text{kneighbors}(X)$
3. $\text{sim}_i = 1 - \text{dist}[i, 1]$ (排除自身后的最近邻相似度)
4. 标记:

$$s = \big(\text{sim}_i > 0.9\big) \quad (\text{副本} \approx 1.0,\ \text{正常} \approx 0.2\text{-}0.5)$$

### 2.3 上下文错配 unrelated (AUC 0.923) — 损失波动

**特征**: `loss_curvature` (0.830) > `loss_std` (0.827) > `grad_norm_mean` (0.764)

**算法**:

1. 计算正常样本 $\text{loss\\_std}$ 的 95th 百分位 $q_{\text{std}}$
2. 标记:

$$s = \big(\text{loss\\_std} > q_{\text{std}}\big) \land \big(\text{loss\\_slope 不显著为负}\big)$$

> 判别逻辑: 正常难样本轨迹平滑下降 (斜率负、波动小); 错配样本波动大、下降慢 → 曲率异常。

### 2.4 混合噪音 (AUC 0.850) — 组合分类器

特征: 全部 19 维 → $\text{StandardScaler}$ → 逻辑回归 ($\max\\_iter = 2000$) 或随机森林 ($n\\_estimators = 200$), 70/30 划分评估。

### 2.5 keyword (AUC 0.531) — **不可行**, 需模型外手段

训练侧 19 个指标与正常样本完全重叠。可选方向: NER 实体一致性校验 (实体频率异常)、反事实扰动 (替换实体后 loss 是否骤降)、外部知识库比对。

---

## 3. 通用检测流水线 (部署建议)

**输入**: 训练语料 $D$, 干净校验集 $C$ (与 $D$ 不相交, ~2.7% 规模)

```
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

## 4. 计算成本 (实测, RTX PRO 6000 Blackwell 96GB)

**关键事实: 逐样本梯度不是"额外反向传播"** — 微批=1 训练本来就要对每个样本反向, 算法只是反向前快照累积梯度、反向后做差 (扁平向量拷贝 + 点积), 实测每样本 ~12ms / 总 160ms ≈ **5-8% 训练开销**。token 级逐 token 归因才是昂贵操作, 仅用于离线小样本分析。

| 特征 | 成本 | 说明 |
|---|---|---|
| `loss` / `entropy` / `user_loss` / `frac_hard` / `max_token_loss` | ~0% | 训练/诊断前向的顺带产物 |
| `grad_norm` / `cos_sim_ref` / `cos_sim_global` / `update_contrib` | **+5-8% 训练时间** | 快照+差分的拷贝与点积; 反向本身是训练的一部分 |
| loss 轨迹特征 (curvature/rank/converge_epoch) | 0% | 训练后从已存 per-epoch loss 派生 |
| 诊断特征 (1/8 抽样, 前向-only) | ~30s/epoch | 每 epoch 一次抽样前向 |
| `text_nn_sim` (TF-IDF + kNN) | ~2 分钟/15K (CPU) | 完全不需要模型 |
| token 级逐 token 梯度 (top-24/样本) | 离线 60+60 样本 ~3-5 分钟/数据集 | 每 hard token 一次反向, **不要在线计算** |

**部署降级选项** (连 5-8% 都不可接受时):

1. **纯前向特征**: `user_loss` + `entropy` + loss 轨迹单独即可检测 garbled (AUC 0.97+), 零梯度开销;
2. **加大微批** (8/16): 损失类特征不受影响; 梯度类特征退化为批级 (可用于 batch 级筛查);
3. **数据侧优先**: duplicate 用 TF-IDF 去重 (CPU), 完全离线。

---

*实现参考: `train.py` (特征捕获), `analyze.py` (派生特征与分类), `analyze.py` (token 级)。验证数据: `results/ratio10/auc_univariate.csv`, `results/ratio10/detection_multivariate.csv`。*
