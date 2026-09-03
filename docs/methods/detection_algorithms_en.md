# Noise Detection Algorithms (validated specification)

> All algorithms validated on the 10%-ratio experiment (Qwen2.5-3B + LoRA, dolly-15k).
> Goal: separate noisy samples from normal ones during / after training.
> Validated AUC: garbled 0.9996 · duplicate 0.974 · unrelated 0.923 · mixed 0.850 · keyword 0.531 (infeasible).

---

## 0. Notation & prerequisites

- Sample $x = (p, r)$: prompt $p$, assistant response $r$; label-token set $L$, user-token set $U$
- Micro-batch = 1 (prerequisite for exact per-sample gradients); per-sample loss & gradient recorded
- $\mathbf{v}$: Adam second moment ($\exp\\_avg\\_sq$), snapshotted after each optimizer step
- Reference direction $\mathbf{g}^\ast$: mean LoRA gradient over $N_{\text{ref}}=200$ clean held-out samples, computed pre-training, normalized: $\\|\mathbf{g}^\ast\\|_2 = 1$
- Threshold guidance: use **percentile-adaptive thresholds** on the normal-sample distribution (default 95th percentile); absolute values depend on the model/data

---

## 1. Per-sample feature computation

### 1.1 Computed live during training (every sample × every epoch)

**Input**: sample $x$, LoRA params $\theta$, accumulated window gradient $\mathbf{g}_{\text{acc}}$

Steps (micro-batch = 1):

1. Snapshot accumulated grads as a flat vector before backward: $\mathbf{b} \leftarrow \text{flat}(\theta.\text{grad})$
2. CE over label tokens: $\text{loss} = \mathrm{CE}\big(\text{model}(p, r),\\, L\big)$
3. Backward: $\theta.\text{grad}.\text{backward}()$
4. Snapshot new grads: $\mathbf{a} \leftarrow \text{flat}(\theta.\text{grad})$
5. **Exact per-sample gradient** (micro-batch=1 difference):

$$\delta = \mathbf{a} - \mathbf{b}$$

6. **Gradient norm**:

$$\text{grad\\_norm} = \\|\delta\\|_2$$

7. **Alignment with the clean reference direction** (LESS-style influence):

$$\text{cos\\_sim\\_ref} = \frac{\langle \delta,\\, \mathbf{g}^\ast \rangle}{\\|\delta\\|_2 \\, \\|\mathbf{g}^\ast\\|_2}$$

8. **Within-window gradient conflict**:

$$\text{cos\\_sim\\_global} = \frac{\langle \delta,\\, \mathbf{g}_{\text{acc}} \rangle}{\\|\delta\\|_2 \\, \\|\mathbf{g}_{\text{acc}}\\|_2}$$

9. **Adam-normalized update contribution** (B matrices only — A grads vanish while B is zero-initialized, so element-wise normalization explodes):

$$\text{update\\_contrib} = \frac{\\|\delta_B\\|_2}{\big\\|\sqrt{\mathbf{v}_B}\big\\|_2 + 10^{-8}}$$

### 1.2 End-of-epoch diagnostics (sample 1/8, forward-only)

**Input**: sample $x$, forward logits $\mathbf{Z} \in \mathbb{R}^{L \times V}$

1. **Full-sequence next-token CE** (targets must be REAL token ids, not $-100$ — ignored positions return 0):

$$\text{ce}[t] = -\log p\big(\text{next\\_id}[t] \mid x_{<t}\big)$$

2. **Prompt loss** (corruption pollutes the input → this spikes):

$$\text{user\\_loss} = \frac{1}{|U|} \sum_{t \in U} \text{ce}[t]$$

3. **Label-token entropy**:

$$\text{entropy} = \frac{1}{|L|} \sum_{t \in L} \Big( -\textstyle\sum_v p(v \mid t)\\,\log p(v \mid t) \Big)$$

4. **Hard-token fraction** (threshold $\tau = 4.0$):

$$\text{frac\\_hard} = \frac{\big|\\{t \in L : \text{ce}[t] > \tau\\}\big|}{|L|}$$

5. **Max token loss**: $\text{max\\_token\\_loss} = \max_{t \in L} \text{ce}[t]$
6. **Per-token loss shape**: $\text{skew}$ / $\text{kurt}$ (skewness / kurtosis of $\\{\text{ce}[t] : t \in L\\}$)

### 1.3 Post-training derived (trajectory features)

Per-epoch loss sequence $l_0, l_1, \ldots, l_{E-1}$ ($E$ = epochs):

$$\text{loss\\_mean} = \frac{1}{E}\sum_{e=0}^{E-1} l_e, \qquad \text{loss\\_last} = l_{E-1}, \qquad \text{loss\\_std} = \sqrt{\frac{1}{E}\sum_{e}(l_e - \bar{l})^2}, \qquad \text{loss\\_slope} = l_{E-1} - l_0$$

$$\text{converge\\_epoch} = \min\\{ e : l_e < 2.0 \\} \quad (\text{else } E)$$

$$\text{loss\\_rank} = \frac{1}{E}\sum_{e=0}^{E-1} \text{percentile}_e(l_e)$$

**Loss-trajectory curvature** ($e^2$ coefficient of a quadratic least-squares fit; vectorized as $\mathbf{c} = \mathbf{y}\\,\mathbf{X}^{+}$ with $\mathbf{X} = [\mathbf{1},\\, \mathbf{e},\\, \mathbf{e}^2] \in \mathbb{R}^{E \times 3}$):

$$\text{loss\\_curvature} = c_0 \quad \text{where} \quad [l_e] \approx c_2 e^2 + c_1 e + c_0$$

**Gradient variability & reference-alignment trend**:

$$\text{grad\\_norm\\_cv} = \frac{\sigma(\text{grad\\_norm}_e)}{\mu(\text{grad\\_norm}_e)}, \qquad \text{cos\\_ref\\_trend} = \text{cos\\_ref}_{E-1} - \text{cos\\_ref}_{0}$$

### 1.4 Data-side feature (no training needed)

$$\text{text\\_nn\\_sim}(x) = 1 - \min_{x' \neq x}\; \cos\!\big(\text{TF-IDF}(x),\\, \text{TF-IDF}(x')\big)$$

TF-IDF: 1-2 grams, $\min\\_df = 10$, $\text{sublinear\\_tf} = \text{True}$, $\max\\_features = 200{,}000$.

---

## 2. Per-noise-type detection algorithms

### 2.1 garbled (AUC 0.9996) — training-dynamics probe

**Features** (by discriminative power): `loss_curvature` (0.985) > `user_loss` (0.979) > `entropy` (0.971) > `frac_hard` (0.954)

**Algorithm**:

1. Compute the 95th percentile of each metric over normal samples: $q_{\text{ul}},\ q_{\text{ent}},\ q_{\text{curv}}$
2. Flag (OR-combination for recall; a single metric already reaches AUC > 0.97):

$$s = \big(\text{user\\_loss} > q_{\text{ul}}\big) \lor \big(\text{entropy} > q_{\text{ent}}\big) \lor \big(\text{loss\\_curvature} > q_{\text{curv}}\big)$$

Optional reinforcement (5-epoch setup): $\text{converge\\_epoch} = E$ (never converges), or $\text{frac\\_hard}$ that never drops.

### 2.2 duplicate (AUC 0.974) — text-similarity dedup

**Feature**: `text_nn_sim` (0.939) dominates; **training-side metrics are useless (loss AUC 0.37, inverted direction)**

**Algorithm**:

1. $X \leftarrow \text{TfidfVectorizer}(1\text{-}2\text{gram},\\, \min\\_df{=}10,\\, \text{sublinear\\_tf},\\, \max\\_features{=}200\text{K})(\text{texts})$
2. $(\text{dist}, \\_) \leftarrow \text{NearestNeighbors}(k{=}2,\\, \text{metric}{=}\text{cosine}).\text{fit}(X).\text{kneighbors}(X)$
3. $\text{sim}_i = 1 - \text{dist}[i, 1]$ (nearest neighbor excluding self)
4. Flag:

$$s = \big(\text{sim}_i > 0.9\big) \quad (\text{copies} \approx 1.0;\ \text{normal} \approx 0.2\text{-}0.5)$$

### 2.3 unrelated (AUC 0.923) — loss volatility

**Features**: `loss_curvature` (0.830) > `loss_std` (0.827) > `grad_norm_mean` (0.764)

**Algorithm**:

1. Compute the 95th percentile $q_{\text{std}}$ of $\text{loss\\_std}$ over normal samples
2. Flag:

$$s = \big(\text{loss\\_std} > q_{\text{std}}\big) \land \big(\text{loss\\_slope not strongly negative}\big)$$

> Rationale: genuinely hard normal samples descend smoothly (negative slope, low volatility); mismatched samples are volatile with slow descent → abnormal curvature.

### 2.4 mixed (AUC 0.850) — combined classifier

Features: all 19 dims → $\text{StandardScaler}$ → LogisticRegression ($\max\\_iter = 2000$) or RandomForest ($n\\_estimators = 200$); evaluate with a 70/30 split.

### 2.5 keyword (AUC 0.531) — infeasible; needs model-external tools

All 19 training-side metrics overlap completely with normal samples. Promising directions: NER entity-consistency checks (abnormal entity frequencies), counterfactual perturbation (does loss drop sharply when the entity is swapped back), external-knowledge verification.

---

## 3. General detection pipeline (deployment recipe)

**Input**: training corpus $D$, clean validation set $C$ (disjoint from $D$, ~2.7% size)

```
Phase A (pre-training, once):
  1. Compute reference direction g* on C (forward + backward)
Phase B (during training; detect within epoch 0-1, per sample in real time):
  2. Train with micro-batch=1; compute the 6 features of §1.1 per sample
  3. At each epoch end: compute §1.2 diagnostic features on the 1/8 subsample
  4. At each epoch end: one forward pass over ALL samples → loss_rank
Phase C (classification & cleaning):
  5. Pick the §2 algorithm per noise type, or classify with the 19-dim LR
  6. Use adaptive 95th-percentile thresholds from the normal samples
  7. Remove (or down-weight) flagged samples, resume training
Notes:
  - Detect as early as possible: AUC decays monotonically over epochs
    (garbled: 0.985 → 0.865)
  - Tune thresholds per noise type; never use a single global threshold
  - Training-side and data-side features are complementary: duplicates are
    only detectable from the data side
```

---

## 4. Computational cost (measured, RTX 5090)

**Key fact: per-sample gradients are NOT an extra backward pass.** With
micro-batch=1 the training backward already runs per sample; the algorithm
only snapshots the accumulated gradients before backward and subtracts after
(flat-vector copies + dot products), measured at ~12 ms of ~160 ms per sample
≈ **5-8% training overhead**. Token-level per-token attribution is the only
expensive operation and should stay offline on small samples.

| Feature | Cost | Note |
|---|---|---|
| `loss` / `entropy` / `user_loss` / `frac_hard` / `max_token_loss` | ~0% | by-product of the training/diagnostic forward |
| `grad_norm` / `cos_sim_ref` / `cos_sim_global` / `update_contrib` | **+5-8% training time** | snapshot+diff copies and dots; the backward itself is part of training |
| loss-trajectory features (curvature/rank/converge_epoch) | 0% | derived post-training from stored per-epoch losses |
| diagnostic features (1/8 subsample, forward-only) | ~30 s/epoch | one subsample forward per epoch |
| `text_nn_sim` (TF-IDF + kNN) | ~2 min / 15K (CPU) | no model needed |
| token-level per-token gradients (top-24/sample) | offline, 60+60 samples ~3-5 min/dataset | one backward per hard token; **do not run online** |

**Deployment fallbacks** (when even 5-8% is too much):

1. **Forward-only features**: `user_loss` + `entropy` + loss trajectory alone detect garbled (AUC 0.97+), zero gradient overhead;
2. **Larger micro-batch** (8/16): loss-side features unaffected; gradient features degrade to batch-level (usable for batch-level screening);
3. **Data-side first**: deduplicate with TF-IDF on CPU, fully offline.

---

*Implementation: `scripts/2_train/train.py` (feature capture), `scripts/3_analysis/analyze_detection.py` (derived features & classifiers), `scripts/3_analysis/analyze_token_level.py` (token-level). Validation data: `results/ratio10/auc_univariate.csv`, `results/ratio10/detection_multivariate.csv`.*
