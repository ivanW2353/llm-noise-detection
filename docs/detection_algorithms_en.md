# Noise Detection Algorithms (validated specification)

> All algorithms validated on the 10%-ratio experiment (Qwen2.5-3B + LoRA, dolly-15k).
> Goal: separate noisy samples from normal ones during / after training.
> Validated AUC: garbled 0.9996 · duplicate 0.974 · unrelated 0.923 · mixed 0.850 · keyword 0.531 (infeasible).

---

## 0. Notation & prerequisites

- Sample $x = (p, r)$: prompt $p$, assistant response $r$; label-token set $L$, user-token set $U$
- Micro-batch = 1 (prerequisite for exact per-sample gradients); per-sample loss & gradient recorded
- $v$: Adam second moment (exp_avg_sq), snapshotted after each optimizer step
- Reference direction $\mathbf{g}^*$: mean LoRA gradient over $N_{\text{ref}}=200$ clean held-out samples, computed pre-training, normalized to unit length
- Threshold guidance: use **percentile-adaptive thresholds** on the normal-sample distribution (default 95th percentile); absolute values depend on the model/data

---

## 1. Per-sample feature computation

### 1.1 Computed live during training (every sample × every epoch)

**Input**: sample $x$, LoRA params $\theta$, accumulated window gradient $\mathbf{g}_{\text{acc}}$

```
1. before ← flat(θ.grad)                      # snapshot accumulated grads (flat vector)
2. loss ← CE(model(p, r), L)                    # CE over label tokens only
3. θ.grad.backward()
4. after ← flat(θ.grad)
5. δ ← after − before                          # exact per-sample gradient (micro-batch=1)
6. grad_norm ← ||δ||₂
7. cos_sim_ref ← ⟨δ, g*⟩ / (||δ||₂ · ||g*||₂)   # alignment with the clean reference direction
8. cos_sim_global ← ⟨δ, g_acc⟩ / (||δ||₂ · ||g_acc||₂)   # within-window gradient conflict
9. update_contrib ← ||δ_B||₂ / (||√v_B||₂ + 1e-8)          # B matrices only (A grads vanish while B=0)
```

### 1.2 End-of-epoch diagnostics (sample 1/8, forward-only)

**Input**: sample $x$, forward logits $\mathbf{Z} \in \mathbb{R}^{L \times V}$

```
1. ce[t] ← −log softmax(Z[t])[next_id[t]]      # full-sequence next-token CE (REAL ids, not -100)
2. user_loss ← mean{ ce[t] : t ∈ U }           # prompt loss
3. entropy ← mean{ −Σ_v p(v|t)·log p(v|t) : t ∈ L }   # label-token entropy
4. frac_hard ← |{ t ∈ L : ce[t] > 4.0 }| / |L|
5. max_token_loss ← max{ ce[t] : t ∈ L }
6. skew/kurt ← skewness/kurtosis of {ce[t] : t ∈ L}
```

### 1.3 Post-training derived (trajectory features)

Let the per-epoch loss sequence be $l_0, ..., l_{E-1}$ ($E$ = epochs):

```
loss_mean    ← mean(l_e)
loss_last    ← l_{E-1}
loss_std     ← std(l_e)
loss_slope   ← l_{E-1} − l_0
converge_epoch ← min{ e : l_e < 2.0 }, else E
loss_rank    ← mean_e( percentile_e(l_e) )
loss_curvature ← a from quadratic least-squares fit of [l_e]:
                X = [1, e, e²] (E×3); coeffs = y @ pinv(X)ᵀ; take the e² coefficient
grad_norm_cv ← std(grad_norm_e) / mean(grad_norm_e)
cos_ref_trend ← cos_ref_{E-1} − cos_ref_0
```

### 1.4 Data-side feature (no training needed)

```
text_nn_sim(x) ← 1 − min_{x' ≠ x} cosine( TF-IDF(x), TF-IDF(x') )
TF-IDF: 1-2 grams, min_df=10, sublinear_tf=True, max_features=200,000
```

---

## 2. Per-noise-type detection algorithms

### 2.1 garbled (AUC 0.9996) — training-dynamics probe

**Features** (by discriminative power): `loss_curvature` (0.985) > `user_loss` (0.979) > `entropy` (0.971) > `frac_hard` (0.954)

**Algorithm**:
```
Input: per-sample user_loss, entropy, loss_curvature
1. Compute the 95th percentile of each metric over normal samples: q_ul, q_ent, q_curv
2. Flag s = (user_loss > q_ul) OR (entropy > q_ent) OR (loss_curvature > q_curv)
   # single metric already reaches AUC > 0.97; OR-combination for recall
Optional reinforcement (5-epoch setup): converge_epoch = E (never converges), or frac_hard that never drops
```

### 2.2 duplicate (AUC 0.974) — text-similarity dedup

**Feature**: `text_nn_sim` (0.939) dominates; **training-side metrics are useless (loss AUC 0.37, inverted direction)**

**Algorithm**:
```
Input: all sample texts
1. X ← TfidfVectorizer(ngram(1,2), min_df=10, sublinear_tf, max_features=200K)(texts)
2. dist, _ ← NearestNeighbors(k=2, metric="cosine").fit(X).kneighbors(X)
3. sim_i ← 1 − dist[i, 1]            # nearest neighbor excluding self
4. Flag s = (sim_i > 0.9)             # copies ≈ 1.0; normal ≈ 0.2-0.5
```

### 2.3 unrelated (AUC 0.923) — loss volatility

**Features**: `loss_curvature` (0.830) > `loss_std` (0.827) > `grad_norm_mean` (0.764)

**Algorithm**:
```
Input: per-sample cross-epoch loss trajectory and grad_norm series
1. Compute the 95th percentile q_std of loss_std over normal samples
2. Flag s = (loss_std > q_std) AND (loss_slope not strongly negative)
   # genuinely hard normal samples: smooth descent (negative slope, low volatility)
   # mismatched samples: volatile, slow descent → abnormal curvature
```

### 2.4 mixed (AUC 0.850) — combined classifier

```
Features: all 19 dims → StandardScaler → LogisticRegression (max_iter=2000) or
          RandomForest (n_estimators=200); evaluate with a 70/30 split
```

### 2.5 keyword (AUC 0.531) — infeasible; needs model-external tools

All 19 training-side metrics overlap completely with normal samples. Promising directions: NER entity-consistency checks (abnormal entity frequencies), counterfactual perturbation (does loss drop sharply when the entity is swapped back), external-knowledge verification.

---

## 3. General detection pipeline (deployment recipe)

```
Input: training corpus D, clean validation set C (disjoint from D, ~2.7% size)
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

## 4. Complexity

| Step | Complexity |
|---|---|
| Per-sample gradient features | ~1% training overhead (flat-buffer fill + dots per sample) |
| Diagnostic features | one 1/8-subsample forward per epoch (~30 s/epoch) |
| text_nn_sim | TF-IDF + kNN on 15K samples: ~2 minutes (CPU) |
| Classifier | seconds (LR/RF) |

---

*Implementation: `scripts/train.py` (feature capture), `scripts/analyze_detection.py` (derived features & classifiers), `scripts/analyze_token_level.py` (token-level). Validation data: `results/auc_univariate_ratio10.csv`, `results/detection_multivariate_ratio10.csv`.*
