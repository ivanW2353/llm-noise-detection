# Impact of Noisy Samples on LLM Fine-tuning: Per-Sample Metric Tracking & Detection Analysis

> Experiment period: 2026-08-12 ~ 2026-08-14
> Base model: Qwen2.5-3B-Instruct (LoRA r=32, 59.9M trainable params) · Training data: databricks-dolly-15k
> Noise ratio: 10% · 5 epochs · 14,611 training samples per run · single RTX 5090

---

## 1. Experimental Setup

### 1.1 Research questions

Two core questions:
1. **Impact**: how much do four types of data noise (garbled text / duplicates / context mismatches / keyword swaps) affect the fine-tuning process and the final model capability?
2. **Detection**: using only per-sample metrics trackable during training (loss / gradients / entropy / ...), can noisy samples be separated from normal ones? Which metrics work for which noise type?

### 1.2 Datasets (6, identical sample order, fixed seed)

| Dataset | Noise construction | Training rows | Noisy samples |
|---|---|---|---|
| `clean` | original data (baseline) | 14611 | 0 |
| `garbled` | 10% samples corrupted with mojibake: Unicode substitution (~12%/char), insertion (~3%), adjacent-char swaps (~2%); whitespace structure preserved | 14611 | 1461 |
| `duplicate` | 10% (1461) extra rows that are byte-identical copies | 16072 | 1461 |
| `unrelated` | 10% samples keep their instruction but receive a fluent, correct response taken from a *different* category | 14611 | 1461 |
| `keyword` | 10% samples with only numbers / years / proper nouns replaced (person names, song titles → random names / orgs), grammar preserved | 14611 | 1461 |
| `mixed` | the four types above at 2.5% each, 10% total (366 copies) | 14976 | 1461 |

- All datasets share the same sample order and sample_ids (duplicate/mixed copies appended at the end) → per-sample metrics are directly comparable across the 6 runs;
- 400 **shared clean held-out samples** (`heldout.jsonl`, disjoint from training) serve to: (a) compute the reference gradient direction before training (LESS-style influence baseline); (b) evaluate held-out generalization loss every 200 steps;
- Every sample carries `noise_label` / `noise_type` / `category` (dolly's 8 task categories).

### 1.3 Training configuration

| Setting | Value | Rationale |
|---|---|---|
| Micro-batch | 1 | per-sample gradients are exactly captured (snapshot accumulated grads before backward, subtract) |
| Gradient accumulation | 16 | one optimizer step per 16 samples (4,570~5,025 steps per run) |
| Learning rate | 2e-4, cosine decay + 3% warmup | identical across all 6 runs |
| Optimizer | AdamW (betas 0.9/0.999) | |
| Precision | bf16 + flash-attention-2 | |
| Sequence length | 1024 (truncation keeps the assistant response) | avoids NaN loss from zero label tokens |
| Duration | 3.3~3.9 h per run | clean 3.69 h / mixed 3.91 h / duplicate 3.58 h |

### 1.4 Recorded metrics (three levels, 19+ features)

**Sample-level metrics** — captured per sample per epoch during training
(micro-batch=1 difference method: snapshot the accumulated gradients before
backward as $\mathbf{b}$, after backward as $\mathbf{a}$; the difference
$\delta = \mathbf{a} - \mathbf{b}$ is the sample's exact gradient; total
overhead only +5-8% of training time), 6 features:

**① loss** — mean cross-entropy over label tokens:

$$\text{loss} = -\frac{1}{|L|}\sum_{t \in L} \log p_\theta(\text{next\_id}[t] \mid x_{<t})$$

- **Meaning**: how hard this sample is to fit right now; the most direct quantity in training monitoring;
- **Detection intuition**: "unlearnable" noise (garbled) stays high forever; but there is an **inverted trap** — noise that is quickly memorized (duplicate copies) shows *lower* loss than normal samples (detection AUC 0.37, direction reversed);
- **Observed**: garbled highest throughout (0.70 at epoch 4 vs clean 0.51); duplicate lowest (0.43).

**② grad_norm** — L2 norm of the sample's LoRA gradient: $\text{grad\_norm} = \|\delta\|_2$

- **Meaning**: how hard this sample "pushes" the parameters;
- **Detection intuition**: large gradients aligned with the reference direction = valuable hard samples; anomalous magnitudes = noise suspects;
- **Observed**: unrelated elevated (0.764), duplicate depressed (0.343, inverted), garbled 0.829.

**③ cos_sim_ref** — cosine similarity with the **clean reference direction** (LESS-style influence):

$$\text{cos\_sim\_ref} = \frac{\langle \delta,\, \mathbf{g}^{\ast} \rangle}{\|\delta\|_2 \, \|\mathbf{g}^{\ast}\|_2}$$

where $\mathbf{g}^{\ast}$ is the mean LoRA gradient over 200 held-out clean samples computed pre-training (unit vector).

- **Meaning**: the angle between this sample's gradient and the "clean training direction" — near 1 means it pushes the model along the clean direction; near 0 or negative means it conflicts;
- **Implementation**: computed once before training, reused throughout; it is an efficient approximation of LESS influence (Xia et al. 2024);
- **Observed**: moderate univariate AUC (0.58-0.62) but an important member of the combined feature set (top LR feature weights).

**④ cos_sim_global** — cosine with the current accumulation-window (16-sample) gradient:

$$\text{cos\_sim\_global} = \frac{\langle \delta,\, \mathbf{g}_{\text{acc}} \rangle}{\|\delta\|_2 \, \|\mathbf{g}_{\text{acc}}\|_2}$$

- **Meaning**: within-batch gradient consistency; negative = this sample conflicts with the dominant update direction of the surrounding 15 samples;
- **Observed**: relatively effective for duplicate (0.610) — copies systematically conflict with the window's other samples.

**⑤ update_contrib** — Adam-normalized update contribution (B matrices only):

$$\text{update\_contrib} = \frac{\|\delta_B\|_2}{\big\|\sqrt{\mathbf{v}_B}\big\|_2 + 10^{-8}}$$

- **Motivation**: raw gradient norms ignore Adam's historical scale; dividing by the second-moment RMS reflects how large this sample's push is *relative to recent gradient magnitudes*;
- **Implementation detail**: B matrices only — LoRA B is zero-initialized, so A's gradients vanish early and element-wise normalization explodes (measured 2.3e8); B-only is mandatory;
- **Observed**: garbled 0.836 / unrelated 0.724 / keyword 0.636 / duplicate 0.330 (inverted).

**⑥ tokens** — number of label tokens: a sequence-length control variable for stratified analysis (longer samples have more stable losses).

**Diagnostic-level metrics** — forward-only pass on a 1/8 subsample at each epoch end (~30 s/epoch, `diag_epoch*.jsonl`), all based on the full-sequence next-token CE ($\text{ce}[t]$; targets must be REAL token ids — using $-100$ makes cross\_entropy return 0, which was the root cause of the user_loss-always-zero bug):

**⑦ max_token_loss** — $\max_{t \in L} \text{ce}[t]$: the largest per-token loss in a sample; captures "local extremes" — corrupted samples have individual tokens with extreme losses;

**⑧ frac_hard** — fraction of tokens with loss > 4.0: global hardness; highest for garbled and barely decays with training (epoch 4: 4.84% vs clean 4.36%); lowest for duplicate (3.82%, fully memorized);

**⑨ user_loss** — mean prompt loss $\frac{1}{|U|}\sum_{t \in U} \text{ce}[t]$: an **input-side signal** — garbled corrupts the input too → spikes (AUC 0.979); keyword/unrelated only alter the output → this stays normal (AUC ~0.5, negative information);

**⑩ entropy** — mean next-token entropy over label tokens: model certainty about its output; extremely high on corrupted tokens (AUC 0.971); low on memorized ones (duplicate 0.406);

**⑪ token_loss_skew/kurt** — skewness/kurtosis of the per-token loss distribution: counter-intuitively, garbled makes *almost all* tokens hard → the distribution becomes flat → skew near 0 (AUC 0.064) — "a signal of having no signal";

**⑫ top-32 hard-token details** — position / token id / loss (`token_diag_epoch*.jsonl`): for offline token-level localization and attribution.

**Derived features** — computed post-training from the per-epoch sequences (zero cost):

**⑬ loss_mean / loss_last / loss_std / loss_slope** — level / final / cross-epoch volatility / first-to-last change: $\text{loss\_std}$ is the main feature for unrelated (0.827) — genuinely hard samples descend smoothly while mismatched samples oscillate;

**⑭ converge_epoch** — $\min\{e : l_e < 2.0\}$ (else E): convergence speed — 61% of garbled samples never converge vs duplicate copies converging in 0.32 epochs on average, a perfect mirror;

**⑮ loss_rank** — mean within-epoch loss percentile: removes global level drift, comparable across runs and epochs;

**⑯ loss_curvature** — quadratic-fit coefficient of the loss trajectory ($[l_e] \approx c_2 e^2 + c_1 e + c_0$, take $c_2$): **the single strongest feature across the whole experiment** — garbled 0.985 / unrelated 0.830 / keyword 0.669, jointly capturing "cannot learn" and "volatile learning" anomalies;

**⑰ grad_norm_cv / cos_ref_trend** — gradient volatility ($\sigma/\mu$) / reference-alignment trend (last minus first): temporal evolution of gradient and direction information;

**⑱ text_nn_sim** — TF-IDF (1-2 grams) nearest-neighbor cosine similarity: a **data-side feature completely independent of training** — the only effective tool for duplicate (0.939): copies have similarity ≈1.0 while normal samples ≈0.2-0.5.

**Token level (offline; 60 noise + 60 normal samples per dataset)**: for each sample, the top-24 hardest label tokens are individually back-propagated (`autograd.grad`) to obtain exact per-token LoRA gradient norms and cosine similarities (cost: one backward per hard token — offline on small samples only).

---

## 2. Training Dynamics: How Noise Affects Training

### 2.1 Training loss trajectory (per-epoch mean)

| run | epoch 0 | epoch 1 | epoch 2 | epoch 3 | epoch 4 | vs clean (final) |
|---|---|---|---|---|---|---|
| clean | 1.366 | 1.127 | 0.861 | 0.642 | 0.514 | — |
| garbled | **1.669** | 1.386 | 1.093 | 0.848 | **0.702** | +37% |
| unrelated | 1.494 | 1.248 | 0.896 | 0.641 | 0.498 | −3% |
| keyword | 1.427 | 1.164 | 0.894 | 0.665 | 0.533 | +4% |
| mixed | 1.496 | 1.207 | 0.904 | 0.662 | 0.525 | +2% |
| duplicate | 1.349 | 1.077 | 0.794 | 0.557 | **0.425** | **−17%** |

![Training loss trajectory](../results/charts/loss_trajectory_ratio10.png)

**Interpretation:**
1. **garbled keeps the highest loss throughout** — +22% over clean at epoch 0, widening to +37% by epoch 4. Corrupted samples have near-zero information density; the model can never "learn" them, so they keep inflating the mean;
2. **duplicate converges to the lowest loss** (0.425, 17% below clean) — copies are memorized precisely within the first epoch. **Low training loss is here a memorization/overfitting signal, not a health signal**;
3. unrelated / keyword / mixed track clean almost exactly — **mean loss is completely insensitive to these semantic noise types**, which successfully disguise themselves as merely difficult samples.

### 2.2 Convergence analysis (converge_epoch: first epoch with loss < 2.0)

| run | noise mean | normal mean | noise "never converged" | normal "never converged" |
|---|---|---|---|---|
| garbled | **4.06** | 0.63 | **61%** | 4% |
| unrelated | 1.37 | 0.63 | 3% | 4% |
| keyword | 1.07 | 0.63 | 7% | 4% |
| mixed | 1.45 | 0.61 | 14% | 4% |
| duplicate | **0.32** | 0.60 | 2% | 3% |

**Interpretation:** convergence speed is an excellent discriminator:
- **garbled**: 61% of noisy samples never drop below 2.0 after 5 epochs — "cannot learn" is the defining property of corruption;
- **duplicate**: copies converge in 0.32 epochs on average, *faster* than normal samples — a perfect mirror image of garbled: **one never converges, the other converges instantly**;
- unrelated/keyword converge only 0.4~0.7 epochs slower — weak but present signal.

### 2.3 Per-epoch diagnostics: hard-token fraction (frac_hard, loss > 4)

| run | end of epoch 0 | end of epoch 2 | end of epoch 4 |
|---|---|---|---|
| garbled | **7.40%** | 5.80% | **4.84%** |
| keyword | 7.17% | 5.48% | 4.48% |
| unrelated | 7.26% | 5.51% | 4.36% |
| clean | 7.07% | 5.35% | 4.36% |
| mixed | 7.49% | 6.31% | 4.34% |
| duplicate | 7.01% | 5.44% | **3.82%** |

**Interpretation:** hard-token fractions fall over training in every run (the model adapts); garbled stays highest throughout (corrupted tokens are *always* hard tokens), duplicate ends lowest (memorized → no hard tokens left). All runs start nearly equal — the differences are amplified *during* training.

### 2.4 Held-out clean loss (generalization damage, every 200 steps)

| run | initial (step 200) | final | increase |
|---|---|---|---|
| clean | 1.628 | 2.051 | +0.423 |
| keyword | 1.627 | **2.044** | **+0.417 (smallest)** |
| garbled | 1.629 | 2.059 | +0.430 |
| mixed | 1.624 | 2.081 | +0.457 |
| unrelated | 1.626 | 2.090 | +0.465 |
| duplicate | 1.626 | **2.143** | **+0.517 (largest)** |

![Held-out loss trajectory](../results/charts/tb_heldout_trajectory_ratio10.png)

**Interpretation:**
1. Held-out loss rises in **every** run, including clean — 5-epoch LoRA fine-tuning on dolly-15k is itself overfitting (+0.42 is the "baseline overfitting");
2. **duplicate overfits the most** (+0.517, 22% more than clean) — memorized copies squeeze generalization further;
3. **keyword overfits the least** (+0.417) — consistent with §5: swapping a few words is nearly harmless;
4. unrelated (+0.465) damages more than garbled (+0.430) — **semantic mismatches mislead the model more than surface corruption**, one of the most counter-intuitive yet important findings of this study.

### 2.5 Per-layer gradient norms (final training window; first / middle / last layers)

| run | layer 0 (near input) | layer 18 (middle) | layer 35 (near output) |
|---|---|---|---|
| garbled | **3.00** | 3.27 | **5.10** |
| unrelated | **3.15** | 3.07 | 3.91 |
| keyword | 2.57 | 3.17 | 3.86 |
| clean | 2.13 | 3.21 | 3.82 |
| mixed | 2.04 | 3.46 | 3.70 |
| duplicate | **1.29** | **1.65** | **1.83** |

![Per-run layer gradient norms](../results/charts/tb_layer_gradnorm_ratio10.png)

**Interpretation:** garbled/unrelated keep much higher gradient norms in the shallow layers (+41%/+48% over clean at layer 0) — noise keeps stimulating large updates in the input-encoding layers; duplicate shows the smallest norms everywhere (gradients vanish once memorized).

---

## 3. Sample-Level Noise Detection

### 3.1 Full univariate AUC table (noise vs normal samples, same run; 19 metrics)

| Metric | garbled | duplicate | unrelated | keyword | mixed |
|---|---|---|---|---|---|
| loss_mean | 0.955 | **0.369** | 0.724 | 0.627 | 0.627 |
| loss_last | 0.865 | 0.372 | 0.575 | 0.572 | 0.553 |
| loss_std | 0.780 | 0.516 | 0.827 | 0.649 | 0.695 |
| loss_slope | 0.206 | 0.487 | 0.167 | 0.343 | 0.302 |
| converge_epoch | 0.941 | 0.458 | 0.714 | 0.602 | 0.648 |
| loss_rank | 0.936 | 0.355 | 0.699 | 0.617 | 0.606 |
| **loss_curvature** | **0.985** | 0.432 | **0.830** | **0.669** | 0.691 |
| grad_norm_mean | 0.829 | 0.343 | 0.764 | 0.639 | 0.580 |
| grad_norm_cv | 0.166 | 0.578 | 0.435 | 0.451 | 0.438 |
| cos_ref_mean | 0.583 | 0.497 | 0.575 | 0.569 | 0.517 |
| cos_ref_trend | 0.436 | 0.369 | 0.456 | 0.451 | 0.421 |
| cos_global_mean | 0.579 | 0.610 | 0.503 | 0.497 | 0.554 |
| update_contrib_mean | 0.836 | 0.330 | 0.724 | 0.636 | 0.579 |
| max_token_loss | 0.809 | 0.352 | 0.650 | 0.613 | 0.585 |
| frac_hard | 0.954 | 0.369 | 0.719 | 0.634 | 0.610 |
| **user_loss** | **0.979** | 0.510 | 0.488 | 0.550 | 0.584 |
| **entropy** | **0.971** | 0.406 | 0.637 | 0.638 | 0.630 |
| token_loss_skew | 0.064 | 0.520 | 0.532 | 0.437 | 0.467 |
| **text_nn_sim** | 0.358 | **0.939** | 0.725 | 0.472 | 0.716 |

**Per-noise-type reading:**

- **garbled (easiest to detect)**: locked by `loss_curvature` (0.985), `user_loss` (0.979), `entropy` (0.971) — corruption pollutes both input and output, and the model shows "cannot learn + uncertain" on both sides. Note `token_loss_skew` ≈ 0.06: corruption makes *almost all* tokens hard (rather than a few extreme ones), so skewness carries no signal;
- **duplicate**: `text_nn_sim` (0.939) stands alone; every training-side metric has AUC ≤ 0.61, and `loss_mean` (0.369) is **below 0.5** — copies have *lower* loss than normal samples, inverting the training-side signal. **For duplicates, the data-side feature (text similarity) is the only effective tool; training dynamics are not just weak but opposite in direction**;
- **unrelated**: `loss_curvature` (0.830), `loss_std` (0.827), `grad_norm_mean` (0.764) — cross-epoch loss volatility and curvature expose "fluent but mismatched" responses;
- **keyword**: every metric sits at AUC 0.47~0.67 — **no single metric can separate them**; sample-level detection fails here;
- **mixed**: features dilute each other; `text_nn_sim` (0.716) still catches the duplicate subset.

### 3.2 Detectability over training time (per-epoch loss AUC)

| run | epoch 0 | epoch 1 | epoch 2 | epoch 3 | epoch 4 |
|---|---|---|---|---|---|
| garbled | **0.985** | 0.969 | 0.930 | 0.889 | 0.865 |
| unrelated | 0.829 | 0.760 | 0.671 | 0.601 | 0.575 |
| keyword | 0.672 | 0.623 | 0.604 | 0.584 | 0.572 |
| duplicate | 0.435 | 0.365 | 0.325 | 0.314 | 0.372 |

**Key finding: detectability decays monotonically with training.** Garbled's loss AUC drops from 0.985 at epoch 0 to 0.865 at epoch 4; unrelated from 0.829 to 0.575 — the model gradually "adapts" to the noise, shrinking the loss gap. **Practical implication: data cleaning should happen early (within the first epoch), not after training completes.**

### 3.3 Multivariate classifiers (LR / Random Forest, 19 features, 70/30 split)

| Noise type | LR AUC | RF AUC | Accuracy | Confusion (TN,FP/FN,TP) | Verdict |
|---|---|---|---|---|---|
| garbled | **0.9996** | 0.9996 | 99.3% | 248,2 / 0,22 | near-perfect separation |
| duplicate | 0.974 | 0.973 | 95.3% | 273,3 / 11,12 | strong |
| unrelated | 0.923 | 0.887 | 94.1% | 247,9 / 7,9 | strong |
| mixed | 0.850 | 0.827 | 92.1% | 245,10 / 12,12 | moderate |
| keyword | **0.531** | 0.551 | (all-normal) | 255,1 / 16,0 | **not separable** |

![RF ROC curves](../results/charts/roc_multivariate_ratio10.png)

**Interpretation:** combining all 19 features, the first four noise types reach practical separability (AUC ≥ 0.85). The keyword classifier with AUC 0.53 classifies *everything as normal* (zero recall in the confusion matrix) — which is the honest outcome: keyword noise overlaps completely with normal samples in this feature space.

### 3.4 Noise vs normal distributions of key metrics

<center>

| Loss & gradient | Input-side features |
|---|---|
| ![loss_mean](../results/charts/metric_dist/metric_dist_loss_mean_ratio10.png) | ![user_loss](../results/charts/metric_dist/metric_dist_user_loss_ratio10.png) |
| ![grad_norm](../results/charts/metric_dist/metric_dist_grad_norm_mean_ratio10.png) | ![entropy](../results/charts/metric_dist/metric_dist_entropy_ratio10.png) |
| ![cos_ref](../results/charts/metric_dist/metric_dist_cos_ref_mean_ratio10.png) | ![text_nn_sim](../results/charts/metric_dist/metric_dist_text_nn_sim_ratio10.png) |

</center>

> All 19 per-metric figures are in `results/charts/metric_dist_*_ratio10.png`; each is a box-plot of 5 noise types × (noise / normal). Visually: garbled's user_loss/entropy barely overlap with normal; duplicate's text_nn_sim is bimodal (≈1.0 for copies); keyword overlaps everywhere.

### 3.5 PCA projection of sample features

![PCA projection](../results/charts/pca_metrics_ratio10.png)

**Interpretation:** on the first two standardized PCs, garbled (red) forms a cleanly separated cluster from normal (grey); duplicate (blue) separates along the text_nn_sim direction; unrelated (green) overlaps partially; keyword (purple) is fully embedded inside the normal cluster — consistent with the AUC conclusions.

### 3.6 Transferability across task types (stratified by dolly's 8 categories; RF AUC)

| category | n | #noise | RF AUC |
|---|---|---|---|
| closed_qa | 650 | 44 | **0.987** |
| creative_writing | 295 | 27 | 0.979 |
| information_extraction | 488 | 32 | 0.977 |
| brainstorming | 761 | 37 | 0.977 |
| general_qa | 790 | 56 | 0.943 |
| summarization | 527 | 55 | 0.942 |
| open_qa | 1342 | 102 | 0.919 |
| classification | 693 | 57 | **0.870 (hardest)** |

**Noise × category matrix (excerpt)**

| category | duplicate | garbled | keyword | unrelated |
|---|---|---|---|---|
| open_qa | 0.980 | 0.992 | 0.535 | 0.925 |
| brainstorming | 0.946 | **1.000** | 0.431 | 0.972 |
| classification | 0.667 | **1.000** | 0.604 | 0.896 |
| summarization | 0.870 | 1.000 | 0.396 | 1.000 |

**Interpretation:**
1. The method works across all 8 task types (AUC 0.87~0.99) — no per-category recalibration needed;
2. **classification is the hardest** — short structured responses weaken token-level signals (entropy / user_loss / frac_hard), leaving loss-trajectory features as the main handle;
3. **garbled approaches 1.0 in every category** — the most universal target; keyword stays weak everywhere (0.40~0.60), confirming the blind spot is task-independent.

---

## 4. Token-Level Detection (exact per-token gradient attribution)

### 4.1 Method

Per noisy dataset, sample **60 noise + 60 normal** samples; run the *final* model of that run forward (retaining the graph); take the top-24 hardest label tokens of each sample; for each token individually `autograd.grad(loss_t, lora_params, retain_graph=True)`; derive `hard_loss_mean`, `hard_gradnorm_mean`, `hard_cos_ref_mean` (cosine to the clean reference direction), `pos_std` (spread of hard-token positions).

### 4.2 Detection AUC

| Feature | garbled | duplicate | unrelated | keyword |
|---|---|---|---|---|
| hard_loss_mean | **0.767** | 0.414 | 0.582 | 0.486 |
| hard_gradnorm_mean | **0.767** | 0.414 | 0.601 | 0.502 |
| hard_cos_ref_mean | 0.624 | 0.588 | 0.553 | 0.478 |
| pos_std | 0.571 | 0.461 | 0.533 | 0.411 |

<center>

| garbled | duplicate |
|---|---|
| ![garbled per-token losses](../results/charts/token_curve_ratio10_garbled.png) | ![duplicate per-token losses](../results/charts/token_curve_ratio10_duplicate.png) |
| unrelated | keyword |
| ![unrelated per-token losses](../results/charts/token_curve_ratio10_unrelated.png) | ![keyword per-token losses](../results/charts/token_curve_ratio10_keyword.png) |

</center>

> Each figure shows position–loss scatter plots of the top-k hardest tokens for 3 noise samples of the corresponding type (hardest tokens only, not the full sequence).

**Interpretation:**
1. **garbled remains the most separable at token level** (0.77) — corrupted positions produce locally extreme losses and anomalous gradients;
2. **duplicate's token-level AUC is below 0.5** — its tokens are perfectly memorized (very low loss) and indistinguishable from normal ones. Its detectability is 100% data-side (text similarity); training dynamics fail at token level too;
3. Token-level AUCs are generally far below sample-level ones (0.77 vs 0.9996) — a single hard token has limited signal-to-noise; **sample-level aggregation (across tokens and epochs) is the reliable detection scale**;
4. `pos_std` is weak for every type (~0.4~0.57) — the spatial arrangement of hard tokens is not discriminative.

### 4.3 Known limitation

The garbled-localization check (`loc_mismatch_frac`) returned 0 — character-level corruption shifts tokenization boundaries, so position-aligned comparison against clean text fails. A sequence-alignment approach (e.g., edit-distance alignment) is needed to correctly locate corrupted tokens; left as future work.

---

## 5. Impact on Final Model Capability (7 models × 7 benchmarks)

### 5.1 Overall comparison

| Model | MMLU | GSM8K | HellaSwag | ARC | BBH | TruthfulQA | Winogrande |
|---|---|---|---|---|---|---|---|
| clean | 0.6295 | 0.5413 | 0.2715 | 0.7995 | 0.0741 | 0.1922 | 0.5383 |
| garbled | 0.6354 | 0.5269 | 0.2664 | 0.8080 | 0.0944 | 0.1873 | 0.5359 |
| duplicate | 0.6309 | 0.5125 | 0.2732 | 0.7918 | 0.0778 | 0.1983 | 0.5525 |
| unrelated | 0.6241 | 0.4981 | 0.2705 | 0.7901 | 0.0833 | 0.1824 | 0.5335 |
| keyword | 0.6333 | 0.5231 | 0.2750 | 0.7986 | 0.0759 | 0.1848 | 0.5241 |
| mixed | 0.6315 | **0.5732** | 0.2673 | 0.7952 | 0.0907 | 0.1836 | 0.5375 |
| **base (no SFT)** | **0.6637** | **0.7460** | 0.2745 | **0.8311** | 0.0611 | **0.1934** | **0.5856** |

### 5.2 Key findings

1. **Noise damage is far smaller than the damage of fine-tuning itself**: all six fine-tuned models cluster tightly (MMLU range 0.011), while the **base model beats every fine-tuned model on 4/7 benchmarks** — most notably GSM8K (0.746 vs ~0.52, −22 pts) and ARC (0.831 vs ~0.79). SFT on dolly-15k hurts general ability, and this effect entirely swamps the 10% noise differences;
2. **unrelated is the worst on average** (MMLU −0.005, GSM8K −0.043, ARC −0.009 vs clean) — fluent but context-mismatched responses mislead the model the most, consistent with its held-out damage (+0.465);
3. **garbled barely harms MMLU** (0.635 > clean 0.630) — the easiest-to-detect noise is the least harmful: the model quickly learns to "ignore" corrupted samples (high loss, but no misleading knowledge);
4. **duplicate is slightly *better* on Winogrande / TruthfulQA** (+0.014/+0.006) — memorization yields a small positive effect on a few tasks, at the cost of the largest generalization damage;
5. **BBH is the exception**: fine-tuned models beat the base (0.074~0.094 vs 0.061) — dolly's instruction style helps structured reasoning (see 5.5).

### 5.3 Question-level flip analysis (MMLU: noise models vs clean model)

| Model | flipped questions / 14042 | flip rate | jointly-correct questions |
|---|---|---|---|
| unrelated | 2133 | **15.2%** | 7735 |
| keyword | 1815 | 12.9% | 7959 |
| garbled | 1772 | 12.6% | 7995 |
| mixed | 1750 | 12.5% | 7979 |
| duplicate | 1571 | **11.2%** | 8064 |

**Interpretation:** even though overall accuracies are nearly identical, **roughly 1 in 7~9 questions flips between any noise model and the clean model** — noisy models make *different* mistakes. unrelated flips the most (15.2%) and shares the fewest correct answers — it genuinely alters the model's knowledge and reasoning paths; duplicate flips the least (11.2%), staying closest to clean.

### 5.4 MMLU 57-subject breakdown

**Base-model subject profile:** strongest — marketing (0.88), high_school_world_history (0.87), high_school_government_and_politics (0.87); weakest — college_mathematics (0.35), global_facts (0.36), moral_scenarios (0.37). The base is humanities/commonsense-skewed; math reasoning is its weak spot.

**Most-damaged 3 subjects per noise type (difference vs clean):**

| Model | Most damaged subjects |
|---|---|
| garbled | high_school_computer_science (−0.060) / business_ethics (−0.050) / global_facts (−0.050) |
| duplicate | electrical_engineering (−0.055) / astronomy (−0.046) / high_school_computer_science (−0.040) |
| unrelated | electrical_engineering (−0.076) / high_school_computer_science (−0.060) / jurisprudence (−0.055) |
| keyword | global_facts (−0.040) / formal_logic (−0.040) / anatomy (−0.037) |
| mixed | anatomy (−0.082) / global_facts (−0.060) / electrical_engineering (−0.055) |

**Interpretation:**
- Subject-level mean differences are only ±0.005 — **no noise type causes systematic subject-level damage**;
- The most-affected subjects are **fact-heavy** (global_facts, anatomy) and **technical** (electrical_engineering, computer_science) — consistent with noisy fact-type samples;
- A curious regularity: every noisy run scores *higher* than clean on **college_mathematics** (+0.05~+0.11) — the mild regularizing effect of noise against overfitting shows up most clearly in the base model's weakest subject.

### 5.5 Other group breakdowns

**BBH (27 tasks)**: SFT's largest gains over base — sports_understanding (+0.40), boolean_expressions (+0.15), causal_judgement (unchanged); largest loss — object_counting (−0.20). Fine-tuned BBH means (0.074~0.094) all beat base (0.061).

**HellaSwag (192 activity groups)**: noise-vs-clean mean differences ≤ 0.007; the largest single-group deviations (small sample sizes — interpret cautiously): unrelated on Getting a tattoo (−0.50), keyword on Hand washing clothes (−0.38). No systematic pattern.

**TruthfulQA (39 categories)**: SFT's own effect dwarfs noise — vs base, the clean model loses on Mandela Effect (−0.167) and Conspiracies (−0.120) while gaining on Distraction (+0.214) and Subjective (+0.111); noise-vs-clean category differences < 0.02.

### 5.6 Confidence & generation behavior (per-question raw records)

**MC confidence (margin = second-best nll − best nll):**

| Model | margin when correct | margin when wrong | ratio (calibration) |
|---|---|---|---|
| base | 4.918 | 1.293 | **3.80 (best calibrated)** |
| clean | 3.844 | 1.169 | 3.29 |
| garbled | 4.266 | 1.314 | 3.25 |
| duplicate | 4.037 | 1.276 | 3.16 |
| unrelated | 4.747 | 1.500 | 3.16 |
| keyword | 4.564 | 1.392 | 3.28 |
| mixed | 4.636 | 1.325 | 3.50 |

**Interpretation:** margins discriminate in every model (~4 when correct vs ~1.3 when wrong); the clean model is the most "hesitant" (lowest correct-answer margin, 3.84); noisy models are *not* less confident than clean — noise affects specific knowledge rather than global behavior.

**Generation length:** base averages 109 tokens/question vs ~54 for fine-tuned models — **dolly SFT makes answers markedly more concise** (dolly responses average 2~3 sentences) — a behavioral change far larger than anything the noise causes.

---

## 6. Conclusions & Discussion

### 6.1 Core conclusions

1. **Sample-level detectability ranking**: garbled (0.9996) > duplicate (0.974) > unrelated (0.923) > mixed (0.850) > **keyword (0.531, infeasible)**;
2. **Feature-to-noise mapping**:
   - garbled → input- and output-side features (user_loss / entropy / loss_curvature); training dynamics alone suffice;
   - duplicate → **must use data-side features** (text_nn_sim); training dynamics are not just weak but *inverted* (loss AUC 0.37);
   - unrelated → cross-epoch loss volatility & curvature (loss_std / curvature), moderate strength;
   - keyword → requires entity-aware detection (e.g., NER consistency); none of our 19 metrics work;
3. **Detectability is anti-correlated with harm**: the easiest-to-detect garbled noise barely harms the model (MMLU +0.006), while the hardest-to-detect semantic noise (unrelated / keyword) is potentially the most damaging. **Real-world data cleaning should prioritize the hard-to-detect semantic noise**;
4. **Detection window matters**: detectability decays monotonically with training (garbled loss AUC: 0.985 → 0.865 from epoch 0 to 4) — **data cleaning should happen early**;
5. **The absolute impact of 10% pollution is small**: all six fine-tuned models are nearly indistinguishable on benchmarks, and all underperform the base (dolly SFT itself hurts general ability) — under LoRA + mid-size data, 10% noise is swamped by the SFT effect;
6. **Noise still leaves observable traces**: duplicate's overfitting (held-out +0.517, 22% above baseline), unrelated's highest question-flip rate (15.2%) and largest MMLU/GSM8K drops — the impact is real, just smaller than fine-tuning itself;
7. The method is robust across task types (8 categories, AUC 0.87~0.99; garbled ≈1.0 everywhere).

### 6.2 Limitations & future work

1. **keyword detection blind spot** — entity-aware detection needed (NER consistency, counterfactual perturbation);
2. **garbled localization** — position-based comparison fails; needs sequence alignment;
3. **Noise-ratio extrapolation** — 10% conclusions may not hold at 5%/20%; the detectability decay curve (§3.2) suggests detection gets *harder* at lower ratios. The 5% data is ready: `bash run_experiment.sh --ratio 0.05 --tag ratio05 --reuse-clean` (reuses the clean run, saves ~3 h);
4. **Single dataset / model** — conclusions rest on dolly-15k + Qwen2.5-3B + LoRA; classification-style datasets, larger models, and full fine-tuning need re-validation (category stratification already gives preliminary transferability evidence);
5. **Eval protocol** — absolute HellaSwag / TruthfulQA scores are low (5-shot / 0-shot interacting with the chat template); between-model comparisons remain valid, but absolute values should be cited cautiously;
6. **Question-level flips (§5.3) point to a deeper question**: overall accuracy hides per-question differences — visualizing/attributing *how* noisy models err differently from clean models is a promising follow-up.

### 6.3 Reproduction

```bash
# data + training + evaluation + analysis (10% is the default tag)
python scripts/make_noise.py
bash run_all.sh
bash run_all_eval.sh
python scripts/analyze_detection.py
python scripts/analyze_token_level.py
# other ratios: bash run_experiment.sh --ratio 0.05 --tag ratio05 --reuse-clean
```

---

*This report is compiled from artifacts produced by the experiment pipeline; all raw data lives in `results/` (evaluation details, per-question records, detection tables & figures) and `<data_root>/runs/ratio10/` (per-sample metrics, per-token diagnostics, layer norms, TensorBoard events).*


---

## 7. The 5% Ratio Experiment (ratio05): Dose-Response Validation

> Identical setup to ratio10 (model / epochs / hyper-parameters / seed); only the
> noise ratio drops to 5% (731 noisy samples per type). The clean run and the
> clean/base eval results are reused from ratio10 (byte-identical data & model).

### 7.1 Training dynamics

| run | epoch 0 | epoch 1 | epoch 2 | epoch 3 | epoch 4 | (10% final) |
|---|---|---|---|---|---|---|
| garbled | 1.526 | 1.257 | 0.977 | 0.746 | 0.609 | (0.702) |
| unrelated | 1.437 | 1.194 | 0.876 | 0.641 | 0.504 | (0.498) |
| keyword | 1.403 | 1.151 | 0.878 | 0.654 | 0.523 | (0.533) |
| mixed | 1.438 | 1.176 | 0.895 | 0.666 | 0.533 | (0.525) |
| duplicate | 1.358 | 1.104 | 0.824 | 0.596 | 0.467 | (0.425) |

- Trajectory shapes match the 10% experiment (garbled highest, duplicate lowest);
- **Final held-out loss**: mixed 2.035 (lowest) < clean 2.051 < garbled 2.054 <
  keyword 2.059 < unrelated 2.063 < **duplicate 2.091 (highest)** — duplicate's
  overfitting damage stays worst at 5% (+0.040 vs clean, roughly half of the
  10% damage +0.092, i.e. nearly linear in the ratio).

### 7.2 Detection results (5%)

| Noise type | LR AUC | RF AUC | Best univariate (AUC) | vs 10% LR |
|---|---|---|---|---|
| garbled | **0.999** | 0.999 | loss_curvature (0.986) | 0.9996 → 0.999 (flat) |
| duplicate | **0.972** | 0.991 | text_nn_sim (0.963) | 0.974 → 0.972 (flat) |
| unrelated | **0.956** | 0.903 | loss_curvature (0.846) | 0.923 → 0.956 (up) |
| mixed | **0.737** | 0.916 | text_nn_sim (0.716) | 0.850 → 0.737 (down) |
| keyword | **0.464** | 0.541 | loss_curvature (0.703) | 0.531 → 0.464 (still infeasible) |

**Category stratification (RF)**: closed_qa 0.993 / summarization 0.976 /
information_extraction 0.949 / open_qa 0.931 / brainstorming 0.871 /
general_qa 0.871 / **classification 0.710 (hardest; lower than the 10% value
of 0.870)** — short structured responses are even harder to separate at low ratios.

**Token level (top-24 hard tokens)**: garbled hard_loss / hard_gradnorm AUC
**0.79 / 0.81 (higher than the 10% values of 0.77)** — with less noise the
model adapts less to corruption, so corrupted tokens stand out more;
duplicate stays below 0.5 (inverted direction persists); unrelated / keyword
~0.5 (not separable, as at 10%).

### 7.3 Dose-response: key findings

1. **Detection is ratio-insensitive (except mixed)**: garbled / duplicate /
   unrelated AUCs are nearly identical at 5% and 10% — their signal mechanisms
   (token-level damage / text duplication / loss-trajectory curvature) do not
   depend on the ratio, so the detector transfers directly to low-pollution
   scenarios;
2. **unrelated hurts MMLU MORE at 5%**: 0.611 (−0.019 vs clean) at 5% vs 0.624
   (−0.005) at 10% — **harm is non-monotonic in the ratio**. Hypothesis: at
   higher ratios the model learns to identify and hedge against mismatched
   samples; at lower ratios each mismatch is trusted as a genuine example,
   making each one more misleading;
3. **duplicate's overfitting damage is roughly linear**: held-out excess over
   clean is +0.040 at 5% vs +0.092 at 10%;
4. **keyword's blind spot is ratio-independent** (0.46 at 5% vs 0.53 at 10%) —
   entity-level tampering needs model-external tools;
5. **mixed is the only clear degradation** (0.85 → 0.74): smaller noise subsets
   dilute every feature more.

### 7.4 Benchmark comparison (5%, 7 models)

| Model | MMLU | GSM8K | HellaSwag | ARC | BBH | TruthfulQA | Winogrande |
|---|---|---|---|---|---|---|---|
| clean | 0.6295 | 0.5413 | 0.2715 | 0.7995 | 0.0741 | 0.1922 | 0.5383 |
| garbled | 0.6296 | 0.5087 | 0.2729 | 0.7901 | 0.0778 | 0.1848 | 0.5478 |
| duplicate | 0.6327 | 0.5049 | 0.2753 | 0.7978 | 0.0833 | 0.1873 | 0.5627 |
| unrelated | **0.6106** | 0.5481 | 0.2735 | **0.7782** | 0.0852 | **0.1665** | 0.5249 |
| keyword | 0.6295 | 0.5428 | 0.2652 | 0.7952 | 0.0796 | 0.1995 | 0.5241 |
| mixed | 0.6330 | 0.5148 | 0.2731 | 0.7875 | 0.0907 | 0.2020 | 0.5320 |
| base | 0.6637 | 0.7460 | 0.2745 | 0.8311 | 0.0611 | 0.1934 | 0.5856 |

- **unrelated remains the most harmful at 5%**: MMLU −0.019 / ARC −0.021 /
  TruthfulQA −0.026 (all largest drops), consistent with and stronger than at 10%;
- Other noise types stay within ≤ 0.006 of clean — noise damage is still far
  smaller than the fine-tuning damage itself (base leads everywhere);
- One exception: unrelated's GSM8K at 5% (0.548) beats its 10% value (0.498) —
  small-sample fluctuation, needs verification.

### 7.5 Conclusion

The 5% experiment answers the ratio-sensitivity question: **detection remains
effective at realistic pollution levels (garbled 0.999 / duplicate 0.972 /
unrelated 0.956); detectability does not decay with the ratio; keyword's blind
spot and mixed-dilution are the only weak points.** It also reveals that the
dose-response is non-monotonic — semantic-mismatch noise (unrelated) has a
higher marginal harm at lower ratios, further supporting "data cleaning should
prioritize semantic-level noise".

