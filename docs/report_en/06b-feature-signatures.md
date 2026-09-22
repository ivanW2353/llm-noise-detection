## 6b. Feature Signatures of Each Noise Type: How Training Dynamics and Text Features Vary by Type

This file covers **Thread 2** of the report's three research threads: setting aside "can it be detected" and "is it harmful downstream," it asks only **what "fingerprint" each of the seven noise types leaves in training-dynamics features and text features, and whether these fingerprints match the theoretical predictions Chapter 5 wrote down before the results came in**.

**Relationship to Chapter 5's theoretical predictions**: Section 5.2 grouped the seven noise types into three mechanism classes — **Class A, anomalous types** (garbled/unrelated/truncation, which conflict with linguistic statistical structure and are predicted to show high loss, large gradients, and directional deviation), **Class B, hyper-typical types** (template/duplicate, which get perfectly memorized by the model and are predicted to show anomalously low loss and anomalously fast convergence), and **Class C, mildly perturbed types** (near_duplicate/keyword, predicted to leave almost no trace in training dynamics). This file verifies section by section whether these predictions hold, and where they need correcting.

---

### 6b.1 Landscape of Detection Difficulty: What's Visible, What Isn't

![Detection difficulty by noise type](../../results/charts/within_type_auc.png)

#### 6b.1.1 What "in-domain detection AUC" means: method and sample construction

"In-domain detection AUC" answers a very specific binary-classification question: **for a given noise type T, can training-dynamics features alone separate "samples injected with noise T" from "fully clean samples"?** It does not measure "can all 7 noise types be identified simultaneously" — it is 7 independent binary-classification tasks, each reporting its own AUC.

Concrete construction (corresponding to `cv_auc()` inside `analyze.py::cross_type_transfer`, producing the diagonal of `cross_type.csv`):

- **Positive samples**: rows in dataset T (e.g. `garbled`) where `noise_type == 'garbled'`; **negative samples**: rows in the same dataset T where `noise_type == 'none'` (the uncontaminated portion of that same dataset). Both come from the same training run (same dataset/same `tag`), never mixed across datasets.
- **Features**: all 37 numeric columns in `per_sample_metrics.csv` (training-trajectory / token-diagnostic / text-similarity features defined in Appendix 8.1-8.3), requiring `dropna(subset=features)` — a row is only included if all 37 features are non-null.
- **Classifier and validation**: after `StandardScaler` standardization, 5-fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` cross-validation, training one `RandomForestClassifier(n_estimators=200)` per fold; the final AUC pools each fold's out-of-fold predicted probabilities and computes a single AUC over the whole population (not the average of 5 separate AUCs).
- **This is a supervised convention, measuring the signal ceiling.** RF training uses the true noise label, so this section's AUC answers "how much separable signal is actually present in training dynamics," **not** "how much can be reached label-free." The latter is the `iforest`/`zscore` route in Section 6b.4 (`analyze.py::unsupervised_metrics` → `unsupervised.csv`), which never uses labels at all. The gap between the two can be enormous: templating is 0.999 in this section but only 0.522 under Section 6b.4's `iforest` — same samples, same features, the only difference is whether a label points the direction, which is exactly the source of Section 6b.4's direction-reversal trap. Keep track of which convention a given number belongs to.

**An easily-overlooked but important detail — the sample count involved is far smaller than the dataset's total size**: 13 of the 37 features are the token-level diagnostic features described in Section 8.2 (`max_token_loss`, `frac_hard`, `hard_loss_mean`, etc.), produced only during the pure-forward diagnostic pass run at `diag_subsample=8` (one row in 8), covering about 12.5%. `dropna` requiring all 37 features to be non-null effectively keeps only the small subset of samples that happen to land in the diagnostic subsample *and* show a hard token in at least one epoch. Measured, the actual sample sizes participating in this section and in Sections 6b.2/6b.6, and in the `iforest`/`zscore` columns of Section 6b.4's table, are:

| Dataset | Total n | Noise n_noise |
|---|---|---|
| duplicate | 995 | 89 |
| garbled | 904 | 85 |
| keyword | 906 | 87 |
| near_duplicate | 906 | 87 |
| template | 906 | 87 |
| truncation | 905 | 86 |
| unrelated | 904 | 85 |

That is, each dataset uses only about 900-1000 samples (roughly 6-7% coverage relative to the full 14,611-16,072 rows), and the positive/negative ratio has been reshuffled by the subsampling process (noise samples generally make up 9%-10%, close to the original injection ratio, showing the subsampling itself carries no systematic bias toward noisy vs. clean samples — but the absolute sample count is indeed much smaller). Section 6b.4's `memo_signed` column, since it uses only 6 features with 100% coverage (no token-diagnostic features), is computed on the full sample population — **the two columns in the same comparison table do not share the same sample size**, worth keeping in mind while reading the table.

**Contrast with Chapter 5's predictions**: Class A (garbled/unrelated/truncation) falls into high/medium/low AUC tiers under the supervised convention respectively (garbled highest, consistent with the "outlier detection works" prediction); Class C (near_duplicate/keyword) has clearly low in-domain AUC (keyword's in-domain CV is only 0.577, the hardest to detect in this section and arguably the whole report), consistent with the "signal isn't in training dynamics" prediction; Class B (template/duplicate) is, under the supervised convention, actually among the two easiest types (template 0.999) — this seems to contradict the "outlier detection fails" prediction, but the contradiction is resolved once the convention is accounted for: with labels, RF is told which direction to look, and Class B's "hyper-typical" trace is equally separable signal, just **pointing in the opposite direction from ordinary outlier detection** — exactly the mechanism Section 6b.4 unpacks.

---

### 6b.2 Cross-Noise-Type Transfer: Can Detectors Generalize

![Cross-type detector transfer matrix](../../results/charts/cross_type_heatmap.png)

The figure above is the 7×7 transfer matrix under `dolly-ratio10`: each row is the noise type the detector was trained on, each column is the noise type it's tested on, and the diagonal is Section 6b.1's in-domain AUC.

**Method**: for every off-diagonal cell `(source type → target type)`, the code (`analyze.py::fit_transfer`) trains one `LogisticRegression(max_iter=2000)` and one `RandomForestClassifier(n_estimators=200)` on the **source-type** dataset (`StandardScaler` is `fit` only on the source data, then only `transform`ed when applied to the target dataset — never refit — so this measures the detector's own transfer ability, not the convenience of re-standardizing), evaluates both on the target-type dataset, and takes the higher of the two AUCs as the cell's final value. The matrix's **diagonal is not "retrained and retested on itself" — it directly reuses the in-domain AUC already obtained from Section 6b.1's 5-fold cross-validation**, so that the diagonal and off-diagonal are comparable on the same scale (otherwise the diagonal would be inflated by training and testing on the same batch of data).

Concretely, a few representative off-diagonal cells (all from `results/dolly-ratio10/cross_type.csv`): `template→garbled` is only 0.403 (well below the 0.5 random baseline, i.e. reversed); `unrelated→template` is only 0.321 (also reversed, and even more extreme than `template→garbled`); while `template→duplicate` is as high as 0.896 — showing that a "detector trained on templating" does not perform poorly on every target type, but rather transfers strongly and positively to structurally similar exact-duplication, while failing completely (or reversing) on the structurally very different garbled type.

**Core findings:**

- **Diagonal mean 0.846, off-diagonal mean 0.675** — an average cross-type transfer loss of about 17-20 points, showing that a substantial portion of training dynamics' discriminative signal for noise is **type-specific**; there is no "universal detector" that transfers directly to a new noise type with no retraining.
- **Templating is the most isolated type**: a detector trained on templating transfers to the other 6 types with AUC generally hovering in 0.40-0.59 (some even below the random baseline, e.g. transferring to garbled reaches only 0.40); conversely, detectors trained on other types and tested on templating also generally perform weakly (e.g. `unrelated→template` is only 0.32, clearly below the 0.5 random baseline, i.e. **direction reversal** — the detector's judgment on templated samples is systematically backwards). This is the same mechanism as the "direction-reversal trap" discussed in Section 6b.4, showing up in the cross-type setting.
- **Garbled transfers most easily**: detectors trained on most other types still reach 0.93-0.98 when tested on garbled, showing garbled's training-dynamics anomaly (e.g. persistently elevated loss) is "universal" enough not to depend on any type-specific feature pattern.
- **Keyword substitution is weak in both directions**: not only is its in-domain CV only 0.577, both transferring in and transferring out give generally low AUC — consistent with "keyword substitution is the hardest type to detect."

**Practical implication**: deploying a label-free detector in production should not rely on training with samples of one known noise type and expecting it to cover unknown types; each possible noise pattern needs its own calibration, or priority should go to the feature-attribution results in Section 6b.6 of this file to design more general rules. This finding is also the direct motivation for [06c-detectability.md](06c-detectability.md) Section 6c.4's "boundaries under an unknown noise type" experiment.

---

### 6b.3 Cross-Ratio Transfer: Almost No Loss

![Cross-ratio transfer](../../results/charts/cross_ratio_transfer.png)

Contrasting with Section 6b.2's cross-type transfer is transfer across noise **ratio**: fixing the noise type, does a detector trained at 10% noise ratio transfer to 5% (and vice versa)?

**Method**: uses the same `fit_transfer` function as Section 6b.2's cross-type transfer (`analyze.py::cross_ratio_transfer`), differing only in how "source/target" are paired — here source and target are two datasets of the **same noise type at different noise ratios** (e.g. fitting on `dolly-ratio10/garbled`, and "`dolly-ratio10→dolly-ratio5`" means fitting on `dolly-ratio10` and evaluating on the corresponding `dolly-ratio5/garbled`), rather than being paired by noise type as in Section 6b.2; the two directions (`dolly-ratio10→dolly-ratio5`, `dolly-ratio5→dolly-ratio10`) are trained independently, and the diagonal (not shown in the table, i.e. "in-domain AUC within the same tag") again reuses Section 6b.1's already-computed in-domain CV result rather than recomputing it.

| Noise type | dolly-ratio10→dolly-ratio5 AUC | dolly-ratio5→dolly-ratio10 AUC |
|---|---|---|
| template | 1.000 | 1.000 |
| garbled | 0.996 | 0.997 |
| duplicate | 0.990 | 0.986 |
| unrelated | 0.985 | 0.948 |
| truncation | 0.966 | 0.880 |
| near_duplicate | 0.895 | 0.790 |
| keyword | 0.898 | 0.751 |

**Core finding: cross-ratio transfer AUC is uniformly ≥0.75, mostly ≥0.88, and retention (transfer AUC / in-domain AUC) is mostly ≥1.0 — transfer costs almost no accuracy, and in some cases even improves it** — e.g. keyword substitution's `dolly-ratio10→dolly-ratio5` reaches 0.898, far above dolly-ratio5's own in-domain CV of 0.634; near_duplicate's `dolly-ratio10→dolly-ratio5` reaches 0.895, far above its own in-domain CV of 0.759. This shows that pooling training-dynamics data across two noise ratios actually learns a more robust decision boundary, rather than overfitting to one particular noise density.

**Conclusion in contrast with Section 6b.2**: training dynamics' discriminative signal for "noise type" is stable across noise ratios (even showing positive transfer), and detectors do not need to be retrained per ratio; **the real transfer bottleneck is across noise types, not across noise ratios.** This is practically meaningful for deployment — if production noise ratios fluctuate over time, there's no need to retrain the detector because of it.

---

### 6b.4 The Direction-Reversal Trap: Memorized Noise Needs a "Signed" Rule

![The direction-reversal trap](../../results/charts/direction_reversal.png)

This is one of the project's most important methodological findings, and the most direct empirical validation of Section 5.2's "Class B hyper-typical noise" prediction. Conventional unsupervised outlier detection (IsolationForest, etc.) assumes "noise = anomaly = outlier," but for samples **perfectly memorized by the model** (templating, exact duplication), training dynamics actually behave **more regularly, less like an outlier** (loss converges rapidly to well below normal, gradient norm shrinks quickly) — the opposite of the "outlier" intuition.

**Exact construction** (`analyze.py::memorization_score`, `MEMO_FEATS`): `memo_signed` uses only 6 trajectory features with 100% coverage — `loss_mean, loss_last, loss_std, loss_curvature, converge_epoch, grad_norm_mean` — and instead of the "data-driven, direction learned by the algorithm" style of iforest, it **fixes the sign coefficient of all 6 features to -1 a priori** (with no fitting whatsoever), taking a robust z-score of each feature, multiplying by -1, and averaging. In other words, this rule explicitly encodes a prior assumption: "the lower the loss, the faster the convergence (the smaller `converge_epoch`), the smaller the gradient — the more likely this is a perfectly-memorized hyper-typical noise sample," rather than letting the model discover the outlier direction itself — this is also the fundamental reason it can reverse iforest's failure on templating: iforest is direction-agnostic general anomaly detection, `memo_signed` is a purpose-built rule with a prior direction.

**A note on sample size**: since `memo_signed`/`low_loss_only` (the "signed memorization rule" column below, from `results/dolly-ratio10/memorization.csv`) relies only on these 6 fully-covered features, it is computed on the **full sample population** (roughly 14,611-16,072 rows, slightly more for duplicate since that type adds rows); whereas the left column below, "general unsupervised iforest AUC" (from `results/dolly-ratio10/unsupervised.csv`), follows the 37-feature-all-non-null constraint from Section 6b.1.1 and is computed only on the roughly 900-1000-row diagnostic subsample. **The two columns in the same table differ in sample size by about 15×, and are two different sample populations — not the same samples scored by two different methods.** This is, alongside Section 6b.1.1, another place in this report requiring an explicit note on sample-size convention.

| Noise type | General unsupervised iforest AUC | Signed memorization rule memo_signed AUC |
|---|---|---|
| garbled | 0.936 | 0.017 |
| unrelated | 0.703 | 0.274 |
| duplicate | 0.612 | 0.654 |
| mixed | 0.662 | 0.380 |
| keyword | 0.552 | 0.330 |
| truncation | 0.598 | 0.352 |
| near_duplicate | 0.599 | 0.346 |
| **template** | **0.522 (near random)** | **0.925 (near perfect)** |

**Templating is the most typical example**: the general unsupervised method gives an AUC of only 0.522, nearly equivalent to random guessing; but switching to a signed rule **whose direction is fixed a priori by the "memorization hypothesis"** (rather than fit per dataset) — i.e. assuming in advance that "lower loss, faster convergence, smaller gradient = more likely memorized hyper-typical noise," rather than letting the algorithm learn the outlier direction itself — the AUC immediately jumps to 0.925.

Conversely, garbled — a genuinely "high-loss, unlearnable" type of noise — is completely defeated by the signed memorization rule (AUC only 0.017, i.e. **almost exactly backwards**), because this noise type's nature is that the model cannot learn it, not that the model has memorized it.

**Practical implication**: there is no universal unsupervised anomaly detector that covers all types of "training anomaly"; at minimum, two broad noise categories need to be distinguished — "unlearnable" types (garbled, etc., use conventional outlier detection) and "memorized/hyper-typical" types (templating, exact duplication, needing a signed prior rule) — and treated accordingly. The detector-selection/deployment decisions driven by this mechanism finding are in [06c-detectability.md](06c-detectability.md) Section 6c.2.

---

### 6b.5 Early Detection: Usable Signal Before Training Even Finishes

![Early detection](../../results/charts/early_detection.png)

All analyses above are based on trajectory features collected after the full 5-epoch training run completes. This section tests: if the trajectory is truncated to the first k epochs and detection metrics recomputed, how early can usable signal be obtained? This section essentially re-verifies Section 6b.4's direction-reversal mechanism along the time axis, checking whether it exists from the very start of training or only emerges later.

**Method** (`analyze.py::early_detection_sweep`): this is not a "training was actually stopped after 1 epoch" experiment, but a **post-hoc truncation simulation** on logs from a fully completed 5-epoch run — `max_epoch` is set in turn to 0, 1, 2, 3, 4 (i.e. keeping only the records of the first 1/2/3/4/5 epochs), and `build_table` is recalled on the truncated data to recompute features, then `unsupervised_metrics` and `memorization_score` are rerun. To keep different truncation points comparable, only "core" features available at every truncation point are used (excluding token-diagnostic features that need at least 2 epochs to compute a slope/trend), so the curves in the figure show "what detection performance would look like if training had only run this many rounds, recomputed with the same feasible feature set" — not a genuine early-stopping experiment. Whether real early-stopped training would give different results because the optimization trajectory itself would change is not verified in this report.

Both subplots now plot all 8 datasets (7 noise types + mixed), making it easy to directly compare the same noise type's trend under the two scoring methods, rather than each subplot showing only a partial selection of types.

**Left plot (general unsupervised iforest, all 8 datasets):**

- **Garbled**: AUC already reaches 0.901 at epoch 1, and is 0.929 at epoch 5 — strong from the very first epoch, with small marginal gain from further training.
- A **counter-intuitive phenomenon**: templating (0.713→0.564), exact duplication (0.616→0.520), unrelated (0.715→0.632), truncation (0.601→0.569) all show AUC **declining** to varying degrees as training proceeds under the general iforest rule, with templating declining the most — this is the same mechanism as Section 6b.4's "direction reversal" showing up along the time axis: the longer these "hyper-typical/memorized" samples are trained, the less they look like outliers.
- Keyword substitution (0.555→0.579) and near_duplicate (0.592→0.602) are roughly flat with small fluctuations; mixed (0.750→0.702, itself a composite signal from 7 injected noise types) trends somewhere between the "declining group" and the "flat group."

**Right plot (signed memorization rule memo_signed, all 8 datasets):**

- **Templating**: AUC=0.91 already at epoch 1, nearly as good as after the full 5 epochs (0.925) — again close to saturation from the first epoch.
- Exact duplication: rises with fluctuation from 0.59 at epoch 1 to 0.65 at epoch 5, strengthening slightly with more training.
- Unrelated (0.24-0.27), garbled (only 0.02-0.03 throughout, almost completely reversed), keyword substitution (0.32-0.33), near_duplicate (0.34-0.35) all remain far below the left plot's iforest performance on these types throughout, showing that the `memo_signed` rule with its prior direction really is only effective for "memorized" noise (templating, exact duplication), and unsuitable for "unlearnable" types (garbled) and mildly-rewritten types (keyword substitution, near_duplicate, unrelated).
- Truncation (0.271→0.352) and mixed (0.354→0.380) rise slightly with more training, but remain clearly below their performance under left-plot iforest (around 0.6) throughout, showing that early detection for these two noise types should rely on general iforest rather than memo_signed.

**Practical implication**: garbled and templating can be scored and removed right after the first training epoch finishes, saving the compute of the remaining 4 epochs on samples that are already clearly noise; the other types (especially keyword substitution, near_duplicate, truncation) have no "early-stopping shortcut" and still need full training or at least several rounds of accumulated signal. This finding corroborates this file's Section 6b.9 external-baseline comparison ("proxy metrics from only the first epoch match or beat the 5-epoch detector").

---

### 6b.6 Feature Attribution: What Is the Detector Actually Looking At

![Feature attribution](../../results/charts/feature_attribution.png)

Everything before this answers "can it be detected." This section uses permutation importance (rather than RF's biased built-in impurity importance) to answer "which feature is the detector using."

**Exact computation** (`analyze.py::feature_attribution`): reuses exactly the same 5-fold cross-validation RF setup as Section 6b.1.1 (the same roughly 900-1000-row diagnostic subsample, the same `RandomForestClassifier(n_estimators=200)`), but after each fold trains, instead of reading RF's built-in `feature_importances_` (based on impurity reduction, with a known systematic bias toward high-cardinality/continuous features), it calls `sklearn.inspection.permutation_importance(n_repeats=20, scoring='roc_auc')` — shuffling one feature column's values within the test fold 20 times, measuring how much AUC drops relative to the unshuffled baseline each time, taking the mean over 20 shuffles as that feature's importance in that fold (with the standard deviation reflecting variance across the 20 shuffles); the final value averages again across the 5 folds. This way "importance" directly means "how much shuffling this feature drops AUC," rather than "how many times this feature was used to split in a tree" — closer to a causal notion of contribution.

Using keyword substitution (the hardest type to detect among all 7) as an example, the full top-8 feature attribution ranking (`results/dolly-ratio10/feature_attribution.csv`):

| Rank | Feature | importance | importance_std |
|---|---|---|---|
| 1 | `loss_slope` | 0.0201 | 0.0198 |
| 2 | `hard_loss_max` | 0.0080 | 0.0126 |
| 3 | `cos_global_slope` | 0.0073 | 0.0046 |
| 4 | `hard_pos_jaccard` | 0.0069 | 0.0050 |
| 5 | `entropy` | 0.0060 | 0.0060 |
| 6 | `max_token_loss` | 0.0051 | 0.0059 |
| 7 | `frac_hard` | 0.0043 | 0.0074 |
| 8 | `token_loss_kurt` | 0.0027 | 0.0031 |

This table shows directly where the difficulty of detecting keyword substitution lies: the top feature `loss_slope` has an importance of only 0.020 (compare exact duplication's top feature `text_nn_sim` at 0.155, nearly 8× larger), and its `importance_std` (0.0198) is **almost equal to the importance itself** (0.0201) — meaning across different folds of the 5-fold cross-validation, this feature's importance is extremely unstable, appearing important in one fold and nearly useless in another. This is not a case of a feature "hiding deep" — keyword substitution, which only swaps 1-2 entity words while leaving sentence structure entirely unchanged, simply leaves no stable, reproducible trace in either training dynamics or text statistics (echoing the raw-text examples shown in Section 8.6).

**Most important finding — two noise types' detection signal comes almost entirely from something other than training dynamics:**

| Noise type | Top feature | Importance | 2nd-place feature | Importance | Gap |
|---|---|---|---|---|---|
| duplicate | `text_nn_sim` (text similarity, a data-level feature) | 0.155 | `cos_global_last` | 0.0092 | 17× |
| unrelated | `text_nn_sim` | 0.084 | `loss_slope` | 0.0102 | 8× |

`text_nn_sim` is TF-IDF nearest-neighbour cosine similarity, coming from the **raw text itself**, unrelated to training dynamics. This means the high AUCs reported in Sections 6b.1 and 6b.2 for exact duplication and topic mismatch are, in essence, **not a contribution of the training-dynamics detection methodology** — they come from "using a ready-made text-similarity feature for detection, with model training basically standing idle." This is an exception that needs to be honestly flagged under the report's core narrative of "detecting noise via training dynamics."

**The other 5 types' top features are genuine training-dynamics quantities:**

- Garbled is dominated by `user_loss` (0.015) — garbled text drives up loss over the whole response span.
- Keyword substitution is dominated by `loss_slope` (0.020), but the standard deviation is nearly equal to the mean — consistent with its "hardest to detect" conclusion; no single feature stably contributes signal.
- Truncation is dominated by `loss_slope`/`loss_std` (0.029/0.028).
- Near-duplication is dominated by `hard_loss_max`/`max_token_loss` (0.024/0.024), token-level hard-example features.
- **Templating** is another notable case: own-AUC is near-perfect (0.999), yet every feature's permutation importance is tiny (the highest, `hard_id_uniq`, is only 0.0019) — showing that when a type is "too easy to detect," redundant/correlated features split the marginal contribution among themselves; this doesn't indicate unreliable detection, just the normal appearance of permutation importance on a near-saturated task.

**The deployment decision driven by this finding** (a single feature replacing a mixed scheme) belongs to Thread 3 — see the feature-group ablation experiment in [06c-detectability.md](06c-detectability.md) Sections 6c.3.1-6c.3.2.

---

### 6b.7 How Much Overlap Is There Among These Metrics?

Before asking "which data is necessary" (the topic of [06c-detectability.md](06c-detectability.md)), a more basic question about the feature signatures themselves needs answering: **how independent are these 19 fully-covered metrics from each other?** Two conclusions elsewhere in this report rely on the judgment that "they're highly correlated" — the next section of this file (6b.8) uses it to explain why RF doesn't care about dropping any single metric, and 06c Section 6c.3.3 uses it to explain why IF actually improves when features are dropped (dimensional dilution). Both were previously assertions, never measured. Script: `analyze.py::feature_correlation()`, output `results/dolly-ratio10/feature_correlation.csv`.

Measured three ways (Spearman rather than Pearson, since several of these quantities are visibly heavy-tailed and Pearson would understate monotonic-but-nonlinear relationships):

| Dataset | Mean \|ρ\| | Fraction of pairs \|ρ\|>0.7 | PCA components for 90% variance | Effective dimensionality | VIF>10 |
|---|---|---|---|---|---|
| template | 0.344 | 15.2% | 7 | 4.91 | 10/19 |
| garbled | 0.323 | 11.1% | 6 | 4.84 | 11/19 |
| duplicate | 0.321 | 14.0% | 6 | 5.02 | 10/19 |
| truncation | 0.319 | 12.3% | 7 | 4.96 | 10/19 |
| mixed | 0.317 | 12.3% | 7 | 5.09 | 10/19 |
| near_duplicate | 0.312 | 12.3% | 7 | 5.07 | 10/19 |
| keyword | 0.310 | 11.1% | 6 | 4.97 | 11/19 |
| unrelated | 0.303 | 12.3% | 7 | 5.28 | 10/19 |
| **pooled across all** | **0.315** | **12.3%** | **7** | **5.08** | **10/19** |

**Finding one: 19 metrics only actually span about 5 effective dimensions.** Effective dimensionality (participation ratio, `1/Σλ²`) is stable at 4.8-5.3, PCA needs only 6-7 principal components to explain 90% of the variance, and half the metrics have VIF > 10 (the conventional near-collinearity threshold). This result is highly consistent across all 8 datasets, not an artifact of one particular noise type. **"Highly correlated" now has a quantitative form**, and it directly explains 06c Section 6c.3.3's finding: the extra 14 dimensions carry almost no new information, and in an unsupervised outlier distance they are pure noise.

**Finding two: redundancy strictly follows "data-source family."** Grouping metrics by which raw signal they're derived from (loss / grad_norm / cos_ref / text / update), within-family mean |ρ| = 0.481, cross-family only 0.257. This matches expectation — four quantities describing the same loss curve are not four independent measurements to begin with. The most redundant pairs are already nearly the same quantity:

| Metric pair | \|ρ\| | Note |
|---|---|---|
| `loss_last` ↔ `loss_min` | **0.999** | Under 5-epoch training, the final value is almost always the minimum |
| `loss_std` ↔ `loss_slope` | **0.994** | For a monotonically decreasing curve, the volatility is determined by the slope |
| `loss_mean` ↔ `loss_rank` | 0.987 | The latter is the former's percentile rank, nearly monotonically equivalent |
| `grad_norm_mean` ↔ `update_contrib_mean` | 0.987 | The contribution share is itself computed as gradient norm normalized |

**Finding three: only a small handful are genuinely independent.** Ranked by "maximum correlation with any other metric," the most independent are `cos_ref_std` (0.253), `cos_ref_last` (0.413), `text_nn_sim` (0.568), `grad_norm_cv` (0.637). This matches 06c Section 6c.3.3's greedy-selection results — the features chosen early tend to come from different families, rather than swapping in another statistic from the same family.

**Methodological implication**: if collection were redesigned, keeping only one of each pair — `loss_last`/`loss_min`, `loss_std`/`loss_slope`, `grad_norm_mean`/`update_contrib_mean` — would change no conclusion. Keeping all of them currently is for historical reasons (features were added first, overlap measured later), not intentional redundancy.

**An implementation bug this measurement caught**: `analyze.py` used to assign `cos_ref_trend` directly as `cos_ref_slope` (correlation coefficient exactly 1.000), with zero difference between the two columns across 133,168 rows — the so-called "20 features" was actually only 19. It had already contaminated Section 6b.6's attribution table: permutation importance would randomly split the same signal between the two columns — on keyword, `cos_ref_trend` was +0.0059 (ranked 4th) while its twin `cos_ref_slope` was −0.0009. That assignment has been removed; the numbers in this section and in Sections 6b.6 and 6b.8 are all based on the recomputed, fixed results.

---

### 6b.8 Single-Feature Leave-One-Out: Which Metric Is Truly Irreplaceable?

06c Sections 6c.3.1-6c.3.2's ablation is done at the **category** level (the whole token-diagnostic group, the whole trajectory-feature group, `text_nn_sim` alone), answering the Thread-3 question of "is this category of data worth collecting." This section answers a question closer to the feature signatures themselves: among the 19 fully-covered features, **what happens when one is removed at a time**? Which are genuinely carrying signal, and which are just along for the ride?

This question needs to be asked separately on two detection routes, because their sensitivity mechanism to "one fewer dimension" is completely different:

- **RF route** (supervised, Section 6b.1 convention): out-of-fold AUC from `StratifiedKFold(5)` + `RandomForestClassifier(200)`, the same construction as the `cross_type.csv` diagonal.
- **IF route** (label-free, Section 6b.4 convention): per-dataset `IsolationForest(300)` AUC on standardized features, i.e. the `iforest` row of `unsupervised.csv`.

Both routes run on **the same features, the same samples** (the diagnostic-subsample population), so the RF Δ and IF Δ for the same metric can be directly compared. Script: `analyze.py::single_feature_ablation()`, output `results/dolly-ratio10/single_feature_ablation.csv` (336 rows = 8 datasets × 21 conditions × 2 routes). Convention: **Δ = AUC after removal − full AUC**, negative meaning "removing it made things worse" (the metric carries signal), positive meaning "removing it made things better" (the metric dilutes signal).

![Single-feature leave-one-out: RF and IF have completely different sensitivity to "one fewer dimension"](../../results/charts/single_feature_ablation.png)

**Finding one: `text_nn_sim` is the only irreplaceable metric; the other 19 are individually dispensable.**

| Dataset | IF full AUC | IF minus `text_nn_sim` | RF full AUC | RF minus `text_nn_sim` |
|---|---|---|---|---|
| duplicate | 0.533 | **-0.093** | 0.984 | -0.046 |
| unrelated | 0.639 | **-0.079** | 0.944 | **-0.085** |
| template | 0.551 | **-0.052** | 0.994 | -0.001 |
| mixed | 0.714 | **-0.051** | 0.838 | **-0.052** |
| near_duplicate | 0.607 | -0.002 | 0.696 | -0.014 |
| keyword | 0.590 | -0.003 | 0.676 | -0.007 |
| truncation | 0.573 | +0.004 | 0.787 | -0.006 |
| garbled | 0.932 | +0.012 | 0.996 | -0.001 |

It is the only feature causing a 0.05-0.09-magnitude loss on both routes, concentrated on duplicate/unrelated/mixed — precisely the types where Section 6b.6's attribution analysis found "over 90% of the signal comes from static text similarity," with two independent methods reaching the same conclusion. On garbled, removing it actually helps slightly (+0.012), because garbled's signal is entirely in the training trajectory, and the text-similarity dimension is pure noise for it.

**Finding two: on the RF route, no single metric is critical — all 144 cells have |Δ| under 0.01.**

Excluding `text_nn_sim`, across RF's 144 "dataset × removed-feature" cells (18 removable features × 8 datasets, the 19 fully-covered features minus `text_nn_sim` itself), **not one** exceeds |Δ| of 0.01, average |Δ| only 0.0011, maximum 0.0056 (mixed, removing `cos_ref_last`). The reason is that these 19 trajectory features are highly correlated — `loss_mean`/`loss_last`/`loss_min`/`loss_rank` all describe different facets of the same loss curve. RF has labels to point the way, so removing one just makes it learn to use a correlated substitute — the signal loops back in from another dimension. **Practical implication: a production supervised detector can safely drop any single trajectory feature to cut collection cost, but cannot drop `text_nn_sim`.**

**Finding three: "more features is better" is wrong on the IF route — removing a feature often raises AUC.**

Among the same 144 cells (excluding `text_nn_sim`), **46** have |Δ| > 0.01 (RF has 0), average |Δ| = 0.0096, 9× RF's. The key is that the sign goes **both ways**:

| Dataset | Biggest gain from removal | Δ | Biggest loss from removal | Δ |
|---|---|---|---|---|
| template | `loss_last` | **+0.034** | `grad_norm_cv` | -0.061 |
| duplicate | `converge_epoch` | **+0.041** | `cos_ref_last` | -0.026 |
| unrelated | `grad_norm_cv` | **+0.014** | `converge_epoch` | -0.013 |
| near_duplicate | `grad_norm_cv` | **+0.010** | `converge_epoch` | -0.014 |

The same `converge_epoch`: removing it on duplicate raises AUC by 4.1 points, but on unrelated/near_duplicate drops it by 1.3/1.4 points respectively. IF has no labels, so every dimension enters the outlier distance indiscriminately, and **dimensions unrelated to the current noise type are pure dilution.** Which dimensions count as "unrelated" depends on the noise type — and the noise type is exactly what's unknown in the label-free setting. This is why "just pick a better feature subset for IF" isn't simple, a point developed further in [06c-detectability.md](06c-detectability.md) Section 6c.3.3.

**A methodological caution**: `IsolationForest` internally uses `n_jobs=-1` multithreading, so floating-point summation order isn't fixed and rerunning the same data gives AUC jitter at the 1e-2 level — rerunning the "biggest gain/loss from removal" table above, the specific winning feature name can change (e.g. template's biggest-gain item has appeared as both `converge_epoch` and `loss_last` across different reruns, both around +0.03～+0.05). **The directional conclusion is stable, but exact rankings of "which single feature is first" should not be over-interpreted** — this section and 06c Section 6c.3.3 both cite the results of one particular recomputation.

This provides independent evidence for 06c Section 6c.4.5's `pooled` scorer taking a max rather than a mean: since unrelated dimensions dilute rather than cancel out, averaging the three legs would let the two silent legs drown out the one that actually responds — max is needed to preserve the signal.

**A previously unrecorded asymmetry: removing `text_nn_sim` on template drops IF by 5.2 points but RF by only 0.1 point.** Template's IF AUC is already only 0.551 (due to direction reversal), and this 5.2-point drop shows its weak label-free signal depends heavily on the text-similarity dimension; RF, having labels, can dig 0.994 out of pure trajectory features alone, with no need for `text_nn_sim` at all. This is the micro-mechanism behind the 0.994-vs-0.551 gulf between Section 6b.1's "supervised = signal ceiling" and Section 6b.4's "label-free = production-reachable": the signal genuinely exists in the trajectory, but the label-free scorer can't read its direction and has to fall back on the residual textual clue instead.

---

### 6b.9 External Baseline Comparison: This Project's Training-Dynamics Detector vs. EL2N/GraNd/TracIn Proxies From the Literature

Earlier sections compared this project's detectors (`iforest`/`zscore`/`memo_signed`/`pooled`) against each other, but never against the methods commonly cited in the data-pruning/influence-function literature. `baselines.py` (not tracked in Git, see that file's docstring and CLAUDE.md's "do not push to GitHub" convention — kept locally only, for report reference) reuses the already-collected trajectory data, with no retraining needed, to replicate three proxy metrics:

- **`el2n_proxy`**: uses only **the first epoch's** loss. EL2N's (Paul et al. 2021) original definition is the L2 norm of the error vector early in training; causal language models have no fixed label vocabulary to construct a softmax error vector from, so loss is the most direct analogue.
- **`grand_proxy`**: uses only **the first epoch's** gradient norm (GraNd, same paper).
- **`tracin_proxy`**: directly reuses the already-computed `cos_ref_mean` (a sample's own gradient direction cosine similarity to the held-out reference direction, averaged over the full trajectory) — this is itself a TracIn-style (Pruthi et al. 2020) self/cross-influence score, just not previously named as such.
- **`loss_rank`**: an existing feature, included as a reference point.

**`dolly-ratio10` (8 injected-noise datasets):**

| Dataset | `el2n_proxy` | `grand_proxy` | `tracin_proxy` | `loss_rank` | This project's best existing method (`iforest`/`zscore`/`memo_signed`) |
|---|---|---|---|---|---|
| duplicate | 0.587 | 0.510 | 0.534 | **0.666** | 0.612 (iforest) |
| garbled | **0.983** | 0.813 | 0.585 | 0.940 | 0.936 (iforest) |
| keyword | **0.682** | 0.569 | 0.566 | 0.619 | 0.552 (iforest) |
| mixed | 0.640 | 0.603 | 0.507 | 0.564 | 0.662 (iforest) |
| near_duplicate | **0.657** | 0.565 | 0.607 | 0.617 | 0.599 (iforest) |
| template | **0.931** | 0.765 | 0.877 | 0.911 | 0.925 (memo_signed, signed rule) |
| truncation | 0.684 | **0.716** | 0.580 | 0.518 | 0.657 (zscore_max) |
| unrelated | **0.771** | 0.635 | 0.558 | 0.632 | 0.703 (iforest) |

("This project's best existing method" is taken from the highest value per dataset in `results/dolly-ratio10/unsupervised.csv` and `memorization.csv`; full data is in `results/dolly-ratio10/external_baselines.csv`.)

**`oasst-wild` (wild noise, full 68,762 rows):**

| Baseline | AUC |
|---|---|
| `grand_proxy` | **0.805** |
| `el2n_proxy` | 0.760 |
| `tracin_proxy` | 0.604 |
| `loss_rank` | 0.545 |

**Core finding: using only the first epoch's loss/gradient norm (`el2n_proxy`/`grand_proxy`) matches or exceeds this project's current 5-epoch, multi-feature detector on several types** — garbled (0.983 vs 0.936), keyword (0.682 vs 0.552), near_duplicate (0.657 vs 0.599), unrelated (0.771 vs 0.703) all show `el2n_proxy` ahead, using only 1/5 of the training rounds. This aligns exactly with, and mutually corroborates, this file's Section 6b.5 "early detection" finding: noise signal mostly shows up early in training, and this project's current label-free production scheme need not necessarily run all 5 epochs to obtain usable signal. On `oasst-wild`, `grand_proxy`/`el2n_proxy` again lead while `tracin_proxy`/`loss_rank` lag behind, the same pattern.

**`tracin_proxy` (i.e. `cos_ref_mean`) used alone is the weakest of all four baselines** (0.51-0.61 on dolly-ratio10, 0.604 on oasst-wild), showing that this quantity — often cited in the literature as an influence-function "gold standard" — has limited value as a standalone detection signal; it is better suited as one leg of a combined score (like `pooled`) than as an independent metric.

**A convention caveat that must be reported together with these numbers:** as elsewhere in this report, AUC in this section is `analyze.py::auc()`'s `max(value, 1-value)`, i.e. direction is unknown — a high AUC does not mean it's ready to deploy directly. `el2n_proxy` reaches 0.931 on template, but Section 6b.4's direction-reversal trap may equally apply to it (`iforest` on the same dataset happens to have exactly the wrong direction, measured AUC only 0.522, and [06c-detectability.md](06c-detectability.md) Section 6c.5.2's targeted cleaning is accordingly worse than random removal); whether `el2n_proxy` really points in the correct direction of "noise = high loss," and whether it can be used directly for real cleaning, is not verified in this report — its sign needs to be confirmed first, the same as with `memo_signed`, before drawing conclusions from this table's numbers alone.

---

### 6b.10 How Much of the Wild-Noise Detection Signal Is a Response-Length Confound?

OASST2's human `quality` label correlates with response length (correlation coefficient −0.20, noisy responses have a median length of 101 characters vs. 674 overall), and length is an obvious confound for any feature derived from a loss curve (short responses have fewer label tokens, so the loss trajectory's shape naturally differs). `analyze.py::length_confound()` quantifies this warning: for `pooled_detector` (`cleaning_loop.py`'s combined score) and 19 fully-covered trajectory features, four numbers are reported each — `raw_auc` (original), `length_auc` (using only response length, the ceiling of the confound itself, identical for every row), `residual_auc` (AUC computed on residuals after linearly regressing out length), and `stratified_auc` (weighted average after stratifying by length decile, assuming no linear relationship).

| Signal | raw_auc | length_auc | residual_auc | stratified_auc | Residual-signal fraction* |
|---|---|---|---|---|---|
| `pooled_detector` | 0.700 | 0.799 | 0.657 | **0.589** | 44.5% |
| `text_nn_sim` | 0.507 | 0.799 | 0.533 | 0.554 | 26.9% |
| `loss_mean` | 0.614 | 0.799 | 0.605 | **0.633** | 117% |
| `loss_std` | 0.803 | 0.799 | 0.750 | 0.588 | 29.0% |
| `loss_slope` | 0.802 | 0.799 | 0.748 | 0.587 | 28.8% |
| `grad_norm_mean` | 0.790 | 0.799 | 0.742 | 0.592 | 31.7% |
| `converge_epoch` | 0.682 | 0.799 | 0.602 | 0.606 | 58.2% |
| `cos_ref_mean` | 0.604 | 0.799 | 0.581 | 0.567 | 64.4% |
| `update_contrib_mean` | 0.787 | 0.799 | 0.720 | 0.579 | 27.5% |

*Residual-signal fraction = (stratified_auc − 0.5) / (raw_auc − 0.5), measuring how much of the original "above random" portion survives stratified correction; the full 20 rows (all 19 features + `pooled_detector`) are in `results/oasst-wild/length_confound.csv`.

**Core finding one: the length confound's ceiling (0.799) is already close to `pooled_detector`'s raw AUC (0.700).** After stratified correction, `pooled_detector` drops to 0.589, retaining only 44.5% of the original "above random" signal — most of the signal genuinely comes from length, but the corrected value is still meaningfully above 0.5, showing there is real training-dynamics signal beyond length, just at less than half the apparent magnitude.

**Core finding two: contamination by length is uneven across features, splitting into two groups.** The group `loss_std`/`loss_slope`/`grad_norm_mean`/`update_contrib_mean` all have raw AUC close to the 0.799 length ceiling, and collapse together to 0.58-0.59 after stratification (retaining only 27%-32%) — almost pure length proxies; the group `loss_mean`/`cos_ref_mean`/`converge_epoch` has lower raw AUC but doesn't drop after stratification (or drops much less, retaining 58%-117%), showing these features capture signal relatively independent of length. `pooled_detector` sits between the two groups, showing the combined score partly relies on the more length-contaminated legs.

**Core finding three: `text_nn_sim` has almost no independent contribution on wild noise** (raw_auc only 0.507, near random; stratified 0.554), sharply contrasting with AGENTS.md's record that "`text_nn_sim` dominates on the injected duplicate/unrelated types" — the same feature is the strongest signal for "copy-paste/topic-mismatch" injected noise, which can be judged at the text level, yet nearly fails on wild noise assessed by human quality judgment, showing its explanatory power is highly type-specific and cannot be generalized to "equally effective on wild noise."

**Conclusion:** for `oasst-wild`'s label-free detection capability, a more careful estimate should use stratified_auc (`pooled_detector` about 0.589) rather than raw_auc (0.700) — a meaningful part of the latter is simply rediscovering the already-known regularity that "shorter responses score lower," not the discriminative power of training dynamics itself. This confound result is the key explanatory basis for [06c-detectability.md](06c-detectability.md) Section 6c.6's "why wild-noise closed-loop cleaning has a negative downstream gain."

---

**Overall conclusion for Thread 2**: the A/B/C mechanism classification Chapter 5 wrote down before the results came in is largely validated — Class A (garbled/unrelated/truncation) indeed matches the "outlier detection works" prediction, Class B (template/duplicate) indeed needs a signed rule to be detected, Class C (near_duplicate/keyword) indeed leaves almost no stable trace in training dynamics; the one place needing correction is that duplicate/unrelated's high AUC actually comes mainly from static text similarity rather than training dynamics itself (Section 6b.6), showing the basic assumption that "every noise type leaves a trace in training dynamics" does not fully hold for these two types, which need extra data-level features to make up the gap. These feature-signature findings together determine which detection method should be used for each type in [06c-detectability.md](06c-detectability.md).
