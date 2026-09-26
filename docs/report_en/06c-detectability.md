## 6c. Label-Free Separability: Precision, Boundaries, and Real Benefit

This file covers **Thread 3** of the report's three research threads: given the per-type feature signatures already established in [06b-feature-signatures.md](06b-feature-signatures.md), it asks the question that actually determines whether any of this is useful in production — **can noisy samples be picked out without looking at labels at all, how precise is that selection, and does retraining after removal actually make things better?**

**Relationship to Section 5.3's theoretical prediction**: Section 5.3 pointed out a structural difficulty built into the label-free setting — Class A noise needs an "outlier-seeking" scorer, Class B noise needs a "hyper-typical-seeking" scorer, and choosing between the two requires already knowing which class a sample belongs to, which is exactly what label-free detection is not allowed to know. This difficulty runs through every section of this file: 6c.1/6c.2 show what to do when the type *is* known; 6c.3's ablations ask whether, once a method is chosen, the raw feature set feeding it can be trimmed; 6c.4 confronts the boundary head-on when the type is *not* known; 6c.5/6c.6 verify whether — once a direction is chosen correctly or incorrectly — the actual act of cleaning is worth doing at all.

---

### 6c.1 Cleaning-Precision Lift: A High AUC Doesn't Guarantee a Useful Removal Budget

![Cleaning-precision lift](../../results/charts/precision_lift.png)

AUC measures overall ranking quality, but in practice only a limited slice of the data can ever be removed (e.g. the top 10%). The metric that actually matters at that point is **P@10% lift**: within the removed top 10%, how many times more often is a sample truly noise, compared to removing 10% at random?

**Exact computation** (`analyze.py::precision_at_k` / `precision_lift_table`): for a dataset of size n, `k = round(0.10 * n)`; sort all samples by descending anomaly score, take the top-k; `p_at_10` = the fraction of those k samples that are true noise; `random_p` = the dataset's overall true-noise fraction (the expected hit rate of "blind random removal"); `lift = p_at_10 / random_p`. Three label-free candidate scorers are compared:

- `zscore_max` / `zscore_mean`: a robust z-score per feature, `(x - median) / (1.4826 * MAD)` (MAD = median absolute deviation); the sample's anomaly score is the **max** or **mean**, respectively, of the absolute z-scores across features.
- `iforest`: after `StandardScaler` standardization, a separate `IsolationForest(n_estimators=300)` is fit per dataset (no sharing of models or pooling of data across datasets); the negative of `score_samples` (larger = more anomalous) is the score.

The full raw result across every dataset × every method on `dolly-ratio10` (the summary table elsewhere in this report shows only the highest-lift row per dataset, which can hide the fact that the same dataset can look completely different under a different scoring method):

| Dataset | Method | n | n_noise | AUC | P@10% | random_p | lift |
|---|---|---|---|---|---|---|---|
| garbled | iforest | 904 | 85 | 0.936 | 0.556 | 0.094 | **5.91** |
| garbled | zscore_mean | 904 | 85 | 0.873 | 0.311 | 0.094 | 3.31 |
| garbled | zscore_max | 904 | 85 | 0.664 | 0.089 | 0.094 | 0.95 |
| mixed | iforest | 919 | 92 | 0.662 | 0.228 | 0.100 | **2.28** |
| mixed | zscore_mean | 919 | 92 | 0.633 | 0.174 | 0.100 | 1.74 |
| mixed | zscore_max | 919 | 92 | 0.604 | 0.054 | 0.100 | 0.54 |
| unrelated | iforest | 904 | 85 | 0.703 | 0.189 | 0.094 | **2.01** |
| unrelated | zscore_mean | 904 | 85 | 0.596 | 0.167 | 0.094 | 1.77 |
| unrelated | zscore_max | 904 | 85 | 0.604 | 0.111 | 0.094 | 1.18 |
| truncation | zscore_max | 905 | 86 | 0.657 | 0.178 | 0.095 | **1.87** |
| truncation | zscore_mean | 905 | 86 | 0.610 | 0.144 | 0.095 | 1.52 |
| truncation | iforest | 905 | 86 | 0.598 | 0.156 | 0.095 | 1.64 |
| template | zscore_mean | 906 | 87 | 0.803 | 0.165 | 0.096 | **1.72** |
| template | iforest | 906 | 87 | 0.522 | 0.066 | 0.096 | 0.69 |
| template | zscore_max | 906 | 87 | 0.878 | 0.022 | 0.096 | 0.23 |
| near_duplicate | iforest | 906 | 87 | 0.599 | 0.132 | 0.096 | **1.37** |
| near_duplicate | zscore_max | 906 | 87 | 0.507 | 0.121 | 0.096 | 1.26 |
| near_duplicate | zscore_mean | 906 | 87 | 0.549 | 0.110 | 0.096 | 1.14 |
| keyword | zscore_max | 906 | 87 | 0.520 | 0.110 | 0.096 | **1.14** |
| keyword | zscore_mean | 906 | 87 | 0.525 | 0.110 | 0.096 | 1.14 |
| keyword | iforest | 906 | 87 | 0.552 | 0.099 | 0.096 | 1.03 |
| duplicate | zscore_mean | 995 | 89 | 0.542 | 0.050 | 0.089 | **0.56** |
| duplicate | iforest | 995 | 89 | 0.612 | 0.050 | 0.089 | 0.56 |
| duplicate | zscore_max | 995 | 89 | 0.604 | 0.040 | 0.089 | 0.45 |

(Bold marks the "best method" row for each dataset shown in summary tables elsewhere; `n`/`n_noise` match [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.1's diagnostic-subsample population, since this unsupervised scoring is likewise computed only where all 37 features are non-null.) The full table exposes two things a summary table hides: **garbled's lift swings enormously across the three methods** (iforest 5.91× vs. zscore_max 0.95× — the same dataset going from "clearly useful" to "worse than random" depending only on which scorer is used); and **template's `zscore_max` (0.23×) is among the single worst combinations in the entire table**, even though switching that same dataset to `zscore_mean` reaches 1.72× — showing that "picking the right method" matters just as much as "picking the right dataset."

**Best-method summary, both noise ratios:**

| Noise type | dolly-ratio10 best-method lift | dolly-ratio5 best-method lift |
|---|---|---|
| garbled | 5.91 (iforest) | 7.40 (iforest) |
| mixed | 2.28 (iforest) | 2.96 (iforest) |
| unrelated | 2.01 (iforest) | 2.49 (iforest) |
| truncation | 1.87 (zscore_max) | 2.32 (zscore_max) |
| template | 1.72 (zscore_mean) | 1.74 (zscore_mean) |
| near_duplicate | 1.37 (iforest) | 1.99 (zscore_mean) |
| keyword | 1.14 (zscore_max) | 1.99 (zscore_max) |
| **duplicate** | **0.56 (zscore_mean)** | **0.89 (zscore_max)** |

**A key warning**: exact duplication has an in-domain AUC as high as 0.986 ([06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.1), yet its P@10% lift is below 1 at both ratios (0.56× at dolly-ratio10, 0.89× at dolly-ratio5) — meaning that if the top 10% by this detector's score were removed, the fraction of true noise actually caught would be **worse than removing 10% completely at random**. The likely cause is that duplicate samples' score distribution is concentrated somewhere other than the extreme tail the top-10% cutoff selects — AUC, a global ranking metric, hides this kind of local failure under a fixed, small budget.

**Methodological implication**: ranking methods by AUC and ranking them by P@10% lift can yield different "best method" conclusions — using AUC alone to choose a cleaning strategy is biased. Any method claimed to be production-ready for automatic cleaning should report budget-constrained precision, not AUC alone.

---

### 6c.2 Best Method Per Noise Type: A Current-State Summary

Earlier sections separately discuss each type's detection difficulty, feature attribution, and direction-reversal behavior (full detail in [06b-feature-signatures.md](06b-feature-signatures.md)), but nowhere collect "which method should be used for this type, what raw data does that method depend on, and is all of that data actually necessary" in one place. This section is that summary; Section 6c.3 below upgrades the "necessity" judgment from qualitative attribution to quantitative ablation evidence.

**The `mixed` row is qualitatively different from the other seven**: the other 7 rows all assume the noise type is known and the method calibrated to it; `mixed` is the "composition unknown" case, handled in full in Section 6c.4, with Section 6c.4.5 reporting the measured cost of a three-leg parallel scorer.

One convention distinction needs stating up front: the AUCs reported in earlier sections (throughout [06b-feature-signatures.md](06b-feature-signatures.md) and Section 6c.1 of this file) (`unsupervised.csv`/`memorization.csv`/`feature_attribution.csv`) are all computed on the **diagnostic-subsample table** (roughly 900-1200 rows per dataset, a 12.5% sample), where all features — including token-level diagnostics and `cos_global_*` — are available; whereas label-free closed-loop cleaning (Section 6c.5) via `cleaning_loop.py` scores and removes on the **full training set** (14,611+ rows), where token-level diagnostics and `cos_global_*` fall short of 100% coverage, leaving only 19 fully-covered features (Appendix Sections 6.1 and 8.3). The two don't always agree: for garbled/template, the fully-covered features happen to be sufficient, so the reported AUC and production precision roughly track each other; but for near_duplicate/keyword, a large share of the usable signal sits inside the token-diagnostic data, so the reported AUC looks better than what production can actually achieve.

| Noise type | Best label-free method (full-training-set AUC) | Supervised ceiling (RF within-type AUC, reference only) | Main raw-data category relied on | Necessity conclusion |
|---|---|---|---|---|
| garbled | iforest + full-coverage features, 0.932 | 0.998 | Full-coverage trajectory features (`loss_curvature`/`loss_rank`, etc.) | Trajectory features already sufficient; neither `text_nn_sim` nor token diagnostics are necessary (Section 6c.3's ablation: adding/removing either makes no notable difference) |
| **template** | **must use memo_signed**, 0.925 (iforest alone is only 0.551, nearly random) | 0.999 | memo_signed's 6 signed trajectory features; but Section 6c.3's ablation shows `text_nn_sim` alone reaches 0.805 | Signed trajectory features are the necessary and sufficient current-production solution; `text_nn_sim` is an independent second signal source — not necessary, but usable as cross-validation |
| duplicate | iforest + full-coverage features, 0.533 (**weak**, and diluted) | 0.986 | In theory, `text_nn_sim` alone would suffice | **Current production scheme is not optimal** — Section 6c.3's ablation shows `text_nn_sim` (zscore) alone reaches 0.938, 0.41 higher than the current 19-feature mixed iforest; the remaining features are essentially dilution |
| unrelated | iforest + full-coverage features, 0.639 | 0.925 | `text_nn_sim` dominant, trajectory features contribute meaningfully too | Partially redundant — `text_nn_sim` alone reaches 0.783 (Section 6c.3), already beating the current mixed scheme, but trajectory features still add incremental value, so it can't be simplified to a single feature the way duplicate can |
| truncation | zscore_max/iforest + full-coverage features, around 0.58 | 0.763 (the ceiling itself is not high) | Full-coverage trajectory features and token diagnostics each contribute a share; no single strong feature | Nothing left to trim — every available category is already in use and the result is still limited; this is genuine detection weakness, not a feature-selection problem |
| near_duplicate | iforest + full-coverage features, 0.614 (weak) | 0.674 (likewise on the low side) | Token-level diagnostic signal is strongest (0.641) but unavailable in production; `text_nn_sim` contributes almost nothing among the full-coverage features | **Insufficient data-category coverage** — the signal that actually works lives in token diagnostics, which only cover 12.5% of samples; the current full-coverage data is close to its ceiling for this type |
| keyword | iforest/zscore + full-coverage features, 0.55-0.59 (near random) | 0.577 (the ceiling itself is low) | Every category contributes weakly (Section 6c.3: `text_nn_sim`, full-coverage trajectory, and token diagnostics all sit between 0.50 and 0.59) | **Not a wrong method choice — none of the currently available raw-data categories are enough.** A 1-2 word substitution barely perturbs the TF-IDF vector or the training trajectory; a genuinely new word-level substitution-detection feature is needed |
| mixed | **`pooled` three-leg parallel, 0.728** (P@10% 0.321; `iforest` alone 0.714/0.270) | No attribution/within-type data available (`feature_attribution.csv`/`cross_type.csv` have no `mixed` row) | All three raw-data categories: full-coverage trajectory (outlier side) + signed trajectory (memorization side) + `text_nn_sim` (static text side) | Section 6c.4.5's measurement: the parallel combination beats any single method on precision, at the cost of the removal budget being dominated by duplicate/garbled; this is the recommended scheme for "composition unknown," not the precision-optimal one |

---

### 6c.3 Feature Ablation: Which Data Is Necessary, Which Metric Is Irreplaceable?

The correlation analysis showing the 19 fully-covered metrics span only about 5 effective dimensions is in [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.7; the single-feature leave-one-out ablation showing `text_nn_sim` is the only irreplaceable metric is in that same file's Section 6b.8. This section builds on both conclusions to answer two questions that feed directly into deployment decisions: **which categories of raw data are actually worth collecting, and can the feature set be narrowed to a minimal collection?**

#### 6c.3.1 Motivation and design

Two of Section 6c.2's "necessity conclusions" currently rest only on the attribution-importance ranking ([06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.6), not on quantitative evidence: is duplicate/unrelated's detection signal really being "diluted" by unrelated features? How much would near_duplicate/keyword gain from adding token-level diagnostics unavailable in production, and is it worth re-engineering the collection pipeline to cover them at full scale?

Both questions can be answered **without retraining anything**: the diagnostic-subsample table (`results/dolly-ratio10/per_sample_metrics.csv`) already contains all three categories of raw data — full-coverage trajectory features, `text_nn_sim`, and token-level diagnostics (each with different coverage, but drawn from the same overlapping 900-1200-row subsample population) — so it's enough to re-score with `analyze.py::unsupervised_metrics()` using different `features=` subsets, which runs in seconds on CPU across all datasets. Script: `analyze.py::feature_group_ablation()`, output `results/dolly-ratio10/feature_ablation.csv`.

Five ablation conditions:

| Condition | Feature scope | Available to `cleaning_loop.py` today |
|---|---|---|
| `full_diag` | All numeric columns (including token diagnostics, `cos_global_*`) | No — this is exactly the raw `unsupervised.csv` result cited in earlier sections (throughout [06b-feature-signatures.md](06b-feature-signatures.md) and Section 6c.1 of this file) |
| `full_coverage` | 19 fully-covered features (Sections 6.1 and 8.3; the feature set `cleaning_loop.py` actually uses in production) | **Yes, current production scheme** |
| `no_text` | `full_coverage` minus `text_nn_sim` (18 features) | Yes |
| `text_only` | `text_nn_sim` alone | Yes |
| `token_only` | Token-level diagnostics alone (13 features, only 12.5%-subsample coverage) | **No** — unavailable in production; included only to quantify "how much it would be worth if it were available" |

#### 6c.3.2 Results

![Feature ablation: is text_nn_sim diluted, and how much would token-level diagnostics add?](../../results/charts/feature_ablation.png)

(Chart values are the best AUC among zscore_max/zscore_mean/iforest under each condition; `full_coverage` is the current production scheme, `text_only`/`token_only` are the two control groups, the latter marked to emphasize "unavailable in production, reference only.")

Three quantitative conclusions:

1. **The current production scheme genuinely dilutes signal for duplicate and unrelated.** Duplicate scored with `text_nn_sim` (zscore) alone reaches AUC 0.938, 0.41 above the current 19-feature mixed iforest (0.528); unrelated scored with `text_nn_sim` alone reaches 0.783, 0.14 above the current scheme (0.641). This is not an inference from an attribution ranking — it's a direct head-to-head comparison: mixing the other 18 weak features into the same unsupervised scorer actively drags down a signal that was already strong on its own. The `no_text` condition (duplicate 0.564, unrelated 0.571) confirms this further: with `text_nn_sim` removed, the remaining trajectory features are weak on their own too, confirming that "dilution" really is the current production scheme's problem, rather than the trajectory features hiding some other value.
2. **Template's high detectability is the sum of two independent signals, not memorization alone.** `text_nn_sim` alone reaches AUC 0.805 on template — templated noise naturally reuses a fixed template, producing high text-level similarity as a byproduct, which is a completely independent clue from "memorized / abnormally fast convergence" (the signal `memo_signed` covers). The current `memo_signed` scheme (0.925) is already good enough; this finding mainly explains *why* template is so easy to detect rather than suggesting a new improvement direction.
3. **Near_duplicate and keyword are genuine methodological blind spots, not cases of the wrong scoring method.** Even adding production-unavailable token-level diagnostics, near_duplicate's AUC only reaches 0.641; keyword's three conditions all land between 0.50 and 0.59, with `text_nn_sim` (0.531) and token diagnostics (0.514) performing similarly weakly. None of the currently available raw-data categories are sufficient for these two "mild, localized perturbation" noise types — no recombination of existing data will fix this; dedicated new features are needed (e.g. per-token localized-substitution detection, rather than whole-passage or whole-trajectory statistics).

#### 6c.3.3 Minimal Feature Set: 3 Features Beat All 19 on the Label-Free Route

[06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.8's leave-one-out ablation answers "what happens if this one feature is removed" — the wrong question for **deciding what to collect**, since highly-correlated features can each individually be dispensable while the whole group is jointly necessary; 19 features that each "look useless alone" could easily be jointly necessary. Answering "what's the minimum needed" requires the opposite approach: starting from nothing, greedily adding whichever feature raises AUC the most at each step, and recording the whole path — the k-th point on that path is "the best achievable score using exactly k features." Script: `analyze.py::minimal_feature_set()`, output `results/dolly-ratio10/minimal_feature_set.csv`.

**A convention caveat first**: greedy selection uses the very labels it is being scored against, so the AUC reported for a given k is **optimistic** — it measures "how much signal a small feature set can carry at its best," not a label-free-usable feature-selection recipe (Section 3.1 explicitly forbids selecting features by label). The honest reading is "if the noise type is already known, this is a lower bound on collection cost." The IF column reports `auc_dir = max(auc, 1-auc)`, since a label-free outlier score dropping to 0.02 doesn't mean uselessness — it means the direction is reversed (Section 6b.4).

![Minimal feature set: the greedy forward-selection AUC path](../../results/charts/minimal_feature_set.png)

**Conclusion 1 (the most important in this section): on the label-free route, 3 features comprehensively beat all 19 — 8 out of 8 datasets, no exceptions, average gain 0.132.**

| Dataset | IF with 3 features | IF with all 19 | Gain |
|---|---|---|---|
| template | **0.875** | 0.551 | **+0.324** |
| duplicate | **0.744** | 0.533 | **+0.211** |
| unrelated | **0.840** | 0.639 | **+0.201** |
| truncation | 0.650 | 0.573 | +0.077 |
| mixed | **0.791** | 0.714 | **+0.077** |
| keyword | 0.665 | 0.590 | +0.075 |
| garbled | **0.977** | 0.932 | +0.045 |
| near_duplicate | 0.650 | 0.607 | +0.043 |

This is not a marginal improvement — it means **the current production scheme (feeding all 19 fully-covered features into IsolationForest) is systematically holding itself back.** Template jumps from a near-random 0.551 to 0.875, and duplicate from 0.533 to 0.744 — these are exactly the two types with the worst direction-reversal problems in [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.4, showing that part of "direction reversal" is actually **dimensional dilution**: unrelated dimensions flatten the outlier distance, drowning the two or three dimensions actually carrying signal. Section 6b.8's observation that "removing a single feature often makes IF better" is the small-scale version of exactly this pattern; this section shows its ceiling.

**Conclusion 2: on the RF route, 1 feature reaches 95% of the full-19 ceiling, but reaching 99% needs 3-6.**

| Dataset | Full 19 | Features needed for 95% | Features needed for 99% |
|---|---|---|---|
| garbled | 0.996 | 1 | 3 |
| duplicate | 0.984 | 1 | 6 |
| template | 0.994 | 2 | 3 |
| unrelated | 0.944 | 3 | 5 |
| keyword | 0.676 | 1 | 5 |
| mixed | 0.838 | 3 | 6 |
| near_duplicate | 0.696 | 4 | 6 |
| truncation | 0.787 | 5 | not reached |

With labels available, features are highly substitutable for one another, so the first one or two already capture most of the signal; the remaining 1-5 points have to be squeezed out slowly across additional dimensions. Truncation still hasn't reached 99% at k=6, the one type genuinely needing a wide feature surface.

**Conclusion 3: the first feature selected exposes each type's underlying mechanism, and the two routes disagree on which feature to pick first.**

| Dataset | RF first pick | IF first pick |
|---|---|---|
| duplicate | `text_nn_sim` | `grad_norm_cv` |
| unrelated | `text_nn_sim` | `text_nn_sim` |
| garbled / template / mixed | `loss_curvature` | `loss_curvature` / `loss_min` / `loss_curvature` |
| keyword / near_duplicate | `converge_epoch` | `converge_epoch` |
| truncation | `converge_epoch` | `loss_slope` |

`loss_slope` appears in 5 of the 8 datasets' k=3 sets, the single most broadly useful feature; `text_nn_sim` and `converge_epoch` each appear 3 times. Duplicate shows the largest disagreement between routes — RF goes straight for `text_nn_sim` (with a label, it knows to check "does this look like something else"), while IF instead first picks `grad_norm_cv`, because with no label it has no way to know "similarity" is the suspicious direction, and can only search for anomaly in gradient fluctuation instead.

**Deployment implication**: if the noise type is known and calibrated, label-free scoring **should use only 2-3 features, not 19** — higher precision and cheaper collection. But "which 3" depends on the noise type (every row of the table above differs), and the type being unknown is exactly the premise of the label-free setting — which loops back to Section 5.3's structural difficulty. The `pooled` scorer currently still feeds all 19 fully-covered features into its iforest leg; this section shows that leg has an unexploited 0.04-0.32 improvement available, an explicit follow-up direction (Section 7.3).

#### 6c.3.4 Deployment recommendations

- **High priority, low cost**: add a third `method` option (e.g. `text_sim`) to `cleaning_loop.py`, scoring duplicate/unrelated directly with `text_nn_sim`'s zscore; expected to substantially improve removal precision, and since `text_nn_sim` is already a fully-covered feature, no new data collection is needed. (**Partially done already**: the `pooled` scorer added in Section 6c.4.5 folds `text_nn_sim`'s |z| in as an independent parallel leg, achieving per-type AUC 0.946 and 97.4% recall for duplicate within `mixed`; but `pooled` is designed for "composition unknown" scenarios — a single-type, already-calibrated setting would still benefit from a pure `text_nn_sim`-only option at higher precision.)
- **High priority**: narrow the `iforest` leg from 19 features down to 2-3 (Section 6c.3.3) — the one change that needs no new data and no new method, just fewer features fed in, for a 0.04-0.34 gain.
- **Not recommended to invest in right now**: near_duplicate/keyword's improvement needs new features rather than a new scoring method, a larger development effort; for now this is explicitly recorded as a known limitation (folded into Section 7.2), to be revisited once there's clear downstream-benefit evidence (analogous to [06a-harm-ranking.md](06a-harm-ranking.md)'s validation for template).

---

### 6c.4 Boundaries of the Method Under an Unknown Noise Type

Every section above assumes the noise type is known — every type has a corresponding injected dataset and a trained detector. Production doesn't grant this assumption. Real dirty data mixes multiple noise types together, and likely includes types outside these 7 entirely. This section answers two questions: **what happens when an existing detector encounters a mixed stream? What happens when it encounters a type it has never seen at all?**

#### 6c.4.1 Why Section 6b.2's transfer matrix can't answer this

[06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.2's 7×7 cross-type transfer matrix explicitly skips the `mixed` dataset at the code level (`analyze.py` contains `if ds in ('clean', 'mixed'): continue`). It answers "can a detector trained on type A find type B" — but every test in that matrix still faces a **single** noise type against a 90%-clean background. Production is "one stream containing all 7 noise types at once, 6 of them never seen" — the distribution itself is different.

A supplementary experiment (`analyze.py::transfer_to_mixed()`, output `results/dolly-ratio10/transfer_to_mixed.csv`) evaluates all 7 single-type detectors against `mixed`. The protocol matches Section 6b.2 exactly (LR and RF each trained, `StandardScaler` fit only on the source domain, the higher of the two AUCs taken). `mixed` has 14,819 rows total, 13,419 clean and 1,400 noisy, each row carrying its own `noise_type` label (near_duplicate 211, template 206, truncation 204, keyword 197, unrelated 197, garbled 194, duplicate 191), which allows slicing within the mixed stream by type.

Two coverage regimes are reported: `full_coverage` (19 fully-covered features, the full 14,819-row population — what `cleaning_loop.py` can actually compute in production) and `full_diag` (37 features, the ~919-row diagnostic subsample, matching the convention used elsewhere in the report). The main text below uses `full_coverage`.

![Boundaries under an unknown noise type](../../results/charts/unknown_noise.png)

#### 6c.4.2 Conclusion One: Detectors Are "Narrow," Not "Broken"

| Detector | Overall AUC on mixed stream | Overall P@10% | AUC on its own type alone | Within-domain AUC (Section 6b.1) | Retention |
|---|---|---|---|---|---|
| near_duplicate | 0.730 | 0.297 | 0.726 | 0.674 | 1.08 |
| keyword | 0.713 | 0.310 | 0.711 | 0.577 | 1.23 |
| unrelated | 0.701 | 0.367 | 0.974 | 0.925 | 1.05 |
| truncation | 0.688 | 0.248 | 0.848 | 0.763 | 1.11 |
| duplicate | 0.668 | 0.256 | 0.980 | 0.986 | 0.99 |
| garbled | 0.662 | 0.298 | 0.998 | 0.998 | 1.00 |
| template | 0.563 | 0.187 | 0.960 | 0.999 | 0.96 |

(Random baseline: overall P@10% = 0.094; per-type-slice P@10% ≈ 0.015)

The overall column drops uniformly to 0.563-0.730, which looks like the detectors are failing on the mixed stream. But once evaluation is restricted to "its own type vs. clean," AUC climbs back to 0.711-0.998, with **retention of 0.96-1.23 — no type shows real decay.** These two sets of numbers together admit only one explanation: the AUC drop is not the detector breaking down under a changed distribution — it's that 6 of 7 noise types in the denominator are simply things the detector was never trained to find. Each detector is still doing exactly what it was built to do; that job just covers only 14% of the noise in the mixed stream.

A counterintuitive detail worth flagging: keyword and near_duplicate have retentions of 1.23 and 1.08 respectively — they actually perform **better** inside the mixed stream than in their own dedicated dataset. The reason is that `mixed`'s clean background population is much larger (13,600 rows vs. roughly 1,300 in the single-type dataset), and AUC, as a ranking metric, is more stable on larger samples; part of the within-domain 5-fold CV's low 0.577 score is small-sample variance. This doesn't change the qualitative conclusion that "keyword is the hardest to detect," but it does mean [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.1's absolute numbers for the two hardest types are somewhat conservative.

#### 6c.4.3 Conclusion Two: Ensembling Rescues the Overall Picture, But Not an Unseen Type

Taking the per-sample max of all 7 detectors' z-scored outputs (`ensemble_all7`):

| Scheme | Overall AUC on mixed stream | P@10% |
|---|---|---|
| 7-detector parallel max | **0.837** | **0.467** |
| Best single detector (near_duplicate) | 0.730 | 0.297 |
| `text_nn_sim` zscore (no calibration needed) | 0.660 | 0.255 |
| Random | 0.500 | 0.094 |

Ensembling pushes P@10% from 0.297 to 0.467 — under a 10% cleaning budget, 47 of every 100 removed rows are true noise, a lift near 5× relative to random. But this number **is not achievable in production**: it assumes every noise type present in the mixed stream has already been identified and given a trained detector — exactly the assumption production violates, so 0.837 should be read as an upper bound, not a plan.

To ask the real question requires leave-one-out: for each target type, use only the **other 6** detectors in parallel, then evaluate only on the "that type vs. clean" slice. The target type is now genuinely unseen by every detector in the pool.

| Unseen type | Leave-one-out ensemble AUC | Leave-one-out ensemble P@10% | `text_nn_sim` AUC | `text_nn_sim` P@10% |
|---|---|---|---|---|
| garbled | **0.935** | 0.112 | 0.396 | 0.002 |
| unrelated | 0.826 | 0.059 | **0.869** | 0.087 |
| duplicate | 0.769 | 0.036 | **0.974** | 0.137 |
| truncation | **0.747** | 0.048 | 0.555 | 0.015 |
| near_duplicate | **0.670** | 0.029 | 0.490 | 0.012 |
| keyword | **0.668** | 0.040 | 0.482 | 0.018 |
| template | 0.416 | 0.008 | **0.866** | 0.049 |

(Random baseline: AUC = 0.5, P@10% ≈ 0.015; bold marks the better of the two per row)

Three things worth noting.

**First, an unseen type is on average still somewhat detectable, but the variance is enormous.** The 7 leave-one-out values range from 0.416 to 0.935, median 0.747. The claim that "there is cross-type-shared noise signal in training dynamics" holds, but whether any *specific* unknown type gets caught by the existing detector pool is essentially unpredictable.

**Second, template drops to 0.416 under leave-one-out — below random.** This is a direct instance of [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.4's direction-reversal trap, and its most dangerous form on an unseen type. Templated samples are highly typical, quickly memorized by the model — low loss, fast convergence; the other 6 detectors have all learned that "noise = high loss, slow convergence, unstable gradient," so they rank templated samples toward the **cleanest** end. Below-random means cleaning by this score would **preferentially retain** templated noise — worse than not detecting it at all. `text_nn_sim`, by contrast, reaches 0.866 on the same slice, since templated samples are highly similar to their text-level neighbors, a static signal this feature catches directly.

**Third, the two routes are complementary, not interchangeable.** The 4 types where the leave-one-out ensemble wins (garbled 0.935/0.396, truncation 0.747/0.555, near_duplicate 0.670/0.490, keyword 0.668/0.482) are exactly where `text_nn_sim` is at or below random — garbled looks nothing like natural language at the character level, but its embedding-neighbor similarity isn't anomalous, while training dynamics expose it immediately. Conversely, the 3 types where `text_nn_sim` wins (duplicate 0.974, unrelated 0.869, template 0.866) are all types where "the text itself looks suspicious." Taking the better of the two per type gives 7 AUCs — 0.935/0.866/0.974/0.747/0.869/0.670/0.668 — every one above 0.66, none below random.

#### 6c.4.4 Deployment recommendations

For an unknown noise type, actionable steps in increasing order of cost:

1. **Run both legs, in parallel, never just one.** The trajectory-detector pool plus `text_nn_sim` zscore, scored simultaneously, take the higher. This is the only combination in the table above that eliminates every below-random cell, at near-zero cost (`text_nn_sim` is already a fully-covered feature, no new collection needed). Section 6c.4.5 wires this recommendation into `cleaning_loop.py` and reports the measured label-free cost — precision does improve, but "eliminating every below-random cell" doesn't fully carry over under label-free conditions.
2. **Build in a safety valve against direction reversal.** The one-directional assumption "noise = high loss" has a clear failure case on unseen types (template at 0.416). [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.4's `memo_signed` is exactly a bidirectional scorer, treating "abnormally easy to learn" as equally suspicious. Applied to unknown noise, bidirectional scoring should be the default, erring toward over-flagging for human review rather than one-directional removal.
3. **Don't trust a single detector's overall AUC.** The 0.563-0.730 range in the first column above invites the misdiagnosis "the detector is mediocre" — the truth is it covers only 14% of the noise and is nearly perfect on that 14%. Evaluating a heterogeneous stream requires slicing by type, or "narrow" gets misdiagnosed as "broken."
4. **The incremental cost of a newly identified type is low.** Once a previously-unknown noise type is manually identified, injecting a batch of samples and training a dedicated detector moves it from the 0.416/0.670 tier straight to the 0.96-1.00 tier (Section 6c.4.2's "own type alone" column). The whole process needs no base-model retraining, just one LoRA fine-tuning run with metrics logging (about 3.1 hours at dolly scale, Section 8.5) — "continuous discovery + incremental detectors" is a more realistic path than chasing one universal detector.

An unverified limitation: this section's "unseen types" are still one of the 7 types this project itself injected, merely unseen by the detector pool. Genuinely wild noise (user mis-pastes, encoding errors, cross-language contamination, machine-translation artifacts, etc.) may differ from these 7 in both text distribution and training dynamics — the leave-one-out range of 0.416-0.935 cannot be extrapolated directly to it. This is folded into Section 7.2's limitations list.

#### 6c.4.5 Wiring the Recommendation Into Production Code: the `pooled` Scorer

Section 6c.4.4's first recommendation is already implemented in `cleaning_loop.py` as a third `method` option, `pooled`. It standardizes three legs and takes the per-sample max:

- `iforest` z-score — the direction-agnostic outlier side, 19 fully-covered features;
- `memo_signed` z-score — the fixed-sign "abnormally easy to learn" side, 6 features (a subset of the 19 above);
- `text_nn_sim`'s **|z|** — the static text side. This leg takes the absolute value rather than a single direction: a high `text_nn_sim` means "nearly identical to its neighbors" (exact duplication, templating), a low one means "unlike anything else" (garbled, topic mismatch) — both ends are suspicious. The other two legs are one-directional.

Max rather than mean is used because each leg is silent (score near 0) on types it cannot see; averaging would let two silent legs dilute the one leg that is actually raising an alarm.

**A convention caveat**: Section 6c.4.3's leave-one-out test uses **supervised** detectors (trained on LR/RF against other types' labels), measuring "what happens when an existing detector meets a new type"; here, all three legs are **fully label-free**, measuring "what's achievable with nothing but a pile of unlabeled dirty data." The two sets of numbers are not directly comparable, but they point in the same direction.

Measured on `mixed` under a 10% budget (`analyze.py::pooled_scorer_compare()`, output `results/dolly-ratio10/pooled_scorer_compare.csv`):

| Scorer | Overall AUC | Removal precision P@10% | Lift vs. random | Below-random per-type cells |
|---|---|---|---|---|
| `pooled` | **0.728** | **0.321** | **3.81×** | **0** |
| `iforest` | 0.714 | 0.270 | 3.20× | **0** |
| `text_nn_sim` \|z\| alone | 0.636 | 0.255 | 3.02× | 2 |
| `memo_signed` | 0.380 | 0.128 | 1.52× | 5 |
| random | 0.500 | 0.084 | 1.00× | — |

(The lift denominator is the **actual random-removal** noise fraction among the 1,482 removed rows, 8.43%, taken from `metadata.json`'s `random_precision`, matching Section 6c.5.3's convention. Using the theoretical full-population noise rate 9.45% as denominator instead gives lifts of 3.40×/2.86×/2.70×/1.36× for the same four rows — same ordering, slightly lower absolute values.)

Per-type AUC (that type vs. clean):

| Type | `iforest` | `memo_signed` | `text_nn_sim` \|z\| | `pooled` |
|---|---|---|---|---|
| duplicate | 0.689 | 0.651 | **0.967** | 0.946 |
| garbled | **0.972** | 0.008 | 0.513 | 0.935 |
| unrelated | 0.743 | 0.230 | 0.809 | **0.842** |
| template | 0.657 | **0.866** | 0.736 | 0.818 |
| keyword | **0.641** | 0.319 | 0.513 | 0.548 |
| truncation | **0.685** | 0.255 | 0.472 | 0.526 |
| near_duplicate | **0.625** | 0.318 | 0.465 | 0.505 |

Three points that must be stated honestly:

**First, `pooled` wins on precision, and in this particular run eliminates every below-random cell.** It lifts P@10% from `iforest`'s 0.270 to 0.321 (about a 19% relative improvement); the weakest per-type case, near_duplicate, rises from `iforest`-alone's 0.625 to 0.505 — just barely above the 0.5 random line, technically "not below random" but still far from truly reliable — 0.505 is essentially indistinguishable from random, only no longer actively reversed the way `memo_signed` is. Section 6c.4.3's leave-one-out finding that "ensembling can eliminate below-random cells" holds directionally here too, but the cost is that near_duplicate is still effectively undetected, just not actively wrong. The label-free `memo_signed` leg is itself below random on 5 of 7 types (garbled only 0.008 — it ranks garbled toward the cleanest end), and contributes far more noise into the max operation than a supervised detector pool would. `pooled` should therefore be read as "trading precision for coverage under unknown noise," not a free improvement.

**Second, `memo_signed` used alone on the mixed stream is disastrous (AUC 0.380, worse than random by 0.12).** This is not a bug — it is the boundary of its design: the fixed-sign rule only works for memorized, hyper-typical noise; 5 of 7 types in the mixed stream are ordinary outlier-type noise, and the rule actively **reverses their ranking**. [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.4 already noted that this rule's AUC on non-memorized noise necessarily falls below 0.5 — this is that cost showing up in a real mixed-stream scenario; template's single-type removal precision of 0.531 (vs. `iforest`'s 0.040) does not extrapolate to the mixed setting.

**Third, the removal budget gets eaten by the easiest types.** Looking at the type composition of the 1,482 removed rows: `pooled` captured 186 duplicate rows (97.4% of that type) and 141 garbled rows (72.7%), but only 14 template rows (6.8%), despite template's respectable 0.818 per-type AUC. The reason is a limited budget combined with duplicate's generally higher score — **a good per-type AUC does not guarantee capture under a shared budget.** This is the same phenomenon as Section 6c.1's "AUC high doesn't guarantee a useful removal budget," made worse in the mixed setting because types compete for the same limited slots.

Taken together, the production recommendation is layered: **when the noise type is known and calibrated, use a single dedicated method** (`memo_signed` for template, `iforest` for garbled, `text_nn_sim` directly for duplicate/unrelated) for clearly higher precision; **when the composition is unknown or mixed, use `pooled`** — currently the only option that avoids actively reverse-ranking any of the 7 types, but one must accept a precision ceiling around 0.32 and a budget that gravitates toward the most salient types, with weak-signal types like near_duplicate remaining barely usable even when "not below random." The full closed-loop retrain-and-evaluate on `mixed` is in Section 6c.5.3 below.

---

### 6c.5 Label-Free Closed-Loop Cleaning: From "Can Detect" to "Cleaning Actually Works"

**This is the project's only set of experiments that advances from "ranking scores" to an actual cleaning action** — a qualitatively different step from everything before it, which only ever validated "can noise be told apart" (AUC/lift numbers), never "does retraining after cleaning actually improve anything." This section runs three closed loops: 6c.5.1 on garbled (easiest to detect, but near-zero downstream harm, detailed in [06a-harm-ranking.md](06a-harm-ranking.md)), 6c.5.2 on template (easy to detect and genuinely harmful downstream, with an added comparison of the cost of choosing the wrong scoring direction), and 6c.5.3 on mixed noise (using Section 6c.4.5's `pooled` scorer).

#### 6c.5.1 First Validation: Garbled Noise, High Precision but Zero Downstream Gain

![Closed-loop cleaning hit precision](../../results/charts/cleaning_precision.png)

This experiment uses `cleaning_loop.py`'s `iforest` scorer. It is one of three options (the other two being `memo_signed` and Section 6c.4.5's `pooled`); it was chosen because garbled is a textbook outlier-type noise, for which `iforest`'s "noise = outlier" assumption holds exactly — [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.4 and Section 6c.4.5 above document how that same assumption fails for memorized noise and mixed streams respectively.

Using `garbled@dolly-ratio10` (the full 14,611-row training set, true noise fraction 9.999%) as the example: pure unsupervised IsolationForest (no noise label used at all, 19-dimensional base trajectory features + `text_nn_sim`, fit separately on the full dataset) scores and removes under a 10% budget.

**Method and feature list** (`cleaning_loop.py::build`): unlike [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.1.1's "in-domain detection AUC," which uses 37 features computed only on the ~900-1000-row diagnostic subsample, scoring here must happen on the **full training set** (otherwise most true noise samples would never even be candidates for removal, making cleaning meaningless), so the feature subset used is "every feature that is non-null for every row in this dataset," excluding all token-level diagnostic features (subsampled, heavily missing) and `cos_global_*` (compared within the same optimizer-step window against other samples, structurally inconvenient to reuse for per-sample scoring), leaving 19 features with 100% coverage:

`text_nn_sim, loss_mean, loss_last, loss_std, loss_slope, loss_min, converge_epoch, loss_curvature, loss_rank, grad_norm_mean, grad_norm_last, grad_norm_std, grad_norm_slope, cos_ref_mean, cos_ref_last, cos_ref_std, cos_ref_slope, grad_norm_cv, update_contrib_mean`

Concrete procedure: after `StandardScaler` standardization of these 19 dimensions, `IsolationForest(n_estimators=300, random_state=42)` is fit on the full dataset, taking the negative of `score_samples` as the anomaly score (no label used, purely score-based ranking); under a 10% budget, `n_drop = round(0.10 * n)` — the top `n_drop` scoring samples form the "targeted removal" set, and an equal number of samples independently drawn at random with the same seed forms the "random removal" control; the true `noise_type != 'none'` fraction within each removal set is the "targeted precision" and "random precision" reported below.

- **Targeted removal precision: 52.1%** (of the 1,461 removed rows, 761 are genuinely injected garbled noise)
- **Random removal precision: 9.2%** (closely matching the true noise fraction of 10.0%, as expected)
- **Lift: about 5.7×**

**Downstream benchmark comparison (retraining and evaluation both complete)**: retrain from scratch on the `train_targeted` (after targeted removal) and `train_random` (random-removal control) training sets, compared against the uncleaned baseline (`eval_dolly-ratio10_garbled.json`) and clean baseline (`eval_dolly-ratio10_clean.json`) in a four-way comparison:

![Actual impact of closed-loop cleaning on downstream performance](../../results/charts/cleaning_downstream.png)

| Version | 7-benchmark average accuracy |
|---|---|
| Clean baseline | 0.4389 |
| Uncleaned baseline (garbled) | 0.4388 |
| Targeted-removal retrain | 0.4364 |
| Random-removal retrain | 0.4357 |

**Core finding — the 5.7× precision advantage did not translate into any observable downstream improvement; both retrained versions in fact score slightly below the uncleaned baseline**: targeted removal (0.4364) is indeed slightly above random removal (0.4357), the expected direction (more precisely removing true noise should beat blind removal), but neither exceeds the uncleaned baseline (0.4388), let alone the clean baseline (0.4389); this gap (0.0007-0.0032) itself falls within the "the 7-benchmark average range is itself narrow (0.42-0.44)" noise band already noted in [06a-harm-ranking.md](06a-harm-ranking.md) Section 6a.1, and shouldn't be over-read as a statistically meaningful decline.

The real explanation was already laid down in [06a-harm-ranking.md](06a-harm-ranking.md) Section 6a.1: **garbled noise's true downstream harm is already essentially zero** (uncleaned baseline 0.4388 vs. clean baseline 0.4389, a gap of just 0.0001) — it was chosen for this closed-loop demonstration because it's the easiest type to detect (AUC > 0.98), not because it's the most harmful. When a noise type is "easy to detect" but "carries little real downstream harm to begin with," cleaning it away naturally yields no downstream gain; and removing 10% of the training data — however precisely targeted — also reduces the effective training set by 10%, so if that "data-loss" side effect slightly exceeds the "denoising" benefit, cleaning can end up scoring marginally lower than not cleaning at all — exactly the 0.4388 → 0.4364/0.4357 pattern observed here.

This result overturns the implicit assumption the closed-loop experiment started with ("higher detection/removal precision must mean better downstream performance") and is itself a valuable negative result: **whether cleaning yields a downstream gain depends on whether this noise type genuinely drags down downstream performance, not merely on how high detection/removal precision is.** [06a-harm-ranking.md](06a-harm-ranking.md) Section 6a.1 already identified template as the one type satisfying both "easy to detect" and "genuinely harmful downstream" (an 8.95-point GSM8K drop relative to clean) — so the next section reruns the same closed loop on template.

#### 6c.5.2 Switching to Template: Cleaning Produces a Positive Downstream Gain for the First Time

Garbled's negative result has two possible explanations — either cleaning itself doesn't help, or the wrong noise type was chosen for the test. Disambiguating requires rerunning on a type known to carry real harm. Template is the only type that qualifies: [06a-harm-ranking.md](06a-harm-ranking.md) Section 6a.1 measured an 8.95-point GSM8K drop relative to clean, and [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.1's supervised AUC is as high as 0.999, so detection signal is not in question.

Template also offers a comparison garbled cannot: it is memorized noise, on which **the two scorers point in exactly opposite directions** ([06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.4), so under identical budget, seed, and training configuration, the "correctly-directed scorer" and the "wrongly-directed scorer" can be compared head-to-head:

| Removal method | Scorer | Feature count | Targeted precision | Random control | Lift vs. random |
|---|---|---|---|---|---|
| Random control | none | — | — | 9.24% | 1.00× |
| Targeted (wrong direction) | `iforest` | 19 | **4.04%** | 9.24% | **0.44×** |
| Targeted (correct direction) | `memo_signed` | 6 | **53.11%** | 9.24% | **5.75×** |

`iforest`'s 0.44× on template isn't "mediocre" — it's **worse than blindly removing at random by more than half.** It ranks by "noise = outlier," and templated noise is exactly the batch of samples with the lowest loss and fastest convergence, so the ranking is systematically reversed: removal preferentially protects the noise and preferentially discards clean samples. This is the direction-reversal trap from [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.4 showing its real cost in an actual cleaning action.

`n_total = 14611`, `n_drop = 1461` (10% budget), `n_keep = 13150`, `seed = 42`; all three training sets use identical LoRA configuration, 5 epochs each.

**Five-way downstream benchmark comparison** (clean baseline / uncleaned / random removal / `iforest` targeted removal / `memo_signed` targeted removal):

![Template closed-loop cleaning: scorer direction determines whether cleaning helps at all](../../results/charts/cleaning_template.png)

| Benchmark | n | Clean | Uncleaned<br>(template) | Random | Targeted<br>`iforest` | Targeted<br>`memo_signed` | memo − uncleaned | iforest − uncleaned |
|---|---|---|---|---|---|---|---|---|
| mmlu | 14042 | 0.6332 | 0.6434 | 0.6349 | 0.6391 | 0.6374 | -0.0061 | -0.0043 |
| **gsm8k** | 1319 | **0.5497** | **0.4602** | 0.4526 | **0.4193** | **0.5231** | **+0.0629** | **-0.0409** |
| hellaswag | 10042 | 0.2770 | 0.2776 | 0.2773 | 0.2758 | 0.2765 | -0.0011 | -0.0018 |
| arc | 1172 | 0.8046 | 0.8157 | 0.8123 | 0.7961 | 0.8123 | -0.0034 | -0.0196 |
| bbh | 540 | 0.0926 | 0.0500 | 0.0611 | 0.0685 | 0.0685 | +0.0185 | +0.0185 |
| truthfulqa | 817 | 0.1787 | 0.1873 | 0.1775 | 0.1787 | 0.1860 | -0.0012 | -0.0086 |
| winogrande | 1267 | 0.5367 | 0.5478 | 0.5343 | 0.5328 | 0.5454 | -0.0024 | -0.0150 |
| **7-benchmark average** | — | **0.4389** | **0.4260** | 0.4214 | **0.4157** | **0.4356** | **+0.0096** | **-0.0102** |

**Core finding one: cleaning produces a genuinely positive downstream gain for the first time, under the correct scorer, and the gain is concentrated exactly where the real damage was.** After `memo_signed` cleaning, GSM8K rises from 0.4602 back to **0.5231**, recovering **70.3%** of the 8.95-point "uncleaned → clean" gap; the 7-benchmark average rises from 0.4260 to 0.4356, recovering **74.4%** of its gap. This lines up perfectly with the diagnosis in [06a-harm-ranking.md](06a-harm-ranking.md) Section 6a.1: template's real harm falls almost entirely on GSM8K, so the cleaning benefit shows up almost entirely there too, while the other 6 benchmarks stay within ±0.006 (BBH's +0.0185 shouldn't be read on its own, given n=540). This closes the question garbled's negative result left open: **closed-loop cleaning is genuinely effective; garbled showed no gain because it was the wrong type to test on, not because the method doesn't work.**

**Core finding two: a wrong scorer direction makes cleaning worse than not cleaning, and worse than random removal too.** `iforest`-cleaned drops the 7-benchmark average to 0.4157 and GSM8K to 0.4193 — 4.09 points below the uncleaned baseline (0.4602), and also below equal-sized random removal (0.4526). Read together, the three numbers form a strict monotonic chain: **correct direction (0.5231) > uncleaned (0.4602) > random removal (0.4526) > wrong direction (0.4193).** A wrongly-directed cleaning pass isn't "a discounted benefit" — it's **actively doing negative work**: it spends 10% of the budget preferentially removing clean samples, layering a second, biased data loss on top of the noise that was already there.

**Core finding three: removal precision does predict downstream gain this time, but only within a type.** Precision 53.11% → downstream +0.0096; precision 9.24% (random) → -0.0046; precision 4.04% → -0.0102 — a clean monotonic relationship. But this does not overturn Section 6c.5.1's garbled conclusion: garbled's equally-high precision (52.1%) produced no gain. **Precision only converts into gain given that "this noise type genuinely carries downstream harm."** Precision determines whether the noise can be found at all; the noise's own harm level determines whether finding it is worth anything. Both conditions must hold simultaneously — the complete conclusion these two experiments together provide.

**Methodological implication**: the key decision in label-free cleaning isn't "whether to clean" but "whether the scorer's direction is correct" — and that direction is exactly what cannot be read off unlabeled data (the entirety of Section 6c.4 is about this difficulty). On template, `memo_signed` vs. `iforest` is the difference between 0.5231 and 0.4193 — same data, same budget, same training configuration, the only variable being which scorer was chosen. This is the direct motivation behind Section 6c.4.5's decision to parallelize three legs into `pooled`: when the noise composition is unknown, it's better to accept a lower precision ceiling than to risk choosing the wrong direction outright.

#### 6c.5.3 The `pooled` Closed Loop on Mixed Noise: Best Ranking Quality, Negative Downstream Gain

Sections 6c.5.1 and 6c.5.2 both assumed the noise type was known. Production doesn't grant that assumption, so this section runs Section 6c.4.5's `pooled` scorer through the full closed loop on `mixed`, down to downstream evaluation: `n_total = 14819`, `n_drop = 1482` (10% budget), `n_keep = 13337`, `seed = 42`, targeted and equal-sized random arms each trained for 5 epochs.

Ranking quality is the best of the three scorers: **targeted removal precision 32.32%, random control 8.43%, lift 3.83×** (`iforest` alone: 26.79%; `memo_signed` alone: only 12.82%).

| Benchmark | n | Clean | Uncleaned<br>(mixed) | Random | Targeted<br>`pooled` | pooled − random |
|---|---|---|---|---|---|---|
| mmlu | 14042 | 0.6332 | 0.6321 | 0.6349 | 0.6303 | -0.0046 |
| gsm8k | 1319 | 0.5497 | 0.5345 | 0.5534 | 0.5155 | **-0.0379** |
| hellaswag | 10042 | 0.2770 | 0.2740 | 0.2710 | 0.2711 | +0.0001 |
| arc | 1172 | 0.8046 | 0.8046 | 0.8029 | 0.7927 | -0.0102 |
| bbh | 540 | 0.0926 | 0.0741 | 0.0944 | 0.0778 | -0.0166 |
| truthfulqa | 817 | 0.1787 | 0.1909 | 0.1775 | 0.1848 | +0.0073 |
| winogrande | 1267 | 0.5367 | 0.5627 | 0.5470 | 0.5517 | +0.0047 |
| **7-benchmark average** | — | **0.4389** | **0.4390** | **0.4402** | **0.4320** | **-0.0082** |

**This is a negative result, and its direction is the reverse of Section 6c.5.2's.** `pooled`'s 3.83× lift is the highest of any experiment in this report, yet downstream performance is 0.0082 below equal-sized random removal, and 0.0070 below not cleaning at all. GSM8K makes this especially visible: random removal (0.5534) is actually the highest of the four conditions, while `pooled` scores only 0.5155.

Three explanations, which are also the real value of this negative result:

**First, `mixed`'s uncleaned baseline is already no worse than the clean baseline** (0.4390 vs. 0.4389). Each of the 7 sub-types makes up about 1.4%, and the only one carrying genuine downstream harm is the roughly 206 template rows (1.4% of the total). Following garbled's lesson from Section 6c.5.1 — harmless noise means no cleaning gain — `mixed` falls squarely into that category overall: the 10% data loss buys back nothing.

**Second, the removal budget is captured by the most salient types, and those happen to be the harmless ones.** Converting `pooled`'s per-type P@10% (`results/dolly-ratio10/pooled_scorer_compare.csv`) into an estimate of where the 1,482 removal slots actually went:

| Type | Per-type P@10% | Approx. slots captured | Downstream harm |
|---|---|---|---|
| duplicate | 0.1367 | ~203 | Near zero |
| garbled | 0.1095 | ~162 | Near zero (measured directly in Section 6c.5.1) |
| unrelated | 0.0712 | ~106 | Not separately measured |
| truncation | 0.0279 | ~41 | Not separately measured |
| **template** | **0.0228** | **~34** | **The only confirmed-harmful type (GSM8K -8.95 points)** |
| keyword | 0.0235 | ~35 | Not separately measured |
| near_duplicate | 0.0191 | ~28 | Not separately measured |

Duplicate and garbled together take 365 slots — about a quarter of the entire budget — while both carry essentially zero downstream harm; the genuinely harmful template gets only about 34 slots, 2.3% of the budget. **High ranking quality prioritized the wrong thing** — precision measures "is what's removed noise," not "is what's removed *harmful* noise." Outlier degree and downstream harm are nearly uncorrelated in this data, and `pooled` ranks by the former.

**Third, this exposes a fundamental limitation of P@10% as a scorer-selection metric.** The lift ordering among the three scorers on `mixed` is `pooled` (3.83×) > `iforest` (3.18×) > `memo_signed` (1.52×), but the downstream-gain ordering has no relationship to it. Taken together, the three closed loops in this section establish the complete criterion: **a cleaning gain requires all three conditions at once — the noise genuinely carries downstream harm (6c.5.1), the scorer's direction is correct (6c.5.2), and the removal budget actually lands on the harmful portion of the noise (this section).** The first two were already known; the third is specific to mixed streams and only becomes visible once the experiment goes all the way to downstream evaluation.

This also means Section 6c.4.5's characterization of `pooled` as "trading precision for coverage" doesn't go far enough — on the mixed stream, the coverage it buys doesn't convert into benefit, because that coverage isn't weighted by harm. Harm-weighted (rather than anomaly-weighted) removal is the next step listed in Section 7.3.

---

### 6c.6 Wild-Noise Closed-Loop Cleaning: Negative Downstream Gain, for a Different Reason Than Injected Noise

Section 6c.5's three closed loops (garbled/template/mixed) are all built on **injected** noise — the noise label is written by code, the type is known, and "is it genuinely harmful" can be measured type by type. `oasst-wild` is this report's only closed loop on **naturally occurring** noise: the noise label comes from a human `quality` score (below 0.30 flagged as noisy, per `wild_data.py`). Using the `pooled` scorer (the 26-dimensional feature set recorded in `metadata.json`, max of `iforest`+`memo_signed`+`text_nn_sim`) to remove under a 10% budget from the full 68,762-row training set: targeted removal precision 26.56% (random control 9.63%, lift about 3.0× relative to the true noise rate of 8.83%), then targeted/random arms each trained for 5 epochs and run through all 7 downstream benchmarks.

| Benchmark | n | Uncleaned<br>(wild) | Targeted<br>`pooled` | Random | pooled − uncleaned | pooled − random |
|---|---|---|---|---|---|---|
| mmlu | 14042 | **0.6041** | 0.5754 | 0.6197 | **-0.0287** | **-0.0443** |
| gsm8k | 1319 | 0.4496 | **0.4920** | 0.4716 | **+0.0424** | +0.0204 |
| hellaswag | 10042 | 0.2713 | 0.2653 | 0.2685 | -0.0060 | -0.0032 |
| arc | 1172 | 0.7696 | 0.7577 | 0.7705 | -0.0119 | -0.0128 |
| bbh | 540 | 0.0574 | 0.0574 | 0.0889 | 0.0000 | -0.0315 |
| truthfulqa | 817 | 0.2203 | 0.2093 | 0.1799 | -0.0110 | +0.0294 |
| winogrande | 1267 | 0.5328 | 0.5178 | 0.5375 | -0.0150 | -0.0197 |
| **7-benchmark average** | — | **0.4150** | **0.4107** | **0.4195** | **-0.0043** | **-0.0088** |

**Core finding: this is the only one of the four closed loops where "random removal beats not cleaning, and targeted removal is worst of all."** Random removal (0.4195) is actually slightly higher than the uncleaned baseline (0.4150), while targeted removal (0.4107) is at the bottom — the opposite ordering from Section 6c.5.2's template chain of "correct direction > uncleaned > random > wrong direction"; here the ordering becomes **random > uncleaned > targeted**, with `pooled` failing to show the positive gain a "correctly-directed" scorer should produce, instead trailing the random baseline slightly.

**Breaking it down by benchmark shows this negative result comes almost entirely from MMLU:** targeted removal on MMLU (0.5754) is 2.87 points below uncleaned and 4.43 points below random removal — the largest and most asymmetric drop of the 7 benchmarks; on GSM8K, by contrast, targeted removal (0.4920) is actually the highest of the three, 4.24 points above uncleaned. Looking at GSM8K alone would suggest "cleaning works," but the 7-benchmark average is dominated by MMLU's drop — **per-benchmark effects are not consistent, and the average alone hides this internal split** (echoing [06a-harm-ranking.md](06a-harm-ranking.md) Section 6a.1's warning that the narrow 7-benchmark average range is easily skewed by a single benchmark's swing).

**This negative result has a different mechanism than Section 6c.5's and needs [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.10's length-confound finding to explain it, rather than the same "one of two preconditions missing" framework:** Section 6c.5's negative results (garbled's harmlessness, `mixed`'s budget misallocation) both occur in settings where "the detection direction itself was correct, the cleaning target simply carried no real harm." Here, the `pooled` scorer itself carries a known contamination — [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.10 measured that 44.5% of its "better than random" signal comes from a response-length confound rather than genuine quality judgment. In the wild-noise setting, the human `quality` score itself already correlates with short responses (correlation -0.20, [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.10), and `pooled` likely amplifies this correlation, systematically removing a batch of "short but reasonable" responses rather than genuinely low-quality ones — if this mis-removed batch happens to matter disproportionately for training on knowledge-dense multiple-choice tasks like MMLU, that would explain why the downstream loss concentrates on MMLU rather than spreading evenly across all 7. **This is a hypothesis; no dedicated ablation in this report verifies "how much of what was removed was mis-removed due to the length confound" — left for future work.**

**Methodological implication**: Section 6c.5 established that a cleaning gain needs "the noise genuinely carries harm" plus "the scorer's direction is correct," implicitly treating "direction correct" as a binary judgment (right/wrong). `oasst-wild` surfaces a new failure mode — the scorer can be "partially correct" (it does achieve a 3.0× targeted-removal lift on real wild noise, clearly better than random), but **the signal it captures is contaminated by a confound unrelated to true quality (length)**, so even though ranking quality looks usable, the removal action itself still yields a negative return. This suggests that "the scorer's direction is correct" should, in the wild-noise setting, be split into two independent checks: "the ranking direction is correct" and "the signal the ranking depends on carries no systematic confound." [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.10's length-stratification method provides a way to check the latter, but this report has not yet verified whether re-scoring and re-cleaning after removing the confound would turn the result positive — another direction listed in Section 7.3.

---

**Overall conclusion for Thread 3**: whether noisy samples can be separated label-free is, honestly, "it depends." When the type is known and calibrated, precision can be pushed quite high (garbled 52.1%, template 53.11%, both several times the random rate). When the type is unknown or mixed, parallelizing multiple scorers avoids the worst case of "completely undetectable," but the precision ceiling drops substantially (`pooled` on mixed: about 32%). More importantly, high detection precision does not equal a cleaning benefit — the benefit depends on whether the noise genuinely carries harm (6c.5.1), whether the scorer's direction is chosen correctly (6c.5.2), and whether the removal budget actually lands on the truly harmful noise (6c.5.3); the wild-noise setting adds a further requirement that the scoring signal itself carry no systematic confound (6c.6). This is the full, real-world unfolding of the structural difficulty Section 5.3 anticipated — "the direction is unknowable" — as it plays out in an actual cleaning action.
