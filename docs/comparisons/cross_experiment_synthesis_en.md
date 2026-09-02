# Cross-Experiment Synthesis: Noise-Detection Findings across Three Studies

> Combining: **qa-noise-experiment** (2026-07, SQuAD v1 extractive QA, 87K samples, 1.5B) ·
> **dynanoise** (2026-08, dolly-15k generative, 1.5B/3B, loss-dynamics signals) ·
> **llm-noise-detection** (this repo, dolly-15k, 3B, per-sample gradients + 40 features)

## 1. Noise-type mapping across experiments

| Noise family | qa-noise | dynanoise | this repo | detectability (this repo) | cleaning P@10% | harm |
|---|---|---|---|---|---|---|
| surface corruption | random_word | A (BPE mojibake) | garbled | **0.996** | **0.937** | **mildest** |
| consistent pattern / shortcut | fixed_wrong | E ("42") | template | **0.9995 (easiest)**, fully inverted | 0.819 | **catastrophic** (GSM8K −23%) |
| redundancy | — | C (redundant) | duplicate | 0.982 (data-side only) | 0.721 | overfitting damage |
| semantic mismatch | random_replacement | B (fluent wrong) | unrelated | 0.931 | 0.631 | moderate (generative) |
| information loss | — | — | truncation | 0.888 | 0.340 | mild |
| light paraphrase | — | — | near_duplicate | **0.733**, text_nn_sim fails (0.492) | 0.266 | no measurable harm |
| subtle tampering | — | D (one fact changed) | keyword | **0.70-0.73 (hardest)** | **0.281** | appears at higher ratios |

> Detectability is the multivariate AUC (better of LR/RF, 5-fold CV); P@10% is the
> precision when the top-scoring 10% is dropped (random baseline 0.10).
> The "TBD" for truncation / near_duplicate was resolved on 2026-09-02:
> **the predicted TF-IDF miss on near_duplicate is confirmed** (0.492), so semantic
> embeddings are needed; and keyword's "infeasible 0.531" is corrected to "hardest,
> 0.70-0.73" (the old value was deflated by two artifacts — the diagnostic subsample
> and a single train/test split; see the main report §3.7).

## 2. Eight convergent findings

1. **"Noise = high loss" is a wrong default and a recurring trap.** dynanoise:
   unlearnable noise has *lower* loss_cv than clean (0.013 vs 0.041); fixing the
   direction (`-loss_cv`) raised the hit rate from 3.8% to 86.5%. This repo:
   duplicate loss is *below* normal (AUC 0.37). And template (measured
   2026-09-02) is the most extreme case in all three experiments: **every
   loss/entropy feature inverts** — loss_mean AUC **0.101**, hard_loss_mean
   0.096, loss_curvature 0.102, entropy 0.139 (all ≈0.90 once flipped). A
   fixed response template is memorized outright, so its loss sits far *below*
   normal samples. The pattern: **the more learnable the noise, the more
   extreme the inversion**. Direction must be validated empirically; use
   bidirectional (z-score) joint schemes.

2. **Token-level signals are the most robust family across experiments.**
   dynanoise $\text{token\\_loss\\_top20}$ is perfectly stable across 1.5B/3B
   (AUROC 0.947 ± 0.001). This repo: `entropy` / `frac_hard` / `user_loss`
   (0.95~0.98) drive garbled detection; per-token gradient attribution
   (0.77) confirms token-level information is usable. Sample-level
   aggregation across tokens/epochs is the reliable scale.

3. **Noise harm is determined by the task type** — the strongest
   cross-experiment contrast:
   - Extractive QA (SQuAD): 50% swapped answers → EM −0.6% (context contains
     the answer, the model falls back to it)
   - Generative (dolly): 10% mismatched responses → GSM8K −0.043 (response IS
     the knowledge)
   - Shortcut noise (fixed_wrong): 50% → EM −41.8% (linear, R²≈0.99)
   Task type must be part of any noise-impact analysis.

4. **Consistent-pattern (shortcut) noise is the most dangerous and the
   highest-value detection target.** It is learnable, so loss_cv is weak
   (0.67) but IFD works (0.90, dynanoise Phase 5). This repo's original 4-type
   design lacked this family; it is now filled by `template` (extra10):
   detection RF **0.9995** — the *easiest* type — with the largest harm
   (GSM8K −23%). The "high harm + detectable" quadrant is no longer empty.

5. **Detectability and cleaning gains are decoupled — filtering has a
   ceiling.** dynanoise: precise filtering (99.8% noise hit) improved
   MT-Bench (+0.48) barely more than random 10% drop (+0.41) in three
   independent runs. This repo: 10% noise barely moves benchmarks (MMLU
   spread 0.011 across 6 models, all below base). This repo also identifies a
   **second source of the ceiling — insufficient precision**: at a 10%
   cleaning budget, unrelated's precision is only 0.631 and keyword's 0.281,
   so even a healthy-looking AUC (0.93) means 37% of what you drop is clean
   data. Only garbled reaches zero collateral damage (P@5% = 1.000). Detection
   is valuable for *data governance / quality monitoring*, not for
   "clean-the-data-and-get-a-better-model"; if you do want cleaning gains,
   target the high-harm families (shortcut / high-ratio label noise).

6. **Controlled-experiment signal directions hold on natural data.**
   dynanoise Phase 6 (lmsys-chat-1m, 50K): Spearman($\text{token\\_top20}$, $\text{loss\\_mu}$) = $−0.78$, matching the controlled direction (AUROC 0.946).

7. **But "the signal transfers to unlabeled data" is not "you can detect
   without labels"** (added 2026-09-02). Every AUC the three experiments have
   reported comes from a **supervised** detector. This repo measured four
   label-free scorers (IsolationForest / bidirectional z-score / Mahalanobis),
   which bounds the scope of finding 6:

   | noise type | supervised | best label-free | label-free P@10% |
   |---|---|---|---|
   | garbled (surface) | 0.996 | **0.955** (iforest) | 0.640 |
   | unrelated (semantic) | 0.933 | 0.722 (iforest) | 0.243 |
   | duplicate (memorizable) | 0.982 | 0.699 (iforest) | 0.083 |
   | **template (consistent pattern)** | **0.988** | 0.633 (outlier) → **0.9994 (signed + concentration)** | **0.059 → 0.836** |

   Under **generic outlier** detection only surface corruption survives — and the
   consistent-pattern family that all three experiments call the highest-value
   target (template / fixed_wrong / shortcut) is precisely the **least
   detectable** one. This is mechanically necessary and is the other face of
   finding 1: noise that gets memorized inverts on the loss side, which means it
   is not an *outlier* but **over-typical** (lower loss/entropy, more
   concentrated loss), sitting at the center of the distribution — no
   single-population outlier model can find it.

   **But that same diagnosis is the fix, and it has now been verified**: ask "is
   this sample *too easy*?" instead of "is it an outlier?". A **signed**
   memorization rule (low loss + fast convergence + low gradient, direction fixed
   a priori by hypothesis rather than fitted) takes template from 0.633 back to
   **0.887**, and adding the scale-free top-20% concentration feature reaches
   **0.9994 (P@10% 0.836)** — close to the 0.988 supervised ceiling. So the
   **label-free spectrum is a U shape**: both ends are detectable
   (consistent-pattern via hyper-typicality, surface corruption via outlierness),
   while the semantic middle (unrelated / keyword / near_duplicate) stays ≤0.77
   under either rule. *That* is the remaining real gap.

   The price: you must **know which end you are hunting** to pick the sign — or, if
   you expect several families at once, spend the budget two-tailed, which beats
   the one-sided rule on mixed runs (0.293 vs 0.083) precisely because mixed
   contamination populates both tails (§4.4). That is much weaker than "labeled
   seeds for every type" (one prior hypothesis vs a batch of annotations), but it
   is not zero knowledge. Note also that dynanoise Phase 6 proved the
   *correlational direction* of a signal holds on natural data, not that noise
   separates without labels; the two are routinely conflated.

   **Conclusion**: in label-free settings, drop generic outlier detection in
   favour of signed rules — hunt memorized noise with a **positively-signed**
   joint rule (low loss + low entropy + high concentration) and surface
   corruption with an outlier model. To remove the "know which end" precondition,
   calibrate each feature's sign against a clean seed set (semi-supervised, not
   yet tested).

8. **Detector transfer: changing the ratio is free, changing the type breaks
   it** (added 2026-09-02). Cross-ratio 10%↔5% transfer retains 0.995–1.156
   of the within-run AUC in both directions across 5 datasets — prevalence
   drift is not a problem and a detector can be deployed across ratios
   directly (keyword actually *gains* 0.11 because the training noise set
   doubles). Cross-type is the opposite: off-diagonal mean 0.688 vs diagonal
   0.899; duplicate↔garbled are **mutually inverse predictors** (0.46/0.48)
   and template is the worst transfer target (0.17–0.44). Each noise family
   needs its own labeled seed. This is dynanoise's "different noise types need
   different signals" (loss_cv fails on shortcut where IFD works) showing up
   one level higher, at the detector.

## 3. Detection difficulty spectrum (merged)

```
detectable ◄──────────────────────────────────────────────────────────► hard to detect
consistent-pattern      surface corruption  semantic mismatch  light paraphrase  subtle tampering
(template/fixed_wrong)  (garbled)           (unrelated)        (near_duplicate)  (keyword)
AUC 0.9995  P@10% 0.82  0.996  0.937        0.931  0.631      0.733  0.266      0.70-0.73  0.281
data-side / IFD /       training dynamics   partially          data-side feature weak signal
inverted loss                               detectable         fails (TF-IDF)
(catastrophic, -23%)    (mildest harm)      (moderate harm)    (no measurable harm) (harmless at low ratios)
```

**Key insight**: detectability does not monotonically correlate with harm —
of the two easiest types one is the most dangerous (template) and one the most
harmless (garbled), and the hardest (keyword) is harmless at low ratios. The
real *value zone* is consistent-pattern noise (detectable AND catastrophic)
and high-ratio semantic noise (hard to detect AND starting to hurt).

> Annotating with precision@10% rather than AUC alone is necessary:
> unrelated's 0.931 looks close to garbled's 0.996, but the actual cleaning
> precision is 0.631 vs 0.937 — AUC systematically oversells usability
> (this repo, measured 2026-09-02).

## 4. Improvements borrowed into this repo (ranked by value)

### 4.1 Add consistent-pattern noise `template` (was "shortcut", highest value) — ✅ done (extra10)
- Construction: all noise samples get one fixed wrong response template.
- **Exceeded expectations**: detection RF **0.9995** (the easiest type), largest
  harm (GSM8K −23%) — the "high harm + detectable" quadrant is filled.
- **Side finding**: its loss/entropy features are **fully direction-inverted**
  (loss_mean AUC 0.101), the most extreme inversion across all three
  experiments (§2.1); `hard_pos_jaccard` 0.808 (hard-token positions are
  constant) is a fingerprint unique to this type.

### 4.2 Add the `token_loss_top20` concentration signal — ✅ done (2026-09-02, `analyze_token_concentration.py`)
- Computed offline as the true top-20% loss share from the stored top-32 token
  detail (zero GPU cost, no retraining).
- **Result: it is template's strongest single feature (0.9994** vs the previous
  best, hard_loss_mean at 0.904**) and the only template feature that does not
  invert** — a memorized template concentrates what loss remains onto a few
  divergence positions, and **a concentration ratio is scale-free**, so it
  survives the collapse of every absolute loss level.
- **But it is not a general signal**: on garbled / keyword / near_duplicate it
  still loses to the existing entropy / frac_hard / max_token_loss. dynanoise's
  "most model-stable signal" **reproduces here but is type-specific** — it is
  essentially a consistent-pattern detector.
- Coverage measured honestly: an exact top-20% needs ≤32 stored tokens, which
  holds for 86–89% of samples (median length 41–50 tokens); `top20_share_ok`
  (NaN when truncated) agrees with `top20_share` to 4 decimals on template, so
  truncation is not the source of the result.

### 4.3 Add the IFD signal (Instruction-Following Difficulty) — ✅ done and evaluated (all three tags)
- Formula: $\text{IFD} = \dfrac{L(A \mid Q)}{L(A)}$ (one extra forward pass on
  the prompt-only input).
- **Correction to the dynanoise reading**: IFD *means* differ enormously across
  types (template 0.005 vs garbled 0.534, >100×), but the **univariate AUC is
  only 0.55–0.80**, below the loss/entropy features of the same type — so "IFD
  is the single most discriminative metric across types" does not hold once
  measured (a mean gap ≠ separability).
- **The real value is incremental**: the ratio itself adds almost nothing
  (+0.000~+0.012), but adding numerator and denominator `L(A|Q)` / `L(A)` as
  **two independent features** on top of the 13-dim trajectory set gives
  template +0.028 (0.967→0.995), unrelated +0.024, 7-way mixed +0.022 — the
  ratio compresses two useful degrees of freedom into one.
- **Recommendation**: keep as an optional feature column for
  template/unrelated, not in the default `METRIC_ORDER` (the gain is
  concentrated in a few types, the cost is global).

### 4.4 Make bidirectional signals explicit — ✅ tested, **the answer is no**
- RF already learns the "low loss" direction for duplicate/template
  implicitly; report §7.2 and §2.1 record the inversion explicitly.
- The thresholding variant has now been measured (§3.9): the bidirectional |z|
  scorer (`zscore_max`) is the **worst** of the four label-free scorers
  (template 0.418, duplicate 0.598) — taking absolute values does buy
  direction-independence, but at the cost of **discarding the direction
  information that carried the signal**.
- The two-tailed *budget* variant (5% from each end) splits by scenario: it fails
  for a **single** noise family (template 0.092, below the 0.10 random baseline —
  one tail is pure clean data, so half the budget is wasted) but **wins under
  multi-family contamination** (mixed 0.293 vs the one-sided 0.083, 2.4× random),
  because a mixed run has noise at both tails.
- **The revised recommendation is now verified**: a threshold detector must use
  **signed** directions, calibrated per noise type. Measured (this repo, §3.13):
  a signed memorization rule takes template from 0.633 → **0.887**, and
  **0.9994 (P@10% 0.836)** with the scale-free concentration — while the *same*
  sign vector scores **0.015 (fully inverted)** on garbled. The sign is exactly
  where the prior knowledge of "which noise family am I hunting" lives. Practical
  recipe: **signed one-sided when hunting one known family, two-tailed when you
  expect several**.

### 4.5 Cleaning-gain control run (test dynanoise's "ceiling") — **still not done, the biggest remaining gap**
- Two runs: remove the top-10% by detection score vs remove a random 10% →
  compare on the validation set.
- Expected: a tiny gain, reproducing dynanoise's random_drop ≈ precise filter.
- This repo has now closed the "detection rate" and "cleaning precision" links
  (§3.7), but "is the model actually better after cleaning?" can still only
  cite dynanoise.
- Cost: 2 runs + 2 evals (~10h).

### 4.6 Natural-data signal validation (reproduce Phase 6) — ✅ done (2026-09-02 evening)
- **dynanoise Phase 6**: lmsys-chat-1m (50K), Spearman(token_loss_top20, loss_mu) = **−0.78**
- **This project §3.14**: lmsys-chat-1m (n=15,404), ratio10 clean model, Spearman = **−0.839** (stronger), p < 1e-300
- **Two independent experiments** on different models/data both observe: token-level concentration negatively correlates with sample difficulty
- **But this only proves "signal correlation direction transfers," not "can detect noise without labels"** — both dynanoise Phase 6 and our §3.14 are **pure correlation measurements**; natural data has no ground truth and no AUC can be computed; §3.9/§3.13's label-free detection is a separate question, with a U-shaped spectrum (both ends detectable, middle not)
- Side finding: token_top20 and loss_cv are **nearly perfectly colinear** (ρ=+0.982) — they measure the same thing; keep only one in a feature set

### 4.7 RHO-style reference comparison
- dynanoise uses a holdout model as the RHO baseline; this repo uses
  cos_sim_ref (LESS-style) as the reference direction.
- Could add an RHO-style column in the analysis (needs 1 holdout model,
  ~3.5h) as a control baseline.

## 5. One-line summary

> Across three experiments: **loss-dynamics signals are real but their
> directions often invert (the more learnable, the more extreme the
> inversion); token-level features are the most stable; harm is
> task-dependent; consistent-pattern noise is the most dangerous target;
> cleaning gains have a ceiling (both because low-ratio noise does little
> harm and because cleaning precision is insufficient)** — detection's value
> lies in data governance and quality monitoring, and the highest-value
> targets are consistent-pattern and high-ratio semantic noise, not the
> easiest-to-detect surface corruption.

> Three additions from 2026-09-02: **AUC systematically oversells cleaning
> usability** (unrelated AUC 0.931 → P@10% only 0.631); **mixing multiple noise
> types does not dilute per-type signal** (each subtype's detectability inside
> the mixed run is ≥ its own single-type run), which means a real
> multi-contaminant scenario should be scored per type and unioned, rather than
> trained as one unified binary classifier; and **label-free detection is
> U-shaped, not monotone** — generic outlier models catch only surface
> corruption, but a signed hyper-typicality rule also catches consistent-pattern
> noise (template 0.9994 / P@10% 0.836), leaving the **semantic middle** as the
> genuine remaining gap.
