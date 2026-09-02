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

## 2. Six convergent findings

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

### 4.2 Add the `token_loss_top20` concentration signal — partially substituted (open)
- dynanoise's most model-stable signal; not implemented directly here
  (existing frac_hard / max_token_loss / entropy are complementary
  approximations).
- Approximate substitute already in place: `hard_loss_mean` (mean loss of the
  top-k hard tokens) works for garbled 0.859 / template 0.904.
- Still worth adding: compute the true top-20% share offline from the stored
  top-32 token detail (~zero cost, no retraining).

### 4.3 Add the IFD signal (Instruction-Following Difficulty) — ✅ implemented (`compute_ifd.py`), but **not yet in the feature set**
- Formula: $\text{IFD} = \dfrac{L(A \mid Q)}{L(A)}$ (one extra forward pass on
  the prompt-only input).
- Results confirm dynanoise: IFD is the single most discriminative metric
  across types (template 43× easier to follow, garbled 2.7× harder).
- **Open**: only extra10 has IFD data, and it is not part of `METRIC_ORDER`;
  should be run for ratio10/ratio05 and fed into the classifier.

### 4.4 Make bidirectional signals explicit — docs covered, thresholding still open
- RF already learns the "low loss" direction for duplicate/template
  implicitly; report §7.2 and §2.1 now record the inversion explicitly.
- Open: a **threshold-based** detector (as opposed to a classifier) needs both
  high- and low-direction z-score features per metric.

### 4.5 Cleaning-gain control run (test dynanoise's "ceiling") — **still not done, the biggest remaining gap**
- Two runs: remove the top-10% by detection score vs remove a random 10% →
  compare on the validation set.
- Expected: a tiny gain, reproducing dynanoise's random_drop ≈ precise filter.
- This repo has now closed the "detection rate" and "cleaning precision" links
  (§3.7), but "is the model actually better after cleaning?" can still only
  cite dynanoise.
- Cost: 2 runs + 2 evals (~10h).

### 4.6 Natural-data signal validation (reproduce Phase 6) — script written but **never run**
- `scripts/natural_signal_validation.py` is implemented; lmsys-chat-1m is
  cached locally.
- Compute token-level signals with the trained clean model, Spearman-correlate
  against loss_mu.
- Cost: ~1h GPU, purely confirmatory; gives external validity to the
  "deployable on real unlabeled data" claim.

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

> Two additions from 2026-09-02: **AUC systematically oversells cleaning
> usability** (unrelated AUC 0.931 → P@10% only 0.631), and **mixing multiple
> noise types does not dilute per-type signal** (each subtype's detectability
> inside the mixed run is ≥ its own single-type run) — the latter means a real
> multi-contaminant scenario should be scored per type and unioned, rather
> than trained as one unified binary classifier.
