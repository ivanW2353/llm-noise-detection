# Cross-Experiment Synthesis: Noise-Detection Findings across Three Studies

> Combining: **qa-noise-experiment** (2026-07, SQuAD v1 extractive QA, 87K samples, 1.5B) ·
> **dynanoise** (2026-08, dolly-15k generative, 1.5B/3B, loss-dynamics signals) ·
> **llm-noise-detection** (this repo, dolly-15k, 3B, per-sample gradients + 19 features)

## 1. Noise-type mapping across experiments

| Noise family | qa-noise | dynanoise | this repo | detectability (this repo) | harm |
|---|---|---|---|---|---|
| surface corruption | random_word | A (BPE mojibake) | garbled | **0.9996 (easiest)** | **mildest** |
| consistent pattern / shortcut | fixed_wrong | E ("42") | template (added) | data-side / IFD | **catastrophic** |
| redundancy | — | C (redundant) | duplicate | 0.974 (data-side only) | overfitting damage |
| semantic mismatch | random_replacement | B (fluent wrong) | unrelated | 0.923 | moderate (generative) |
| subtle tampering | — | D (one fact changed) | keyword | **0.531 (infeasible)** | appears at higher ratios |
| information loss | — | — | truncation (added) | TBD | TBD |
| light paraphrase | — | — | near_duplicate (added) | TBD (TF-IDF expected to miss) | TBD |

## 2. Six convergent findings

1. **"Noise = high loss" is a wrong default and a recurring trap.** dynanoise:
   unlearnable noise has *lower* loss_cv than clean (0.013 vs 0.041); fixing the
   direction (`-loss_cv`) raised the hit rate from 3.8% to 86.5%. This repo:
   duplicate loss is *below* normal (AUC 0.37). Direction must be validated
   empirically; use bidirectional (z-score) joint schemes.

2. **Token-level signals are the most robust family across experiments.**
   dynanoise `token_loss_top20` is perfectly stable across 1.5B/3B
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
   (0.67) but IFD works (0.90, dynanoise Phase 5). Our experiment lacks this
   noise family — a known design gap (now added: `--with-shortcut`).

5. **Detectability and cleaning gains are decoupled — filtering has a
   ceiling.** dynanoise: precise filtering (99.8% noise hit) improved
   MT-Bench (+0.48) barely more than random 10% drop (+0.41) in three
   independent runs. This repo: 10% noise barely moves benchmarks. Detection
   is valuable for *data governance / quality monitoring*, not for
   "clean-the-data-and-get-a-better-model".

6. **Controlled-experiment signal directions hold on natural data.**
   dynanoise Phase 6 (lmsys-chat-1m, 50K): Spearman(token_top20, loss_mu) =
   −0.78, matching the controlled direction (AUROC 0.946).

## 3. Detection difficulty spectrum (merged)

```
detectable ◄───────────────────────────────────────────────► undetectable
consistent-pattern    surface corruption    semantic mismatch    subtle tampering
(duplicate/fixed_wrong)(garbled)            (unrelated)          (keyword)
data-side / IFD       training dynamics     partial              nearly impossible
(overfit / catastrophic)(mildest harm)      (moderate harm)      (harmless at low ratios)
```

**Key insight**: detectability does not monotonically correlate with harm —
the *value zone* is consistent-pattern noise (detectable AND catastrophic)
and high-ratio semantic noise (hard to detect AND starting to hurt).

## 4. Improvements borrowed into this repo (status)

| # | Improvement | status |
|---|---|---|
| 1 | `--with-extra`: template + truncation + near_duplicate (+ 7-way mixed) | implemented & CPU-tested |
| 2 | `token_loss_top20` concentration signal | planned (diagnostic already computes per-token CE) |
| 3 | `compute_ifd.py` (post-hoc IFD, detects shortcut) | written; run after GPU frees up |
| 4 | explicit bidirectional z-scores | AUC already direction-agnostic; docs updated |
| 5 | cleaning-gain control run | planned after ratio05 |
| 6 | `natural_signal_validation.py` (reproduces Phase 6 on lmsys-1m) | written; needs GPU |
| 7 | RHO-style reference comparison | optional (~3.5h holdout training) |

## 5. One-line summary

> Across three experiments: loss-dynamics signals are real but directions
> often invert; token-level features are the most stable; harm is
> task-dependent; consistent-pattern noise is the most dangerous target;
> cleaning gains have a ceiling — detection serves data governance, and the
> highest-value targets are consistent-pattern and high-ratio semantic noise,
> not the easiest-to-detect surface corruption.
