# Noisy Samples in LLM Fine-tuning: Detection Methods and Impact Analysis

> Setup: Qwen2.5-3B-Instruct LoRA (r=32, 59.9M trainable params) fine-tuned on databricks-dolly-15k
> Coverage: **7 noise types** × 2-3 contamination ratios (10%/5%) · 5 epochs · per-sample gradient tracking (micro-batch=1 snapshot-diff, +5-8% overhead)
> Data: 3 experiment groups (ratio10/ratio05/extra10), 251,334 per-sample records, 59-dim feature space, 7 downstream benchmarks
> Key finding: most noise types are detectable, but cleaning gains are limited; detection difficulty is decoupled from harm; label-free detection follows a U-shaped spectrum

---

## 0. Key Findings

1. **Detectability tiers by type** (RF AUC, supervised): **garbled 0.996-0.999 / template 1.000** (easiest) → duplicate 0.967-0.988, unrelated 0.909-0.940, truncation 0.818 (moderate) → near_duplicate 0.687-0.733, **keyword 0.52-0.56 (hardest, near-random)**.
2. **Detection difficulty is decoupled from harm**: the easiest-to-detect garbled is nearly harmless (MMLU -0.001~+0.006); the hardest-to-detect keyword is also harmless at 10%; meanwhile **the most harmful type, template (GSM8K -0.125, relative -23%), is simultaneously the easiest to detect** — harm is not a function of how hard a noise type is to catch.
3. **Label-free detection follows a U-shaped spectrum**: generic outlier scorers (IsolationForest, etc.) only work on surface corruption (garbled 0.955); memorized, consistent-pattern noise (duplicate/template) converges to "over-typical" rather than "outlier", so generic outlier detectors fail on it (template 0.494, below random); switching to a **signed memorization rule** plus a scale-free token-concentration feature restores template to **0.9994**, near the supervised ceiling; semantic noise (unrelated/keyword/near_duplicate) stays ≤0.72 under both label-free paradigms — the real blind spot.
4. **Detectors transfer across ratios but not across noise types**: 10%↔5% bidirectional retention is 0.995-1.156 (contamination-rate drift is not a problem); but off-diagonal cross-type retention averages only 0.688 vs 1.000 on the diagonal, and a detector trained on duplicate is actually inverted on garbled (0.469) — **detectors must be trained and scored per noise type**.
5. **Training dynamics contain a direction-inversion trap**: duplicate/template are memorized quickly, so their loss is *lower* than clean (duplicate loss_mean AUC only 0.369, 0.631 after flipping; template loss_mean AUC 0.101, 0.899 after flipping) — "higher loss = noise" is not a safe assumption.
6. **Detection power decays over training**: for unrelated, RF AUC rises from 0.881 at epoch 0 to 0.916 over 5 full epochs, yet P@10% precision is best early; garbled reaches 98%+ of its full-trajectory AUC using only epoch 0-1 features (0.980 vs 0.987) — **cleaning should happen early in training**.
7. **AUC systematically overstates cleaning precision**: under a realistic cleaning operation (drop the top-10% scored samples), garbled goes from AUC 0.999 to P@10% 0.937 (6.3% false-positive drop rate), but unrelated drops from AUC 0.909 to P@10% 0.629 (37.1% false positives), and duplicate from 0.967 to 0.722 (27.8%) — a healthy-looking AUC does not guarantee an acceptable false-positive rate.
8. **Absolute impact is usually small**: at 10% contamination the six noise models' MMLU scores cluster tightly (range 0.011), and all fall below the un-fine-tuned base model — dolly SFT's own generalization cost outweighs differences between noise types. The one exception is template's systematic damage to GSM8K.

![Detection AUC by noise type](../results/charts/detection_auc_by_type.png)

---

## 1. Experimental Design

### 1.1 Research Questions

1. **Detection**: using only per-sample metrics trackable during training, can the 7 noise types be separated from clean samples? Which features work for which type? Do detectors transfer across ratios/types? How much detection power survives without labels?
2. **Impact**: how much do different noise types and ratios affect training dynamics and final downstream capability? Does detection difficulty track actual harm?

### 1.2 Unified Construction of Seven Noise Types

Built from databricks-dolly-15k (15,011 examples) across three experiment groups, same seed and sample ordering:

| Type | Construction | Mechanism | Ratio / Source |
|---|---|---|---|
| garbled | Unicode substitution/insertion/character swap, whitespace preserved | Surface corruption | ratio10 (10%), ratio05 (5%) |
| duplicate | Exact byte-for-byte duplicate rows | Memorizable noise (direction-inverted) | ratio10, ratio05 |
| unrelated | Response replaced with a fluent answer from a different category | Semantic mismatch | ratio10, ratio05 |
| keyword | Only numbers/years/proper nouns replaced, grammar/structure intact | Subtle tampering | ratio10, ratio05 |
| template | Response replaced with a fixed, wrong reply template | Consistent pattern (direction-inverted, most dangerous) | extra10 (10%) |
| truncation | Response truncated to its first half | Information loss | extra10 (10%) |
| near_duplicate | WordNet synonym-substitution paraphrase | Light duplication, evades surface-similarity detection | extra10 (10%) |
| mixed | Even split of the group's own types | — | all three groups |

- ratio10/ratio05 cover duplicate/garbled/unrelated/keyword at 10% and 5% in parallel, with identical sample-ID sets (only the noise subset differs) → cross-ratio per-sample comparable.
- extra10 covers template/truncation/near_duplicate at 10%.
- Each run trains on 14,611 samples (extra10 slightly different due to type composition), with 400 shared clean held-out samples (`heldout.jsonl`) used both for the reference-gradient direction and for held-out generalization tracking.

### 1.3 Training Configuration

| Setting | Value |
|---|---|
| Exact per-sample gradients | micro-batch 1 + grad accumulation 16, pre/post-backward snapshot diff, +5-8% overhead |
| Optimizer / precision | AdamW, lr 2e-4, cosine + 3% warmup, bf16 + flash-attention-2 |
| Sequence length | 1024 (truncated, assistant span preserved) |
| Schedule | 5 epochs, 4570-5025 steps/run, ~3.4-3.9h/run, single RTX 5090 |

### 1.4 Feature Space (59 dims)

- **Training-dynamics features** (13 dims, full-sample, snapshot-diff per epoch): mean/last/std/slope/curvature of loss, grad_norm, cos_sim_ref, plus converge_epoch;
- **Data features**: text_nn_sim (max TF-IDF similarity to the rest of the training set), n_tokens;
- **Diagnostic features** (1/8 subsample, 40 dims): token-level entropy, hard-token (max-gradient token) position statistics, top-20%/top-8%/top-32% loss concentration, IFD numerator/denominator;
- Full definitions in Appendix B.

---

## 2. Training Dynamics: How Noise Gets Learned

### 2.1 Loss Trajectory Tiers (ratio10, epoch-4 endpoint)

| Type | loss_mean (full run) | loss_last (endpoint) | converge_epoch | vs. clean |
|---|---|---|---|---|
| **template** | 0.098 | 0.025 | 0.02 | far below — instant memorization |
| duplicate | 0.595 | 0.234 | 0.36 | below — fast memorization |
| clean (none) | 0.894 | 0.495 | 0.62 | baseline |
| truncation | 1.039 | 0.387 | 0.84 | slightly above |
| keyword | 1.215 | 0.666 | 1.09 | above |
| near_duplicate | 1.272 | 0.766 | 1.40 | above |
| unrelated | 1.406 | 0.514 | 1.33 | above (endpoint close to clean) |
| garbled | 3.322 | 2.379 | 4.05 | far above — never fully learned |

The two direction-inverted types (template, duplicate) sit below clean for the entire run — they are over-memorized, not "hard to learn." **garbled** is the opposite extreme, remaining high-loss through epoch 4. The remaining types (keyword/unrelated/near_duplicate/truncation) sit in between, slightly above clean but not at either extreme.

### 2.2 Held-out Generalization

The held-out loss trajectory (`tb_heldout_loss.csv`) shows that only template noticeably raises held-out loss on the clean reference set, matching its large GSM8K drop (§5). The remaining noise types do not visibly separate from clean on this curve.

**Key insight**: memorizable noise (duplicate/template) is an *inverted* signal in loss space — the more dangerous pattern is learned faster and ends up at a lower loss. Any detector assuming "high loss = noise" will fail, or actively flip, on these two types.

---

## 3. Detection Methodology

Detection methods split into three paradigms by whether they use labels and whether they assume a feature's direction.

### 3.1 Supervised Detection (requires labeled samples of the target type)

Method: LR / RF classifier, 70/30 split. Best result per type (higher of the two models/tags):

| Type | RF AUC | P@10% | Top-3 features | Mechanism |
|---|---|---|---|---|
| **template** | **1.000** | 0.819 | hard_loss_mean, loss_mean, loss_curvature | consistent pattern, loss position highly stable |
| **garbled** | **0.999** | 0.937 | loss_ep0, loss_curvature, loss_ep1 | double-sided corruption, separable at epoch 0 |
| duplicate | 0.967-0.988 | 0.467-0.722 | text_nn_sim, cos_global_last, inverted loss features | relies on data-side similarity; training-side inverted |
| unrelated | 0.909-0.940 | 0.405-0.629 | loss_slope, loss_curvature, loss_ep0 | cross-epoch loss fluctuation |
| truncation | 0.818 | 0.340 | loss_std, loss_slope, mean_loss_std | length/position leakage |
| near_duplicate | 0.687-0.733 | 0.266 | max_token_loss, hard_loss_max, mean_loss | weak signal; text_nn_sim fails on it (0.49) |
| **keyword** | **0.52-0.56** | ~random | frac_hard_std, loss_ep0, loss_curvature | grammar/structure preserved, near-undetectable |

**Findings**:
1. **Detection difficulty is decoupled from harm** — among the two easiest types, one is nearly harmless (garbled), the other is the most dangerous (template); the hardest type (keyword) is also harmless at 10%.
2. **Each type depends on different features**: garbled relies on token-level entropy and early-epoch loss; duplicate relies almost entirely on the data-side `text_nn_sim` (0.939), with training-side features inverted; template relies on "hard-token position is constant" (hard_loss_mean); truncation relies on length-correlated loss fluctuation.
3. **AUC overstates usable precision** — see §3.5 (P@10% vs AUC).

### 3.2 Label-free Detection: Generic Outlier Models

Method: IsolationForest / Mahalanobis / two-sided z-score — no labels required, assuming noise is an "outlier."

| Type | Supervised RF | Best label-free (mostly IsolationForest) | P@10% (label-free) |
|---|---|---|---|
| garbled | 0.999 | **0.955** | 0.640 |
| unrelated | 0.909 | 0.722 | 0.243 |
| duplicate | 0.967 | 0.556-0.699 | 0.053-0.083 |
| keyword | 0.522 | 0.572 | 0.170 |
| **template** | 1.000 | **0.494** ← below random | ~0.06 |
| truncation | 0.818 | 0.582 | — |
| near_duplicate | 0.687 | 0.620 | — |

**Mechanism**: memorized noise (duplicate/template) is not an "outlier" but "over-typical" — its loss/entropy sits far below clean, at the center of the distribution, where a single-population outlier model finds nothing. garbled is the only type that stays strong label-free, because it genuinely is far from the distribution's center.

### 3.3 Signed Memorization Rules (partial fix)

Assumption: if we already know we're hunting for "over-memorized" samples, use a positive-direction rule instead — low loss + fast convergence + low gradient norm, with direction fixed by hypothesis rather than fit.

| Type | Generic outlier (iforest) | memo_signed | **memo + top20_conc** |
|---|---|---|---|
| **template** | 0.494 (below random) | 0.887 | **0.9994** |
| duplicate | 0.556-0.699 | 0.625 | 0.475-0.633 |
| garbled | 0.955 | **0.015** ← fully inverted | 0.110 |
| unrelated | 0.722 | 0.225 | 0.513 |
| keyword | 0.572 | 0.346 | 0.435 |
| truncation | 0.582 | 0.370 | 0.514 |
| near_duplicate | 0.620 | 0.361 | 0.481 |

**Key findings**:
- The sign itself encodes the prior of "which noise type are we hunting" — the same `memo_signed` rule scores 0.887 on template but only 0.015 on garbled (fully inverted), because garbled is exactly "low typicality," the opposite of what the rule assumes.
- Adding the scale-free `top20_share` (token-loss concentration) pushes template to **0.9994 / P@10% 0.836**, close to the 0.988-1.000 supervised ceiling, and it's the only unflipped template feature (direction AUC 0.9994 vs. 0.0032 flipped).
- Semantic noise (unrelated/keyword/near_duplicate) stays ≤0.72 under both label-free rules — the genuine detection blind spot.

**The label-free U-shaped spectrum**:

```
Detectable ◄──────────────────────────────────────────► Hard to detect
Consistent pattern      Surface corruption      Semantic / subtle noise
(template) 0.9994       (garbled) 0.955          ≤0.72 (both paradigms fail)
(signed+conc)           (iforest)
```

### 3.4 Detector Transferability

**Cross-ratio** (10%↔5%, `transfer_cross_ratio.csv`): bidirectional retention 0.995-1.156 — contamination-rate drift is not a problem; keyword even improves (0.995→1.156) from doubled training samples.

**Cross-type** (`transfer_cross_type.csv`, 4×4 matrix × 2 ratios): diagonal (same-type train/test) is always 1.000; off-diagonal average is only 0.688. Extreme cases:
- Detector trained on duplicate, tested on garbled: retention 0.493 (ratio10);
- Detector trained on garbled, tested on duplicate: retention 0.463 — **mutually inverted predictions**;
- keyword is the best transfer *source* (0.94-0.98 retention onto garbled/unrelated) but the worst transfer *target* (only 0.709-0.713 retention when the source detector is trained on duplicate).

**Conclusion**: detectors must be trained and scored per noise type, then combined by taking the union of flagged samples; a single binary classifier fails broadly under multi-type contamination.

### 3.5 Detection Power Decays Over Training

From `detector_epoch_budget.csv` (ratio10), training detectors on cumulative per-epoch features:

| Type | epoch 0-1 RF AUC | full 5-epoch RF AUC | % of full achieved at epoch 0-1 |
|---|---|---|---|
| garbled | 0.980 | 0.987 | 99.3% |
| unrelated | 0.881 | 0.916 | 96.2% |
| duplicate | 0.922 | 0.957 | 96.3% |
| keyword | 0.605 | 0.649 | 93.2% |
| mixed | 0.879 | 0.908 | 96.8% |

Epoch 0-1 features alone reach 93-99% of the full-trajectory detector's performance — **cleaning should happen early (epoch 0-1)**; the later cleaning is attempted, the higher the cost and the lower the benefit.

### 3.6 Mixed Contamination Does Not Dilute Single-Type Signal

`mixed_subtype_dilution.csv` shows each subtype's detectability inside a 4-way (ratio10/ratio05) or 7-way (extra10) mixed run is at least as high as in its own single-type run (e.g. duplicate: 0.998 inside mixed vs. 0.981 alone; keyword: 0.773 vs. 0.688) — real-world multi-type contamination does not weaken detectors; mixed's overall lower AUC is purely a label-aggregation artifact.

---

## 4. Token-Level Detection and Feature Increments

- **Token-level entropy / hard-token analysis**: garbled's hard_loss_mean AUC reaches 0.958, the second-strongest feature after loss_ep0; unrelated 0.778; keyword only 0.677 — all weaker than sample-level aggregate features. Token-level analysis mainly adds interpretability, not independent discriminative power.
- **template's fingerprint**: hard-token position statistics (`hard_pos_std_mean`/`n_hard`) reach AUC 0.886 (0.114 before flipping) — hard-token positions are highly constant, confirming the "consistent pattern" mechanism.
- **IFD (Instruction Following Difficulty) value is incremental, not independently discriminative**: the IFD ratio itself has a weak univariate AUC (0.55-0.80: garbled 0.800, template 0.761 inverted, duplicate 0.618 inverted, keyword 0.580, unrelated 0.553), all below same-type loss/entropy features; but adding the numerator/denominator `L(A|Q)`/`L(A)` as two separate features (instead of the ratio) to the 13-dim trajectory set lifts template to 0.9884 (+0.017, `detector_ablation.csv`), with smaller gains on truncation/near_duplicate — the ratio collapses two useful degrees of freedom into one, discarding information.
- **Diagnostic (40-dim) feature increment over trajectory (13-dim) features** (`detector_ablation.csv`, diag_subsample): template 0.967→0.999 (+0.032), truncation 0.776→0.858 (+0.082), near_duplicate 0.682→0.763 (+0.081) — diagnostic features add the most value for moderate-difficulty types.

---

## 5. Impact on Model Capability

### 5.1 Overall Evaluation (ratio10, 7-benchmark excerpt)

| Model | MMLU | GSM8K | ARC | vs. clean (MMLU) |
|---|---|---|---|---|
| base (not fine-tuned) | 0.6637 | 0.7460 | 0.8311 | +0.034 |
| clean | 0.6295 | 0.5413 | 0.7995 | baseline |
| garbled | 0.6354 | 0.5269 | 0.8080 | +0.006 |
| keyword | 0.6333 | 0.5231 | 0.7986 | +0.004 |
| mixed | 0.6315 | 0.5732 | 0.7952 | +0.002 |
| duplicate | 0.6309 | 0.5125 | 0.7918 | +0.001 |
| unrelated | 0.6241 | 0.4981 | 0.7901 | -0.005 |
| **template** (extra10) | 0.6314 | **0.4162** | 0.7901 | -0.002 (MMLU) / **-0.125 GSM8K (relative -23%)** |

**Key findings**:
1. **Absolute impact is usually small**: at ratio10 the six noise models' MMLU scores cluster tightly (range 0.011, 0.624-0.635), and all fall below the un-fine-tuned base (0.664) — dolly SFT's own cost to general capability dwarfs the differences between noise types.
2. **template is the sole exception**: GSM8K drops from clean's 0.541 to 0.416 (relative -23%), while MMLU/ARC are barely affected — a systematic error (fixed wrong template) is learned as a "shortcut," whereas random-style errors (garbled/duplicate/keyword) are mostly absorbed or averaged out by SFT.
3. **unrelated's harm is non-monotonic in ratio**: at ratio05 (5%) MMLU is 0.6106 vs. 0.6241 at ratio10 (10%) — 5% is *worse*, suggesting a small amount of semantic mismatch is enough to disrupt confidence calibration (see below).

### 5.2 Confidence Analysis (MMLU margin, correct-answer logprob minus best-wrong-answer logprob)

| Model | Mean margin on correctly-answered items |
|---|---|
| ratio10 clean | 4.770 |
| ratio10 unrelated (10%) | 5.940 |
| ratio05 unrelated (5%) | 2.855 |

5% unrelated noise sharply lowers confidence on correctly-answered items (4.77→2.86), while 10% actually raises it — suggesting the impact of low-ratio semantic-mismatch noise is not a simple dilution effect, and may relate to non-monotonic training dynamics; the exact mechanism is left to future work.

---

## 6. Methodological Discussion

### 6.1 The Gap Between Cleaning Precision and AUC

Under a realistic cleaning operation (drop the top-10% scored samples), precision from `detector_precision_at_k.csv`:

| Type | RF AUC | P@10% | False-positive rate (1-P@10%) | Random baseline |
|---|---|---|---|---|
| garbled | 0.999 | 0.937 | 6.3% | 0.100 |
| duplicate | 0.967 | 0.722 | 27.8% | 0.091 |
| unrelated | 0.909 | 0.629 | 37.1% | 0.100 |
| keyword | 0.522-0.562 | ~0.25-0.28 | ~72-75% | 0.100 |

**Conclusion**: AUC systematically overstates usable cleaning precision — only garbled reaches near-zero false positives in practice; other types with "healthy" AUC (0.9+) still discard a large fraction of clean samples when actually cleaning. Decisions should be based on precision at the intended budget, not AUC alone.

### 6.2 Signal Consistency Validation on Natural Data

Validated on real conversational data from lmsys-chat-1m (15,404 valid samples, post-hoc scored with the ratio10_clean model, `results/natural_validation.csv`), checking the internal direction between core signals:

| Signal pair | Spearman r | p |
|---|---|---|
| token_top20 vs loss_mu | **-0.839** | <1e-300 |
| loss_cv vs loss_mu | **-0.861** | <1e-300 |
| token_top20 vs loss_cv | **+0.982** | <1e-300 |

**Interpretation**: token concentration is strongly negatively correlated with mean loss — the better the model handles a sample, the more its loss concentrates in a few hard tokens; and concentration moves almost perfectly with the loss coefficient of variation (0.982), indicating both measure the same underlying quantity on natural data. This matches the mechanism given for `top20_share` in §3.3 as a "scale-free concentration feature": it captures the *shape* of the loss distribution rather than its absolute level, which is why it survives the absolute-loss collapse on memorized samples.

**Limitation**: this validates **internal consistency between signals** (natural data has no noise labels), so it cannot be used to infer detection AUC on real-world noise.

### 6.3 The Direction-Inversion Trap

| Type | Raw AUC (loss_mean) | AUC after flipping |
|---|---|---|
| duplicate | 0.369 | 0.631 |
| template | 0.101 | 0.899 |

Mechanism: memorizable noise converges to extremely low loss — "over-typical," not "hard to learn." **Any detector must empirically verify feature direction; "higher loss = noise" cannot be assumed** — this is also why the `memo_signed` rule in §3.3 must fix its direction per noise type rather than take an absolute value or a single global sign.

---

## 7. Conclusions

### 7.1 Key Findings

1. Sample-level noise detection is technically feasible for most types — 5 of 7 types reach supervised AUC ≥0.82 (garbled/template/duplicate/unrelated/truncation); keyword and near_duplicate remain clearly harder.
2. Detection's real value is in data governance (quality monitoring, auditing, anomaly discovery) rather than "cleaning makes the model meaningfully better" — for most types at 10% contamination, absolute downstream impact is under one percentage point, and imperfect cleaning precision incurs a real false-positive cost.
3. **The highest-priority governance target is consistent-pattern noise** (template/shortcut-style): it is both the easiest to detect (0.9994 achievable label-free) and the most catastrophic (GSM8K -23%) — the one type worth building a dedicated label-free monitoring rule for.
4. The methodological boundary is clear: supervised detection covers all 7 types but needs labels; label-free detection only covers the two extremes (surface corruption + consistent pattern) — the semantic-noise middle (unrelated/keyword/near_duplicate) has no label-free solution under the current feature set and still requires labeling or semi-supervised methods.

### 7.2 Limitations

1. Single model (Qwen2.5-3B) + single dataset (dolly-15k) + single LoRA configuration; cross-model/cross-dataset generalization is untested.
2. Noise is synthetically constructed; real-world noise patterns may be subtler and more mixed.
3. No direct "clean-then-retrain" gain comparison yet (a cleaning-gain experiment is running on GPU; results will be added in a future update).

### 7.3 Future Work

1. Cleaning-gain comparison: top-10%-by-score removal vs. random 10% removal, comparing downstream performance after retraining (in progress).
2. Natural-data signal validation: internal consistency already validated on lmsys-chat-1m (§6.2); what remains is detection-AUC validation on a **natural dataset with real noise labels**.
3. Cross-model / cross-dataset generalization validation.

---

## Appendix A: Reproduction

Training commands, data paths, and random seeds are documented in `README.md` and the header comments of `scripts/2_train/train.py`. Core scripts:
- `scripts/2_train/train.py` — per-sample gradient-tracked training
- `scripts/2_train/evaluate.py` — 7-benchmark evaluation
- `scripts/rebuild_analysis_inventory.py` — data inventory generator (`results/data_inventory.json`)
- `scripts/generate_report_tables.py` — source of this report's tables (`docs/report_tables.md`)

## Appendix B: Feature Definitions

Full mathematical definitions of the 59 features (loss/grad_norm/cos_sim_ref formulas, token-level entropy and hard-token definitions, IFD formula, etc.) are in the `src/` module docstrings and the per-sample feature computation in `scripts/2_train/train.py`.
