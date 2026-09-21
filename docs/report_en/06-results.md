## 6. Experimental Process and Results: Index

This chapter used to be a single file organized in "chronological order of methodological development" (12 experiment groups, roughly 28 numbered subsections). To let the report answer the project's three cross-cutting research questions more directly, this chapter has been split into three new files, organized by **research thread**. This file keeps the original filename (so old links keep working) and now only carries the introduction, the shared-infrastructure notes, navigation, and the section cross-reference table — the body has moved entirely into the three files below.

**Split principle**: original section numbers (6.2, 6.6, 6.9, 6.11.5, 6.13.2, 6.16, etc.) are **unchanged** in the new files, only redistributed across them. Mechanistic findings (e.g. the direction-reversal trap, `text_nn_sim` dominance) go under Thread 2, which answers "what do the feature signatures look like"; the detector-design/deployment decisions those findings drive go under Thread 3, which answers "can it be separated." The closed-loop cleaning experiments (6.13/6.16) have their full body under Thread 3 (since their design purpose is to validate cleaning feasibility); Thread 1 only cites the "does this harm exist, how large" numbers from them, marked "see 06c Section X for detail." **No data conclusion (AUC, lift, percentage point) was altered during the move** — only narrative order and file ownership changed.

### The Three Threads

1. **[06a. Downstream harm ranking](06a-harm-ranking.md) (Thread 1)**: which noise type has the biggest impact on training outcomes? — Stepping back from "can it be detected" to ask first "is it worth detecting."
2. **[06b. Feature signatures of each noise type](06b-feature-signatures.md) (Thread 2)**: what metric signatures does each noise type have, and how do training dynamics / text features vary by type? — The largest part of this chapter, covering the full landscape of detection difficulty, cross-type/cross-ratio transfer, the direction-reversal trap, feature attribution, external baselines, and the length confound in wild noise.
3. **[06c. Label-free separability](06c-detectability.md) (Thread 3)**: given these feature signatures, can noisy samples be separated out label-free, how accurate is it, and does training after cleaning actually improve? — Moving from ranking quality (AUC/lift) to a real cleaning action (closed-loop retraining + downstream evaluation).

Three conventions run through all three files (detailed in Section 3.5), and recur repeatedly:

- **Supervised vs. label-free**: Sections 6.2 and 6.3 in 06b use a random forest trained on true labels, measuring a **signal ceiling**. Every other section's `iforest` / `memo_signed` / `pooled` never uses labels at all, measuring what **production can actually reach**. The gap between the two for the same noise type can be enormous (template: 0.999 vs. 0.522) — that gap is itself the subject of 06b Section 6.6.
- **Full coverage vs. diagnostic subsample**: most AUCs are computed on the ~900-1200 row diagnostic subsample (37 features), while closed-loop cleaning runs on the full 14,611+ rows (19 features). The former is optimistic, especially for near_duplicate.
- **Ranking quality vs. downstream benefit**: most sections in 06b/06c measure "can this rank noise to the top"; only 06c Sections 6.13/6.16 actually remove samples and retrain. The two can decouple — this is the central finding of 06c's final sections.

### 6.1 Experimental Setup and Output Files (shared infrastructure across all three threads)

Chapter 4 already covers the pipeline and commands in full; this section only collects the three things worth having at hand while reading results.

**Data scale.** Each dataset has 14,611 training rows (`duplicate` has 16,072 due to appended copies), plus a 400-row held-out set (200 for the reference gradient, 200 for held-out loss monitoring, shared across all 9 datasets). Noise ratios `dolly-ratio10` = 10% and `dolly-ratio5` = 5%, both fully run across all 9 datasets.

**Which data source feeds which section.**

| Output file | Produced by | Used by (new file location) |
|---|---|---|
| `results/{tag}/per_sample_metrics.csv` | `analyze --kind features` | Input to every later analysis |
| `results/{tag}/cross_type.csv` | `analyze --kind cross_type` | 06b 6.2 (diagonal), 6.3 (off-diagonal) |
| `results/{tag}/unsupervised.csv` | `analyze --kind unsupervised` | 06c 6.5, 06b 6.6 |
| `results/{tag}/precision_lift.csv` | `analyze --kind precision_lift` | 06c 6.5 |
| `results/{tag}/memorization.csv` | `analyze --kind memorization` | 06b 6.6 |
| `results/{tag}/early_*.csv` | `analyze --kind early_*` | 06b 6.7 |
| `results/{tag}/feature_attribution.csv` | `analyze --kind feature_attribution` | 06b 6.8 |
| `results/eval/eval_{tag}_{dataset}.json` | `evaluate` | 06a 6.9, 06c 6.13 |
| `results/{tag}/feature_ablation.csv` | `analyze.py::feature_group_ablation()` | 06c 6.11.2-6.11.3 |
| `results/{tag}/feature_correlation.csv` | `analyze.py::feature_correlation()` | 06b 6.11.1 |
| `results/{tag}/single_feature_ablation.csv` | `analyze.py::single_feature_ablation()` | 06b 6.11.4 |
| `results/{tag}/minimal_feature_set.csv` | `analyze.py::minimal_feature_set()` | 06c 6.11.5 |
| `results/{tag}/transfer_to_mixed.csv` | `analyze.py::transfer_to_mixed()` | 06c 6.12 |
| `results/{tag}/pooled_scorer_compare.csv` | `analyze.py::pooled_scorer_compare()` | 06c 6.12.5 |
| `datasets/{tag}/cleaning_loop/{name}/metadata.json` | `clean` | 06a/06c 6.13 |
| `results/{tag}/external_baselines.csv` | `baselines.py` (not tracked in Git) | 06b 6.14 |
| `results/oasst-wild/length_confound.csv` | `analyze.py::length_confound()` | 06b 6.15 |

**Downstream evaluation conventions.** 7 benchmarks: MMLU (n=14042), GSM8K (1319), HellaSwag (10042), ARC (1172), BBH (540), TruthfulQA (817), WinoGrande (1267). The "7-benchmark average" reported throughout is an **unweighted arithmetic mean** of these 7 accuracies — so BBH (n=540) and MMLU (n=14042) carry equal weight, and any single-benchmark swing is amplified. 06a Section 6.9 and 06c Section 6.13 therefore always report the per-benchmark table alongside the average, not the average alone.

Except for 06c Sections 6.13/6.16, every analysis in all three files recomputes from an existing trajectory without retraining; Sections 6.13/6.16 are the only experiments in the whole report that actually modify the training data and retrain.

### Navigation Table

| File | Thread | Core question | Main sections |
|---|---|---|---|
| [06a-harm-ranking.md](06a-harm-ranking.md) | Thread 1: harm ranking | Which noise type hurts training the most | 6.9 (full), 6.13.1-6.13.3/6.16 (summary + reference) |
| [06b-feature-signatures.md](06b-feature-signatures.md) | Thread 2: feature signatures | What do each type's training dynamics / text features look like | 6.2, 6.2.1, 6.3, 6.4, 6.6, 6.7, 6.8, 6.11.1, 6.11.4, 6.14, 6.15 |
| [06c-detectability.md](06c-detectability.md) | Thread 3: detectability | Can it be separated label-free, how accurate, does cleaning actually pay off | 6.5, 6.10, 6.11.2-6.11.3, 6.11.5-6.11.6, 6.12.1-6.12.5, 6.13.1-6.13.3 (full), 6.16 (full) |

### Section Cross-Reference Table

| Original # | Title | Now in |
|---|---|---|
| 6.1 | Setup and output files | This file (above) |
| 6.2 / 6.2.1 | Landscape of detection difficulty / in-domain AUC method | [06b](06b-feature-signatures.md) |
| 6.3 | Cross-type transfer | [06b](06b-feature-signatures.md) |
| 6.4 | Cross-ratio transfer | [06b](06b-feature-signatures.md) |
| 6.5 | Cleaning-precision lift | [06c](06c-detectability.md) |
| 6.6 | The direction-reversal trap | [06b](06b-feature-signatures.md) |
| 6.7 | Early detection | [06b](06b-feature-signatures.md) |
| 6.8 | Feature attribution | [06b](06b-feature-signatures.md) |
| 6.9 | Real downstream impact | [06a](06a-harm-ranking.md) |
| 6.10 | Best method per noise type, summary | [06c](06c-detectability.md) |
| 6.11.1 | Feature correlation / redundancy | [06b](06b-feature-signatures.md) |
| 6.11.2-6.11.3 | Feature-group ablation design and results | [06c](06c-detectability.md) |
| 6.11.4 | Single-feature leave-one-out ablation | [06b](06b-feature-signatures.md) |
| 6.11.5 | Minimal feature set | [06c](06c-detectability.md) |
| 6.11.6 | Deployment recommendation | [06c](06c-detectability.md) |
| 6.12.1-6.12.5 | Boundaries under an unknown noise type (incl. the `pooled` scorer) | [06c](06c-detectability.md) |
| 6.13.1 | garbled closed-loop cleaning | Full text in [06c](06c-detectability.md); harm-angle summary in [06a](06a-harm-ranking.md) |
| 6.13.2 | template closed-loop cleaning (direction comparison) | Full text in [06c](06c-detectability.md); harm-angle summary in [06a](06a-harm-ranking.md) |
| 6.13.3 | `pooled` closed loop on mixed | Full text in [06c](06c-detectability.md); harm-angle summary in [06a](06a-harm-ranking.md) |
| 6.14 | External baselines (EL2N/GraNd/TracIn) | [06b](06b-feature-signatures.md) |
| 6.15 | Length confound in wild noise | [06b](06b-feature-signatures.md) |
| 6.16 | Wild-noise closed-loop cleaning | Full text in [06c](06c-detectability.md); harm-angle summary in [06a](06a-harm-ranking.md) |
