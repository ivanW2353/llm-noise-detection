## 6. Experimental Process and Results: Index

This chapter is organized into three files, one per cross-cutting research thread; this file carries the introduction, the shared-infrastructure notes, and navigation, with the body in the three files below.

Mechanistic findings (e.g. the direction-reversal trap, `text_nn_sim` dominance) fall under Thread 2, which answers "what do the feature signatures look like"; the detector-design/deployment decisions those findings drive fall under Thread 3, which answers "can it be separated." The closed-loop cleaning experiments (06c Sections 6c.5/6c.6) have their full body under Thread 3 (their design purpose is to validate cleaning feasibility); Thread 1 only cites the "does this harm exist, how large" numbers from them.

### The Three Threads

1. **[06a. Downstream harm ranking](06a-harm-ranking.md) (Thread 1)**: which noise type has the biggest impact on training outcomes? — Stepping back from "can it be detected" to ask first "is it worth detecting."
2. **[06b. Feature signatures of each noise type](06b-feature-signatures.md) (Thread 2)**: what metric signatures does each noise type have, and how do training dynamics / text features vary by type? — The largest part of this chapter, covering the full landscape of detection difficulty, cross-type/cross-ratio transfer, the direction-reversal trap, feature attribution, external baselines, and the length confound in wild noise.
3. **[06c. Label-free separability](06c-detectability.md) (Thread 3)**: given these feature signatures, can noisy samples be separated out label-free, how accurate is it, and does training after cleaning actually improve? — Moving from ranking quality (AUC/lift) to a real cleaning action (closed-loop retraining + downstream evaluation).

Three conventions run through all three files (detailed in Section 3.5), and recur repeatedly:

- **Supervised vs. label-free**: Sections 6b.1 and 6b.2 in 06b use a random forest trained on true labels, measuring a **signal ceiling**. Every other section's `iforest` / `memo_signed` / `pooled` never uses labels at all, measuring what **production can actually reach**. The gap between the two for the same noise type can be enormous (template: 0.999 vs. 0.522) — that gap is itself the subject of 06b Section 6b.4.
- **Full coverage vs. diagnostic subsample**: most AUCs are computed on the ~900-1200 row diagnostic subsample (37 features), while closed-loop cleaning runs on the full 14,611+ rows (19 features). The former is optimistic, especially for near_duplicate.
- **Ranking quality vs. downstream benefit**: most sections in 06b/06c measure "can this rank noise to the top"; only 06c Sections 6c.5/6c.6 actually remove samples and retrain. The two can decouple — this is the central finding of 06c's final sections.

### 6.1 Experimental Setup and Output Files (shared infrastructure across all three threads)

Chapter 4 already covers the pipeline and commands in full; this section only collects the three things worth having at hand while reading results.

**Data scale.** Each dataset has 14,611 training rows (`duplicate` has 16,072 due to appended copies), plus a 400-row held-out set (200 for the reference gradient, 200 for held-out loss monitoring, shared across all 9 datasets). Noise ratios `dolly-ratio10` = 10% and `dolly-ratio5` = 5%, both fully run across all 9 datasets.

**Which data source feeds which section.**

| Output file | Produced by | Used by |
|---|---|---|
| `results/{tag}/per_sample_metrics.csv` | `analyze --kind features` | Input to every later analysis |
| `results/{tag}/cross_type.csv` | `analyze --kind cross_type` | 06b 6b.1 (diagonal), 6b.2 (off-diagonal) |
| `results/{tag}/unsupervised.csv` | `analyze --kind unsupervised` | 06c 6c.1, 06b 6b.4 |
| `results/{tag}/precision_lift.csv` | `analyze --kind precision_lift` | 06c 6c.1 |
| `results/{tag}/memorization.csv` | `analyze --kind memorization` | 06b 6b.4 |
| `results/{tag}/early_*.csv` | `analyze --kind early_*` | 06b 6b.5 |
| `results/{tag}/feature_attribution.csv` | `analyze --kind feature_attribution` | 06b 6b.6 |
| `results/eval/eval_{tag}_{dataset}.json` | `evaluate` | 06a 6a.1, 06c 6c.5 |
| `results/{tag}/feature_ablation.csv` | `analyze.py::feature_group_ablation()` | 06c 6c.3.1-6c.3.2 |
| `results/{tag}/feature_correlation.csv` | `analyze.py::feature_correlation()` | 06b 6b.7 |
| `results/{tag}/single_feature_ablation.csv` | `analyze.py::single_feature_ablation()` | 06b 6b.8 |
| `results/{tag}/minimal_feature_set.csv` | `analyze.py::minimal_feature_set()` | 06c 6c.3.3 |
| `results/{tag}/transfer_to_mixed.csv` | `analyze.py::transfer_to_mixed()` | 06c 6c.4 |
| `results/{tag}/pooled_scorer_compare.csv` | `analyze.py::pooled_scorer_compare()` | 06c 6c.4.5 |
| `datasets/{tag}/cleaning_loop/{name}/metadata.json` | `clean` | 06a/06c 6c.5 |
| `results/{tag}/external_baselines.csv` | `baselines.py` (not tracked in Git) | 06b 6b.9 |
| `results/oasst-wild/length_confound.csv` | `analyze.py::length_confound()` | 06b 6b.10 |

**Downstream evaluation conventions.** 7 benchmarks: MMLU (n=14042), GSM8K (1319), HellaSwag (10042), ARC (1172), BBH (540), TruthfulQA (817), WinoGrande (1267). The "7-benchmark average" reported throughout is an **unweighted arithmetic mean** of these 7 accuracies — so BBH (n=540) and MMLU (n=14042) carry equal weight, and any single-benchmark swing is amplified. 06a Section 6a.1 and 06c Section 6c.5 therefore always report the per-benchmark table alongside the average, not the average alone.

Except for 06c Sections 6c.5/6c.6, every analysis in all three files recomputes from an existing trajectory without retraining; Sections 6c.5/6c.6 are the only experiments in the whole report that actually modify the training data and retrain.

### Navigation Table

| File | Thread | Core question | Main sections |
|---|---|---|---|
| [06a-harm-ranking.md](06a-harm-ranking.md) | Thread 1: harm ranking | Which noise type hurts training the most | 6a.1, 6a.2.1-6a.2.4 |
| [06b-feature-signatures.md](06b-feature-signatures.md) | Thread 2: feature signatures | What do each type's training dynamics / text features look like | 6b.1 (incl. 6b.1.1), 6b.2, 6b.3, 6b.4, 6b.5, 6b.6, 6b.7, 6b.8, 6b.9, 6b.10 |
| [06c-detectability.md](06c-detectability.md) | Thread 3: detectability | Can it be separated label-free, how accurate, does cleaning actually pay off | 6c.1, 6c.2, 6c.3.1-6c.3.4, 6c.4.1-6c.4.5, 6c.5.1-6c.5.3, 6c.6 |
