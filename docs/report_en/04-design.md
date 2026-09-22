## 4. Experimental Design

This section spells out the execution path for the whole experiment: the command for each step, its output files, and a few design choices that affect how the results should be read. Every later analysis is recomputed from the intermediate files produced here, with no need to retrain.

### 4.1 Overall Pipeline

```
① Build datasets        cli.py data      → datasets/{tag}/{dataset}/train.jsonl
② LoRA fine-tuning       cli.py train     → runs/{tag}/{dataset}/metrics/*.jsonl
③ Aggregate per-sample   cli.py analyze --kind features
   metrics                                → results/{tag}/per_sample_metrics.csv
④ Run analyses           cli.py analyze --kind {unsupervised,cross_type,...}
                                          → results/{tag}/{kind}.csv
⑤ Closed-loop cleaning   cli.py clean     → datasets/{tag}/cleaning_loop/{name}/train_{targeted,random}.jsonl
⑥ Retrain + evaluate     cli.py train / evaluate (loop back to step ②)
```

Steps ①-④ answer "can it be detected"; ⑤-⑥ answer "does cleaning actually help." **Only ⑤-⑥ ever touch the training data**; everything before that recomputes from an existing trajectory, so changing the analysis method is cheap while changing the data or training configuration costs a full round of GPU time.

### 4.2 Step ①: Building the datasets and injecting noise

```bash
python3 cli.py data --source dolly --tag dolly-ratio10 --ratio 0.10 \
  --datasets clean,garbled,template,duplicate,unrelated,truncation,near_duplicate,keyword,mixed
```

Starting from dolly-15k, 400 rows are first carved out as a held-out set (`datasets/{tag}/heldout.jsonl`, shared by all 9 datasets; 200 rows for the reference gradient, 200 for held-out loss monitoring), leaving 14,611 training rows. The injection logic lives in `data.py::apply`: `np.random.default_rng(seed=42)` draws `int(len(rows) * ratio)` sample indices, transforms the sampled rows, and leaves everything else untouched.

Two details affect how later numbers should be read:

**`duplicate` is the only type that changes the row count.** The other 6 types replace rows in place (`[fn(r) if i in ids else r for ...]`), so row count is unchanged; `duplicate` instead does `rows + [_duplicate_copy(...)]`, i.e. appends copies, so the `duplicate` dataset has 16,072 rows rather than 14,611, and its noise fraction is 1461/16072 ≈ 9.1% rather than 10%.

**`mixed`'s per-subtype counts fluctuate.** `data.py:175` injects each of the 7 subtypes independently at `ratio/len(types)` = 0.10/7 ≈ 1.43%, reseeding as `seed + offset` each time. The theoretical count per type is 14611 × 0.0143 ≈ 209; the measured counts range from 191 to 211 (near_duplicate 211, template 206, truncation 204, keyword 197, unrelated 197, garbled 194, duplicate 191), the deviation coming from independent sampling occasionally hitting the same row twice, with the later draw overwriting the earlier one. Total noise is 1400 rows, 9.4% of 14,819. [06c](06c-detectability.md) Section 6c.4's per-type slicing is built on these labels.

9 datasets: `clean` (no injection, used as the downstream comparison baseline), 7 single-type datasets, and `mixed`. Two noise ratios, `dolly-ratio10` (10%) and `dolly-ratio5` (5%), the latter used to confirm the conclusions are not an artifact of one particular ratio ([06b](06b-feature-signatures.md) Section 6b.3).

This step is the shared data foundation for all three threads: the injected noise samples are what Thread 2's feature signatures and Thread 3's scorers observe; the `clean` baseline is the comparison anchor Thread 1 uses to measure downstream harm.

### 4.3 Step ②: LoRA fine-tuning and per-sample metric collection

```bash
python3 cli.py train --tag dolly-ratio10 --dataset garbled --model hf-lora
```

Qwen2.5-3B-Instruct + LoRA (r=32, alpha=64, dropout=0.05), 5 epochs, `micro_batch=1` + `grad_accum=16` (effective batch 16), lr=2e-4, `max_len=1024`. Single NVIDIA RTX PRO 6000 Blackwell GPU (~98GB), 9 datasets queued serially, roughly 1.5 hours each (Section 8.5 has the measured breakdown).

`micro_batch=1` is not a performance choice; it is **the precondition for per-sample gradient tracking** — only when a single sample makes up one entire forward-backward pass can gradient norm and cosine similarity to the reference direction be attributed to that sample. This is the data foundation the whole method rests on, and the reason it is slower than ordinary fine-tuning.

Each sample writes one record per epoch (`runs/{tag}/{dataset}/metrics/per_sample.jsonl`), with fields including:

| Field | Meaning |
|---|---|
| `loss` | The sample's training loss this epoch |
| `grad_norm` | The L2 norm of the sample's gradient |
| `cos_sim_ref` | Cosine similarity between the sample's gradient and the "reference gradient" (the average gradient direction over 200 held-out samples) |
| `cos_sim_global` | Cosine similarity between the sample's gradient and the average gradient of other samples in the same optimizer step |
| `update_contrib` | The sample's share of contribution to this parameter update |

Two further **subsampled** outputs exist: `diag_epoch*.jsonl` (aggregated diagnostic-layer quantities) and `token_diag_epoch*.jsonl` (the top-32 highest-loss tokens per sample, as `[position, token_id, loss]` triples). Both are collected at `diag_subsample=8`, i.e. one row in 8 gets a pure forward diagnostic pass — **about 12.5% coverage**.

This coverage gap runs through the whole report and is the easiest trap in reading the numbers: **13 of the 37 features have only 12.5% coverage**, and since `dropna` requires every feature to be non-null, any analysis using the full feature set effectively runs on only about 900-1200 rows. [06c](06c-detectability.md) Section 6c.2 opens with an explanation of this gap, and [06c](06c-detectability.md) Section 6c.4 reports both the `full_diag` (37 features, ~919 rows) and `full_coverage` (19 features, all 14,819 rows) populations side by side.

This step primarily serves Thread 2 — per-sample training dynamics are the direct source of every downstream feature.

### 4.4 Step ③: Aggregating the per-sample metric table

```bash
python3 cli.py analyze --tag dolly-ratio10 --kind features
```

`analyze.py::build_table` stitches four sources into one wide table, `results/{tag}/per_sample_metrics.csv`:

1. **Trajectory aggregation** (`_load_run_metrics`): pivots `per_sample.jsonl` by `sample_id × epoch`, computing `_mean`/`_last`/`_std`/`_slope` (last minus first) for each of the loss/grad_norm/cos_ref/cos_global groups. Loss additionally gets `loss_min`, `converge_epoch` (first epoch with loss < 2.0, or 5 if never reached), `loss_curvature` (the constant term of a quadratic fit over epochs), and `loss_rank` (mean of the per-epoch percentile rank).
2. **Diagnostic-layer aggregation** (mean over `diag_epoch*.jsonl` per sample).
3. **Token-level statistics** (`_load_token_features`): computes `hard_loss_mean`/`hard_loss_max`/`hard_pos_peak`/`hard_id_uniq`/`hard_pos_jaccard` etc. from the top-32 hard-token triples.
4. **Static text features** (`textsim.text_nn_sim`): not dependent on training, computed directly from `train.jsonl` as each sample's text similarity to its nearest neighbour. This is the only feature with 100% coverage that needs no GPU; both [06c](06c-detectability.md) Section 6c.3's ablation and Section 6c.4's unlabeled baseline use it.

The true label `noise_type` is read from `train.jsonl` **only for evaluation**; no scorer ever uses it — this is the precise meaning of "label-free": the label exists on the evaluation side, not on the detection side.

This step primarily serves Thread 2 (turning per-sample trajectories into an analyzable feature table), and is also the direct input to every Thread 3 scorer (`iforest`/`memo_signed`/`pooled`).

### 4.5 Step ④: Per-analysis conventions

| Analysis | `--kind` | Method notes |
|---|---|---|
| In-domain detection difficulty ([06b](06b-feature-signatures.md) Section 6b.1) | `unsupervised` | IsolationForest and MAD z-score are fit **independently per dataset**, never merged across datasets (otherwise garbled's extreme magnitude would inflate template's outlier baseline) |
| Cross-type transfer (Section 6b.2) | `cross_type` | Trains LR + RF on the source domain, `StandardScaler` is `fit` only on the source domain and only `transform`ed on the target domain; the higher of the two AUCs is taken. The diagonal reuses the 5-fold CV result so it is on the same scale as the off-diagonal |
| Cross-ratio transfer (Section 6b.3) | `cross_ratio` | dolly-ratio10 ↔ dolly-ratio5, both directions |
| Cleaning-precision lift ([06c](06c-detectability.md) Section 6c.1) | `precision_lift` | P@10% ÷ random baseline |
| Direction reversal ([06b](06b-feature-signatures.md) Section 6b.4) | `memorization` | Fixed sign rule, **never refit per dataset** |
| Early detection (Section 6b.5) | `early_unsupervised` / `early_memorization` | `build_table(max_epoch=k)` truncates the trajectory to simulate "trained for only k epochs," with no need to actually stop training early |
| Feature attribution (Section 6b.6) | `feature_attribution` | Permutation importance, `n_repeats=20` |

**Cross-type transfer and `mixed`**: `cross_type` explicitly skips `mixed` in code (`if ds in ('clean','mixed'): continue`), so it cannot answer the "mixed stream" question. [06c](06c-detectability.md) Section 6c.4 fills this gap with a separate routine, `analyze.py::transfer_to_mixed()`.

Most of this step's analyses (in-domain detection, cross-type/cross-ratio transfer, direction reversal, early detection, feature attribution) primarily serve Thread 2; `precision_lift` ([06c](06c-detectability.md) Section 6c.1) is a deployment-facing metric that directly serves Thread 3.

### 4.6 Steps ⑤-⑥: Closed-loop cleaning and retraining

```bash
python3 cli.py clean --tag dolly-ratio10 --dataset template --method memo_signed --budget 0.10
python3 cli.py train    --tag dolly-ratio10 --dataset cleaning_loop_targeted_template_signed \
  --train-file datasets/dolly-ratio10/cleaning_loop/template_memo_signed/train_targeted.jsonl --model hf-lora
python3 cli.py evaluate --tag dolly-ratio10 --dataset cleaning_loop_targeted_template_signed --model hf-lora
```

Key design choices in `cleaning_loop.py::build`:

**Only full-coverage features are used.** Scoring targets the **full training set**, not the diagnostic subsample — otherwise most real noise would fall outside the candidate pool to begin with, making removal pointless. The cost is being restricted to the 19 features that are non-null across the whole dataset, excluding every token-level diagnostic and `cos_global_*`. This is the source of what [06c](06c-detectability.md) Section 6c.2 calls "reported AUC and production precision don't always agree."

**Three scorers, corresponding to three noise hypotheses** (see [06b](06b-feature-signatures.md) Section 6b.4 and [06c](06c-detectability.md) Section 6c.4.5 for detail): `iforest` (noise = outlier, undirected), `memo_signed` (noise = anomalously easy to learn, fixed negative sign), and `pooled` (the three legs standardized and combined by per-sample maximum, for settings where the noise composition is unknown).

**A matched random-removal control is mandatory.** Removing 10% of the samples reduces the training set by 10% regardless of whether the removal is smart; comparing only against the "uncleaned baseline" cannot separate "cleaning benefit" from "data-quantity loss." So every `clean` run produces both `train_targeted.jsonl` (score-based removal) and `train_random.jsonl` (same `n_drop`, same seed, random removal), and both get retrained. [06c](06c-detectability.md) Section 6c.5's four-way comparison (clean baseline / uncleaned / targeted removal / random removal) comes from this.

**Removal precision is a post-hoc statistic, not part of scoring.** `targeted_precision` is the fraction of the removed set whose true `noise_type != 'none'`, written into `metadata.json` for analysis; the scoring process never sees it.

Step ⑤ primarily serves Thread 3 (producing the cleaned candidate training set); step ⑥'s retrain-and-evaluate serves both Thread 1 (whether downstream performance actually improves, i.e. whether the noise is actually harmful) and Thread 3 (validating the cleaning scheme's real benefit in a true training loop).

### 4.7 Known limitations of the design

The following are design-level limitations, not execution errors, and should be kept in mind when reading the conclusions:

- **12.5% subsample coverage** makes token-level features unavailable at full scale, and [06c](06c-detectability.md) Section 6c.3's ablation shows these features happen to be near_duplicate's only effective signal source — that type's reported AUC is optimistic relative to what production can reach.
- **A single base model at a single scale** (Qwen2.5-3B + LoRA r=32, 5 epochs). The downstream harm from noise is small to begin with ([06a](06a-harm-ranking.md) Section 6a.1: 7-benchmark averages cluster in a 0.42-0.44 band); larger models or longer training could plausibly widen the gap.
- **All 7 noise types are programmatically injected**, with known, consistent perturbations. [06c](06c-detectability.md) Section 6c.4's "unseen types" are still one of these 7 (merely unseen by the detector pool); genuinely wild noise is outside this study's scope.
- **Closed-loop retraining does not cover near_duplicate or keyword**—both have weak detection signal to begin with, making any closed-loop gain hard to interpret; the closed loops that are complete cover garbled, template, mixed ([06c](06c-detectability.md) Section 6c.5) and oasst-wild ([06c](06c-detectability.md) Section 6c.6).
---
