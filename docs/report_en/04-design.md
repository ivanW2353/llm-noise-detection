## 4. Experimental Design

This section lays out the full execution path: the command for each step, the files it produces, and the design choices that affect how the results should be read. Every analysis in later sections is recomputed from the intermediate files produced here, with no retraining required.

### 4.1 Overall pipeline

```
(1) Build datasets       cli.py data      -> data/{tag}/{dataset}/train.jsonl
(2) LoRA fine-tune       cli.py train     -> runs/{tag}/{dataset}/metrics/*.jsonl
(3) Aggregate per-sample cli.py analyze --kind features
                                          -> results/{tag}/per_sample_metrics.csv
(4) Run analyses         cli.py analyze --kind {unsupervised,cross_type,...}
                                          -> results/{tag}/{kind}.csv
(5) Closed-loop cleaning cli.py clean     -> data/{tag}/cleaning_loop/{name}/train_{targeted,random}.jsonl
(6) Retrain + evaluate   cli.py train / evaluate (back to step 2)
```

Steps (1)-(4) answer "can it be detected"; (5)-(6) answer "does cleaning actually help". **Only (5)-(6) touch the training data**; everything before that is recomputation over existing trajectories, so changing an analysis method is nearly free while changing the data or training config costs a full GPU round.

### 4.2 Step (1): dataset construction and noise injection

```bash
python3 cli.py data --source dolly --tag ratio10 --ratio 0.10 \
  --datasets clean,garbled,template,duplicate,unrelated,truncation,near_duplicate,keyword,mixed
```

Starting from dolly-15k, 400 rows are split off as a holdout set (`data/{tag}/heldout.jsonl`, shared by all 9 datasets; `ref_samples=200` for the reference gradient, `heldout_samples=200` for held-out loss monitoring), leaving 14,611 rows as the training set. Injection happens in `data.py::apply`: `np.random.default_rng(seed=42)` draws `int(len(rows) * ratio)` sample indices, the selected samples get the corresponding transform, and the rest pass through untouched.

Two details affect later readings:

**`duplicate` is the only type that changes the row count.** The other 6 replace in place (`[fn(r) if i in ids else r for ...]`), leaving the count unchanged; `duplicate` goes through `rows + [_duplicate_copy(...)]`, appending copies, so the `duplicate` dataset has 16,072 rows rather than 14,611 — and its noise fraction is therefore 1461/16072 ≈ 9.1%, not 10%.

**Per-type counts in `mixed` fluctuate.** `data.py:175` injects each of the 7 subtypes independently at `ratio/len(types)` = 0.10/7 ≈ 1.43%, reseeding with `seed + offset` each time. The theoretical count per type is 14611 × 0.0143 ≈ 209; the measured counts are 191-211 (near_duplicate 211, template 206, truncation 204, keyword 197, unrelated 197, garbled 194, duplicate 191), the deviation coming from independent draws occasionally hitting the same row, where the later transform overwrites the earlier one. Total noise is 1400 rows, 9.4% of 14,819. The per-type slicing in Section 6.12 rests on these labels.

Nine datasets in all: `clean` (no injection, the downstream comparison baseline), the 7 single types, and `mixed`. Two noise ratios, `ratio10` (10%) and `ratio5` (5%), the latter to confirm the conclusions are not an accident of one particular ratio (Section 6.4).

### 4.3 Step (2): LoRA fine-tuning and per-sample metric collection

```bash
python3 cli.py train --tag ratio10 --dataset garbled --model hf-lora
```

Qwen2.5-3B-Instruct + LoRA (r=32, alpha=64, dropout=0.05), 5 epochs, `micro_batch=1` with `grad_accum=16` (effective batch 16), lr=2e-4, `max_len=1024`. A single NVIDIA RTX PRO 6000 Blackwell (~98GB), 9 datasets run serially at roughly 1.5 hours each (Section 8.5 has the measured breakdown).

`micro_batch=1` is not a performance choice but **the precondition for per-sample gradient tracking** — only when one sample constitutes an entire forward-backward pass can the gradient norm and the cosine similarity to a reference direction be attributed to that sample. This is the data foundation of the whole method, and also why it is slower than ordinary fine-tuning.

One record per sample per epoch lands in `runs/{tag}/{dataset}/metrics/per_sample.jsonl`:

| Field | Meaning |
|---|---|
| `loss` | That sample's training loss this epoch |
| `grad_norm` | L2 norm of that sample's gradient |
| `cos_sim_ref` | Cosine similarity between that sample's gradient and the "reference gradient" (mean gradient direction over 200 holdout samples) |
| `cos_sim_global` | Cosine similarity between that sample's gradient and the mean gradient of other samples in the same optimizer step |
| `update_contrib` | That sample's share of the parameter update |

Two further **subsampled** outputs: `diag_epoch*.jsonl` (diagnostic-layer aggregates) and `token_diag_epoch*.jsonl` (the `[position, token_id, loss]` triples of each sample's 32 highest-loss tokens). Both are collected at `diag_subsample=8` — one in every 8 samples gets a pure forward diagnostic pass — giving roughly **12.5% coverage**.

That coverage gap runs through the entire report and is the easiest trap when reading numbers: **13 of the 37 features have only 12.5% coverage**, and `dropna` requires all of them present, so any analysis using the full feature set actually runs on roughly 900-1,200 rows. Section 6.10 opens with an explanation of this framing difference, and Section 6.12 reports both views side by side: `full_diag` (37 features, ~919 rows) and `full_coverage` (19 features, all 14,819 rows).

### 4.4 Step (3): assembling the per-sample metric table

```bash
python3 cli.py analyze --tag ratio10 --kind features
```

`analyze.py::build_table` joins four sources into one wide table, `results/{tag}/per_sample_metrics.csv`:

1. **Trajectory aggregates** (`_load_run_metrics`): pivot `per_sample.jsonl` by `sample_id × epoch`, then compute `_mean`/`_last`/`_std`/`_slope` (last minus first) for each of loss/grad_norm/cos_ref/cos_global. Loss additionally gets `loss_min`, `converge_epoch` (first epoch where loss < 2.0, recorded as 5 if never reached), `loss_curvature` (constant term of a quadratic fit over epochs), and `loss_rank` (mean within-epoch percentile rank).
2. **Diagnostic-layer aggregates** (`diag_epoch*.jsonl`, averaged per sample).
3. **Token-level statistics** (`_load_token_features`): `hard_loss_mean`/`hard_loss_max`/`hard_pos_peak`/`hard_id_uniq`/`hard_pos_jaccard` and others, from the top-32 hard-token triples.
4. **Static text feature** (`textsim.text_nn_sim`): independent of training, computed straight from `train.jsonl` as each sample's text similarity to its nearest neighbor. It is the only feature with 100% coverage that needs no GPU, and it carries both the ablation in Section 6.11 and the calibration-free baseline in Section 6.12.

The true `noise_type` label is read from `train.jsonl` **for evaluation only**; no scorer ever uses it. That is the precise meaning of "label-free" here: labels exist on the evaluation side, never on the detection side.

### 4.5 Step (4): what each analysis actually does

| Analysis | `--kind` | Key protocol points |
|---|---|---|
| Within-domain difficulty (Section 6.2) | `unsupervised` | IsolationForest and MAD z-score fit **independently per dataset**, never pooled (a global fit would let garbled's extreme magnitudes raise the outlier baseline for a low-variance dataset like template) |
| Cross-type transfer (Section 6.3) | `cross_type` | Train LR + RF on the source, `StandardScaler` fit on the source only and `transform`-ed on the target, higher AUC of the two. The diagonal reuses the 5-fold CV result so it is on the same scale as the off-diagonal |
| Cross-ratio transfer (Section 6.4) | `cross_ratio` | ratio10 ↔ ratio5, both directions |
| Cleaning-precision lift (Section 6.5) | `precision_lift` | P@10% ÷ random baseline |
| Direction reversal (Section 6.6) | `memorization` | Fixed-sign rule, **direction never re-fit per dataset** |
| Early detection (Section 6.7) | `early_unsupervised` / `early_memorization` | `build_table(max_epoch=k)` truncates the trajectory to simulate "trained only k epochs", with no need to actually stop early |
| Feature attribution (Section 6.8) | `feature_attribution` | Permutation importance, `n_repeats=20` |

**How cross-type transfer relates to `mixed`**: `cross_type` explicitly skips `mixed` in code (`if ds in ('clean','mixed'): continue`), so it cannot answer the mixed-stream question. Section 6.12 fills that gap with a separate script, `analyze.py::transfer_to_mixed()`.

### 4.6 Steps (5)-(6): closed-loop cleaning and retraining

```bash
python3 cli.py clean --tag ratio10 --dataset template --method memo_signed --budget 0.10
python3 cli.py train    --tag ratio10 --dataset cleaning_loop_targeted_template_signed \
  --train-file data/ratio10/cleaning_loop/template_memo_signed/train_targeted.jsonl --model hf-lora
python3 cli.py evaluate --tag ratio10 --dataset cleaning_loop_targeted_template_signed --model hf-lora
```

The design decisions inside `cleaning_loop.py::build`:

**Full-coverage features only.** Scoring targets the **entire training set**, not the diagnostic subsample — otherwise most real noise would never be a removal candidate and cleaning would be meaningless. The price is being restricted to the 19 features with no missing values in that dataset, excluding all token-level diagnostics and `cos_global_*`. This is the source of what Section 6.10 calls "reported AUC and production precision don't always agree".

**Three scorers, one per noise assumption** (detailed in Sections 6.6 and 6.12.5): `iforest` (noise = outlier, undirected), `memo_signed` (noise = abnormally easy to learn, fixed negative sign), and `pooled` (three legs standardized and combined by per-sample max, for unknown composition).

**An equal-size random-drop control is mandatory.** Removing 10% of the samples removes 10% of the training data by itself, so comparing only against the uncleaned baseline cannot separate "denoising gain" from "data-volume loss". Each `clean` therefore emits both `train_targeted.jsonl` (removed by score) and `train_random.jsonl` (same `n_drop`, same seed, removed at random), and both get retrained. That is where Section 6.13's four-way comparison (clean baseline / uncleaned / targeted / random) comes from.

**Removal precision is measured after the fact, never used in scoring.** `targeted_precision` is the fraction of the removed set whose true `noise_type != 'none'`; it is written into `metadata.json` for analysis, and the scoring process never sees it.

### 4.7 Known design-level limitations

The following are design choices rather than execution slips, and should be held in mind alongside the conclusions:

- **12.5% subsample coverage** makes the token-level features unusable in a full-dataset setting, and the ablation in Section 6.11 shows those features happen to be near_duplicate's only effective signal — so the reported AUC for that type is optimistic relative to what production can reach.
- **A single base model at a single scale** (Qwen2.5-3B + LoRA r=32, 5 epochs). The downstream harm of noise is small to begin with (Section 6.9: the 7-benchmark averages cluster in 0.42-0.44), so larger models or longer training could plausibly widen the differences.
- **All 7 noise types are programmatically injected**, with known and consistent perturbations. The "unseen types" in Section 6.12 are still among these 7 (merely unseen by the detector pool), and genuinely wild noise is outside what was validated.
- **Closed-loop retraining covers only garbled and template**; the `mixed` closed loop has not been run (listed as a next step in Section 7.3).
---
