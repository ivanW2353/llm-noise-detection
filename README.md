# Noise Effects on LLM Fine-tuning: Sample-Level Metrics & Detection

Study of how four types of data noise affect LLM SFT, and whether per-sample
training metrics (loss / gradient norm / gradient cosine similarity) can
separate noisy samples from clean ones.

## Datasets (6)

Built from `databricks/databricks-dolly-15k` (15011 samples, chat format),
noise ratio 10% (configurable in `config.yaml`):

| dataset    | description                                                    |
|------------|----------------------------------------------------------------|
| `clean`    | original data (baseline)                                        |
| `garbled`  | 10% samples corrupted with mojibake/garbage characters          |
| `duplicate`| 10% extra rows that are exact copies                            |
| `unrelated`| 10% samples whose response is fluent/correct but from a different category (contextually unrelated) |
| `keyword`  | 10% samples with only key entities/numbers replaced             |
| `mixed`    | 10% total noise, evenly split among the four types              |

Every row carries `noise_label`/`noise_type`. Sample order is identical across
datasets (fixed seed) except appended duplicate copies, so per-sample metrics
are directly comparable.

## Training

- Model: `Qwen2.5-3B-Instruct` + LoRA (r=32, all linear modules), bf16,
  flash-attention, 5 epochs, micro-batch 1 + grad-accum 16, lr 2e-4 cosine.
- Per-sample tracking (micro-batch=1): `loss`, `grad_norm` (LoRA gradient L2),
  `cos_sim_ref` (cosine similarity with a pre-training reference direction from
  held-out clean samples, LESS-style influence), `cos_sim_global` (similarity
  with the accumulation-window gradient), `tokens`.
- Epoch-end diagnostic pass (every 8th sample): `max_token_loss`, `frac_hard`
  (fraction of tokens with loss > 4.0).
- TensorBoard: train/loss, grad_norm, cos_ref, cos_global, lr, tokens/sec,
  gpu_mem, per-layer grad norms, LoRA weight/grad histograms, held-out loss.
- 6 runs, same seed/order; only the noise differs.

## Evaluation

Few-shot evaluation of all 6 fine-tuned models + base model on common
validation sets (implemented locally, cached datasets):

| task         | setup            |
|--------------|------------------|
| MMLU         | 5-shot           |
| HellaSwag    | 10-shot          |
| ARC-Challenge| 25-shot          |
| Winogrande   | 5-shot           |
| TruthfulQA   | 0-shot           |
| GSM8K        | 5-shot CoT (chat template) |
| BBH          | 3-shot CoT, 20/task |

## Detection analysis

Combines per-sample metrics with noise labels; per noise type:
- univariate AUC per metric, within-run noise vs normal
- multivariate logistic regression / random forest + ROC + feature importance
- loss trajectories across epochs, metric distributions, PCA scatter

### Metrics recorded per sample

| metric | meaning |
|--------|---------|
| `loss` / `grad_norm` | per-sample CE loss & LoRA gradient L2 norm (every epoch) |
| `cos_sim_ref` | cosine similarity with a pre-training clean reference direction (LESS-style influence) |
| `cos_sim_global` | cosine similarity with the accumulation-window gradient |
| `update_contrib` | sample gradient norm relative to the running Adam-RMS gradient (B params only) |
| `user_loss` | mean CE over the USER (prompt) tokens — separates garbled (prompt corrupted) from keyword/unrelated (prompt intact) |
| `entropy` | mean next-token entropy over label tokens (diagnostic subset) |
| `token_loss_skew/kurt` | shape of the per-token loss distribution (garbled = strongly right-skewed) |
| `max_token_loss` / `frac_hard` | hardest token loss / fraction of tokens with loss > 4.0 |
| `loss_std`, `converge_epoch`, `loss_rank`, `loss_curvature` | derived from per-epoch loss trajectories |
| `grad_norm_cv`, `cos_ref_trend` | gradient variability / reference-alignment trend |
| `text_nn_sim` | TF-IDF nearest-neighbor similarity — direct signal for duplicate (exact copies) and keyword (few words changed) |

Token-level diagnostics (top-k hardest label tokens per sample) are saved
separately (`token_diag_epoch*.jsonl`) and analyzed offline with per-token
exact gradient attribution (`analyze_token_level.py`).

## Changing the noise ratio

Every experiment is isolated under an `experiment_tag` (default none), so
different ratios never overwrite each other:

```bash
bash run_experiment.sh --ratio 0.20            # build + train + eval + analyze
bash run_experiment.sh --ratio 0.20 --train-only
bash run_experiment.sh --ratio 0.20 --eval-only
bash run_experiment.sh --ratio 0.20 --analyze-only
```

Layout for a tagged experiment (e.g. `ratio20`):
- `<data_root>/data/ratio20/train/<dataset>/` (plus a shared `heldout.jsonl`)
- `<data_root>/runs/ratio20/<dataset>/`
- `<repo>/results/eval_ratio20_<dataset>.json` and `*_ratio20.*` analysis outputs

Or step by step:

```bash
python scripts/make_noise.py --ratio 0.20 --tag ratio20
python scripts/train.py --dataset clean --tag ratio20
python scripts/evaluate.py --dataset clean --tag ratio20
python scripts/analyze_detection.py --tag ratio20
python scripts/analyze_token_level.py --tag ratio20
```

## Reproduce

```bash
# 1. build the 6 datasets
python scripts/make_noise.py            # -> <data_root>/data/train/*

# 2. train (one run per dataset, ~3 h each on RTX 5090 at 5 epochs)
python scripts/train.py --dataset clean
# or all: bash run_all.sh

# 3. evaluate all models (after training)
bash run_all_eval.sh                    # -> results/eval_*.json

# 4. analysis
python scripts/analyze_detection.py     # -> results/*.csv + *.png
python scripts/analyze_token_level.py   # -> token-level attribution + AUCs
```

Config: `config.yaml` (paths, noise ratio, hyper-parameters).

## Layout

- `scripts/` – data construction, training, evaluation, analysis
- `results/` – evaluation tables, AUC tables, ROC / distribution plots
- `<data_root>/data/train/` – the 6 datasets
- `<data_root>/runs/<dataset>/` – per-sample metrics, tensorboard, LoRA weights
