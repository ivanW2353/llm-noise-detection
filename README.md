# Noise Effects on LLM Fine-tuning: Sample-Level Metrics & Detection

Study of how four types of data noise affect LLM SFT, and whether per-sample
training metrics (loss / gradient norm / gradient cosine similarity) can
separate noisy samples from clean ones.

## Experiment status

| Experiment | Noise types | Ratio | Status | Highlights |
|---|---|---|---|---|
| `ratio10` (default tag) | garbled, duplicate, unrelated, keyword + mixed (4-way) | 10% | **COMPLETE** | detection garbled 0.9996 / duplicate 0.974 / unrelated 0.923 / keyword 0.531 (blind spot); noise harm << SFT harm; detectability decays with epochs |
| `ratio05` | same 4 types + mixed | 5% | **COMPLETE** | detection is ratio-insensitive (garbled 0.999); unrelated harms MMLU *more* at 5% than 10% |
| `extra10` | + template, truncation, near_duplicate + mixed (7-way) | 10% | **data built; training pending GPU** | fills the "consistent-pattern / info-loss / near-duplicate" quadrants |

## Analysis reports

- **[中文详细分析报告](docs/analysis_report_zh.md)** — 训练动态、样本/token 级检测、验证集影响、逐题分析
- **[English analysis report](docs/analysis_report_en.md)** — training dynamics, sample/token-level detection, benchmark impact, per-question analysis
- **[检测算法规范](docs/detection_algorithms_zh.md) / [detection algorithms](docs/detection_algorithms_en.md)** — 可复现的噪音检测算法 (LaTeX 公式)
- **[跨实验综合](docs/cross_experiment_synthesis_zh.md) / [cross-experiment synthesis](docs/cross_experiment_synthesis_en.md)** — 与 dynanoise / qa-noise 的合并结论
- **[剂量-效应对比](docs/dose_response_zh.md) / [dose response](docs/dose_response_en.md)** — ratio10 vs ratio05
- **[文献综述](docs/literature_review_zh.md)** — 25 篇相关论文

## Datasets (6 core, +3 optional with `--with-extra`)

Built from `databricks/databricks-dolly-15k` (15011 samples, chat format),
noise ratio 10% (configurable in `config.yaml`):

| dataset    | description                                                    |
|------------|----------------------------------------------------------------|
| `clean`    | original data (baseline)                                        |
| `garbled`  | 10% samples corrupted with mojibake/garbage characters          |
| `duplicate`| 10% extra rows that are exact copies                            |
| `unrelated`| 10% samples whose response is fluent/correct but from a different category (contextually unrelated) |
| `keyword`  | 10% samples with only key entities/numbers replaced             |
| `mixed`    | 10% total noise, evenly split among the noise types             |

`--with-extra` adds three more (7-way mixed):
`template` (consistent fixed-answer pattern), `truncation` (response cut at
50%, information loss), `near_duplicate` (light paraphrase via WordNet).

Every row carries `noise_label`/`noise_type`. Sample order is identical across
datasets (fixed seed) except appended duplicate copies, so per-sample metrics
are directly comparable.

## Training

- Model: `Qwen2.5-3B-Instruct` + LoRA (r=32, all linear modules), bf16,
  flash-attention, 5 epochs, micro-batch 1 + grad-accum 16, lr 2e-4 cosine.
- 400 clean samples are held out of every dataset (`heldout.jsonl`, shared
  across datasets) and used for the reference gradient direction and the
  held-out eval loss.
- Per-sample tracking (micro-batch=1): `loss`, `grad_norm` (LoRA gradient L2),
  `cos_sim_ref` (cosine similarity with a pre-training reference direction from
  held-out clean samples, LESS-style influence), `cos_sim_global` (similarity
  with the accumulation-window gradient), `update_contrib` (gradient norm
  relative to the running Adam-RMS, B params only), `tokens`.
- Epoch-end diagnostic pass (every 8th sample): `max_token_loss`, `frac_hard`,
  `user_loss`, `entropy`, `token_loss_skew/kurt`, plus top-k hardest label
  tokens per sample (`token_diag_epoch*.jsonl`).
- TensorBoard: train/loss, grad_norm, cos_ref, cos_global, update_contrib, lr,
  tokens/sec, gpu_mem, per-layer grad norms, LoRA weight/grad histograms,
  held-out loss.
- 6 runs, same seed/order; only the noise differs.

## Evaluation

Few-shot evaluation of all 6 fine-tuned models + base model on common
validation sets (implemented locally, cached datasets):

| task         | setup            |
|--------------|------------------|
| MMLU         | 5-shot (per-subject breakdown, 57 subjects) |
| HellaSwag    | 5-shot           |
| ARC-Challenge| 25-shot          |
| Winogrande   | 5-shot           |
| TruthfulQA   | 0-shot           |
| GSM8K        | 5-shot CoT (chat template) |
| BBH          | 3-shot CoT, 20/task (per-task breakdown) |

Full per-task results (including MMLU per-subject and BBH per-task accuracy)
are saved in each `eval_<model>.json` and aggregated into
`results/eval_comparison.csv` and `results/eval_mmlu_subjects.csv`.

Evaluation is resumable: results are written after every task, and re-running
skips already-completed models/tasks (`--force` redoes a model).

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

Every experiment is isolated under an `experiment_tag` (default `ratio10`), so
different ratios never overwrite each other:

```bash
bash run_experiment.sh --ratio 0.20            # build + train + eval + analyze
bash run_experiment.sh --ratio 0.20 --train-only
bash run_experiment.sh --ratio 0.20 --eval-only
bash run_experiment.sh --ratio 0.20 --analyze-only
```

Layout per experiment tag (e.g. `ratio20`):
- `<data_root>/data/ratio20/<dataset>/` (plus a shared `heldout.jsonl`)
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
# 1. build the 6 datasets (default tag: ratio10)
python scripts/make_noise.py            # -> <data_root>/data/ratio10/*

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
- `docs/` – 9 reports: analysis (zh/en), detection algorithms, cross-experiment synthesis, dose-response, literature review
- `results/eval/` – per-model evaluation results + raw per-question records + comparison tables (per tag)
- `results/charts/` – main figures; `metric_dist/` (38), `token_curve/` (8) subfolders
- `results/token_level/` – per-token attribution records (per tag)
- `results/` – detection-analysis tables (AUC, multivariate, per-category, per tag)
- `<data_root>/logs/` – pipeline logs (train / eval / experiment)
- `<data_root>/data/` – the 6 datasets (tagged dirs for other ratios)
- `<data_root>/runs/<dataset>/` – per-sample metrics, tensorboard, LoRA weights
