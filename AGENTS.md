# AGENTS.md

LLM-noise-detection experiment: 7 noise types injected into dolly-15k (`datasets/{tag}/{dataset}/train.jsonl`),
Qwen2.5-3B-Instruct LoRA SFT with per-sample metric tracking, label-free noise-detection analysis.
Three tags trained so far: `dolly-ratio10` (10% noise, all 9 datasets incl. clean/mixed), `dolly-ratio5` (5% noise,
cross-validation of the dolly-ratio10 findings), and `oasst-wild` (natural noise from OASST2 via `wild_data.py`,
8.83% noise rate from human `quality` ratings rather than an injected perturbation — see `wild_data.py`'s
module docstring for the framing caveats this implies). All experiments/analysis run through a single root-level
codebase + `cli.py` entry point — there is no `src/`, `scripts/1_data/` etc. layer anymore (that layout
was replaced 2026-09-11/13, see `docs/experiment_log.md`).

## Code structure (root-level entry points + facades, three subpackages)

Implementation modules are grouped into three packages; root-level files that other unmovable
consumers (`cli.py`, `textsim.py`, `cleaning_loop.py`, the untracked `baselines.py`) import from
directly stay at the root as thin facades re-exporting from the packages, so those imports never
had to change:

- `datalib/` — `sample.py` (`Sample` dataclass, `Provider` Protocol), `data_io.py` (`Jsonl` read/write, `read()`/`write()`, `load_rows()` incl. `hf://...`, `validate()`), `data_split.py` (`split_holdout()`, `reindex()`, `split_fractions()`), `noise.py` (noise injection: `apply()`, `TRANSFORMS`, `NOISE_TYPES`, corruption helpers).
- `lora/` — `lora_train.py` (`LoRA` class, the real `hf-lora` backend) and `lora_internals.py` (its per-sample gradient/loss/cos-sim helper functions).
- `analysis/` — `metrics_common.py` (shared low-level primitives — `auc()`, `_cv_auc()`, `_if_auc()`, `_fit_transfer()`, `precision_at_k()`, `FULL_COVERAGE_FEATS`, `FEATURE_FAMILY`, `MEMO_FEATS`, `DIAG_COLS`/`TOKEN_COLS`), `metrics_table.py` (`build_table()` — assembles `per_sample_metrics.csv` from raw run metrics + labels + text_nn_sim, supports `max_epoch` truncation — plus `training_metrics()`/`token_metrics()`/`token_metrics_for_tag()`), `detect_unsupervised.py` (`unsupervised_metrics()`, `memorization_score()`, `early_detection_sweep()`, `precision_lift_table()`, `pooled_scorer_compare()` — compares `cleaning_loop.py`'s three scorers, reusing its `_score_*` helpers directly), `detect_transfer.py` (`cross_type_transfer()`, `cross_ratio_transfer()`, `transfer_to_mixed()`), `feature_diagnostics.py` (`feature_attribution()`, `feature_correlation()` — Spearman/PCA/VIF redundancy, `minimal_feature_set()` — greedy forward selection, `single_feature_ablation()` — leave-one-out, `feature_group_ablation()` — text/token/trajectory group ablation, `length_confound()` — wild-noise response-length confound control).

Root-level entry points and facades:

- `settings.py` — config loading (`load(path, tag)` → `Settings`), tag-based path helpers (`data_dir()`, `runs_dir()`, `results_dir()`).
- `data.py` — facade, re-exports from `datalib/`.
- `model.py` — facade: `Model` Protocol, `Mock` (CPU/interface testing) backend, and `create()` factory live here; the real `hf-lora` backend is re-exported from `lora/lora_train.py` (`LoRA` class). `LoRA.fit()` is decomposed into module-level phase functions (`_setup_run`, `_load_or_init_reference`, `_open_run_io`, `_run_epoch`, `_finalize_epoch`, `_finalize_run`, plus `_save_checkpoint`/`_flush_window`) sharing state via a `_TrainState` dataclass. It checkpoints to `run_dir/checkpoint/` at each epoch boundary (adapter + optimizer state + `ref_buf`/`v_buf`/`epoch_stats`, overwritten in place via write-temp-then-rename — latest epoch only, no history kept) and auto-resumes from it if present; a plain re-run of the same `train` command continues from the last completed epoch with no new flag needed, and `checkpoint/` is deleted once the run finishes normally.
- `train.py` — thin `Trainer` orchestration wrapper around `model.create()`.
- `evaluate.py` — downstream benchmark suite (mmlu/gsm8k/hellaswag/arc/bbh/truthfulqa/winogrande), resumable per-task.
- `textsim.py` — `text_nn_sim()`: TF-IDF nearest-neighbor cosine similarity per sample (a **data-level** feature, not a training-dynamics one — see the feature-attribution gotcha below).
- `analyze.py` — facade, re-exports from `analysis/`.
- `cleaning_loop.py` — `build()`: label-free closed-loop cleaning (targeted-drop + random-drop control training sets, full-corpus scale). Three scorers via `method=`: `iforest` (undirected outlier detection), `memo_signed` (fixed-sign hyper-typicality rule for memorized/hyper-typical noise like template/duplicate), `pooled` (max of three standardized legs — iforest z-score, memo_signed z-score, |z| of `text_nn_sim` — for when the noise composition is unknown, e.g. `mixed`).
- `wild_data.py` — builds a natural-noise dataset from OASST2, using human `quality` annotation ratings (not an injected perturbation) as the noise label; not wired through `cli.py` (run directly). Writes `datasets/{tag}/heldout.jsonl` (clean-only) and `datasets/{tag}/wild/train.jsonl`.
- `cli.py` — the **only** entry point for the `data`/`train`/`evaluate`/`analyze`/`clean` subcommands; `run.py` just calls `cli.main()`. `analyze --kind` choices: `features`/`training`/`token`/`unsupervised`/`transfer`/`cross_type`/`cross_ratio`/`precision_lift`/`memorization`/`early_unsupervised`/`early_memorization`/`feature_attribution`/`feature_correlation`/`minimal_feature_set`/`single_feature_ablation`/`transfer_to_mixed`/`feature_group_ablation`/`pooled_scorer_compare` (see Commands below).

**Placement principle**: `scripts/` (gitignored, `/*.sh` + `/scripts/` in `.gitignore`) is supposed to hold
only **orchestration** — what to run, in what order, with what waiting/polling logic. Any code that
implements actual experimental methodology (a detector, a scoring rule, a feature) belongs in a tracked
root-level `.py` module wired through `cli.py`, never in `scripts/`. This regressed twice already
(`cleaning_loop.py`/`early_detection_sweep()`, then `feature_ablation.py`/`feature_correlation.py`/
`minimal_feature_set.py`/`pooled_scorer_compare.py`/`single_feature_ablation.py`/`transfer_to_mixed.py`)
before being moved into `analyze.py` (as `feature_group_ablation()`, `feature_correlation()`,
`minimal_feature_set()`, `pooled_scorer_compare()`, `single_feature_ablation()`, `transfer_to_mixed()`)
and wired through `cli.py analyze --kind`, each verified byte-exact against the prior script's output
before the `scripts/` copies were deleted. `make_report_charts.py` is the one script deliberately left in
`scripts/` untracked (chart generation, not methodology; the PNGs it produces are tracked instead). Watch
for this regressing a third time.

## Layout & data flow

- **Large data lives in the repo dir but is gitignored where noted above** (LoRA weights, logs). Paths are tag-based:
  - `datasets/{tag}/{dataset}/train.jsonl` + shared `datasets/{tag}/heldout.jsonl`; `datasets/benchmarks/bbh/` (test + cot-prompts).
  - `runs/{tag}/{dataset}/{metrics,tb,lora}` — `metrics/per_sample.jsonl` (per-epoch loss/grad_norm/cos_sim), `metrics/diag_epoch*.jsonl` + `metrics/token_diag_epoch*.jsonl` (diagnostic-layer, ~900-row/dataset subsample only).
  - `results/{tag}/per_sample_metrics.csv` (built by `cli.py analyze --kind features`) + per-analysis CSVs (`unsupervised.csv`, `cross_type.csv`, `precision_lift.csv`, `memorization.csv`, `early_unsupervised.csv`, `early_memorization.csv`, `feature_attribution.csv`); `results/transfer_cross_ratio.csv` (cross-tag, at `results/` root); `results/eval/eval_{tag}_{dataset}.json`.
  - `datasets/{tag}/cleaning_loop/{dataset}/{train_targeted,train_random}.jsonl` + `metadata.json` — closed-loop cleaning outputs.
- `experiment_tag` defaults to `dolly-ratio10` in `config.yaml`; every `cli.py` subcommand takes `--tag`.
- GPU: **NVIDIA GeForce RTX 4090, ~49GB**. torch 2.8.0+cu128, transformers 5.13.1, peft 0.19.1. **Single GPU** — only one training/eval job can run at a time; queue others (see tmux convention below).

## Commands

```bash
python cli.py data --source /path/to/train.jsonl --tag dolly-ratio10
python cli.py train --tag dolly-ratio10 --dataset clean --model hf-lora     # ~3.5h, 5 epochs; --model mock for interface checks
python cli.py evaluate --tag dolly-ratio10 --dataset clean --model hf-lora [--force]   # resumable, skips done tasks
python cli.py clean --tag dolly-ratio10 --dataset garbled --budget 0.10     # label-free closed-loop cleaning (targeted + random control)
                     --method iforest|memo_signed|pooled              # default iforest; pooled is for unknown/mixed composition

# analyze --kind:
python cli.py analyze --kind features --tag dolly-ratio10                  # builds per_sample_metrics.csv
python cli.py analyze --kind training --tag dolly-ratio10                  # loss/grad_norm/cos_sim by epoch
python cli.py analyze --kind token --tag dolly-ratio10 [--dataset garbled]  # hard-token stats
python cli.py analyze --kind unsupervised --tag dolly-ratio10               # per-dataset IsolationForest/z-score, label-free
python cli.py analyze --kind memorization --tag dolly-ratio10                # signed hyper-typicality rule (duplicate/template)
python cli.py analyze --kind cross_type --tag dolly-ratio10                  # cross-noise-type detector transfer matrix
python cli.py analyze --kind cross_ratio --tags dolly-ratio10,dolly-ratio5         # cross-noise-ratio detector transfer matrix
python cli.py analyze --kind precision_lift --tag dolly-ratio10               # P@10% lift vs random, from unsupervised.csv
python cli.py analyze --kind early_unsupervised --tag dolly-ratio10           # detection AUC by training-epoch cutoff
python cli.py analyze --kind early_memorization --tag dolly-ratio10           # same, for memo_signed rule
python cli.py analyze --kind feature_attribution --tag dolly-ratio10          # permutation importance per noise type
python cli.py analyze --kind feature_correlation --tag dolly-ratio10          # Spearman/PCA/VIF redundancy (+ _pairs.csv)
python cli.py analyze --kind minimal_feature_set --tag dolly-ratio10          # greedy forward feature selection (rf + iforest routes)
python cli.py analyze --kind single_feature_ablation --tag dolly-ratio10      # leave-one-feature-out, both routes
python cli.py analyze --kind transfer_to_mixed --tag dolly-ratio10            # single-type detectors evaluated against mixed
python cli.py analyze --kind feature_group_ablation --tag dolly-ratio10       # text/token/trajectory feature-group ablation
python cli.py analyze --kind pooled_scorer_compare --tag dolly-ratio10 [--dataset mixed]  # compares cleaning_loop.py's 3 scorers
python cli.py analyze --kind transfer --tags dolly-ratio10,dolly-ratio5             # re-read a saved transfer CSV
```

- Long jobs (train/evaluate, or an orchestration shell script) run in their **own detached tmux session**
  named after the job (`tmux new-session -d -s <job> 'cmd 2>&1 | tee <log>'`), not windows inside one
  shared session. When a job must wait for the single GPU to free up, its orchestration script polls
  with `while tmux has-session -t <blocking_job> 2>/dev/null; do sleep 60; done` at the top, so it starts
  automatically the moment the blocking job's session ends — don't babysit this manually.
- **Never edit a shell script while its tmux session is live and already past the point that reads it**
  (bash reads scripts lazily) — kill and relaunch the session with the corrected script instead of trusting
  a live process to pick up an in-place edit.
- Training/evaluation skip already-completed datasets/tasks — safe to re-run.

## Reuse principles (avoid redundant work)

- Keep per-sample metrics jsonl — all derived features are computed from them in `analyze.py`, never re-train to regenerate analysis.
- `build_table(..., max_epoch=k)` reuses already-collected 5-epoch raw metrics to simulate "early" detection — no need to actually retrain with fewer epochs to answer that question.
- Cross-tag/cross-type comparisons (`cross_type_transfer`, `cross_ratio_transfer`) work directly off `per_sample_metrics.csv` — no retraining needed to test transfer.

## Gotchas (all bit us before)

**Data / feature coverage**
- `per_sample_metrics.csv` has two coverage tiers: diagnostic-layer + token-hard columns (`DIAG_COLS`/`TOKEN_COLS` in `analyze.py`, e.g. `hard_pos_jaccard`, `n_hard`) only populate a ~900-row/dataset diagnostic subsample; basic trajectory columns (`loss_*`, `grad_norm_*`, `cos_ref_*`) cover the full ~14.6k-row corpus. Any full-corpus operation (e.g. `cleaning_loop.py`) must auto-restrict to the full-coverage subset (`[c for c in numeric if sub[c].notna().all()]`), not the richer-but-narrower diagnostic tier.
- A single-epoch trajectory makes `*_std` columns NaN (`pandas.std()` ddof=1 undefined for n=1) — fill with 0 (zero spread is correct for one observation) rather than changing `_load_run_metrics`'s shared std logic.
- `hard_pos_jaccard` needs ≥2 epochs to compute a pairwise jaccard — NaN below that; any epoch-cutoff sweep must recompute its "core feature" list fresh per cutoff, not reuse one fixed list, or `dropna` silently wipes every row at low cutoffs.
- `text_nn_sim` (TF-IDF nearest-neighbor similarity, `textsim.py`) is a **data-level** feature, computed once from the raw text, not a training-dynamics signal — `feature_attribution.csv` shows it single-handedly dominates detection for `duplicate`/`unrelated` (an order of magnitude over the next feature), meaning the high AUC `cross_type_transfer()` reports for those two types is mostly not attributable to training dynamics at all. Don't cite those two types as evidence for the "training-dynamics detection" story without this caveat.

**Analysis methodology**
- Fit unsupervised scorers (IsolationForest etc.) **per dataset**, never pooled across datasets — a global fit lets extreme-magnitude datasets (garbled) shift the outlier baseline for low-variance ones (template), inverting the correct per-type result. Always `StandardScaler` before IsolationForest.
- "Direction reversal trap": memorized/hyper-typical noise (duplicate/template) looks **more normal**, not more anomalous, under generic distance-based unsupervised scoring (low loss, fast convergence) — a plain `|z|`/iforest score can show AUC < 0.5 for these types. The fix is a *signed* rule (`MEMO_FEATS` in `analyze.py::memorization_score()`, sign fixed a priori by the memorization hypothesis, never fit per dataset) rather than an unsigned outlier score. This reappears progressively across training time too: generic iforest AUC for template/duplicate/mixed/unrelated *decreases* over training epochs while `memo_signed` AUC for the same types *increases* — same mechanism, different axis.
- AUC and P@k% can disagree on which method is "best" — always check `precision_lift_table()` (P@10% vs random) before recommending a cleaning method by AUC alone; a high-AUC method can still have lift ≤ 1 at the review-budget slice that matters in practice.
- Permutation importance (`feature_attribution()`), not RF's built-in impurity-based `feature_importances_` — the latter is biased toward high-cardinality/continuous features regardless of true predictive value.
- When a noise type is near-perfectly detectable (template, AUC≈0.999), don't expect any single feature's permutation importance to be large — redundant/correlated features dilute each other's marginal contribution; small importances there mean "saturated detector", not "unreliable detector".

**Training**
- PyYAML parses `2e-4` as a **string** → write `0.0002` in `config.yaml`.
- `torch_dtype=` is deprecated in transformers 5.x → use `dtype=`.
- Qwen chat template returns all-zero `assistant_masks` → build labels via user-prefix token length (`add_generation_prompt=True` prefix trick).
- Truncation must keep the assistant response (truncate the user prefix), else 0 label tokens → NaN loss (0/0).
- `F.cross_entropy(..., reduction="none")` returns **0.0 at `-100` target positions** — for user-side loss (`user_loss`) use real next-token ids as targets.
- LoRA B is zero-initialized → A gradients are 0 early → `update_contrib` must use B-only offsets.
- Per-sample grad capture: snapshot `p.grad` before backward, subtract after (preallocated flat buffer) — don't reintroduce per-param python loops.
- Never run a training/eval command against a real `--dataset` without confirming the `--tag`; tag-scoped paths (`runs/{tag}/...`) are the only thing preventing cross-experiment overwrites.
- Data files contain ONLY training rows; held-out lives in `datasets/{tag}/heldout.jsonl`, never sliced out of `train.jsonl`.
- Interrupted training can be continued by just re-running the same `train` command — resume is auto-detected from `run_dir/checkpoint/` (epoch-boundary checkpoints; no separate flag). Never edit `run_dir/checkpoint/` by hand or delete it mid-run: on resume, `model.py::LoRA.fit()` trims `per_sample.jsonl`/`layer_norms.jsonl` back to the checkpointed epoch/step before appending, so a hand-edited or partially-written checkpoint will desync those files from `state.pt`.

**Evaluation**
- Flash-attn generation is broken with right padding → `tokenizer.padding_side = 'left'`.
- MC scoring caps at `SCORE_MAX_LEN=1024` in `evaluate.py`; `MAX_LEN=2048` is only for generation. Raising the score cap OOMs (logits `[B, 2048, 151936]`).
- GSM8K needs chat-template prompts + `max_new_tokens=512` + parse `####`/`\boxed{...}`/last-number fallback.
- Eval is resumable (per-task save); results dict is nested `{task: {acc, n, subjects/raw}}`.
- BBH data lives at `datasets/benchmarks/bbh/{test,cot-prompts}` (`_load_bbh` in `evaluate.py`).

**Analysis (model internals)**
- `PEFT from_pretrained` sets `requires_grad=False` → re-enable `lora_` params before any backward.
- Load base via `AutoModelForCausalLM` then wrap once with `PeftModel.from_pretrained`; never wrap an already-PeftModel result.
- `logits[0]` is `[L, V]`; shift with `[:-1]` to align with `labels[:, 1:]`.

**GitHub**
- Push failures with `gnutls_handshake`/TLS are transient — retry after a pause; local commits are safe.
