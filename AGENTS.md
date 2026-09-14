# AGENTS.md

LLM-noise-detection experiment: 7 noise types injected into dolly-15k (`data/{tag}/{dataset}/train.jsonl`),
Qwen2.5-3B-Instruct LoRA SFT with per-sample metric tracking, label-free noise-detection analysis.
Two tags trained so far: `ratio10` (10% noise, all 9 datasets incl. clean/mixed) and `ratio5` (5% noise,
cross-validation of the ratio10 findings). All experiments/analysis run through a single root-level
codebase + `cli.py` entry point — there is no `src/`, `scripts/1_data/` etc. layer anymore (that layout
was replaced 2026-09-11/13, see `docs/experiment_log.md`).

## Code structure (root-level modules, no subpackages)

- `settings.py` — config loading (`load(path, tag)` → `Settings`), tag-based path helpers (`data_dir()`, `runs_dir()`, `results_dir()`).
- `data.py` — `Jsonl` read/write, `Provider.rows()` loaders (local file or `hf://...`), noise injection (`apply()`), holdout split.
- `model.py` — `mock` (CPU/interface testing) and `hf-lora` (real Qwen2.5-3B LoRA) backends; per-sample gradient/loss/cos-sim capture lives here.
- `train.py` — thin `Trainer` orchestration wrapper around `model.create()`.
- `evaluate.py` — downstream benchmark suite (mmlu/gsm8k/hellaswag/arc/bbh/truthfulqa/winogrande), resumable per-task.
- `textsim.py` — `text_nn_sim()`: TF-IDF nearest-neighbor cosine similarity per sample (a **data-level** feature, not a training-dynamics one — see the feature-attribution gotcha below).
- `analyze.py` — all post-hoc analysis: `build_table()` (assembles `per_sample_metrics.csv` from raw run metrics + labels + text_nn_sim, supports `max_epoch` truncation), `unsupervised_metrics()`, `memorization_score()`, `cross_type_transfer()`, `cross_ratio_transfer()`, `precision_lift_table()`, `early_detection_sweep()`, `feature_attribution()`.
- `cleaning_loop.py` — `build()`: label-free IsolationForest-based closed-loop cleaning (targeted-drop + random-drop control training sets, full-corpus scale).
- `cli.py` — the **only** entry point; subcommands `data` / `train` / `evaluate` / `analyze` / `clean`. `run.py` just calls `cli.main()`.

**Placement principle (do not regress this)**: `scripts/` (gitignored, `/*.sh` + `/scripts/` in `.gitignore`)
holds only **orchestration** — what to run, in what order, with what waiting/polling logic. Any code that
implements actual experimental methodology (a detector, a scoring rule, a feature) belongs in a tracked
root-level `.py` module wired through `cli.py`, never in `scripts/`. This was corrected once already
(`cleaning_loop.py`/`early_detection_sweep()` were briefly misplaced in `scripts/` before being moved).

## Layout & data flow

- **Large data lives in the repo dir but is gitignored where noted above** (LoRA weights, logs). Paths are tag-based:
  - `data/{tag}/{dataset}/train.jsonl` + shared `data/{tag}/heldout.jsonl`; `data/benchmarks/bbh/` (test + cot-prompts).
  - `runs/{tag}/{dataset}/{metrics,tb,lora}` — `metrics/per_sample.jsonl` (per-epoch loss/grad_norm/cos_sim), `metrics/diag_epoch*.jsonl` + `metrics/token_diag_epoch*.jsonl` (diagnostic-layer, ~900-row/dataset subsample only).
  - `results/{tag}/per_sample_metrics.csv` (built by `cli.py analyze --kind features`) + per-analysis CSVs (`unsupervised.csv`, `cross_type.csv`, `precision_lift.csv`, `memorization.csv`, `early_unsupervised.csv`, `early_memorization.csv`, `feature_attribution.csv`); `results/transfer_cross_ratio.csv` (cross-tag, at `results/` root); `results/eval/eval_{tag}_{dataset}.json`.
  - `data/{tag}/cleaning_loop/{dataset}/{train_targeted,train_random}.jsonl` + `metadata.json` — closed-loop cleaning outputs.
- `experiment_tag` defaults to `ratio10` in `config.yaml`; every `cli.py` subcommand takes `--tag`.
- GPU: **NVIDIA RTX PRO 6000 Blackwell Server Edition, ~98GB**, sm_120. torch 2.8.0+cu128, transformers 5.13.1, peft 0.19.1. **Single GPU** — only one training/eval job can run at a time; queue others (see tmux convention below).

## Commands

```bash
python cli.py data --source /path/to/train.jsonl --tag ratio10
python cli.py train --tag ratio10 --dataset clean --model hf-lora     # ~3.5h, 5 epochs; --model mock for interface checks
python cli.py evaluate --tag ratio10 --dataset clean --model hf-lora [--force]   # resumable, skips done tasks
python cli.py clean --tag ratio10 --dataset garbled --budget 0.10     # label-free closed-loop cleaning (targeted + random control)

# analyze --kind:
python cli.py analyze --kind features --tag ratio10                  # builds per_sample_metrics.csv
python cli.py analyze --kind training --tag ratio10                  # loss/grad_norm/cos_sim by epoch
python cli.py analyze --kind token --tag ratio10 [--dataset garbled]  # hard-token stats
python cli.py analyze --kind unsupervised --tag ratio10               # per-dataset IsolationForest/z-score, label-free
python cli.py analyze --kind memorization --tag ratio10                # signed hyper-typicality rule (duplicate/template)
python cli.py analyze --kind cross_type --tag ratio10                  # cross-noise-type detector transfer matrix
python cli.py analyze --kind cross_ratio --tags ratio10,ratio5         # cross-noise-ratio detector transfer matrix
python cli.py analyze --kind precision_lift --tag ratio10               # P@10% lift vs random, from unsupervised.csv
python cli.py analyze --kind early_unsupervised --tag ratio10           # detection AUC by training-epoch cutoff
python cli.py analyze --kind early_memorization --tag ratio10           # same, for memo_signed rule
python cli.py analyze --kind feature_attribution --tag ratio10          # permutation importance per noise type
python cli.py analyze --kind transfer --tags ratio10,ratio5             # re-read a saved transfer CSV
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
- Data files contain ONLY training rows; held-out lives in `data/{tag}/heldout.jsonl`, never sliced out of `train.jsonl`.

**Evaluation**
- Flash-attn generation is broken with right padding → `tokenizer.padding_side = 'left'`.
- MC scoring caps at `SCORE_MAX_LEN=1024` in `evaluate.py`; `MAX_LEN=2048` is only for generation. Raising the score cap OOMs (logits `[B, 2048, 151936]`).
- GSM8K needs chat-template prompts + `max_new_tokens=512` + parse `####`/`\boxed{...}`/last-number fallback.
- Eval is resumable (per-task save); results dict is nested `{task: {acc, n, subjects/raw}}`.
- BBH data lives at `data/benchmarks/bbh/{test,cot-prompts}` (`_load_bbh` in `evaluate.py`).

**Analysis (model internals)**
- `PEFT from_pretrained` sets `requires_grad=False` → re-enable `lora_` params before any backward.
- Load base via `AutoModelForCausalLM` then wrap once with `PeftModel.from_pretrained`; never wrap an already-PeftModel result.
- `logits[0]` is `[L, V]`; shift with `[:-1]` to align with `labels[:, 1:]`.

**GitHub**
- Push failures with `gnutls_handshake`/TLS are transient — retry after a pause; local commits are safe.
