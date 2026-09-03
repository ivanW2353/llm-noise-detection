# AGENTS.md

LLM-noise-detection experiment: 4 noise types injected into dolly-15k at 10% (tag `ratio10`), LoRA SFT with per-sample metric tracking, noise-detection analysis. All experiments already ran; new work = other ratios / detection improvements.

## Code structure (Phase 1 + Phase 2 refactoring completed 2026-09-02)

**Shared libraries** (`src/`, ~590 lines extracted from 15+ scripts):
- `src/config.py` — configuration loading, `get_tag()`, `get_results_dir()`
- `src/metrics.py` — feature constants (`METRIC_ORDER`, `TRAJ_METRICS`, `DATASETS`)
- `src/data.py` — data loading (`load_metrics()`, `filter_features()`, `get_noise_spec()`)
- `src/detection.py` — supervised detection (`univariate_auc()`, `fit_eval()`, `get_feature_importance()`)
- `src/scorers.py` — label-free scorers (`robust_z()`, `memo_scores()`, `unsupervised_scores()`)
- `src/eval_utils.py` — evaluation metrics (`precision_at_k()`, `safe_auc()`, `lift_at_k()`)

**Analysis scripts** (all migrated to use `src/` modules):
- `analyze_detection.py` — supervised detection (LR/RF), univariate AUCs, feature importance
- `analyze_unsupervised.py` — label-free scorers (IsolationForest, Mahalanobis, z-score)
- `analyze_memorization.py` — signed hyper-typicality rule for memorized noise (§3.13)
- `analyze_transfer.py` — cross-ratio and cross-type detector transfer
- `analyze_token_concentration.py` — true token_loss_top20 concentration (§3.12)
- `analyze_early_detection.py` — early-epoch detection (precision@k by epoch)
- `analyze_all_features.py` — feature exploration using ALL per-sample data

See `scripts/README.md` for the current stage-by-stage command map.

## Layout & data flow

- `scripts/` pipeline (run in order): `make_noise.py` → `train.py` → `evaluate.py` → `analyze_detection.py` / `analyze_token_level.py`; `recompute_diag.py` is a post-hoc fix script.
- Orchestrator: `workflows/run_full_experiment.sh` (one tagged pipeline: data → train → eval → analysis → report).
- **Large data lives in the repo dir but OUTSIDE git** at `data_root=/root/noisedetect` (gitignored: `data/`, `runs/`, `logs/`). Paths are tag-based:
  - `data/{tag}/{dataset}/train.jsonl` (no `train/` level) + shared `data/{tag}/heldout.jsonl`
  - `runs/{tag}/{dataset}/{metrics,tb,lora}`
  - `results/{tag}/` (per experiment: AUC/detection/tb CSVs, `ifd_{ds}.jsonl`, `token_level_{ds}.jsonl`, `eval_*` tables, plus the post-hoc `unsupervised_detection.csv` / `memorization_detection.csv` / `token_concentration.csv` / `feature_exploration.csv`), `results/transfer_cross_{ratio,type}.csv` (cross-tag, at `results/` root), `results/eval/` (per-model json + gitignored `eval_raw_*.jsonl`), `results/charts/` (png + `metric_dist/`, `token_curve/`).
- `experiment_tag` defaults to `ratio10` in `config.yaml`; every script takes `--tag`.
- GPU (since 2026-09-01): **NVIDIA RTX PRO 6000 Blackwell Server Edition, 96GB**, sm_120; previously RTX 5090 32GB. torch 2.8.0+cu128, transformers 5.13.1, peft 0.19.1, datasets 5.x, pandas 3.x. Measured train speed ~2.1 s/step (vs 2.6 s/step on the 5090); bs=1 training still latency-bound (GPU util 40-65%).

## Commands

```bash
python scripts/1_data/make_noise.py --ratio 0.05 --tag ratio05   # build datasets
python scripts/2_train/train.py --dataset clean --tag ratio10      # one run (~3.5h, 5 epochs)
python scripts/2_train/train.py --dataset garbled --smoke          # fast sanity check (~15s)
python scripts/2_train/evaluate.py --dataset clean [--force]       # resumable: skips done tasks
python scripts/3_analysis/analyze_detection.py --tag ratio10
python scripts/3_analysis/analyze_token_level.py                      # needs GPU; ~15 min
python scripts/2_train/recompute_diag.py                           # fixes user_loss in old runs
bash workflows/run_full_experiment.sh ratio05 garbled,duplicate,unrelated,keyword  # full pipeline

# CPU-only post-hoc analyses (no GPU, no retraining; all take --tags a,b,c)
python scripts/3_analysis/analyze_unsupervised.py        --tags ratio10,ratio05,extra10  # label-free scorers vs supervised ceiling
python scripts/3_analysis/analyze_memorization.py  --tags ratio10,ratio05,extra10  # signed hyper-typicality rule (§3.13)
python scripts/3_analysis/analyze_transfer.py            --tags ratio10,ratio05,extra10  # cross-ratio + cross-type transfer
python scripts/3_analysis/analyze_token_concentration.py --tags ratio10,ratio05,extra10  # true token_loss_top20 (§3.12)
python scripts/3_analysis/analyze_all_features.py        --tag ratio10                   # full-feature exploration

# 6. external validity: natural-data signal validation (GPU ~2h, uses clean model)
python scripts/3_analysis/natural_signal_validation.py --model clean --n 20000  # -> results/natural_validation.csv (§3.14)
```

- Tests are lightweight entry-point and scorer checks: `python tests/test_refactored_scripts.py` and `python tests/test_scorers.py`.
- Long jobs run in tmux windows inside the main session `noisedetect`
  (window 0 `chat`, window 1+ per job), never plain nohup:
  `tmux new-window -t noisedetect -n <job> 'cmd 2>&1 | tee <log>'`.
  Don't create separate tmux sessions for background jobs.
- **Never edit `workflows/run_full_experiment.sh` while a tmux pipeline is executing it** — bash reads the script lazily; overwriting the file mid-run feeds torn lines to the running shell (caused a silent pipeline death: `ntinue: command not found` after training finished, eval never started). Edit scripts only between runs.
- `workflows/run_full_experiment.sh` = **one experiment, one command**: build → train → eval → analysis → report, then prints `Experiment complete` (run `compare_ratios.py` separately when needed). Training and evaluation skip completed work; analysis can be re-run to refresh outputs. Re-running the command therefore resumes interrupted training/evaluation safely.
- Analysis scripts auto-detect the trained datasets from `runs/{tag}/*/summary.json` (exclude clean for token-level) — no `--datasets` needed; `compute_ifd.py` now takes `--tag` (no more config-override hack).

## Reuse principles (avoid redundant work)

- **clean run is ratio-independent** (same seed/order) — it can be copied manually
  from `runs/ratio10/clean` when starting another ratio; base eval is tag-independent.
- **Identical datasets across experiments need no retraining**: e.g. the 10%
  garbled/duplicate/unrelated/keyword/mixed (4-way) runs of `ratio10` are
  byte-identical to any future 10% experiment without `--with-extra` — only
  train what's new (the extra10 experiment trains only template/truncation/
  near_duplicate/mixed, ~4 runs instead of 8).
- Training skips datasets whose `summary.json` exists, and evaluation skips models whose results are complete (7 tasks) — safe to re-run the workflow.
- Keep per-sample metrics jsonl — all derived features are computed from them
  in analysis, never re-train to regenerate analysis.

## Gotchas (all bit us before)

**Data / datasets (5.x)**
- `ds[:5]` returns a column-dict, NOT rows → iterate `ds.select(range(5))`.
- HellaSwag `label` is a **string** ("3") → `int(r["label"])`; Winogrande `answer` is a string too.
- MMLU answer is int; arc `answerKey` is a letter string matched against `choices["label"]`.
- TruthfulQA `multiple_choice` has no `category`; map via the `generation` split by question.

**Training**
- PyYAML parses `2e-4` as a **string** → write `0.0002` in config.
- `torch_dtype=` is deprecated in transformers 5.x → use `dtype=`.
- Qwen chat template returns all-zero `assistant_masks` → build labels by user-prefix token length (`add_generation_prompt=True` prefix trick).
- Truncation must keep the assistant response (truncate the user prefix), else 0 label tokens → NaN loss (0/0).
- `F.cross_entropy(..., reduction="none")` returns **0.0 at `-100` target positions** — for user-side loss use real next-token ids as targets (that was the `user_loss` bug).
- LoRA B is zero-initialized → A gradients are 0 early → `update_contrib` must use B-only offsets (element-wise `grad/sqrt(v)` explodes otherwise).
- Per-sample grad capture: snapshot `p.grad` before backward, subtract after (flat buffer `fill_flat`, preallocated — don't re-introduce per-param python loops, they cost ~14k kernel launches/step).
- **Never run `--smoke` without a tag on a real dataset** — smoke writes to `runs/{tag}/_smoke/` now; older smoke runs destroyed a real `runs/clean` (a previous incident). `rm -rf runs/{dataset}` only when nothing is training on it.
- The data files contain ONLY training rows; do NOT slice `rows[400:]` in train.py (held-out lives in `heldout.jsonl`).

**Evaluation**
- Flash-attn generation is broken with right padding → `tokenizer.padding_side = "left"`.
- MC scoring caps at `SCORE_MAX_LEN=1024`; `MAX_LEN=2048` is only for generation. Raising the score cap OOMs (logits `[B, 2048, 151936]`).
- GSM8K needs chat-template prompts + `max_new_tokens=512` + parse `####`/`\boxed{...}`/last-number fallback.
- `nlls` from `score_options` are torch tensors → `float()` before `round()`/math.
- Eval is resumable (per-task save); results dict is nested `{task: {acc, n, subjects/raw}}` — analysis must read `r["mmlu"]["subjects"]`, not `r["subjects"]`.
- **BBH data lives at `data/{data_root}/bbh/{test,cot-prompts}`** (re-downloaded from `suzgunmirac/BIG-Bench-Hard` on 2026-09-01 after the server migration lost `/root/autodl-tmp/less`; keep `BBH_DIR` in evaluate.py pointing there). Verify data integrity via fingerprints: causal_judgement 0.55, navigate 0.50 across all models.
- **Two div-by-zero bugs in evaluate.py fixed 2026-09-01**: ETA prints at s=0 divided by rate=0 (`max(rate, 1e-9)` guards); a crash on the 5th task (bbh) means BBH_DIR data is missing (eval looks "stuck" otherwise).

**Analysis**
- `PEFT from_pretrained` sets `requires_grad=False` → re-enable `lora_` params before any backward (token-level analysis).
- Load base via `AutoModelForCausalLM` then wrap once with `PeftModel.from_pretrained`; never wrap a `build_model()` result (already PeftModel → nested adapter bug).
- `logits[0]` is `[L, V]`; shift with `[:-1]` to align with `labels[:, 1:]`.
- `torch.cuda.memory_allocated()` (TB `gpu_mem_GB`) ≠ nvidia-smi total: cached segments stay reserved (~28GB vs 8GB) — normal, not a leak.
- Host load fluctuates (22→62); training ~160ms/sample, eval ~75-90 min/model. When measuring lines/sec: 16 lines = 1 optimizer step = 16 samples.

**GitHub**
- Push failures with `gnutls_handshake`/TLS are transient — retry after a pause; local commits are safe.
