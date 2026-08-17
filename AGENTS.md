# AGENTS.md

LLM-noise-detection experiment: 4 noise types injected into dolly-15k at 10% (tag `ratio10`), LoRA SFT with per-sample metric tracking, noise-detection analysis. All experiments already ran; new work = other ratios / detection improvements.

## Layout & data flow

- `scripts/` pipeline (run in order): `make_noise.py` → `train.py` → `evaluate.py` → `analyze_detection.py` / `analyze_token_level.py`; `recompute_diag.py` is a post-hoc fix script.
- Orchestrators: `run_all.sh` (6 trainings), `run_all_eval.sh` (7 models), `run_experiment.sh` (one-command full pipeline for a new ratio).
- **Large data lives OUTSIDE the repo** at `data_root=/root/autodl-tmp/noisedetect`. Paths are tag-based:
  - `data/{tag}/{dataset}/train.jsonl` (no `train/` level) + shared `data/{tag}/heldout.jsonl`
  - `runs/{tag}/{dataset}/{metrics,tb,lora}`
  - `results/eval/` (per-model json + `eval_raw_*.jsonl` per-question), `results/charts/` (png), top-level CSVs (detection tables).
- `experiment_tag` defaults to `ratio10` in `config.yaml`; every script takes `--tag`.
- GPU: RTX 5090 32GB, torch 2.8+cu128, transformers 5.13, peft 0.19, datasets 5.x, pandas 3.x.

## Commands

```bash
python scripts/make_noise.py --ratio 0.05 --tag ratio05   # build datasets
python scripts/train.py --dataset clean --tag ratio10      # one run (~3.5h, 5 epochs)
python scripts/train.py --dataset garbled --smoke          # fast sanity check (~15s)
python scripts/evaluate.py --dataset clean [--force]       # resumable: skips done tasks
python scripts/analyze_detection.py --tag ratio10
python scripts/analyze_token_level.py                      # needs GPU; ~15 min
python scripts/recompute_diag.py                           # fixes user_loss in old runs
bash run_experiment.sh --ratio 0.05 --tag ratio05 --reuse-clean  # full pipeline; reuses clean run
```

- No test suite; verification = `--smoke` flags + CPU-only loader checks.
- Long jobs run in tmux windows inside the main session `noisedetect`
  (window 0 `chat`, window 1+ per job), never plain nohup:
  `tmux new-window -t noisedetect -n <job> 'cmd 2>&1 | tee <log>'`.
  Don't create separate tmux sessions for background jobs.
- **Never edit `run_experiment.sh` / `run_all*.sh` while a tmux pipeline is executing it** — bash reads the script lazily; overwriting the file mid-run feeds torn lines to the running shell (caused a silent pipeline death: `ntinue: command not found` after training finished, eval never started). Edit scripts only between runs.
- `run_experiment.sh` chaining: build → train → eval → analysis → prints `ALL DONE` (watch for it in the log; then run `compare_ratios.py` + git push manually).

## Reuse principles (avoid redundant work)

- **clean run is ratio-independent** (same seed/order) — reuse it across ratio
  experiments: `--reuse-clean` copies `runs/ratio10/clean` (saves ~3.5h) and
  reuses its eval results (saves ~1.5h); base eval is tag-independent too.
- **Identical datasets across experiments need no retraining**: e.g. the 10%
  garbled/duplicate/unrelated/keyword/mixed (4-way) runs of `ratio10` are
  byte-identical to any future 10% experiment without `--with-extra` — only
  train what's new (the extra10 experiment trains only template/truncation/
  near_duplicate/mixed, ~4 runs instead of 8).
- **run_experiment.sh auto-skips** datasets whose `summary.json` exists and
  eval models whose results are complete (7 tasks) — safe to re-run anytime.
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

**Analysis**
- `PEFT from_pretrained` sets `requires_grad=False` → re-enable `lora_` params before any backward (token-level analysis).
- Load base via `AutoModelForCausalLM` then wrap once with `PeftModel.from_pretrained`; never wrap a `build_model()` result (already PeftModel → nested adapter bug).
- `logits[0]` is `[L, V]`; shift with `[:-1]` to align with `labels[:, 1:]`.
- `torch.cuda.memory_allocated()` (TB `gpu_mem_GB`) ≠ nvidia-smi total: cached segments stay reserved (~28GB vs 8GB) — normal, not a leak.
- Host load fluctuates (22→62); training ~160ms/sample, eval ~75-90 min/model. When measuring lines/sec: 16 lines = 1 optimizer step = 16 samples.

**GitHub**
- Push failures with `gnutls_handshake`/TLS are transient — retry after a pause; local commits are safe.
