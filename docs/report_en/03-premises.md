## 3. Experimental Premises

This section collects the premises the whole experiment rests on. Some are constraints imposed deliberately (label-free scoring); others are boundaries imposed by circumstance (a single base model). Both directly bound how far the conclusions carry. **Read this section before reading any individual number**, or it is easy to mistake "holds under these premises" for "holds generally."

### 3.1 Hard constraint: noise labels are never used

**Primary impact: Thread 3 (detectability)** — this constraint directly decides whether the question "can noise be separated label-free" is even meaningful; it also draws the line, for Thread 2, that supervised AUC is a signal ceiling, not something production can reach.

This is the first premise behind every method, and the main dividing line from the noisy-label literature (Section 4.3): **no scorer sees `noise_type` at any point**. Labels appear in exactly two places — injecting noise when building the datasets, and computing AUC / precision when evaluating. The detection side never sees them.

The constraint rules out a family of practices that are common in academic settings and unavailable in production:

- No calibrating the scoring direction against a small set of known-clean samples (exactly the difficulty discussed in Section 5.3).
- No selecting feature subsets or tuning scorer hyperparameters by label — [06b](06b-feature-signatures.md) Section 6b.8's single-feature ablation shows that choosing features per noise type would leave another 2-5 points of label-free AUC on the table, but that headroom is unreachable in production.
- No using labels to decide which scorer to apply. [06c](06c-detectability.md) Section 6c.5.2 quantifies what this costs: on template, the scorer that points the wrong way leaves downstream performance *worse* than not cleaning at all.

The sole exception is the **supervised AUC in [06b](06b-feature-signatures.md) Sections 6b.1 and 6b.2**, which deliberately trains a random forest on the labels. Those sections are not a production proposal but a **reference ceiling for the signal**: if even a supervised model cannot separate a noise type, a label-free one certainly cannot. Every supervised number in the report is explicitly marked as such.

### 3.2 The noise is programmatically injected, not wild

**Primary impact: Thread 2, Thread 3** — this premise bounds how far the detector's generalization can be trusted: every mechanistic finding Thread 2 reports (the direction-reversal trap, feature attribution) and every label-free precision figure Thread 3 reports are validated on this known, injected distribution. It does not directly constrain Thread 1, since measuring downstream harm only depends on the fact that a bad row is present in training — not on whether that row came from injection or the wild.

All 7 noise types are injected by `data.py::apply` under fixed rules, so the perturbation is known and consistent. This buys one key advantage and imposes one key limitation:

- **Advantage**: exact ground truth, so detection difficulty can be computed per noise type instead of collapsing into one aggregate number. Chapter 6's per-type reporting depends on this.
- **Limitation**: genuinely wild noise (mis-pasted content, encoding errors, cross-language contamination, machine-translation artifacts, a model's own hallucinated output) may differ from these 7 in both text distribution and training dynamics. The "unseen types" in [06c](06c-detectability.md) Section 6c.4 are still one of these 7, merely unseen by the detector pool — **that section's 0.416-0.935 range cannot be extrapolated to wild data directly**.

Two injection details also affect how numbers read, detailed in Section 4.2: `duplicate` is the only type that changes the dataset's row count (it appends copies rather than replacing in place, so its noise fraction is 9.1% rather than 10%), and `mixed`'s per-subtype counts fluctuate between 191 and 211.

### 3.3 A single base model, a single scale; the task domain has grown from one to two

Every experiment uses Qwen2.5-3B-Instruct + LoRA (r=32) for 5 epochs. The base dataset used to be dolly-15k (instruction-following) alone; `triviaqa-ratio10` (a single factual-QA task, see [06a](06a-harm-ranking.md) Section 6a.3) is now a second task domain, with four noise types in total (`clean`/`wrong_answer`/`refusal`/`confusable_wrong`), all now complete (see Section 3.6). No cross-validation has been run over model scale, LoRA rank, or epoch count.

**The hardware is not uniform: the GPU changed partway through the project.** This report previously claimed a "single RTX PRO 6000"; that premise no longer holds for the later experiments, and the actual assignment is recorded here:

- `dolly-ratio10`, `dolly-ratio5`, `oasst-wild`: **NVIDIA RTX PRO 6000 Blackwell Server Edition (~98GB)**.
- All four `triviaqa-ratio10` datasets: **NVIDIA GeForce RTX 4090 (~49GB)**. The card changed on 2026-09-20, before this tag's earliest training artifact (`clean`'s epoch 0 was written at 09-21 06:24), so the entire QA task domain ran on the 4090 and **no single training run spans two cards**.
- `refusal` was **interrupted and resumed** (`logs/train_refusal.log` records `resuming from epoch 1, step 8624`): epoch 0 finished at 09-24 01:46, the container then crashed and restarted (container start time 09-24 09:22), and the orchestrator resumed epochs 1-4 from the checkpoint. This is **not** a card change — epoch 0 itself took 6.37 hours (19:24:14 → 01:46:20), matching the roughly 6.4 hours of each of epochs 1-4, and the intervening 14.4-hour gap is downtime rather than compute. Only one consequence matters: its `summary.json` `seconds` field (92504.7 s) **covers only the 4 resumed epochs**, not the full wall-clock time of the run.

What the hardware change means for **cross-task-domain comparability**:

- **Engineering figures** (timing, VRAM footprint, GPU utilisation) are directly affected; every place that cites one of those (Sections 3.4, 7.3, and 8.5) now states which card produced it.
- **Feature values** are post-processed from the persisted `per_sample.jsonl` and are determined by the model, the data, and the random seed; bf16 is natively supported on both cards, the same `micro_batch=1`+`grad_accum=16` configuration was used throughout, and no OOM or obvious training instability was observed. Because the card boundary coincides exactly with the task-domain boundary (dolly/oasst on the old card, triviaqa on the new one), **hardware differences and task-domain differences cannot be fully separated in this report** — worth keeping in mind when reading the cross-task-domain comparison in Section 6a.3. That section's conclusion rests on the *shape* of the harm distribution across benchmarks rather than on absolute values, so it is only weakly exposed to the hardware.

### 3.4 `micro_batch=1` is a methodological precondition, not a tuning choice

Attributing a gradient to a single sample requires that sample to constitute one forward-backward pass on its own, hence `micro_batch=1` + `grad_accum=16` (effective batch 16). This is not the outcome of tuning; it is **the precondition for obtaining the `grad_norm` / `cos_sim_ref` / `update_contrib` features at all** (Section 4.3).

The cost is training speed: roughly 3.1 hours per dolly dataset on the RTX PRO 6000, about 27.8 hours for 9 datasets run serially (summed from each `summary.json`'s `seconds` field, consistent with the 187-minute measurement for `keyword` in Section 8.5). [Correction: this section previously read "about 1.5 hours / 13.5 hours", which was off by a factor of two against Section 8.5's own measurement; it has been corrected to the measured values.] On the 4090, triviaqa took about 6.4 hours per epoch, but that gap comes **mainly from dataset size** (137,984 rows, 9.4× dolly's 14,611); per-sample throughput differs by under 15% between the two, so it should not be read as "the card made it 4× slower". Any attempt to reproduce this method has to accept that cost up front. Using only loss-family features (which need no per-sample gradients) allows a fall back to conventional batch training, at the price of losing every gradient-direction signal.

### 3.5 Two populations that must be read separately: full coverage vs. diagnostic subsample

**Primary impact: Thread 3** (this is the gap between reported AUC and what production can actually reach, and it directly bounds how much precision a real cleaning pipeline can deliver); it also affects how the absolute value of a Thread 2 number should be read (the ranking is stable across both populations, but the raw AUC is not).

This is the easiest trap in the whole report, and Sections 4.3 and 4.5 restate it:

| Population | Features | Sample size | Who uses it |
|---|---|---|---|
| **Full coverage** | 19 | All training samples (14,611+) | Closed-loop cleaning ([06c](06c-detectability.md) Section 6c.5), i.e. what production can actually reach |
| **Diagnostic subsample** | 37 | ~900-1200 rows per dataset | Most AUCs in [06b](06b-feature-signatures.md)/[06c](06c-detectability.md), most sections |

The gap comes from `diag_subsample=8`: token-level diagnostics and `cos_global_*` are collected for only 1 sample in 8, about 12.5% coverage. Most analyses use `dropna` requiring every feature to be non-null, so they effectively run on that ~12.5%.

**The asymmetry matters**: [06c](06c-detectability.md) Section 6c.3's ablation shows token-level diagnostics happen to be near_duplicate's only effective signal source, so **that type's reported AUC is optimistic relative to what production can reach**. For garbled and template the signal lives mostly in full-coverage features, so the two populations agree. Before reading any AUC as "what production would achieve," confirm which population it came from.

### 3.6 What is complete and what is not

The conclusions are not uniformly mature; the split is:

- **Both noise ratios are complete**: `dolly-ratio10` (10%) and `dolly-ratio5` (5%), 9 datasets each, with training, analysis, and downstream evaluation all finished.
- **Closed-loop retraining covers three cases**: `garbled` and `template` (both single-type, [06c](06c-detectability.md) Sections 6c.5.1/6c.5.2) and `mixed` (the `pooled` scorer, Section 6c.5.3) — training and evaluation are done for all three.
- **The wild-noise closed loop is also done**: `oasst-wild` ([06b](06b-feature-signatures.md) Sections 6b.9/6b.10, see below), with `cleaning_loop_targeted_wild`/`cleaning_loop_random_wild` training and all 7 downstream benchmarks complete.
- **The second task domain is complete**: `triviaqa-ratio10` (QA, [06a](06a-harm-ranking.md) Section 6a.3), with training, the 7 downstream benchmarks, and the dedicated `qa_correctness` evaluation all finished for its four datasets `clean`/`wrong_answer`/`refusal`/`confusable_wrong`. This tag has only a feature table and a per-sample metrics table; it does **not** have dolly's full suite of cross-type / transfer / closed-loop-cleaning analyses, so it serves to answer one cluster of questions: how task homogeneity and noise content jointly determine downstream harm.
- **Not done**: closed loops for near_duplicate / keyword (detection is weak for both), and multi-seed repeats (so the downstream gains in [06c](06c-detectability.md) Section 6c.5 and the oasst-wild closed loop carry no variance estimate).
