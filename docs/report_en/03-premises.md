## 3. Experimental Premises

This section collects the premises the whole experiment rests on. Some are constraints imposed deliberately (label-free scoring); others are boundaries imposed by circumstance (a single base model). Both directly bound how far the conclusions carry. **Read this section before reading any individual number**, or it is easy to mistake "holds under these premises" for "holds generally."

### 3.1 Hard constraint: noise labels are never used

This is the first premise behind every method, and the main dividing line from the noisy-label literature (Section 4.3): **no scorer sees `noise_type` at any point**. Labels appear in exactly two places — injecting noise when building the datasets, and computing AUC / precision when evaluating. The detection side never sees them.

The constraint rules out a family of practices that are common in academic settings and unavailable in production:

- No calibrating the scoring direction against a small set of known-clean samples (exactly the difficulty discussed in Section 5.3).
- No selecting feature subsets or tuning scorer hyperparameters by label — Section 6.11's single-feature ablation shows that choosing features per noise type would leave another 2-5 points of label-free AUC on the table, but that headroom is unreachable in production.
- No using labels to decide which scorer to apply. Section 6.13 quantifies what this costs: on template, the scorer that points the wrong way leaves downstream performance *worse* than not cleaning at all.

The sole exception is the **supervised AUC in Sections 6.2 and 6.3**, which deliberately trains a random forest on the labels. Those sections are not a production proposal but a **reference ceiling for the signal**: if even a supervised model cannot separate a noise type, a label-free one certainly cannot. Every supervised number in the report is explicitly marked as such.

### 3.2 The noise is programmatically injected, not wild

All 7 noise types are injected by `data.py::apply` under fixed rules, so the perturbation is known and consistent. This buys one key advantage and imposes one key limitation:

- **Advantage**: exact ground truth, so detection difficulty can be computed per noise type instead of collapsing into one aggregate number. Chapter 6's per-type reporting depends on this.
- **Limitation**: genuinely wild noise (mis-pasted content, encoding errors, cross-language contamination, machine-translation artifacts, a model's own hallucinated output) may differ from these 7 in both text distribution and training dynamics. The "unseen types" in Section 6.12 are still one of these 7, merely unseen by the detector pool — **that section's 0.416-0.935 range cannot be extrapolated to wild data directly**.

Two injection details also affect how numbers read, detailed in Section 4.2: `duplicate` is the only type that changes the dataset's row count (it appends copies rather than replacing in place, so its noise fraction is 9.1% rather than 10%), and `mixed`'s per-subtype counts fluctuate between 191 and 211.

### 3.3 A single base model, a single scale, a single task domain

Every experiment uses Qwen2.5-3B-Instruct + LoRA (r=32) for 5 epochs, with dolly-15k as the base dataset. No cross-validation was run over model scale, LoRA rank, epoch count, or base dataset.

This premise bears on two classes of conclusion very differently:

- **Detection-capability conclusions** (Sections 6.2-6.12) are relatively robust. They rest on the mechanism that noisy and clean samples separate in training dynamics, and the ranking is consistent across the 10% and 5% noise ratios (Section 6.4) — ratio cross-validation supplies one layer of robustness evidence, though it is no substitute for cross-validating over model scale.
- **Downstream-harm conclusions** (Sections 6.9 and 6.13) are affected far more. The 7-benchmark averages cluster in a narrow 0.42-0.44 band; the downstream harm of noise is small to begin with. Larger models, longer training, or an evaluation set more sensitive to noise could plausibly widen these differences. **The conclusion "this noise type does nearly zero downstream harm" holds only at the current scale.**

### 3.4 `micro_batch=1` is a methodological precondition, not a tuning choice

Attributing a gradient to a single sample requires that sample to constitute one forward-backward pass on its own, hence `micro_batch=1` + `grad_accum=16` (effective batch 16). This is not the outcome of tuning; it is **the precondition for obtaining the `grad_norm` / `cos_sim_ref` / `update_contrib` features at all** (Section 4.3).

The cost is training speed: roughly 1.5 hours per dataset, about 13.5 hours for 9 datasets run serially (Section 8.5 has the measured breakdown). Any attempt to reproduce this method has to accept that cost up front. Using only loss-family features (which need no per-sample gradients) allows a fall back to conventional batch training, at the price of losing every gradient-direction signal.

### 3.5 Two populations that must be read separately: full coverage vs. diagnostic subsample

This is the easiest trap in the whole report, and Sections 4.3 and 4.5 restate it:

| Population | Features | Sample size | Who uses it |
|---|---|---|---|
| **Full coverage** | 20 | All training samples (14,611+) | Closed-loop cleaning (Section 6.13), i.e. what production can actually reach |
| **Diagnostic subsample** | 37 | ~900-1200 rows per dataset | Most AUCs in Sections 6.2-6.12 |

The gap comes from `diag_subsample=8`: token-level diagnostics and `cos_global_*` are collected for only 1 sample in 8, about 12.5% coverage. Most analyses use `dropna` requiring every feature to be non-null, so they effectively run on that ~12.5%.

**The asymmetry matters**: Section 6.11's ablation shows token-level diagnostics happen to be near_duplicate's only effective signal source, so **that type's reported AUC is optimistic relative to what production can reach**. For garbled and template the signal lives mostly in full-coverage features, so the two populations agree. Before reading any AUC as "what production would achieve," confirm which population it came from.

### 3.6 What is complete and what is not

The conclusions are not uniformly mature; the split is:

- **Both noise ratios are complete**: `ratio10` (10%) and `ratio5` (5%), 9 datasets each, with training, analysis, and downstream evaluation all finished.
- **Closed-loop retraining covers two types only**: `garbled` and `template` (Section 6.13). Scoring for the `mixed` `pooled` closed loop is done and its retraining is in progress; Section 7.3 lists it as a next step.
- **Not done**: closed loops for near_duplicate / keyword (detection is weak for both), multi-seed repeats (so the downstream gains in Section 6.13 carry no variance estimate), and validation against wild noise.
