## 1. Experiment Overview

This project investigates a core question: **without ever using noise labels, can training dynamics captured during LoRA fine-tuning (loss trajectories, gradient norms, cosine similarity, etc.) reveal which training samples were injected as low-quality/anomalous?**

### 1.1 Data and Noise Types

Starting from dolly-15k as the base dataset, 7 noise types are injected, plus `clean` (uncontaminated baseline) and `mixed` (all 7 types combined) — 9 datasets total:

| Noise type | Description |
|---|---|
| `garbled` | Random character substitution/shuffling in the text |
| `template` | Response replaced with a highly repetitive templated sentence |
| `duplicate` | Exact copy of an existing sample |
| `unrelated` | Response mismatched with the question |
| `truncation` | Response cut short, information incomplete |
| `near_duplicate` | Existing sample with a light paraphrase |
| `keyword` | Key entities/terms in the sentence swapped out |

Two experiment tags correspond to two noise ratios:

- **`dolly-ratio10`**: 10% noise ratio. All 9 datasets fully trained, analyzed, and evaluated (training completed 2026-09-13 00:48, downstream evaluation completed 2026-09-14 02:20).
- **`dolly-ratio5`**: 5% noise ratio, used for cross-validation. Training on all 9 datasets completed 2026-09-14 07:46, the 5 analysis tables completed at 09:35. Downstream benchmark evaluation (tmux session `dolly-ratio5_eval`) finished the same day, fully complete (9/9 datasets).

### 1.2 Training Configuration

- Model: Qwen2.5-3B-Instruct, LoRA (r=32) fine-tuning, 5 epochs, per-sample gradient/loss tracking.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (~98GB), single GPU — all training/evaluation jobs are queued and run sequentially.
- After each dataset finishes training, `runs/{tag}/{dataset}/metrics/per_sample.jsonl` is written with per-sample, per-epoch loss/gradient-norm/cosine-similarity metrics; all downstream analysis is recomputed from these files without ever retraining.

### 1.3 Report Structure

The report follows a standard paper structure, in eight chapters, one file each:

| Ch. | Content | When to read it |
|---|---|---|
| 1 | Experiment overview | Now |
| [2](02-related-work.md) | Related work | To see which existing results this builds on, and the division of labour with them |
| [3](03-premises.md) | Experimental premises | **Before reading any number** — the label-free constraint, the single model scale, and the two sample populations |
| [4](04-design.md) | Experimental design | To reproduce the work, or to check how a particular number was computed |
| [5](05-theory.md) | Theoretical analysis | To see why it is designed this way, and which noise types are predicted detectable |
| [6](06-results.md) | Experimental process and results | Setup, data, analysis and results for 12 experiment groups (the body) |
| [7](07-conclusions.md) | Conclusions and future work | To go straight to the findings |
| [8](08-appendix.md) | Appendix | Metric definitions, collection timing, and all raw-data displays |

Chapter 6's 12 groups proceed as "measure capability, then boundaries, then real benefit": 6.2-6.9 examine each facet of detection capability (in-domain difficulty, cross-type and cross-ratio transfer, cleaning-precision lift, the direction-reversal trap, early detection, feature attribution, downstream harm); 6.10-6.12 synthesize and probe boundaries (best-method summary, two layers of feature ablation, and dismantling the "noise type is known" premise); 6.13 is the only experiment that actually removes samples and retrains, producing a real downstream comparison.

**Read Chapter 3 first.** Numbers in this report belong to two different populations (supervised vs. label-free, full coverage vs. diagnostic subsample), and conflating them yields wrong conclusions; Sections 3.1 and 3.5 explain both distinctions.
