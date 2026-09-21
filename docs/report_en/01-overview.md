## 1. Experiment Overview

This project is organized around three parallel research threads. They build on one another yet remain independent — missing any one of them leaves the bigger question ("is noise detection worth doing in practice?") unanswered:

1. **Thread 1 (harm ranking)**: which noise types actually hurt training the most? Even a perfectly detectable noise type is not worth cleaning if it is nearly harmless downstream.
2. **Thread 2 (feature signatures)**: what do the metric signatures of each noise type look like — how do training dynamics and text features vary by type?
3. **Thread 3 (detectability)**: given these signatures, can noisy samples be separated out **without ever using noise labels**, how accurate is that separation, and does cleaning actually improve training?

The three threads are not simply sequential. Thread 2 is the methodological foundation for Thread 3 — without first knowing what the signatures look like, there is no basis for designing a label-free scorer. But Thread 2's central mechanistic finding (the "direction-reversal trap" in Sections 5.2/6.6) shows precisely why Thread 2 and Thread 3 must be discussed separately: high *supervised* detection accuracy (e.g. `template`'s in-domain AUC of 0.999) does not imply an equally accurate *label-free* scheme exists (the same type scores only 0.522 under `iforest`). Thread 1 is independent of both — no matter how detectable or how mature the label-free scheme, cleaning a noise type that is essentially harmless downstream (e.g. `garbled`'s zero downstream gain after closed-loop cleaning, Section 6.13.1) delivers limited real value.

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

The report is split into eight chapters (Chapter 6 is physically split into four files — the index plus 06a/06b/06c — see below). The table below is organized as "chapter × primary thread served": `●` means the chapter is primarily about that thread, `○` means it touches on it but is not the focus, and a blank cell means it is essentially unrelated:

| Ch. | Content | Thread 1<br>harm ranking | Thread 2<br>feature signatures | Thread 3<br>detectability |
|---|---|:---:|:---:|:---:|
| 1 | Experiment overview (now) | ● | ● | ● |
| [2](02-related-work.md) | Related work — what this method builds on, and the division of labour with it | ○ | ● | ● |
| [3](03-premises.md) | Experimental premises — **read before any number**: the label-free constraint, the single model scale, the two sample populations | ○ | ● | ● |
| [4](04-design.md) | Experimental design — to reproduce the work or check how a number was computed | ○ | ● | ● |
| [5](05-theory.md) | Theoretical analysis — why it is designed this way, and which noise types are predicted detectable | ○ | ● | ● |
| [6](06-results.md) | Experimental process and results: index page (routes to 06a/06b/06c) | ○ | ○ | ○ |
| [6a](06a-harm-ranking.md) | Downstream harm ranking — which noise type hurts training the most | ● | | ○ |
| [6b](06b-feature-signatures.md) | Feature signatures of each noise type — what training dynamics / text features look like (the largest part of the report) | | ● | ○ |
| [6c](06c-detectability.md) | Label-free detectability — detection accuracy, ablations, and real downstream gains from closed-loop cleaning | ○ | | ● |
| [7](07-conclusions.md) | Conclusions and future work — to go straight to the findings | ● | ● | ● |
| [8](08-appendix.md) | Appendix — metric definitions, collection timing, all raw-data displays | | ● | ○ |

Internally, the three Chapter-6 sub-files still follow "measure capability, then boundaries, then real benefit": 06b examines the detection difficulty, transferability, direction-reversal trap, and feature attribution of each noise type; 06c builds on that with synthesis, boundary conditions (unknown types), and real closed-loop cleaning gains; 06a is independent of the first two and answers "is it worth cleaning at all" directly from downstream benchmarks. Sections 6.13/6.16 in 06c are the only experiments in the whole report that actually remove samples and retrain, producing a real downstream comparison.

**Read Chapter 3 first.** Numbers in this report belong to two different populations (supervised vs. label-free, full coverage vs. diagnostic subsample), and conflating them yields wrong conclusions; Sections 3.1 and 3.5 explain both distinctions.
