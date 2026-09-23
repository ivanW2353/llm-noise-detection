## 6a. Downstream Harm Ranking: Which Noise Type Hurts Training Outcomes the Most

This file covers **Thread 1** of the report's three research threads: rather than asking "can this noise be detected," it asks **whether model performance on downstream tasks actually gets worse after noise injection, by how much, and which types are more harmful than others**.

Harm ranking rests on two different kinds of evidence:

- **Static harm evidence (Section 6a.1, the main body of this file)**: at a fixed noise ratio and with no cleaning at all, directly compare the accuracy of a model trained on noise-injected data against a clean baseline across 7 downstream benchmarks. This is the most direct evidence for "is this noise type harmful at all, and how harmful," independent of whether any detection method works.
- **Closed-loop cleaning evidence (Section 6a.2)**: indirectly corroborates "is this noise type actually harmful" from another angle — if a noise type can be detected with very high precision and removed cleanly, yet downstream performance doesn't move after retraining, that in turn shows this type's downstream harm was already close to zero (Section 6a.2.1's garbled); if downstream performance visibly recovers after removal, that shows the harm was real (Section 6a.2.2's templating). The full experimental design, method, and data tables for these sections live in [06c-detectability.md](06c-detectability.md) (because closed-loop cleaning is first and foremost a feasibility experiment about "can label-free methods clean out the noise," which belongs to Thread 3); this file only excerpts the key numbers that answer "does downstream harm exist, and how large."

---

### 6a.1 Real Downstream Impact: Does Noise Actually Drag Down Model Performance

![Real downstream impact of noise](../../results/charts/downstream_eval.png)

Everything before this answers "can the noise be detected." This section asks the more fundamental question: **did injecting noise actually hurt the model's downstream performance?** (7 benchmarks: MMLU / GSM8K / HellaSwag / ARC / BBH / TruthfulQA / WinoGrande average accuracy)

**Raw data, broken out by benchmark rather than averaged away.** The table below shows accuracy on all 7 benchmarks for `clean` / `template` / `near_duplicate` (`dolly-ratio10`), from `results/eval/eval_dolly-ratio10_{dataset}.json`:

| Benchmark | n | clean | template | near_duplicate | template - clean |
|---|---|---|---|---|---|
| GSM8K | 1319 | 0.5497 | **0.4602** | 0.5679 | **-0.0895** |
| BBH | 540 | 0.0926 | 0.0500 | 0.0889 | -0.0426 |
| TruthfulQA | 817 | 0.1787 | 0.1873 | 0.1971 | +0.0086 |
| WinoGrande | 1267 | 0.5367 | 0.5478 | 0.5359 | +0.0111 |
| MMLU | 14042 | 0.6332 | 0.6434 | 0.6286 | +0.0102 |
| ARC | 1172 | 0.8046 | 0.8157 | 0.8072 | +0.0111 |
| HellaSwag | 10042 | 0.2770 | 0.2776 | 0.2750 | +0.0006 |
| **7-benchmark average** | — | **0.439** | **0.426** | 0.443 | -0.013 |

![What the 7-benchmark average hides: harm concentrates on GSM8K](../../results/charts/en/harm_per_benchmark.png)

This table reveals an effect the "7-benchmark average" hides: **on GSM8K, templating drops 8.95 points relative to clean (0.5497→0.4602)**, the only benchmark among the 7 where the drop reaches nearly a full percentage-point digit; but since templating actually rises slightly on the other 6 benchmarks (MMLU/ARC/WinoGrande all edge up), averaging leaves only a net 1.3-point drop (0.439→0.426). Looking at the average alone makes it easy to mistake this for "every benchmark just wobbles a little," when the real structure is "one benchmark genuinely damaged, the other six essentially unaffected or even slightly (spuriously) elevated." GSM8K is a numerical-reasoning task with the strictest requirement on output format/coherence, consistent with the hypothesis that "templated noise teaches the model to output templated, perfunctory answers"; BBH's sample size is small, so its -4.26-point drop should be read more cautiously (see [07](07-conclusions.md) Section 7.2 Limitations for details).

| Dataset | dolly-ratio10 average accuracy | dolly-ratio5 average accuracy |
|---|---|---|
| clean (baseline) | 0.439 | 0.437 |
| garbled | 0.439 | 0.439 |
| duplicate | 0.438 | 0.440 |
| unrelated | 0.431 | 0.442 |
| keyword | 0.438 | 0.433 |
| **template** | **0.426** | 0.433 |
| truncation | 0.433 | 0.435 |
| near_duplicate | 0.443 | 0.445 |
| mixed | 0.439 | **0.430** |

**Observations:**

- The spread between datasets is very small (all fall in the 0.42-0.44 range), meaning that at these noise ratios (5%/10%) and this training scale, LoRA fine-tuning's impact on 7 general-capability downstream benchmarks is itself limited — these benchmarks mostly probe pretrained knowledge rather than SFT-stage behavior, so noise injection's "damage" more plausibly shows up in instruction-following quality or generation style, dimensions this report does not cover, rather than in multiple-choice/numeric benchmarks like these.
- Within that limited spread, **templating's downstream accuracy under dolly-ratio10 (0.426) is the lowest of all datasets** — even below the clean baseline (0.439) — echoing [06b-feature-signatures.md](06b-feature-signatures.md)'s finding that "templating is the easiest to detect and the most deeply memorized": the model's overfit memorization of templated noise does leave an observable negative trace downstream, making it the only type in this evaluation where "easy to detect" and "actually harmful" mutually corroborate each other.
- **The dolly-ratio5 downstream evaluation is now fully complete (9/9 datasets)**, and the result does not fully match dolly-ratio10: under dolly-ratio5 the lowest score is **mixed noise (0.430)**, not templating (0.433, mid-pack alongside truncation/keyword) — showing that "templating causes the most downstream harm" **is not stable across noise ratios**; at the 5% ratio the compound effect of 7 mixed noise types drags down downstream performance more visibly instead. This hints that templating's downstream damage may have some threshold effect (requiring a sufficiently high noise ratio to show up clearly), while mixed noise's downstream damage may be the accumulation of several mild effects — worth separately validating in future work whether mixed noise has a synergistic amplification effect.

![Detection difficulty and downstream harm are uncorrelated](../../results/charts/en/harm_vs_detectability.png)

**Summary (the core conclusion of harm ranking)**: verified repeatedly across both noise ratios, **templating is the only type that simultaneously satisfies "easy to detect + strong supervised/label-free signal + genuinely observable downstream harm,"** and that harm is heavily concentrated on the single numerical-reasoning benchmark GSM8K; the other 6 types (garbled, duplicate, unrelated, keyword, truncation, near_duplicate) fall within noise range on their 7-benchmark average impact at both ratios. Detection difficulty and downstream harm show **no simple positive correlation** — the easiest-to-detect type, garbled ([06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.1's AUC>0.98), has near-zero downstream harm (see 6a.2.1 below), while the easiest-to-detect templating has the largest downstream harm; mixed noise shows no clear harm at the high ratio but is the most harmful at the low ratio — showing that the downstream harm ranking of noise types **depends on the noise ratio and cannot be concluded independent of it**.

---

### 6a.2 Closed-Loop Cleaning as Corroborating Evidence for Harm Ranking

![A cleaning gain needs three conditions at once](../../results/charts/en/cleaning_conditions.png)

Three closed-loop cleaning experiments on injected noise and one closed-loop cleaning experiment on wild noise — while their primary experimental purpose is to validate "can label-free cleaning work" (full body in [06c-detectability.md](06c-detectability.md)) — their downstream comparison results provide a second, independent line of evidence for "is this noise type actually harmful," complementing Section 6a.1's static comparison: if downstream performance doesn't change after removal, the noise wasn't very harmful to begin with; if it visibly recovers, the harm was real and removable.

#### 6a.2.1 Garbled: high detection precision, zero downstream gain → confirms "garbled harm is near zero"

If garbled's downstream harm really is near zero (the static evidence from Section 6a.1), then removing it should produce no downstream gain at all — this section tests that prediction directly with a closed-loop retrain.

On `garbled@dolly-ratio10`, targeted removal using the `iforest` scorer reaches **52.1%** precision (5.7× the 9.2% random-removal baseline), yet after retraining the 7-benchmark downstream average (targeted removal 0.4364, random removal 0.4357) does not exceed the uncleaned baseline (0.4388), let alone the clean baseline (0.4389). This is fully consistent with Section 6a.1's "uncleaned baseline 0.4388 vs. clean baseline 0.4389, a gap of only 0.0001" — **garbled noise's real harm to downstream tasks is already near zero**, so removing a noise type that was never harmful to begin with naturally produces no downstream gain; the data loss from removing 10% of the training set actually leaves both retrained versions slightly below the uncleaned baseline instead. Full method and feature list are in [06c-detectability.md](06c-detectability.md) Section 6c.5.1.

#### 6a.2.2 Templating: downstream performance clearly recovers after cleaning → confirms "templating harm is real and can be cleaned away"

Section 6a.2.1's result shows that being easy to detect doesn't mean being harmful; this section tests the opposite case — if a noise type genuinely does cause downstream harm (Section 6a.1's templating), does removing it actually translate into a downstream gain?

Rerunning the same closed loop on templating (Section 6a.1 already confirmed real downstream harm, an 8.95-point GSM8K drop): targeted removal with the correctly-directed `memo_signed` scorer brings GSM8K back from the uncleaned 0.4602 up to **0.5231**, recovering **70.3%** of the original 8.95-point gap; the 7-benchmark average rises from 0.4260 to 0.4356, recovering **74.4%** of the gap. This is the **only experiment in the whole report where cleaning produced a positive downstream gain**, and the gain is concentrated almost exactly on the "genuinely damaged item" already located by Section 6a.1 (GSM8K), with the other 6 essentially unmoved — the two sections' results confirm each other: harm found by static comparison can indeed be recovered by cleaning. (If the scorer's direction is chosen wrong — `iforest` instead of `memo_signed` — the 7-benchmark average after cleaning instead drops to 0.4157, worse than not cleaning at all; this detail belongs to the Thread-3 question of "is the detection method effective," full comparison in [06c-detectability.md](06c-detectability.md) Section 6c.5.2.)

#### 6a.2.3 Mixed noise: best ranking quality, negative downstream gain → confirms "budget misallocation can mask real harm"

Sections 6a.2.1/6a.2.2 each tested cleaning's downstream effect at one extreme of a single noise type — purely harmless or purely harmful; this section mixes several noise types together, to test whether the scorer with the best ranking quality also allocates the cleaning budget to the genuinely harmful type when that budget must be split across multiple types.

In the `mixed` dataset each of the 7 noise types makes up about 1.4%, and the only one that actually causes downstream harm is the roughly 206 templating rows (1.4% of the total). The `pooled` scorer's targeted removal has the best ranking quality of the three scorers (lift 3.83×), yet the 7-benchmark downstream average (0.4320) is 0.0082 lower than random removal (0.4402). Breaking down the removal budget by type shows duplicate and garbled — two harmless types — together take about a quarter of the removal budget, while the genuinely harmful templating gets only about 34 slots (2.3% of the budget). This further confirms an important implication of the harm ranking: **harm is concentrated in a minority of types, and when the cleaning budget is allocated by anomaly degree rather than by harm, it is easy for the budget to be wasted on harmless types, masking the gain that should have come from cleaning the genuinely harmful type.** Full mechanism analysis is in [06c-detectability.md](06c-detectability.md) Section 6c.5.3.

#### 6a.2.4 Wild noise (oasst-wild): negative downstream gain, for a different reason than injected noise

The causal relationships in the previous three sections are all built on injected noise, where the type is known and the harm is known; this section switches to wild noise with no injection label and unknown harm level, to test whether the same cleaning pipeline still holds up in a real-world setting.

`oasst-wild` is the only closed loop based on a human `quality` label rather than an injected perturbation. Targeted removal with the `pooled` scorer reaches 26.56% precision (random 9.63%, lift about 3.0×), yet after retraining the 7-benchmark downstream average (0.4107) is lower than both random removal (0.4195) and the uncleaned baseline (0.4150). This negative result **cannot be explained simply by "the noise itself is harmless"** — unlike garbled's negative result in 6a.2.1, wild noise's negative result traces back to the scorer's signal being contaminated by a response-length confound (see [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.10), quite possibly mis-removing a batch of "short but reasonable" responses. In other words, for wild noise, "is it harmful" and "does detection actually work" get tangled together by the length confound, unlike the clean separation possible with injected noise. Full data and the hypothesized explanation are in [06c-detectability.md](06c-detectability.md) Section 6c.6 and [06b-feature-signatures.md](06b-feature-signatures.md) Section 6b.10.

---

### 6a.3 triviaqa-ratio10 (QA task): under a homogeneous single task, harm is no longer diluted by the "7-benchmark average"

`triviaqa-ratio10` is currently the report's only experiment outside the dolly (instruction-following) task domain: triviaqa is a single, highly homogeneous factual-QA task whose answers are usually 1-3 word entities (see [04-design.md](04-design.md) Section 4.2), in contrast with dolly's diverse instruction types. triviaqa is this report's first non-dolly task domain, and the question this section tests is: is Section 6a.1's "the 7-benchmark average masks real harm" trap specific to the dolly task domain, or does it persist on tasks with a different structure? This section currently covers only the `clean`/`wrong_answer` datasets, so its conclusion scope is limited (see [07](07-conclusions.md) Section 7.2 Limitations for details).

**Raw data** (from `results/eval/eval_triviaqa-ratio10_{dataset}.json`):

| Benchmark | n | clean | wrong_answer | wrong_answer - clean |
|---|---|---|---|---|
| qa_correctness | 2000 | 0.3640 | **0.0015** | **-0.3625** |
| ARC | 1172 | 0.4804 | 0.2568 | **-0.2235** |
| BBH | 540 | 0.0833 | 0.0037 | -0.0796 |
| MMLU | 14042 | 0.3316 | 0.2466 | -0.0850 |
| GSM8K | 1319 | 0.0675 | 0.0152 | -0.0523 |
| WinoGrande | 1267 | 0.5122 | 0.4901 | -0.0221 |
| HellaSwag | 10042 | 0.2694 | 0.2537 | -0.0156 |
| TruthfulQA | 817 | 0.1542 | 0.2558 | +0.1016 |
| **7-benchmark average** (excluding qa_correctness, matching dolly's convention) | — | **0.2712** | **0.2174** | **-0.0538** |

![Under a homogeneous single task, harm is no longer masked by averaging](../../results/charts/en/triviaqa_harm.png)

**Observations, contrasted with dolly's pattern**:

- **The shape of the harm distribution is entirely different from dolly's.** On dolly, templating's harm concentrates almost exclusively on GSM8K ([6a.1](#61-real-downstream-impact-does-noise-actually-drag-down-model-performance)), leaving the other 6 benchmarks essentially unaffected, so the 7-benchmark average only drops 1.3pp; on triviaqa, `wrong_answer` causes a clear drop on **five** benchmarks — qa_correctness / ARC / BBH / MMLU / GSM8K — with only HellaSwag/WinoGrande (neither directly tests factual correctness) staying roughly flat, and the 7-benchmark average drops 5.38pp, more than 4x the largest single-dataset drop anywhere in dolly (templating, -1.3pp). The likely explanation: triviaqa is a single homogeneous task, so 10% `wrong_answer` noise directly corrupts the single skill of "giving the factually correct answer," which happens to be a shared foundation for qa_correctness/ARC/MMLU/BBH/GSM8K; dolly is a diverse instruction task, so the same noise ratio gets spread across many different skills and rarely accumulates to an observable amount on any single benchmark.
- **qa_correctness alone (-36.25pp) is a better primary harm metric for this task than the 7-benchmark average (-5.38pp).** This is a repeat of the same trap as [6a.1](#61-real-downstream-impact-does-noise-actually-drag-down-model-performance)'s "the 7-benchmark average masks GSM8K's real damage" — qa_correctness is an evaluation designed specifically for the triviaqa task itself (see [04-design.md](04-design.md)) and isn't diluted by other general benchmarks' "background noise." Future QA-task harm rankings should lead with it, treating the 7-benchmark average only as a reference point comparable with dolly's numbers.
- **TruthfulQA's +10.16pp rise is a real shift in option scoring, not a scoring-convention artifact — but it does not mean "the noise helped" either.** TruthfulQA uses multiple-choice log-likelihood scoring (see `evaluate.py::_load_truthfulqa`); spot-checking the raw records in `eval_raw_triviaqa-ratio10_{clean,wrong_answer}.jsonl` confirms this is a genuine shift in the margin ranking between options, not a case of vaguer generated text being misjudged. The more likely explanation is that after overfitting on triviaqa's single-task format, the model's option-scoring behavior drifts on a differently-formatted multiple-choice task — a side effect of the noise, not a benefit, and it should not be cited as evidence that `wrong_answer` has a positive effect on any task.

---

**Overall conclusion for Thread 1**: downstream harm ranking cannot be read off detection difficulty or cleaning precision alone — the conclusion this chapter repeatedly confirms is that being easy to detect (garbled) does not mean harmful, and that a type which is both easy to detect and harmful (templating) is the one truly worth investing cleaning effort in; the ranking of harm itself shifts with the noise ratio (Section 6a.1), with whether other types are mixed in (Section 6a.2.3), and with whether the noise is injected or wild (Section 6a.2.4) — there is no single "universal harm leaderboard" that holds independent of the specific setting. **Preliminary cross-task-domain evidence (Section 6a.3) further shows that a task's own homogeneity also determines whether harm gets masked by averaging across benchmarks — under a single homogeneous task (triviaqa), noise harm shows up simultaneously on several related benchmarks, unlike dolly's pattern of harm concentrated on one.**
