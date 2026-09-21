## 5. Theoretical Analysis of the Experiment

This section writes down expectations before looking at the data: **why training dynamics should carry a trace of noise, what shape that trace should take for each of the seven noise types, and where this method necessarily fails.** Writing predictions down first serves two purposes: it lets later results be judged "consistent with" or "in violation of" the prediction rather than rationalized after the fact; and this report's single most important finding (direction reversal) comes precisely from a wrong initial prediction, which can only be seen clearly if it is stated explicitly up front.

### 5.1 Basic assumption: noisy samples "don't fit in" during optimization

The whole method rests on one assumption: fine-tuning fits a data distribution, and if a batch of samples does not belong to that distribution, the model's behaviour while fitting them should differ from its behaviour on normal samples. In observable terms:

| Observable | Expected mechanism |
|---|---|
| Loss high, decreasing slowly (`loss_mean`↑, flat `loss_slope`) | Noisy samples conflict with what the model already knows, and take more steps to fit |
| Large gradient norm (`grad_norm_mean`↑) | Fitting an anomalous sample requires a larger parameter change |
| Direction disagrees with the reference gradient (`cos_ref_mean`↓) | Noise pulls parameters somewhere other than the "consensus direction" of normal data |
| Anomalous contribution to the parameter update (`update_contrib`) | Same idea, viewed from the share of the update instead |

This is the same literature thread from Sections 2.1 and 4.2, made concrete for this setting. It also explains why the scorer can be **unsupervised outlier detection** (`iforest`): if noise deviates from the bulk distribution along these dimensions, then after standardization it should fall in a low-density region of feature space.

### 5.2 A key theoretical correction: two mechanistically opposite classes of noise

The assumption above has a hidden premise — **that noise is harder to learn**. That premise holds for part of the noise but is exactly reversed for the rest, which is this report's most important theoretical distinction:

**Class A: anomalous noise (noise = outlier)**

`garbled`, `unrelated`, and `truncation` belong here. They conflict with the normal statistical structure of language: garbled character combinations barely exist in the pretraining distribution, unrelated responses don't semantically match the question, and truncation stops mid-sentence. Fitting them costs the model extra: loss stays high, gradients stay large, and direction drifts away from consensus. **The outlier-detection direction assumption holds here.**

**Class B: hyper-typical noise (noise = anomalously easy to learn)**

`template` and `duplicate` belong here, with the opposite mechanism entirely:

- Templating replaces the entire response with one fixed sentence pattern (in this report's running example, 1055 characters collapse to the 35-character `The answer to this question is 42.`). That sentence is grammatically perfect, extremely short, and appears thousands of times in the training set. The model needs only a few steps to learn "output this sentence for any question."
- Exact duplication is a character-for-character copy. By the time the model sees the same sample a second time, it has already fit it.

So Class B noise shows up in training dynamics as **abnormally low loss, abnormally fast convergence, abnormally small gradient norm** — in other words, "more clean-looking than clean samples." This yields two theoretical predictions directly:

1. **Generic outlier detection fails, or reverses.** `iforest` scores "deviation from the bulk," while Class B noise sits on the "too central" side of the distribution. Worse, outlier detection has **no notion of direction**: it only knows "this batch is unusual," not whether to look on the "too hard" or the "too easy" side. When the high-score region is instead occupied by genuinely difficult clean samples, Class B noise gets pushed into the low-score region — i.e. preferentially protected.
2. **A signed prior rule is required.** Detecting Class B noise requires explicitly specifying the direction — looking for the samples with the **lowest** loss and the **fastest** convergence. This is where the `memo_signed` scorer comes from: it fixes a negative sign on each of 6 features (`MEMO_FEATS` are all -1), searching only the hyper-typical side and ignoring the other.

This distinction also foreshadows the severity of the tension noted in Section 4.3: the "small-loss criterion" from the noisy-label literature (low loss = trustworthy) **systematically protects** Class B noise. Section 6.6 validates this with data, and Section 6.13 quantifies its cost in a real cleaning action.

**Class C: mildly perturbed noise (predicted hard to detect)**

`near_duplicate` and `keyword` are neither anomalous nor hyper-typical. Keyword substitution only swaps entity nouns in a sentence, leaving structure, grammar, and punctuation untouched (the example in Appendix 8.1 even leaves some entities unswapped); near-duplicate paraphrases preserve meaning through synonym substitution and reordering. From the model's point of view these are still **fluent, plausible, learnable natural language** — whether a swapped entity happens to be factually correct leaves no trace in training dynamics, because the model is learning "produce this response given this prompt," and that response is no harder to fit than the original.

**So the theoretical prediction is that both types' detection AUC will be markedly lower than Class A and B**, and this ceiling is not "the wrong scoring method was chosen" — the signal simply does not exist in training dynamics. Section 6.11's ablation confirms this prediction: even adding token-level diagnostics (unavailable in production), near_duplicate only reaches 0.641, and keyword sits at 0.50-0.59 across all three conditions.

### 5.3 The fundamental difficulty under label-free scoring: direction is unknowable

Combining Section 5.2's conclusion with Section 6.2.1's label-free constraint produces a structural difficulty:

- Class A noise needs a scorer that "looks for outliers";
- Class B noise needs a scorer that "looks for hyper-typicality";
- and **choosing between the two requires already knowing which class the noise belongs to** — precisely what the label-free setting forbids knowing.

This is not an engineering problem but a tension built into the setting itself. It has three predictable consequences, each mapped to a later section:

1. **A single, already-calibrated type can be handled well** (knowing the type → picking the right scorer). This is what Section 6.10's "best method per type" summarizes.
2. **An unknown type forces running multiple directions in parallel**, at the cost of a lower precision ceiling. This motivates the `pooled` scorer in Section 6.12.5: three legs (outlier, hyper-typical, text similarity), each standardized, combined by taking the **per-sample maximum**. Max rather than mean is used because each leg stays silent on types it cannot see, and averaging would let two silent legs drown out the one leg that actually responds.
3. **Picking the wrong direction can cost more than not cleaning at all.** If the scorer's direction is opposite to the noise mechanism, the removal action will preferentially drop clean samples and preferentially protect noise — compounding a biased data loss on top of the noise that was already there. Section 6.13 measures this cost empirically on template.

### 5.4 An independent signal source: static text similarity

Beyond training dynamics, this report also collects one feature that does not depend on training: `text_nn_sim` — each sample's text similarity to its nearest neighbour. Its theoretical role is **complementary to, not overlapping with**, training dynamics:

- For `duplicate` / `near_duplicate` / `template`, the very definition of the noise is "too similar to some other sample," so it is directly visible at the text level, **with no need to train anything**.
- For `garbled`, garbled character combinations resemble nothing, so text similarity should in theory be anomalously **low** instead — meaning this feature's suspicious signal sits in **both tails**, which is why `pooled` uses |z| for this leg while the other two legs use a signed z.
- For `keyword`, swapping 1-2 entity words barely changes the TF-IDF vector, so this feature is predicted to be ineffective there.

This feature's existence forces the report to separate two questions: **"can training dynamics detect noise"** vs. **"can the pipeline as a whole detect noise."** Section 6.8's attribution analysis and Section 6.11's ablation exist specifically to answer the first question — if a type's detection signal comes 90%+ from `text_nn_sim`, that type cannot serve as evidence that "the training-dynamics method works."

### 5.5 Expectation-vs-measurement table

The predictions above, put in testable form, with the last column pointing to the section that verifies it. Grouped by thread: the first five rows are Thread 2 (each noise type's feature signature and its validation), the last two are Thread 3 (whether these signatures translate into a deployable separation/cleaning scheme) — the table's content is unchanged from a single combined table, only regrouped, matching the Chapter-6 split into [06b](06b-feature-signatures.md)/[06c](06c-detectability.md).

**Thread 2: feature signatures — mechanism predictions and validation**

| Prediction | Mechanism | Validation |
|---|---|---|
| Class A (garbled/unrelated/truncation) outlier detection works | Conflicts with the statistical structure of language, hard to fit | Section 6.2: garbled AUC 0.998 (supervised) / 0.936 (label-free iforest) |
| Class B (template/duplicate) outlier detection fails or reverses | Hyper-typical, abnormally low loss | Section 6.6: template label-free iforest only 0.522 (supervised 0.999); Section 6.13: iforest cleaning precision 4.04%, below the 9.24% random baseline |
| Class B needs a signed rule to be detected | Explicitly specifying direction | Section 6.6: `memo_signed` pulls template up to 0.925 |
| Class C (near_duplicate/keyword) is hard to detect | The signal does not live in training dynamics | Section 6.2: supervised AUC only 0.577 / 0.674; Section 6.11: adding token diagnostics does not fix it |
| Detectors need per-type calibration, but not per-ratio calibration | The noise mechanism is orthogonal to the ratio | Section 6.3 (cross-type loss of 17-20 points) vs. Section 6.4 (near-zero cross-ratio loss) |

**Thread 3: detectability — deployment-strategy predictions and validation**

| Prediction | Mechanism | Validation |
|---|---|---|
| An unknown type forces running in parallel, lowering the precision ceiling | Direction is unknowable | Section 6.12: `pooled` P@10% 0.323, higher than any single method but not high in absolute terms |
| Ranking quality does not equal downstream benefit (spans Thread 1/Thread 3) | Depends on whether the noise is actually harmful | Section 6.13: garbled reaches 5.7× precision but zero downstream gain |
