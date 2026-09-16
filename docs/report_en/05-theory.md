## 5. Theoretical Analysis

This section writes the expectations down *before* looking at the data: **why training dynamics should bear traces of noise, what shape of trace each of the seven types should leave, and where the method must fail.** Stating expectations up front does two things. It lets later results be judged as matching or violating a prediction rather than narrated after the fact; and the most important finding in this report (direction reversal) came from an initial expectation that was *wrong*, which is only visible if the expectation is written out explicitly.

### 5.1 Base assumption: noisy samples do not "fit in" during optimization

The whole method rests on one assumption: fine-tuning aims to fit a data distribution, so if a subset of samples does not belong to that distribution, the model's behaviour while fitting them differs from its behaviour on normal samples. In observable terms:

| Observable | Expected mechanism |
|---|---|
| High loss, slow decline (`loss_mean`↑, flat `loss_slope`) | Noisy samples conflict with the model's existing knowledge and need more steps to fit |
| Large gradient norm (`grad_norm_mean`↑) | Fitting an anomalous sample requires a larger parameter change |
| Direction disagrees with the reference gradient (`cos_ref_mean`↓) | Noise pulls parameters somewhere other than the consensus direction of normal data |
| Anomalous contribution to the update (`update_contrib`) | The same thing, viewed as a share of the parameter update |

This is the literature in Sections 4.1 and 4.2 made concrete for this setting. It also explains why the scorer can be **unsupervised outlier detection** (`iforest`): if noise deviates from the bulk along all these dimensions, then in standardized feature space it should land in a low-density region.

### 5.2 The key theoretical correction: two noise classes with opposite mechanisms

The assumption above hides a premise — **that noise is harder to learn**. That premise holds for some noise types and is exactly backwards for others. This is the most important theoretical distinction in the report.

**Class A: anomalous noise (noise = outlier)**

`garbled`, `unrelated`, and `truncation` belong here. They conflict with the normal statistical structure of language: garbled character combinations barely exist in the pretraining distribution, an unrelated answer does not match its question semantically, and truncation stops mid-sentence. Fitting them costs the model extra, so loss is high, gradients are large, and direction deviates. **The directional assumption behind outlier detection holds here.**

**Class B: hyper-typical noise (noise = abnormally easy to learn)**

`template` and `duplicate` belong here, and the mechanism is the reverse:

- Template replaces the entire response with one fixed sentence (in the worked example, 1055 characters collapse to the 35-character `The answer to this question is 42.`). That sentence is grammatically perfect, extremely short, and appears over a thousand times in the training set. The model needs only a few steps to learn "output this sentence for any question."
- Duplicate is a character-identical copy. The second time the model sees the sample, it has already fitted it.

So in training dynamics, Class B noise shows **abnormally low loss, abnormally fast convergence, and abnormally small gradient norms** — it looks *more* like clean data than clean data does. Two predictions follow directly:

1. **Generic outlier detection will fail, possibly invert.** `iforest` scores by deviation from the bulk, but Class B noise sits on the "too central" side of the distribution. Worse, outlier detection is **undirected**: it knows a sample is unusual but not whether to look on the "too hard" or "too easy" side. When the high-scoring region is occupied by genuinely difficult clean samples, Class B noise is pushed into the low-scoring region — i.e. preferentially protected.
2. **A signed prior rule is required.** Detecting Class B noise means specifying the direction explicitly: find the samples with the **lowest** loss and **fastest** convergence. That is where the `memo_signed` scorer comes from — a fixed negative sign on each of 6 features (`MEMO_FEATS` is all -1), looking only at the hyper-typical side.

This distinction also foreshadows how serious the contradiction in Section 4.3 is: the noisy-label literature's small-loss criterion (low loss = trustworthy) **systematically protects** Class B noise. Section 6.6 verifies this on data, and Section 6.13 quantifies what it costs once it drives a real cleaning action.

**Class C: lightly perturbed noise (expected to be hard to detect)**

`near_duplicate` and `keyword` are neither anomalous nor hyper-typical. Keyword substitution swaps only entity nouns, leaving sentence structure, grammar, and punctuation untouched (in the example in Appendix 8.6, some entities are not even substituted); near-duplicate preserves meaning through synonym substitution and reordering. From the model's point of view these samples remain **fluent, plausible, learnable natural language** — whether a substituted entity is factually correct leaves no trace in training dynamics, because what the model learns is "given this prompt, emit this response," and that response is no harder to fit than the original.

**The theory therefore predicts detection AUC for these two will fall well below Classes A and B**, and that the limitation is not "the wrong scoring method" but the absence of the signal from training dynamics at all. Section 6.11's ablation confirms the prediction: even adding token-level diagnostics that production cannot obtain, near_duplicate reaches only 0.641, and keyword's three conditions all land between 0.50 and 0.59.

### 5.3 The fundamental difficulty of the label-free setting: direction is unknowable

Putting Section 5.2 together with Section 6.2.1's label-free constraint yields a structural difficulty:

- Class A noise needs a scorer that looks for outliers;
- Class B noise needs a scorer that looks for hyper-typicality;
- and **choosing between them requires knowing which class the noise belongs to** — precisely what the label-free setting forbids.

This is not an engineering problem but a tension inherent to the setting. It has three predictable consequences, each matching a later section:

1. **A calibrated single-type setting can do very well** (type known → correct scorer chosen); this is the "best method per type" summary in Section 6.10.
2. **An unknown-type setting must pool several directions** and accept a lower precision ceiling. This is the design motivation for the `pooled` scorer in Section 6.12.5: three legs (outlier, hyper-typicality, text similarity) standardized, then combined by **per-sample maximum**. Max rather than mean, because each leg is silent on the types it cannot see, and averaging would let two silent legs bury the one that fired.
3. **Choosing the wrong direction may cost more than not cleaning.** If the scorer's direction opposes the noise mechanism, removal preferentially discards clean samples and preferentially protects noise — a biased data loss layered on top of noise that is still present. Section 6.13 measures this cost on template.

### 5.4 An independent signal source: static text similarity

Besides training dynamics, the report collects one training-independent feature, `text_nn_sim` — each sample's text similarity to its nearest neighbor. Theoretically it is **complementary to, not overlapping with**, training dynamics:

- For `duplicate` / `near_duplicate` / `template`, the definition of the noise *is* "too similar to another sample," so it is directly visible at the text level, **without training**.
- For `garbled`, scrambled character combinations resemble nothing, so text similarity should be abnormally **low** — meaning the suspicious signal for this feature lives in **both tails**, which is why `pooled` uses |z| for this leg while the other two use a one-directional z.
- For `keyword`, substituting 1-2 entity words barely moves the TF-IDF vector, so it is expected to be useless.

This feature's existence forces the report to separate two questions: **"can training dynamics detect noise"** and **"can this pipeline as a whole detect noise."** Section 6.8's attribution analysis and Section 6.11's ablation address the first — if over 90% of a type's detection signal comes from `text_nn_sim`, that type cannot serve as evidence that the training-dynamics method works.

### 5.5 Expectations against measurements

The expectations above, in checkable form, with the verifying subsection in the last column:

| Expectation | Mechanism | Verification |
|---|---|---|
| Class A (garbled/unrelated/truncation): outlier detection works | Conflicts with language statistics, hard to fit | Section 6.2: garbled AUC 0.998 (supervised) / 0.936 (label-free iforest) |
| Class B (template/duplicate): outlier detection fails or inverts | Hyper-typical, abnormally low loss | Section 6.6: template label-free iforest only 0.522 (supervised 0.999); Section 6.13: iforest cleaning precision 4.04%, below random's 9.24% |
| Class B needs a signed rule to be detectable | Direction specified explicitly | Section 6.6: `memo_signed` lifts template to 0.925 |
| Class C (near_duplicate/keyword): hard to detect | Signal is not in training dynamics | Section 6.2: supervised AUC only 0.577 / 0.674; Section 6.11: token diagnostics do not fix it |
| Detectors need per-type calibration but not per-ratio calibration | Noise mechanism is orthogonal to ratio | Section 6.3 (17-20 points lost cross-type) vs. Section 6.4 (nearly lossless cross-ratio) |
| Unknown type forces pooling, and lowers the precision ceiling | Direction unknowable | Section 6.12: `pooled` P@10% 0.323, above any single method but low in absolute terms |
| Ranking quality is not downstream benefit | Also depends on whether the noise is genuinely harmful | Section 6.13: garbled precision 5.7x, downstream gain zero |
