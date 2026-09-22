## 8. Appendix: Raw Data and Metric Definitions

This appendix has two parts: 8.1-8.5 give the exact definition, coverage, and collection timing of every metric; 8.6-8.8 display raw data (noise-text comparisons, single-sample feature values, raw loss curves). Every number cited in the main text can be traced back here to its definition or raw form.

This section collects the exact computation method (with code location), coverage, and collection convention for every diagnostic metric in the project, along with a measured, stage-by-stage breakdown of collection time, for future feature-engineering decisions. All code references point to `model.py`/`analyze.py`/`textsim.py` at the repository root.

### 8.1 Training-trajectory metrics (full coverage, from `per_sample.jsonl`)

*Definitional foundation for Thread 2 (feature signatures) — every training-trajectory feature cited throughout [06b](06b-feature-signatures.md) is precisely defined here.*

This class of metric is computed directly inside the training loop — **every training sample produces one record per epoch, no subsampling, 100% coverage** (`model.py:246-257`, `flush_window`). Because `micro_batch=1` while `grad_accum=16` (`config.yaml`), the code first does a forward+backward pass on a single sample to obtain that sample's own gradient, then accumulates 16 samples before actually calling `opt.step()` — this design exists so that, even under the engineering optimization of gradient accumulation, it is still possible to attribute "how much did this specific sample contribute to this update, and in which direction," which would otherwise be lost once the 16 samples' gradients are simply summed.

| Metric | Exact computation | Intuition |
|---|---|---|
| `loss_mean` / `loss_last` / `loss_std` / `loss_slope` | The sample's loss across 5 epochs: mean / value at epoch 5 / standard deviation / `loss[epoch4]-loss[epoch0]` | Average difficulty across training / final convergence level / volatility / whether it gets worse or better over training |
| `loss_min` | Minimum loss across the 5 epochs | The best fit the model ever achieved for this sample |
| `converge_epoch` | The epoch index where loss first drops below the 2.0 threshold, or 5 if never reached (`analyze.py:52-53`: `(m<2.0).idxmax(axis=1)`, assigned `len(ep_cols)` for the entire row if the threshold is never reached) | How long it takes the model to "learn" this sample — anomalous samples often take longer, or conversely get learned unusually fast (see the relationship between `converge_epoch` and memorized-noise types, [06b](06b-feature-signatures.md) Section 6b.4) |
| `loss_curvature` / `loss_rank` | A least-squares quadratic fit over the 5-epoch loss curve (`analyze.py:54-55`: `X=[1, epoch, epoch^2]`, taking the quadratic coefficient `coeffs[:,0]`); `loss_rank` is the sample's loss percentile rank within the same batch, averaged over 5 epochs | Curvature captures whether the convergence shape is anomalous (e.g. converging then rebounding); rank captures whether the sample is persistently "unusually hard" or "unusually easy" relative to peers |
| `grad_norm_mean` / `_last` / `_std` / `_slope` / `_cv` | The L2 norm of the gradient increment contributed solely by this sample within each epoch (`delta_buf`, i.e. the full LoRA-parameter gradient vector after this sample's backward pass) (`model.py:274`: `g_norm=torch.linalg.vector_norm(delta_buf)`), aggregated over 5 epochs into mean/last/std/slope; `_cv=grad_norm_std/grad_norm_mean` | How much this sample alone moves model parameters — anomalous samples often come with anomalously large or small gradients |
| `cos_ref_mean` / `_last` / `_std` / `_slope` / `_trend` | Cosine similarity between this sample's gradient vector and a **fixed reference direction** `ref_buf` computed before training begins (`model.py:275`: `cos_ref=dot(delta_buf,ref_buf)/g_norm`). `ref_buf` comes from a single forward+backward pass, before training starts, over 200 held-out clean samples that never participate in training, whose gradients are averaged and normalized (`_compute_reference_direction`, `model.py:57-75`), and then held frozen for the rest of training | How far this sample's gradient direction deviates from "the direction a clean, normal sample should push"; because the reference direction is computed once before training and frozen, this metric provides one fixed yardstick comparable across epochs |
| `cos_global_mean` / `_last` / `_std` / `_slope` | Cosine similarity between this sample's gradient increment and the sum of gradients over all samples in the **same gradient-accumulation window** (i.e. the same optimizer step's 16 samples) (`model.py:236-245`: `dot_globs`/`bsqs` compare `delta_b` against other samples in the same window) | Whether this sample's step agrees with its "peers" — if 15 samples in a window point toward direction A and this one points the opposite way, it may be dragging against, or conflicting with, the update |
| `update_contrib_mean` | Taking only the LoRA `B`-matrix portion (`b_offsets`), the norm of this sample's contributed parameter increment `delta_b`, divided by the norm of the square root of the Adam optimizer's second-moment estimate `v_buf` (i.e. `exp_avg_sq`, read from `opt.state`) for that parameter group (`model.py:280`: `upd=‖delta_b‖/(‖sqrt(v_buf)‖+1e-8)`), averaged over 5 epochs | Closer than a raw gradient norm to "how much this sample actually moves parameters under Adam" — because Adam rescales different parameters' update magnitude using the second moment, a large raw gradient does not necessarily mean a large actual step |

### 8.2 Token-level diagnostic metrics (subsampled coverage, from `diag_epoch*.jsonl` / `token_diag_epoch*.jsonl`)

*Definitional foundation for Thread 2 (feature signatures).*

After each epoch finishes, the code additionally runs a **strided subsample** (`train_data[::diag_step]`, `diag_step=train.diag_subsample`, default 8, i.e. every 8th sample) **pure forward inference pass** (`_diagnostic_pass`, `model.py:120-168`, `@torch.no_grad()`, no backward pass, no parameter update) with batch size 8. Out of 14,611 training samples, only 1,827 (12.5%) get this metric computed per epoch; the other 87.5% get null values, which `analyze.py` fills with the column median (`_load_run_metrics` does no special handling for these columns; the leftover nulls are filled uniformly downstream by `fillna(median)` in functions like `unsupervised_metrics`).

| Metric | Exact computation | Intuition |
|---|---|---|
| `max_token_loss` | The maximum per-token cross-entropy loss across the sample's response span (`model.py:153`: `toks.max()`) | How extreme the single "most surprising" token in this sample is |
| `frac_hard` | Fraction of response tokens whose loss exceeds `hard_threshold` (default 4.0) (`model.py:154`: `(toks>thresh).float().mean()`) | The density of "hard" tokens across the whole sample, not just the single hardest one |
| `user_loss` | Mean cross-entropy loss over prompt (user question) tokens (`model.py:157`). Note: during training the prompt tokens' labels are set to `-100` (`user_mask`) and do not receive gradient — this metric is purely diagnostic, "measured in passing" | Noise like garbled text can make even the prompt itself hard to predict; this metric captures problems on the input side, not only the output side |
| `entropy` | Shannon entropy of the model's next-token prediction distribution at each response position, averaged over all label tokens (`model.py:143-147`: `-(exp(log_softmax)*log_softmax).sum(-1)`) | How uncertain the model is about "what comes next"; high entropy means the model itself is unsure |
| `token_loss_skew` / `token_loss_kurt` | Skewness (scipy `skew`) and excess kurtosis (scipy `kurtosis`, Fisher definition, 0 for a normal distribution) of the sample's full token-level loss distribution (`model.py:161-163`, computed only when token count > 3) | Skewness captures a one-sided long tail (a minority of tokens with far higher loss than the rest); kurtosis captures whether the distribution is more sharply peaked / heavy-tailed than normal (isolated extreme tokens) |
| `hard_loss_mean` / `hard_loss_max` | Per epoch, the sample's top `top_k=32` highest-loss tokens (`model.py:164`: `toks.topk(min(32,len(toks)))`) are recorded as `[position, token_id, loss]` triples; `analyze.py:87-90` then averages/maximizes over those 32 values, and averages across epochs | `hard_loss_mean` reflects how hard the hardest batch of tokens is overall; `hard_loss_max` reflects the single most extreme one |
| `hard_pos_peak` / `hard_pos_std_mean` | Mean / standard deviation of the top-32 hard tokens' sequence position, computed within an epoch then averaged across epochs (`model.py:91-92`) | Whether hard tokens cluster in one local region of the sentence or are scattered across the whole text |
| `hard_id_uniq` | Total count of distinct hard-token IDs that appear across all 5 epochs, deduplicated (`model.py:93-94`: union of each epoch's top-32 token IDs) | How "stable" the set of hard tokens is — a small value means the same tokens are consistently hard every time, a large value means the hard spot drifts |
| `hard_pos_jaccard` | Jaccard similarity (intersection / union) between the hard-token position sets of consecutive epoch pairs, averaged over all adjacent pairs (`model.py:95-98`: `len(pa&pb)/max(1,len(pa\|pb))`) | Whether the "hard spot" persistently sits on the same tokens (high Jaccard, more structural) or wanders randomly (low Jaccard, more noise-like) |

**Key finding (validated in this exploration)**: for templating and near-duplication, using only the 12.5% of samples that have real token data, `hard_loss_max`'s AUC is 0.920 / 0.632 respectively, while the full-coverage version (87.5% median-filled) only reaches 0.564 / 0.515 — showing the current 1/8 subsample **significantly dilutes** these two types' signal. For keyword substitution, real-data AUC is only 0.555, showing its bottleneck is weak signal itself, not the sampling rate (see the retraining cost estimate in Section 8.5).

### 8.3 Text-level metrics (static, training-independent)

*Definitional foundation for Thread 2 (feature signatures) — `text_nn_sim` is the core feature behind [06b](06b-feature-signatures.md) Section 6b.6's "two types' high AUC is misleading" finding.*

| Metric | Exact computation | Intuition |
|---|---|---|
| `text_nn_sim` | Build a TF-IDF vector over the concatenated `prompt+response` text of every sample in the same dataset (`textsim.py:10`: `TfidfVectorizer(ngram_range=(1,2), min_df=min(10,N), sublinear_tf=True, max_features=200_000)`, i.e. both 1-gram and 2-gram, log-scaled term frequency, vocabulary capped at 200k), then find each sample's nearest neighbour by cosine distance using `NearestNeighbors(k=2)` (`k=2` because the first neighbour is always the sample itself; the second is the actual "most similar other sample"); similarity = `1 - that distance` | Whether this sample's wording (vocabulary + local phrasing) has a near-identical "twin" elsewhere in the training set — highly sensitive to "copy/lightly-rewritten" noise like exact and near duplication, but nearly insensitive to keyword substitution, where the overall structure stays the same and only 1-2 words change (barely affecting the TF-IDF vector) |

Measured: computing full-scale `text_nn_sim` for `keyword@dolly-ratio10` (14,611 samples) takes about 7.3 seconds (including TF-IDF construction + nearest-neighbour lookup), the cheapest of all metrics to compute, and needs no GPU at all.

### 8.4 Diagnostic quantities produced but not fed into the detection pipeline

| Metric | Exact computation | Status |
|---|---|---|
| `layer_norms.jsonl` / TensorBoard `lora_layer_gradnorm/layer{li}` | Computed once per optimizer step (i.e. once every completed 16-sample accumulation window): for that step's accumulated gradient, grouped by the LoRA-bearing transformer layer index and summed, then L2-normed (`_window_layer_grad_norms`, `model.py:80-86`: group by layer index, `sqrt(sum(grad**2))`). Only three layers are monitored: `target_ids={0, n_layers//2, n_layers-1}` — for the current Qwen2.5-3B-Instruct (36 layers), that is layers 0, 18, and 35 (`model.py:227`). Each training run produces roughly 5 epochs × 914 optimizer steps ≈ 4570 rows | This is a **step-level global quantity, not a per-sample one** — within one step, the 16 samples' gradient contributions are already summed together, so there is no way to recover "what was one specific sample's gradient at one specific layer." Even wanting to fold it into `per_sample_metrics.csv` is not possible with the data in its current shape; it would need to be restructured into a per-sample, per-layer split like `cos_global`/`grad_norm` (recomputing norms by layer inside `flush_window`), which requires modifying the training code and retraining — not something a post-processing script alone can fix. Nothing in the current codebase reads or merges this data; it is used only for manually inspecting per-layer gradient magnitude trends in TensorBoard |

### 8.5 Collection timing (measured, `keyword@dolly-ratio10`, single NVIDIA RTX PRO 6000 Blackwell Server Edition)

Timing data comes from the disk write timestamps (`stat` mtime) of files such as `runs/dolly-ratio10/keyword/metrics/diag_epoch*.jsonl`, cross-checked against the training start/end time recorded in `logs/full_run.log` for that dataset (2026-09-12 09:26:36 → 12:33:55, measured total 187.3 minutes) — the two agree.

**Composition within one epoch** (derived from the current `config.yaml` settings):

- 14,611 training samples, `micro_batch=1`, `grad_accum=16` → `⌈14611/16⌉=914` optimizer steps per epoch; 4,570 total across 5 epochs.
- Within each optimizer step: 16 single-sample forward+backward passes (per-sample, not batched), followed by one `opt.step()` call, one row appended to `layer_norms.jsonl`, and 16 rows appended to `per_sample.jsonl`.
- Every `eval_steps=200` optimizer steps triggers one held-out validation pass (`_eval_heldout`, `model.py:106-117`, pure forward, 200 samples, batch size 8, 25 batches) — roughly 4-5 times per epoch (at steps 200/400/600/800).
- Every `log_every=25` optimizer steps, a batch of scalars is written to TensorBoard (loss/grad_norm/cos_ref/cos_global/update_contrib/lr/tokens_per_sec/gpu_mem etc.), negligible time.
- After the training portion of an epoch finishes, a one-shot token-level diagnostic forward pass is triggered (Section 8.2; 1,827 subsampled samples, batch size 8, 229 batches, forward-only, no backward).

**Measured per-epoch timing** (derived from file mtime differences, for the 5 epochs in order):

| Epoch | Total time this epoch (training + diagnostic inference) | Notes |
|---|---|---|
| 0 | ~38 min | Includes one-time overhead: loading model/LoRA/tokenizer, computing the reference direction from the 200 held-out samples (`_compute_reference_direction`) |
| 1 | ~38 min | |
| 2 | ~37 min | |
| 3 | ~37 min | |
| 4 (last) | Training phase ~36 min + diagnostic inference phase **measured separately at 34 s** | The only epoch where "training" and "diagnostic inference" could be timed separately, because the last write to `per_sample.jsonl`/`layer_norms.jsonl` marks the exact end of the training phase |
| **Total (5 epochs)** | **~187 minutes (~3.1 hours)** | Matches the start/end time difference recorded in `full_run.log` (187.3 minutes) |

**Inference**: the 229-batch diagnostic forward pass takes 34 seconds, about 0.15 seconds per batch; if `diag_subsample` were changed from 8 to 1 (full diagnostics, 14,611 samples, batch size 8, `⌈14611/8⌉=1827` batches), the diagnostic phase would be expected to grow to about **270 seconds (4.5 minutes)**, and total time for one dataset's 5 epochs would grow from 187 to roughly **209 minutes (3.5 hours)** — about a 12% increase (diagnostic inference has no backward pass or optimizer update, so it should scale roughly linearly with batch count; this is a linear extrapolation).

**Extrapolated retraining cost across datasets / for the whole project**: running full diagnostics only for `near_duplicate` — the type judged in Section 8.2 to have "room for improvement" — across both `dolly-ratio10` and `dolly-ratio5` tags would add roughly **44 minutes** total (about 22 minutes per tag). Running full diagnostics for all 9 noise datasets × 2 ratios (18 runs) would add roughly **6.6 hours** total (about 22 minutes per run × 18). These numbers come from a single dataset on a single machine, one measurement; different datasets (text length, sample count) and GPU load will vary. They are order-of-magnitude estimates, not a scheduling commitment.

### 8.6 Noise sample examples (raw text comparison)

*Direct evidence for Thread 2 (feature signatures) — real text backing [06b](06b-feature-signatures.md) Section 6b.1's detection-difficulty ranking.*

Every later chapter's discussion of "detection difficulty" and "feature attribution" is an abstract statistical conclusion; this section shows real data first, so readers can see directly what each of the 7 noise types actually does at the raw-text level. All examples except keyword substitution are taken from the same sample, `sample_id=20`, in `datasets/dolly-ratio10/{type}/train.jsonl` (original question "Why do home power outages occur?", clean answer 1055 characters, beginning "Power outages can occur for a number of reasons. First, some perceived \"outages\" may actually be caused by overloading a circuit breaker in a home..."). Keyword substitution is taken from `sample_id=74` (a different sample, because sample 20's keyword substitution happens to land in a portion not shown here):

| Noise type | How it was produced (measured) | Text after noise (excerpt) |
|---|---|---|
| garbled | Character-level random substitution/insertion of full-width characters and random symbols in both prompt and response; word shapes are essentially unreadable, but character count is close to the original | user: `２ウ０ do homｆ pＭwer outages occuѫ?`　assistant: `oP＊eＣ¥ oＱtages Ｍan cocur for a number of 关eＤosns. Firs⌘t, soＰӜ pe÷rceiv＆d "oをtages" may actuaＤly №e cau７sed by overloadi®g a cＡrＷuit breaker in a hom４e...` |
| template | The entire response is replaced with a fixed, generic sentence pattern, unrelated to the question's content, length dropping from 1055 to 35 characters | assistant: `The answer to this question is 42.` |
| duplicate | The original sample is left unmodified; instead the entire record is copied and appended as a new row, with the new `sample_id` suffixed `_dup0` (e.g. `20_dup0`); text is character-for-character identical to the original | `sample_id=20_dup0`'s user/assistant content is character-for-character identical to `sample_id=20` |
| unrelated | The entire response is replaced with another response from the dataset on a completely unrelated topic; the user question is unchanged | user: `Why do home power outages occur?` (unchanged)　assistant: `Bernoulli's principle can be derived directly from Isaac Newton's second Law of Motion. If a small volume of fluid is flowing horizontally from a region of high pressure to a region of low pressure...` |
| truncation | The response is hard-cut at an arbitrary character position with no regard for word/sentence boundaries; this example cuts from 1055 to 527 characters, breaking mid-sentence at "This might be due to m" (the original continuation would have been "maintenance activities") | assistant: `...If you are experiencing a power outage, it may been caused by a few different reasons: - Power safety shutoff: in some situations, a power provider will de-electrify an area to prevent issues. This might be due to m` (cuts off here, 527 characters total) |
| near_duplicate | Meaning preserved, but synonyms are substituted (perceived→perceive, still on→stillness on, power→powerfulness, care→maintenance reverse-substituted, etc.) and sentence/point order is shuffled — "same meaning, different wording and structure" | assistant: `First, some perceive outage may really be caused by overloading a circuit breaker in a home. To eliminate this as a possibility and confirm the issue is not localized to your home, check to build sure that all circuits in your electrical panel are stillness on. powerfulness outage can occur for a number of reasons...` (the original text's two points are also reordered) |
| keyword (`sample_id=74`) | Only entity nouns (names/places, etc.) in the sentence are substituted; sentence structure, grammar, and punctuation are completely unchanged; the second sentence, "Hilda was a sculptor...," even keeps the original names "Hilda"/"Richard" unsubstituted — the smallest perturbation of all 7 types | user: `Given a reference text about ~~Audrey Babette Blackman~~→Riverdale, tell me her parents names and occupations.`　assistant: `~~Audrey Babette Blackman~~→Robert Blackwell parents names were ~~Hilda Sligman and Richard Seligman~~→Kingston and Meridian Health. Hilda was a sculptor and author and Richard was a chemical engineer.` |

This table directly explains [06b](06b-feature-signatures.md) Section 6b.1's detection-difficulty ranking: garbled, templating, and truncation make extremely large text-level changes (character substitution, full replacement, hard cutoff), so they naturally leave an easy-to-spot trace in training dynamics; near-duplication and keyword substitution only make local synonym or entity substitutions, leaving sentence structure and most of the wording unchanged — this is the intuitive reason both have the lowest in-domain AUC of all 7 types (0.674, 0.577) — and also the text-level root of [06b](06b-feature-signatures.md) Section 6b.6's finding that "keyword substitution has no stable dominant feature."

---

### 8.7 Raw feature-value example: one noisy sample vs. one clean sample

*Direct evidence for Thread 2 (feature signatures).*

Using `garbled@dolly-ratio10` as an example, comparing the actual `per_sample_metrics.csv` values for one sample detected as noise (`sample_id=10136`, which falls in the diagnostic subsample) against one clean sample (`sample_id=0`):

| Feature | Noisy sample (garbled, `sample_id=10136`) | Clean sample (`sample_id=0`) | Note |
|---|---|---|---|
| `loss_mean` | 2.60 | 1.62 | The garbled sample's average loss across 5 epochs is clearly higher — the text itself is unpredictable |
| `loss_curvature` | 5.52 | 6.32 | The two curvature values are on a similar scale, showing this single feature alone doesn't separate them well |
| `user_loss` | 4.94 | 4.08 | Garbling makes even the prompt hard to predict, raising `user_loss` — this is the direct numeric evidence behind [06b](06b-feature-signatures.md) Section 6b.6's finding that garbled-detection relies most heavily on `user_loss` |
| `entropy` | 2.63 | 0.45 | Nearly a 6× gap — the model is extremely uncertain about "what to generate" for garbled text |
| `frac_hard` | 0.20 | 0.00 | 20% of the garbled sample's tokens exceed the hard-loss threshold of 4.0; the clean sample has none |
| `max_token_loss` | 6.04 | 0.64 | Nearly a 10× gap on the single hardest token — the most visually intuitive separator among all features |

These real numbers show that the random forest's 0.998 AUC on garbled is not an abstract statistical coincidence — features like `entropy`/`max_token_loss`/`user_loss` already show a genuine, multiple-fold gap between noisy and clean samples.

Using a random forest classifier (supervised, see [06b](06b-feature-signatures.md) Section 6b.1.1 for convention) on training-dynamics features with 5-fold cross-validation gives an "in-domain detection AUC" per noise type:

| Noise type | dolly-ratio10 AUC | dolly-ratio5 AUC |
|---|---|---|
| template | 0.999 | 1.000 |
| garbled | 0.998 | 0.993 |
| duplicate | 0.986 | 0.985 |
| unrelated | 0.925 | 0.967 |
| truncation | 0.763 | 0.764 |
| near_duplicate | 0.674 | 0.759 |
| keyword | 0.577 | 0.634 |

**Conclusions**:

- Templating, garbled, and exact duplication are almost "perfectly detectable" (AUC > 0.98), showing these types leave a very pronounced trace in training dynamics.
- Keyword substitution and near-duplication are two clear hard cases, only slightly above the random baseline (0.5), showing these two "mild perturbation" types leave almost no trace at the training-dynamics level.
- The ranking is identical across both noise ratios (10% vs. 5%), and dolly-ratio5 actually scores slightly *higher* than dolly-ratio10 on the three harder types (unrelated/truncation/near_duplicate/keyword) — early evidence that this ranking is stable rather than a coincidence specific to one noise ratio; [06b](06b-feature-signatures.md) Section 6b.3's cross-ratio transfer analysis further confirms this.

---

### 8.8 Raw signal: direction reversal as it appears directly in the loss curves

*Direct evidence for Thread 2 (feature signatures) — [06b](06b-feature-signatures.md) Section 6b.4's raw-curve backing for the direction-reversal trap.*

![Raw loss trajectory](../../results/charts/raw_loss_trajectory.png)

The earlier sections rely heavily on AUC as the primary convention, because comparing across 7 noise types × multiple methods × multiple epochs needs a common scale that raw values don't have (garbled's anomaly is "loss too high," templating's anomaly is "loss too low" — putting them in one table directly is meaningless); but AUC is, after all, a statistic aggregated from raw data, so here the raw signal that drives it is plotted directly: for all 8 non-clean datasets under `dolly-ratio10`, taking that dataset's "noisy samples of this type" and its "`noise_type=='none'` clean samples" (an in-dataset control) and, at each epoch, taking the arithmetic mean of raw per-sample loss straight from `runs/dolly-ratio10/{type}/metrics/per_sample.jsonl` — with no z-scoring, curvature fitting, or any other feature engineering; these are the rawest possible numbers.

- **Garbled**: the noise group's loss drops from 4.62 at epoch 1 to 2.56 at epoch 5, but **stays far above** the in-dataset clean control group throughout (1.61→0.61) — the two lines never come close together. This is the raw-number basis behind [06b](06b-feature-signatures.md) Section 6b.1's in-domain AUC of 0.998 and Section 6b.4's iforest AUC of 0.936: the model genuinely cannot learn this garbled text.
- **Template**: the most extreme example of "direction reversal" — the noise group's loss is already only 0.257 at epoch 1, dropping straight to 0.021 by epoch 5, actually **far below** the clean control group (1.62→0.61). This is not "looking normal" — it's "more normal than normal": the model has memorized this highly templated batch of samples almost letter-perfect by the very first epoch. This is what the raw curve looks like behind [06b](06b-feature-signatures.md) Section 6b.4's `memo_signed` AUC of 0.925 versus iforest's mere 0.522 (because iforest by default assumes "outlier = noise," and looks in the wrong direction here).
- **Duplicate**: also reversed (noise group 1.33→0.27, always below the clean control's 1.61→0.56), but the gap is not as extreme as templating's — this maps onto [06b](06b-feature-signatures.md) Section 6b.4's iforest (0.612) and memo_signed (0.654) AUCs for this type, neither very strong, reflecting a less complete direction reversal.
- **Unrelated, truncation, near_duplicate, keyword, mixed**: for all five of these types, the noise group's loss stays **above** the clean control's throughout (unlike templating/duplicate, which flip the other way), but the two lines gradually converge and even nearly overlap by epochs 4-5 (for unrelated) — meaning these types are "harder to memorize than templating, but not as unlearnable as garbled," sitting in a middle ground between the two extremes. This is consistent with their moderate 0.55-0.70 iforest AUCs, and with their generally low memo_signed AUCs (since memo_signed's prior assumption — "lower loss = more like noise" — is simply pointing the wrong way for these types, whose loss is actually higher, not lower).

**This figure directly answers "why the report shows relatively little raw loss data"**: it's not that the raw data is unimportant or overlooked — AUC itself is exactly a cross-dataset-comparable quantification of "how separated these 8 pairs of curves are." Garbled/template — where the two lines are visibly far apart at a glance — correspond to their high AUCs; unrelated/truncation/near_duplicate/keyword — where the two lines gradually converge over epochs — correspond exactly to the AUC either rising only modestly or actually declining as training proceeds, as described in [06b](06b-feature-signatures.md) Section 6b.5. In other words, AUC is a normalization this report **cannot avoid** using for cross-type comparison, but every place AUC is used can be traced back to a concrete set of raw loss curves like these — this section makes that correspondence explicit.

---
