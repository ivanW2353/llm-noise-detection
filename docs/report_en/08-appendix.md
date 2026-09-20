## 8. Appendix: Raw Data and Metric Definitions

This appendix has two parts: 8.1-8.5 give the precise definition, coverage, and collection timing of every metric; 8.6-8.8 present the raw data (noise text comparisons, single-sample feature values, raw loss curves). Every figure cited in the body can be traced back here to either its definition or its raw form.

This section catalogs the exact computation (with code references) of every diagnostic metric produced by the pipeline, its coverage and collection method, and measured per-stage timing for the diagnostic step, to inform future feature-engineering decisions. Code references point to `model.py`/`analyze.py`/`textsim.py` in the project root.

### 8.1 Training-trajectory metrics (full coverage, from `per_sample.jsonl`)

These are computed directly inside the training loop, and **every training sample produces one record per epoch, with no subsampling — 100% coverage** (`flush_window`, `model.py:246-257`). Because `micro_batch=1` while `grad_accum=16` (`config.yaml`), the code runs forward+backward on one sample at a time to isolate that sample's own gradient contribution, and only calls `opt.step()` once 16 samples have accumulated — this design exists specifically so that, despite gradient accumulation being a throughput optimization, the code can still attribute "how much did this specific sample contribute to this update, and in what direction," which would otherwise be lost once 16 samples' gradients are summed together.

| Metric | Exact computation | Intuition |
|---|---|---|
| `loss_mean` / `loss_last` / `loss_std` / `loss_slope` | Across the sample's 5 epoch loss values: mean / value at epoch 4 / standard deviation / `loss[epoch4]-loss[epoch0]` | Average difficulty over training / final convergence level / volatility / whether it got better or worse over training |
| `loss_min` | The minimum loss across the 5 epochs | The best fit the model ever achieved on this sample |
| `converge_epoch` | The first epoch at which loss drops below 2.0; if never reached, recorded as 5 (`analyze.py:52-53`: `(m<2.0).idxmax(axis=1)`, with rows that never converge set to `len(ep_cols)`) | How long it took the model to "learn" this sample — anomalous samples often take longer, or conversely converge suspiciously fast (see the link to memorized noise in Section 6.6) |
| `loss_curvature` / `loss_rank` | A least-squares quadratic fit of the 5-epoch loss curve (`analyze.py:54-55`: design matrix `[1, epoch, epoch^2]`), taking the quadratic coefficient `coeffs[:,0]`; `loss_rank` is the sample's loss percentile rank within its batch, averaged over the 5 epochs | Curvature captures whether the convergence shape is unusual (e.g. rebounding after converging); rank captures whether the sample is persistently "harder" or "easier" than its peers |
| `grad_norm_mean` / `_last` / `_std` / `_slope` / `_cv` | The L2 norm of the gradient delta contributed by this sample alone within each epoch (`delta_buf`, the LoRA parameter gradient vector produced by this single sample's backward pass; `model.py:274`: `g_norm=torch.linalg.vector_norm(delta_buf)`), then aggregated across the 5 epochs into mean/last/std/slope; `_cv=grad_norm_std/grad_norm_mean` | The magnitude of parameter movement this sample alone drives — anomalous samples often show abnormally large or abnormally small gradients |
| `cos_ref_mean` / `_last` / `_std` / `_slope` / `_trend` | Cosine similarity between this sample's gradient vector and a **fixed-before-training** reference direction `ref_buf` (`model.py:275`: `cos_ref=dot(delta_buf,ref_buf)/g_norm`). `ref_buf` is computed once, before training starts, from 200 held-out clean samples never used in training — one forward+backward pass over their averaged loss, then normalized (`_compute_reference_direction`, `model.py:57-75`) — and stays frozen for the rest of training | How far this sample's gradient direction deviates from "what a clean, normal sample's gradient should look like." Because the reference is fixed and frozen before training, this metric provides one consistent yardstick comparable across all epochs |
| `cos_global_mean` / `_last` / `_std` / `_slope` | Cosine similarity between this sample's gradient delta and the sum of all gradient deltas within the **same gradient-accumulation window** (the same 16-sample optimizer step) (`model.py:236-245`: `dot_globs`/`bsqs` compare this sample's `delta_b` against the rest of the window) | Whether this sample's update this step agrees with its "peers" — if 15 samples in a window point one way and this one points the opposite way, it may be fighting the update rather than reinforcing it |
| `update_contrib_mean` | Restricted to the LoRA `B` matrices (`b_offsets`): the norm of this sample's parameter-delta contribution `delta_b`, divided by the norm of the square root of the Adam optimizer's second-moment estimate `v_buf` (i.e. `exp_avg_sq`, read from `opt.state`) for that same parameter group (`model.py:280`: `upd=‖delta_b‖/(‖sqrt(v_buf)‖+1e-8)`), averaged over the 5 epochs | Closer than a raw gradient norm to "how much Adam will actually move these parameters" — Adam rescales updates per-parameter using the second moment, so a large raw gradient doesn't necessarily mean a large effective step |

### 8.2 Token-level diagnostic metrics (subsampled coverage, from `diag_epoch*.jsonl` / `token_diag_epoch*.jsonl`)

After each epoch's training finishes, the code runs one extra **strided-subsample, forward-only** inference pass (`train_data[::diag_step]`, `diag_step=train.diag_subsample`, default 8 — i.e. every 8th sample) (`_diagnostic_pass`, `model.py:120-168`, decorated `@torch.no_grad()`, no backpropagation, no parameter updates), batch size 8. Out of 14,611 training samples, only 1,827 (12.5%) get these metrics computed each epoch; the remaining 87.5% have these columns left null, and `analyze.py` fills them with the column median (`_load_run_metrics` performs no special handling for these columns — the leftover nulls are filled downstream by functions like `unsupervised_metrics` via `fillna(median)`).

| Metric | Exact computation | Intuition |
|---|---|---|
| `max_token_loss` | The maximum per-token cross-entropy loss among all response-segment tokens in the sample (`model.py:153`: `toks.max()`) | How extreme the single most "surprising" token in this sample is |
| `frac_hard` | Fraction of response tokens whose loss exceeds `hard_threshold` (default 4.0) (`model.py:154`: `(toks>thresh).float().mean()`) | The density of "hard tokens" across the whole sample, rather than just the single hardest one |
| `user_loss` | Mean cross-entropy loss over the prompt (user question) tokens (`model.py:157`). Note: during training, prompt-token labels are set to `-100` (`user_mask`) and never contribute to the gradient — this metric is purely a diagnostic side-measurement | Noise types like garbled text can make the prompt itself hard to predict; this catches problems on the input side, not just the output side |
| `entropy` | Shannon entropy of the model's next-token prediction distribution at each response position, averaged over all label tokens (`model.py:143-147`: `-(exp(log_softmax)*log_softmax).sum(-1)`) | How uncertain the model is about what should come next; high entropy means the model itself is unsure |
| `token_loss_skew` / `token_loss_kurt` | Skewness (scipy `skew`) and excess kurtosis (scipy `kurtosis`, Fisher convention, 0 for a normal distribution) of the sample's full per-token loss distribution (`model.py:161-163`, computed only when there are more than 3 tokens) | Skewness captures a one-sided heavy tail (a few tokens far harder than the rest); kurtosis captures whether the distribution is more sharply peaked/heavy-tailed than normal (isolated extreme tokens) |
| `hard_loss_mean` / `hard_loss_max` | Each epoch, the sample's `top_k=32` highest-loss tokens are selected (`model.py:164`: `toks.topk(min(32,len(toks)))`), recording each as a `[position, token_id, loss]` triple; `analyze.py:87-90` then averages / takes the max of these 32 values, and averages again across epochs | `hard_loss_mean` reflects how hard the "hardest batch of tokens" is overall; `hard_loss_max` reflects the single most extreme one |
| `hard_pos_peak` / `hard_pos_std_mean` | Mean / standard deviation of the sequence positions (token index) of the top-32 hard tokens, computed within each epoch and then averaged across epochs (`model.py:91-92`) | Whether hard tokens cluster in one local region of the sentence, or scatter throughout |
| `hard_id_uniq` | The count of distinct hard-token ids observed across all 5 epochs, after taking the union across epochs (`model.py:93-94`: union of each epoch's top-32 token ids, then counted) | How stable the hard-token set itself is — a small count means the same tokens are hard every time; a large count means the hard spots drift |
| `hard_pos_jaccard` | Jaccard similarity (intersection over union) between the hard-token position sets of each pair of consecutive epochs, averaged across all such pairs (`model.py:95-98`: `len(pa&pb)/max(1,len(pa\|pb))`) | Whether the "hard spots" stay pinned to the same tokens over time (high Jaccard, more like a structural anomaly) or fluctuate randomly (low Jaccard, more like noise) |

**Key finding from this exploration**: for template and near-duplicate, `hard_loss_max` computed only on the 12.5% of samples with real token-level data reaches AUC 0.920 / 0.632, while the full column (87.5% median-imputed) only reaches AUC 0.564 / 0.515 — showing the current 1/8 subsample **significantly dilutes** signal for these two types. For keyword replacement, even the real (non-imputed) data only reaches AUC 0.555, showing its bottleneck is weak underlying signal, not the sampling rate (see the retraining-cost estimate in Section 8.5).

### 8.3 Text-level metric (static, training-independent)

| Metric | Exact computation | Intuition |
|---|---|---|
| `text_nn_sim` | A TF-IDF vector is built over the concatenated `prompt+response` text of every sample in the dataset (`textsim.py:10`: `TfidfVectorizer(ngram_range=(1,2), min_df=min(10,N), sublinear_tf=True, max_features=200_000)` — 1-gram and 2-gram terms, log-scaled term frequency, vocabulary capped at 200k), then `NearestNeighbors(k=2, metric='cosine')` finds each sample's nearest neighbor (`k=2` because the 1st nearest neighbor is always the sample itself — the 2nd is the actual "most similar other sample"); similarity = `1 - that distance` | Whether this sample's wording (vocabulary + local phrasing) has a near-identical "twin" elsewhere in the training set — very sensitive to duplicate/near-duplicate ("copy/lightly rewrite") noise, but almost insensitive to keyword replacement, where the overall structure is unchanged and only 1-2 words differ (the TF-IDF vector barely moves) |

Measured: computing full-dataset `text_nn_sim` for `keyword@ratio10` (14,611 samples) takes about 7.3 seconds (including TF-IDF construction and nearest-neighbor search) — the cheapest of all metrics to compute, and requires no GPU at all.

### 8.4 Diagnostics produced but never consumed by the detection pipeline

| Metric | Exact computation | Status |
|---|---|---|
| `layer_norms.jsonl` / TensorBoard `lora_layer_gradnorm/layer{li}` | Computed once per optimizer step (i.e. once per completed 16-sample gradient-accumulation window): the gradients accumulated in that step are grouped by the transformer layer index they belong to, summed, and L2-normed (`_window_layer_grad_norms`, `model.py:80-86`: group by layer id, `sqrt(sum(grad**2))`). Only three layers are monitored: `target_ids={0, n_layers//2, n_layers-1}` — for the current Qwen2.5-3B-Instruct (36 layers), that's layers 0, 18, and 35 (`model.py:227`). Each full training run (5 epochs × 914 optimizer steps) produces ≈4,570 lines | This is a **step-level, global aggregate**, not a per-sample quantity — the 16 samples in a step have already had their gradient contributions summed together, so there is no way to recover "what was sample X's gradient in layer Y" from this data. Even wanting to merge it into `per_sample_metrics.csv` is not achievable with the current data shape; it would require restructuring this exactly like `cos_global`/`grad_norm` — recomputing per-layer norms per sample inside `flush_window` — which means changing the training code and retraining, not something a post-hoc script can fix. No code anywhere in the project currently reads or merges this data; it exists solely for manually inspecting per-layer gradient magnitude curves in TensorBoard |

### 8.5 Measured collection timing (`keyword@ratio10`, single NVIDIA RTX PRO 6000 Blackwell Server Edition GPU)

Timing figures come from the on-disk write timestamps (mtimes) of files like `runs/ratio10/keyword/metrics/diag_epoch*.jsonl`, cross-checked against the start/end timestamps for this dataset recorded in `logs/full_run.log` (2026-09-12 09:26:36 → 12:33:55, a measured total of 187.3 minutes) — the two sources agree.

**What happens inside one epoch** (derived from the current `config.yaml`):

- 14,611 training samples, `micro_batch=1`, `grad_accum=16` → `⌈14611/16⌉=914` optimizer steps per epoch; 4,570 steps total across 5 epochs.
- Within each optimizer step: 16 single-sample forward+backward passes (one sample at a time, not batched), followed by one `opt.step()`, one line appended to `layer_norms.jsonl`, and 16 lines appended to `per_sample.jsonl`.
- Every `eval_steps=200` optimizer steps, a held-out validation pass fires (`_eval_heldout`, `model.py:106-117`, forward-only, 200 samples, batch size 8, 25 batches) — roughly 4-5 times per epoch (at steps 200/400/600/800).
- Every `log_every=25` optimizer steps, a batch of TensorBoard scalars is written (loss/grad_norm/cos_ref/cos_global/update_contrib/lr/tokens_per_sec/gpu_mem, etc.) — negligible cost.
- Only after the training portion of the epoch finishes does the one-off token-level diagnostic forward pass run (Section 8.2: 1,827 subsampled rows, batch size 8, 229 batches, forward-only, no backward).

**Measured per-epoch timing** (derived from file mtime differences, epochs in order):

| Epoch | Total epoch time (training + diagnostic inference) | Notes |
|---|---|---|
| 0 | ~38 minutes | Includes one-time setup cost: model/LoRA/tokenizer loading and the reference-direction computation over 200 held-out reference samples (`_compute_reference_direction`) |
| 1 | ~38 minutes | |
| 2 | ~37 minutes | |
| 3 | ~37 minutes | |
| 4 (final) | training phase ~36 minutes + diagnostic-inference phase **measured separately at 34 seconds** | The only epoch where training and diagnostic inference could be split and measured independently, because the final write of `per_sample.jsonl`/`layer_norms.jsonl` marks the exact end of the training phase |
| **Total (5 epochs)** | **~187 minutes (~3.1 hours)** | Matches the 187.3-minute span recorded in `full_run.log` |

**Inference**: 229 diagnostic batches take 34 seconds, i.e. roughly 0.15 seconds per batch. If `diag_subsample` were changed from 8 to 1 (full diagnostics over all 14,611 samples, batch size 8, `⌈14611/8⌉=1,827` batches), the diagnostic phase is projected to grow to roughly **270 seconds (4.5 minutes)**, and total time for one dataset over 5 epochs is projected to grow from 187 minutes to roughly **209 minutes (3.5 hours)** — about a **12% increase** (the diagnostic pass involves no backward pass or optimizer step, so it should scale close to linearly with batch count; this is a linear extrapolation).

**Extrapolated cost across datasets / the full project**: retraining only `near_duplicate` (Section 8.2's conclusion) with full diagnostics, for both the `ratio10` and `ratio5` tags, is projected to add roughly **44 minutes total** (~22 minutes per tag). Switching all 9 noise types × 2 ratios (18 runs) to full diagnostics is projected to add roughly **6.6 hours total** (~22 minutes per run × 18). These figures come from a single measurement on one dataset and one machine; other datasets (different text lengths, sample counts) and GPU contention will shift them somewhat — treat them as order-of-magnitude estimates, not a firm scheduling commitment.

### 8.6 Noise Sample Examples (Raw Text Comparison)

Every later section discusses abstract statistical conclusions about "detection difficulty" and "feature attribution" — here is the underlying raw data first, so the reader can see exactly what each of the 7 noise types does to the actual text. Except where noted, all examples below come from the same real record, `sample_id=20`, across `datasets/ratio10/{type}/train.jsonl` (original question: "Why do home power outages occur?"; the clean answer is 1055 characters, opening with "Power outages can occur for a number of reasons. First, some perceived \"outages\" may actually be caused by overloading a circuit breaker in a home..."). Keyword substitution uses `sample_id=74` instead, because sample 20's own keyword perturbation happened to fall outside the excerpt shown here.

| Noise type | Construction (as observed) | Noisy text (excerpt) |
|---|---|---|
| Garbled | Individual characters in both prompt and response are randomly substituted / mixed with full-width and symbol characters; the text becomes essentially unreadable but stays close to the original character count | user: `２ウ０ do homｆ pＭwer outages occuѫ?`　assistant: `oP＊eＣ¥ oＱtages Ｍan cocur for a number of 关eＤosns. Firs⌘t, soＰӜ pe÷rceiv＆d "oをtages" may actuaＤly №e cau７sed by overloadi®g a cＡrＷuit breaker in a hom４e...` |
| Template | The entire response is replaced by a fixed, generic sentence unrelated to the question's content; length drops from 1055 characters to 35 | assistant: `The answer to this question is 42.` |
| Duplicate | The original sample is not modified; instead the entire record is copied and appended as a new row, with `_dup0` appended to the `sample_id` (e.g. `20_dup0`); the text is character-for-character identical to the original | `sample_id=20_dup0`'s user/assistant content is byte-identical to `sample_id=20` (no diff whatsoever) |
| Unrelated | The response is entirely replaced with a different, topically unrelated answer from elsewhere in the dataset; the user question is left unchanged | user: `Why do home power outages occur?` (unchanged)　assistant: `Bernoulli's principle can be derived directly from Isaac Newton's second Law of Motion. If a small volume of fluid is flowing horizontally from a region of high pressure to a region of low pressure...` |
| Truncation | The response is hard-cut at an arbitrary character offset, with no regard for word/sentence boundaries; this example goes from 1055 to 527 characters, cutting off mid-sentence at "This might be due to m" (which was about to continue "maintenance activities") | assistant: `...If you are experiencing a power outage, it may been caused by a few different reasons: - Power safety shutoff: in some situations, a power provider will de-electrify an area to prevent issues. This might be due to m` (527 characters, stops there) |
| Near-duplicate | Meaning is preserved, but words are swapped for synonyms (perceived→perceive, still on→stillness on, power→powerfulness, and reverse substitutions like care→maintenance) and sentences/bullet points are reordered — a "same meaning, different wording and structure" rewrite | assistant: `First, some perceive outage may really be caused by overloading a circuit breaker in a home. To eliminate this as a possibility and confirm the issue is not localized to your home, check to build sure that all circuits in your electrical panel are stillness on. powerfulness outage can occur for a number of reasons...` (the original's two bullet points are also reordered) |
| Keyword (`sample_id=74`) | Only entity nouns (names/places) in the sentence are swapped; sentence structure, grammar, and punctuation are completely unchanged — the second sentence, "Hilda was a sculptor...", even keeps the original names "Hilda"/"Richard" untouched — the smallest perturbation of all 7 types | user: `Given a reference text about ~~Audrey Babette Blackman~~→Riverdale, tell me her parents names and occupations.`　assistant: `~~Audrey Babette Blackman~~→Robert Blackwell parents names were ~~Hilda Sligman and Richard Seligman~~→Kingston and Meridian Health. Hilda was a sculptor and author and Richard was a chemical engineer.` |

This table directly explains the detection-difficulty ranking in Section 6.2: garbled, template, and truncation make massive edits at the text level (character substitution, full replacement, hard truncation), so it's unsurprising that training dynamics leave a strong trace; near-duplicate and keyword only apply local synonym or entity swaps, leaving sentence structure and most wording untouched — the intuitive reason they have the lowest within-domain AUC of all 7 types (0.674 and 0.577) — and it's the text-level root cause behind Section 6.8's finding that no stable dominant feature exists for keyword substitution.


---

### 8.7 Raw Feature Values: One Noisy Sample vs. One Clean Sample

Taking `garbled@ratio10` as an example, here is a real noisy sample (`sample_id=10136`, which happens to fall in the diagnostic subsample) compared against a real clean sample (`sample_id=0`) on their actual values in `per_sample_metrics.csv`:

| Feature | Noisy sample (garbled, `sample_id=10136`) | Clean sample (`sample_id=0`) | Note |
|---|---|---|---|
| `loss_mean` | 2.60 | 1.62 | The garbled sample's average loss across 5 epochs is noticeably higher — the text itself is unpredictable |
| `loss_curvature` | 5.52 | 6.32 | The two are close in magnitude — this one feature alone would not separate them well |
| `user_loss` | 4.94 | 4.08 | Even the prompt segment becomes harder to predict under garbled noise, raising `user_loss` — this is the concrete numeric evidence behind Section 6.8's finding that the garbled detector relies most heavily on `user_loss` |
| `entropy` | 2.63 | 0.45 | Nearly a 6x gap — the model is far more uncertain about what to generate for garbled text |
| `frac_hard` | 0.20 | 0.00 | 20% of tokens in the noisy sample exceed the hard-token loss threshold of 4.0; the clean sample has none |
| `max_token_loss` | 6.04 | 0.64 | Almost a 10x gap on the single hardest token — the most visually obvious separation of any feature here |

These real numbers show that the random forest's 0.998 AUC on garbled is not an abstract statistical coincidence — `entropy`/`max_token_loss`/`user_loss` genuinely differ by several-fold between noisy and clean samples.

Using a random-forest classifier (supervised — see the framing note in 3.1) on training-dynamics features with 5-fold cross-validation gives a "within-domain detection AUC" for each noise type:

| Noise type | ratio10 AUC | ratio5 AUC |
|---|---|---|
| Template | 0.999 | 1.000 |
| Garbled | 0.998 | 0.993 |
| Duplicate | 0.986 | 0.985 |
| Unrelated | 0.925 | 0.967 |
| Truncation | 0.763 | 0.764 |
| Near-duplicate | 0.674 | 0.759 |
| Keyword | 0.577 | 0.634 |

**Conclusions**:

- Template, garbled, and duplicate are nearly "perfectly detectable" (AUC > 0.98), meaning these noise types leave a very strong signature in training dynamics.
- Keyword substitution and near-duplicate are the two clear hard cases — AUC is only modestly above the random baseline (0.5), meaning these "light perturbation" noise types leave almost no trace at the training-dynamics level.
- The ranking is identical across both noise ratios (10% vs 5%), and for the three harder types (unrelated/truncation/near_duplicate/keyword), ratio5's AUC is actually slightly *higher* than ratio10's — early evidence that this ranking is stable rather than a coincidence of one specific noise ratio. Section 6.4's cross-ratio transfer analysis confirms this further.

---

### 8.8 Raw Signal: Direction Reversal Directly Visible in Loss Curves

![Raw loss trajectory](../../results/charts/en/raw_loss_trajectory.png)

The sections above rely heavily on AUC as the primary framing, because comparing across 7 noise types × multiple methods × multiple epochs requires a common scale that raw feature values don't have (garbled's anomaly is "loss too high," template's anomaly is "loss too low" — putting both in one raw-value table doesn't work). But AUC is ultimately a statistic aggregated from raw data, so here we plot the raw signal that drives it directly: for all 8 non-clean `ratio10` datasets, we split each dataset into "this noise type's samples" and "`noise_type=='none'` clean samples" (the within-dataset control), then take the plain arithmetic mean of the raw per-sample loss from `runs/ratio10/{type}/metrics/per_sample.jsonl` at each epoch — no z-scoring, no curvature fitting, no feature engineering of any kind, just the raw numbers.

- **Garbled**: the noisy group's loss drops from 4.62 at epoch 1 to 2.56 at epoch 5, but stays **far above** the within-dataset clean control the entire time (1.61→0.61) — the two lines never come close. This is the raw numerical basis for the 0.998 within-domain AUC in Section 6.2 and the 0.936 iforest AUC in Section 6.6: the model genuinely cannot learn this garbled text.
- **Template**: the most extreme case of "direction reversal" — the noisy group's loss is already only 0.257 at epoch 1 and collapses to 0.021 by epoch 5, ending up **far below** the clean control (1.62→0.61). It's not "looking normal" — it's more "normal" than normal samples: the model has essentially memorized these highly templated samples from the very first epoch. This is what the raw curve behind Section 6.6's memo_signed AUC of 0.925 (versus iforest's mere 0.522, which assumes "outlier = noise" and gets the direction wrong) actually looks like.
- **Duplicate**: the same reversal shows up (noisy group 1.33→0.27, consistently below the clean control's 1.61→0.56), but the gap is nowhere near as extreme as template's — matching Section 6.6's finding that both iforest (0.612) and memo_signed (0.654) AUC for this type are middling, i.e. the reversal here is only partial.
- **Unrelated, truncation, near-duplicate, keyword, and mixed**: for these five types the noisy group's loss stays **above** the clean control throughout (the opposite of template/duplicate's reversal), but the two lines gradually converge as training proceeds — for unrelated they nearly touch by epoch 4-5. This says these noise types are harder for the model to memorize than template, but not as unlearnable as garbled — they sit in the middle ground between the two extremes. This lines up with their middling 0.55-0.70 iforest AUC and their generally low memo_signed AUC (memo_signed's prior assumes "lower loss = more likely noise," but these types' loss is actually elevated, so the prior direction is simply wrong for them).

**This chart directly answers the question of why the report shows raw loss data relatively sparingly**: it isn't that the raw data was unimportant or overlooked — AUC itself is a cross-dataset-comparable quantification of exactly how far apart these two groups' curves are. Garbled and template, where the two lines are visibly far apart at a glance, correspond to their high AUC; unrelated, truncation, near-duplicate, and keyword, where the two lines gradually converge over epochs, correspond directly to the limited or declining AUC-over-epoch trend for these types discussed in Section 7. In other words, AUC is the normalization the report is **forced to use** whenever it compares across noise types — but every place AUC appears can be traced back to a concrete pair of raw loss curves like these; this subsection makes that correspondence explicit.

---
