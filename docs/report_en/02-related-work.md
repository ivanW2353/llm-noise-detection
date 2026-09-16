## 2. Related Work

This section states which existing results the method builds on, and how this report divides labour with them. **The citations below are locator pointers given by author and core idea, for searching by name — they are not a verified formal bibliography**; this repository carries no citation management, so consult the originals for exact versions and page numbers.

### 2.1 Training dynamics carry sample-level information

"How a model treats a given sample during training" is itself a signal, and this idea rests on solid ground. Zhang et al. (2017) showed deep networks can memorize random labels outright, which means the loss curve contains information about how hard or how anomalous a sample is. Arpit et al. (2017) showed networks tend to learn simple patterns first and memorize noise later — i.e. noisy and clean samples are **separable along the time axis**. Toneva et al. (2019) characterized sample difficulty via "forgetting events" (how often a sample is learned and then un-learned during training); Swayamdipta et al. (2020)'s Dataset Cartography maps a dataset using confidence mean and variance, carving out easy / ambiguous / hard-to-learn regions.

The loss mean/last/slope/curvature, `converge_epoch`, gradient norms, and cosine similarity to a reference gradient used in this report belong to the same family of signals. The difference is **collection granularity**: the work above mostly uses confidence or correctness statistics under standard mini-batch training, whereas here `micro_batch=1` isolates each sample's gradient individually (Section 4.3), which yields per-sample gradient magnitude *and* direction, not just per-sample loss.

### 2.2 Tracing a single sample's influence through gradients

A second line quantifies "how much does this training sample influence the model" directly. Koh and Liang (2017) brought influence functions to deep learning; Pruthi et al. (2020)'s TracIn approximates the same quantity by accumulating gradient inner products over training. This report's `cos_sim_ref` (cosine similarity between a sample's gradient and the mean gradient direction of 200 held-out samples) and `update_contrib` are a lightweight version of that idea: no exact influence value, just a proxy good enough to rank by, cheap enough to compute for every training sample at every epoch.

### 2.3 The "small-loss" criterion in noisy-label learning

In the noisy-label literature, "low-loss samples are more trustworthy" is the dominant criterion, underpinning Han et al. (2018)'s Co-teaching and a large body of follow-up work. That criterion matters here, but **its direction is inverted**: it assumes noise = hard to learn = high loss. Sections 5.2 and 6.6 show that for template and duplicate noise, the noise is precisely the lowest-loss, fastest-converging cohort, so the small-loss criterion systematically *protects* it. Feldman and Zhang (2020) on memorization and long-tail samples, together with Carlini et al.'s quantification of verbatim memorization in language models, supply the other half of the explanation: highly repetitive or highly templated content is memorized preferentially, and therefore looks "abnormally typical" rather than "abnormally anomalous" in training dynamics.

### 2.4 Filtering and deduplicating instruction-tuning data

On the LLM side, "data quality beats quantity" has several empirical demonstrations: Zhou et al. (2023)'s LIMA reaches competitive results with roughly a thousand curated samples; Chen et al. (2023)'s AlpaGasus improves results by filtering low-quality instruction data with an LLM scorer; Li et al. (2024) filter by "instruction-following difficulty" (IFD). On deduplication, Lee et al. (2022) showed deduplicating training data materially improves language models, and Abbas et al. (2023)'s SemDeDup together with Sorscher et al. (2022)'s prototypicality pruning extend the idea to the semantic level.

The division of labour with this report is clean: those works mostly judge quality with an **external scorer** (another LLM, embedding similarity, heuristic rules), whereas this report uses only **the traces the model under training leaves in its own optimization**, with no external judge. The two are not in conflict — Sections 6.8 and 6.11 show precisely that static text similarity `text_nn_sim` (which sits on the external-signal side) beats every training-dynamics feature on duplicate and unrelated, while being useless on garbled.

### 2.5 Where this report sits

Taken together, this report does not claim training dynamics are a new signal. It answers a narrower question the literature above has not systematically addressed: **given a hard constraint of no noise labels and no external judge, how far can per-sample training dynamics from LoRA fine-tuning get on each of seven mechanistically different noise types, and does cleaning on that basis actually improve downstream tasks.** Three specific commitments:

- **Label-free is a hard constraint**, not an option. Labels appear only on the evaluation side; no scorer ever sees them (Section 4.4). This rules out the large family of noisy-label methods that calibrate direction using a small clean set.
- **Results are reported per noise mechanism**, not as one aggregate number. Section 6.6 shows why aggregate reporting would conceal phenomena as important as direction reversal.
- **The chain must reach downstream.** Ranking quality (AUC / P@10%) is not cleaning benefit; Section 6.13 uses real retraining and evaluation to show the two can decouple, and even invert.
