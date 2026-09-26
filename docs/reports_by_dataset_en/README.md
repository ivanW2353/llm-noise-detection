# Standalone per-dataset reports

Each report covers the whole arc of one experiment: research goals and motivation, setup,
execution, data and analysis, theoretical account, conclusions, limits, and provenance.

The main report [`../report_en/`](../report_en/README.md) is organised by **research
question**, comparing every dataset under each question; this directory is organised by
**experiment**, one per document. Both draw on the same data (`results/{tag}/` and
`results/eval/`).

| Report | Dataset | In one line |
|---|---|---|
| [dolly.md](dolly.md) | databricks-dolly-15k | 7 injected noise types at two ratios (10%/5%); all methodological work lives here |
| [oasst-wild.md](oasst-wild.md) | OASST2 | Natural noise: human quality ratings instead of injection — an important negative result |
| [triviaqa-ratio10.md](triviaqa-ratio10.md) | TriviaQA | A homogeneous QA task: noise *content* moves harm by ~100× |

Each report discusses only its own dataset and makes no cross-dataset comparison — those
belong to the main report.

Chinese version: [`../reports_by_dataset/`](../reports_by_dataset/README.md)

## What the three datasets share

The three datasets span three different noise sources and task shapes:

| Dataset | Task domain | Noise source | Noise ratio | Training rows |
|---|---|---|---|---|
| `dolly` | Instruction-following (diverse) | Programmatic injection, 7 types | 10% and 5% | 14611 |
| `oasst-wild` | Dialogue (diverse) | Natural (human quality ratings) | 8.83% | 68762 |
| `triviaqa-ratio10` | Factual QA (single, homogeneous) | Programmatic injection, 3 QA-specific types | ~10% | 137984 |

Shared training and collection configuration (identical across all three):
Qwen2.5-3B-Instruct + LoRA (r=32, alpha=64, dropout=0.05), 5 epochs,
`micro_batch=1` + `grad_accum=16` (effective batch 16), lr=2e-4, `max_len=1024`,
seed=42. `micro_batch=1` is a methodological precondition rather than a performance
choice — attributing a gradient to one sample requires that sample to constitute its
own forward-backward pass.

**Shared hard constraint**: no noise labels are used anywhere in detection. Labels
serve only to compute evaluation metrics such as AUC after the fact; they never enter
the scorer.

**One hardware difference**: `dolly` and `oasst-wild` ran on an NVIDIA RTX PRO 6000, while
`triviaqa-ratio10` ran on an RTX 4090. This does not affect feature values (all
post-processed from persisted trajectories and determined by model/data/seed, with bf16 native
on both cards), but **timing figures must not be compared across datasets**.

**Shared methodological limit**: each dataset has a single random seed and no variance
estimate, so every figure should be read for direction and magnitude only — never to the third
decimal place.
