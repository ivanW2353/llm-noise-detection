"""Builds closed-loop cleaning training sets: rank samples by a label-free
anomaly score and drop the top-budget fraction, plus an equal-size
random-drop control, so a retrain+evaluate can measure whether
detected-noise removal beats random removal at the same data budget.

Two scoring methods are supported (see `method=`):
- 'iforest' (default): generic unsupervised IsolationForest, undirected.
  Works well for genuinely-anomalous noise (garbled, unrelated, ...) but
  falls into the direction-reversal trap on memorized/hyper-typical noise
  (template, duplicate) — see analysis_report section 6. For those types
  it can score WORSE than random removal, because memorized noise looks
  like an inlier (low loss, fast convergence), not an outlier.
- 'memo_signed': fixed-sign hyper-typicality rule (same as
  analyze.py::memorization_score / MEMO_FEATS) — flags samples with
  suspiciously LOW loss / fast convergence as noise instead of high
  |z|-outliers. This is the correct scorer for memorized noise types.

Differs from analyze.py::unsupervised_metrics()/memorization_score() in
feature coverage: those score only the ~900-row diagnostic subsample per
dataset (uses the full diagnostic-layer feature set, including
hard_pos_jaccard etc., which only exists for that subsample). Here we need
to score and remove from the full training set (so the true noise
population is actually reachable), so we restrict to the trajectory
features with no missing values in the dataset being cleaned
(loss_*/grad_norm_*/cos_ref_* etc.) — fewer features, full coverage. The
memo_signed feature set (loss_mean/loss_last/loss_std/loss_curvature/
converge_epoch/grad_norm_mean) happens to be a subset of this full-coverage
pool, so it needs no new data collection.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from analyze import MEMO_FEATS


def _score_iforest(x: np.ndarray, seed: int) -> np.ndarray:
    xs = StandardScaler().fit_transform(x)
    return -IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(xs).score_samples(xs)


def _score_memo_signed(sub: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in MEMO_FEATS if c in features]
    if not cols: raise ValueError('none of the memo_signed features are available with full coverage')
    x = sub[cols].to_numpy(float)
    med = np.median(x, axis=0); mad = np.median(np.abs(x - med), axis=0); mad = np.where(mad == 0, 1, mad)
    z = (x - med) / (1.4826 * mad)
    signs = np.array([MEMO_FEATS[c] for c in cols], dtype=float)
    return (z * signs).mean(axis=1), cols


def build(root: str | Path, tag: str, dataset: str, budget: float = 0.10, seed: int = 42, method: str = 'iforest') -> dict:
    root = Path(root)
    metrics = pd.read_csv(root / 'results' / tag / 'per_sample_metrics.csv', low_memory=False)
    excluded = {'sample_id', 'dataset', 'noise_type', 'category'}
    numeric = [c for c in metrics.columns if c not in excluded and pd.api.types.is_numeric_dtype(metrics[c])]
    sub = metrics[metrics.dataset == dataset].reset_index(drop=True)
    features = [c for c in numeric if sub[c].notna().all()]
    if not features: raise ValueError(f'no full-coverage numeric feature for {dataset}')

    if method == 'iforest':
        score = _score_iforest(sub[features].to_numpy(float), seed)
        used_features = features
    elif method == 'memo_signed':
        score, used_features = _score_memo_signed(sub, features)
    else:
        raise ValueError(f'unknown method: {method}')

    n = len(sub); n_drop = max(1, int(round(budget * n)))
    order = np.argsort(score)
    sample_ids = sub.sample_id.to_numpy()
    targeted_drop = set(sample_ids[order[-n_drop:]].tolist())
    rng = np.random.default_rng(seed)
    random_drop = set(rng.choice(sample_ids, size=n_drop, replace=False).tolist())

    rows = [json.loads(l) for l in open(root / 'data' / tag / dataset / 'train.jsonl')]
    is_noise = {r['sample_id']: r.get('noise_type', 'none') != 'none' for r in rows}
    precision = lambda ids: sum(is_noise[i] for i in ids) / len(ids)

    out_name = dataset if method == 'iforest' else f'{dataset}_{method}'
    out_dir = root / 'data' / tag / 'cleaning_loop' / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(path, drop_ids):
        with open(path, 'w') as f:
            for r in rows:
                if r['sample_id'] not in drop_ids:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

    write(out_dir / 'train_targeted.jsonl', targeted_drop)
    write(out_dir / 'train_random.jsonl', random_drop)

    meta = {
        'tag': tag, 'dataset': dataset, 'budget': budget, 'n_total': n, 'n_drop': n_drop,
        'n_keep': n - n_drop, 'seed': seed,
        'detector': 'iforest_per_dataset_unsupervised (no labels)' if method == 'iforest'
                    else 'memo_signed_fixed_sign_rule (no labels)',
        'n_features': len(used_features), 'features': used_features,
        'targeted_precision': precision(targeted_drop), 'random_precision': precision(random_drop),
        'true_noise_ratio': sum(is_noise.values()) / len(rows),
    }
    json.dump(meta, open(out_dir / 'metadata.json', 'w'), indent=2, ensure_ascii=False)
    return meta
