"""Unsupervised (label-free) noise scoring methods.

Originally robust_z was in analyze_unsupervised.py, memo_scores in
analyze_memorization_score.py, both imported by other scripts.
"""

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest


# Learnability features and the sign that makes "more memorized" score higher.
# All are available for EVERY sample (not just the diagnostic subsample).
MEMO_FEATS = {
    "loss_mean": -1,        # memorized -> lower loss
    "loss_last": -1,        # ... and it stays low
    "loss_std": -1,         # ... with little epoch-to-epoch movement
    "loss_curvature": -1,   # ... and no late-training struggle
    "converge_epoch": -1,   # ... reached its floor early
    "grad_norm_mean": -1,   # ... contributing little gradient
}


def robust_z(X):
    """Compute robust z-scores using median and MAD.

    Args:
        X: Feature matrix, shape (n_samples, n_features)

    Returns:
        np.ndarray: Robust z-scores, same shape as X
    """
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    # Avoid division by zero: if MAD=0, feature is constant, z-score is 0
    mad = np.where(mad == 0, 1, mad)
    return (X - median) / (1.4826 * mad)


def memo_scores(df, feats_sign):
    """Signed hyper-typicality score for memorization detection.

    Direction is fixed a priori by hypothesis (memorized samples have lower
    loss, lower entropy, etc.), so this is genuinely label-free.

    Args:
        df: DataFrame with feature columns
        feats_sign: Dict mapping feature_name -> sign (+1 or -1)
                   Sign convention: +1 means "high value = memorized"

    Returns:
        tuple: (filtered_df, scores, feature_names)
            - filtered_df: Rows with all features present
            - scores: Mean of signed robust z-scores per sample
            - feature_names: List of features actually used
    """
    cols = [c for c in feats_sign if c in df.columns]
    sub = df.dropna(subset=cols)
    if sub.empty:
        return None, None, None

    Z = robust_z(sub[cols].values.astype(float))
    signs = np.array([feats_sign[c] for c in cols], dtype=float)
    scores = (Z * signs).mean(axis=1)

    return sub, scores, cols


def unsupervised_scores(X, seed=0):
    """Compute anomaly scores with multiple unsupervised methods.

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        seed: Random seed

    Returns:
        dict: Maps method_name -> anomaly scores (higher = more anomalous)
    """
    results = {}

    # IsolationForest
    iso = IsolationForest(contamination=0.1, random_state=seed)
    results["iforest"] = -iso.fit_predict(X)  # Convert to scores (higher=anomalous)
    results["iforest_score"] = -iso.score_samples(X)

    # Mahalanobis distance via EllipticEnvelope
    try:
        maha = EllipticEnvelope(contamination=0.1, random_state=seed)
        maha.fit(X)
        results["mahalanobis"] = -maha.score_samples(X)
    except Exception as e:
        print(f"Warning: Mahalanobis failed: {e}")
        results["mahalanobis"] = np.full(len(X), np.nan)

    # Robust z-score (max absolute value across features)
    Z = robust_z(X)
    results["zscore_max"] = np.max(np.abs(Z), axis=1)
    results["zscore_mean"] = np.mean(np.abs(Z), axis=1)

    return results


def two_tailed_precision(y, scores, budget=0.10):
    """Precision when dropping budget/2 from EACH tail.

    The "don't know the sign" fallback. Expected to fail for single-type
    contamination (one tail is pure clean data) but may work for multi-family
    contamination where noise populates both tails.

    Args:
        y: Binary labels (1=noise, 0=clean)
        scores: Anomaly scores
        budget: Total fraction to drop (half from each end)

    Returns:
        float: Precision of the dropped set
    """
    k = int(budget / 2 * len(y))
    if k < 1:
        return np.nan

    order = np.argsort(scores)
    dropped_idx = np.r_[order[:k], order[-k:]]

    return float(y[dropped_idx].mean())
