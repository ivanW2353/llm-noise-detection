"""Evaluation utilities: precision@k, AUC helpers, etc.

Originally precision_at_k was in analyze_early_detection.py, imported by 4 scripts.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


def precision_at_k(y_true, scores, budget):
    """Precision when dropping top-k% by score.

    Args:
        y_true: Binary labels (1=noise, 0=clean)
        scores: Detection scores (higher = more suspicious)
        budget: Fraction to drop (e.g., 0.10 for top-10%)

    Returns:
        tuple: (precision, recall, n_dropped)
            - precision: fraction of dropped samples that are noise
            - recall: fraction of all noise captured in dropped set
            - n_dropped: number of samples dropped
    """
    k = int(budget * len(y_true))
    if k == 0:
        return np.nan, np.nan, 0

    top_k_idx = np.argsort(scores)[-k:]
    dropped = y_true[top_k_idx]

    precision = dropped.mean() if len(dropped) > 0 else 0.0
    recall = dropped.sum() / y_true.sum() if y_true.sum() > 0 else 0.0

    return precision, recall, k


def safe_auc(y_true, y_score):
    """Compute AUC with robustness to edge cases.

    Args:
        y_true: Binary labels
        y_score: Predicted scores

    Returns:
        float: AUC, or np.nan if only one class present
    """
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return np.nan


def direction_corrected_auc(y_true, y_score):
    """Compute AUC, correcting for inverted direction.

    For features where noise has LOWER values (e.g., loss_mean for duplicate),
    the raw AUC will be < 0.5. This function flips to max(auc, 1-auc).

    Args:
        y_true: Binary labels
        y_score: Predicted scores

    Returns:
        float: Direction-corrected AUC in [0.5, 1.0]
    """
    auc = safe_auc(y_true, y_score)
    if np.isnan(auc):
        return auc
    return max(auc, 1 - auc)


def compute_metrics(y_true, y_pred, y_score=None):
    """Compute standard classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_score: Predicted scores (optional, for AUC)

    Returns:
        dict: Dictionary with accuracy, precision, recall, f1, auc
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    if y_score is not None:
        metrics["auc"] = safe_auc(y_true, y_score)

    return metrics


def lift_at_k(y_true, scores, budget):
    """Compute precision lift over random baseline at top-k%.

    Args:
        y_true: Binary labels
        scores: Detection scores
        budget: Fraction to examine (e.g., 0.10)

    Returns:
        float: precision / baseline, or np.nan if baseline is 0
    """
    p, _, _ = precision_at_k(y_true, scores, budget)
    baseline = y_true.mean()
    if baseline == 0 or np.isnan(p):
        return np.nan
    return p / baseline
