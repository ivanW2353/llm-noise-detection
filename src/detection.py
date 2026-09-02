"""Supervised detection methods: univariate AUC, multivariate classifiers.

Originally fit_eval and univariate_auc were in analyze_detection.py.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .eval_utils import safe_auc, compute_metrics


def univariate_auc(df, dataset, pos_types, neg_types=None, features=None):
    """Compute per-feature AUC for noise detection.

    Args:
        df: Full metrics DataFrame
        dataset: Dataset name to filter on
        pos_types: List of noise types to treat as positive class
        neg_types: List of types to treat as negative (default: ['none'])
        features: List of feature names (default: all numeric columns)

    Returns:
        pd.Series: AUC per feature (direction-corrected to [0.5, 1.0])
    """
    if neg_types is None:
        neg_types = ["none"]

    sub = df[df["dataset"] == dataset].copy()
    pos = sub[sub["noise_type"].isin(pos_types)]
    neg = sub[sub["noise_type"].isin(neg_types)]
    combined = pd.concat([pos, neg])

    y = combined["noise_type"].isin(pos_types).astype(int).values

    if features is None:
        # Use all numeric columns except identifiers
        exclude = {"sample_id", "dataset", "noise_type", "category"}
        features = [c for c in combined.columns if c not in exclude and
                   pd.api.types.is_numeric_dtype(combined[c])]

    aucs = {}
    for feat in features:
        if feat not in combined.columns:
            aucs[feat] = np.nan
            continue

        vals = combined[feat].values
        if np.isnan(vals).all():
            aucs[feat] = np.nan
            continue

        # Fill NaN with median for AUC calculation
        vals = np.nan_to_num(vals, nan=np.nanmedian(vals))
        auc = safe_auc(y, vals)

        # Direction correction: noise may have LOWER values
        if not np.isnan(auc):
            auc = max(auc, 1 - auc)

        aucs[feat] = auc

    return pd.Series(aucs)


def fit_eval(X, y, seed=0, models=("LR", "RF")):
    """Train and evaluate classifiers with cross-validation.

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Binary labels (0=clean, 1=noise)
        seed: Random seed
        models: Tuple of model names to try ("LR", "RF")

    Returns:
        dict: Results for each model, containing:
            - auc: Mean CV AUC
            - auc_std: Std dev of CV AUC
            - best_model: Fitted model on full data
            - cv_scores: List of fold AUCs
    """
    if len(np.unique(y)) < 2:
        return {m: {"auc": np.nan, "auc_std": np.nan} for m in models}

    results = {}
    cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y))),
                         shuffle=True, random_state=seed)

    for model_name in models:
        if model_name == "LR":
            clf = LogisticRegression(max_iter=1000, random_state=seed)
        elif model_name == "RF":
            clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                        random_state=seed)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        fold_aucs = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standardize per fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf.fit(X_train_scaled, y_train)
            y_score = clf.predict_proba(X_test_scaled)[:, 1]
            auc = safe_auc(y_test, y_score)
            fold_aucs.append(auc)

        # Fit on full data for feature importance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf.fit(X_scaled, y)

        results[model_name] = {
            "auc": np.nanmean(fold_aucs),
            "auc_std": np.nanstd(fold_aucs),
            "cv_scores": fold_aucs,
            "best_model": clf,
            "scaler": scaler,
        }

    return results


def get_feature_importance(model, feature_names):
    """Extract feature importance from a fitted model.

    Args:
        model: Fitted sklearn model (LR or RF)
        feature_names: List of feature names

    Returns:
        pd.Series: Feature importance scores, sorted descending
    """
    if hasattr(model, "feature_importances_"):
        # RandomForest
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        # LogisticRegression
        imp = np.abs(model.coef_[0])
    else:
        return pd.Series()

    return pd.Series(imp, index=feature_names).sort_values(ascending=False)


def fit_eval_single_split(X, y, seed=0, models=("LR", "RF")):
    """70/30 split fit for backward compatibility with analyze_detection.py.

    Args:
        X: Feature matrix
        y: Binary labels
        seed: Random seed
        models: Tuple of model names to fit

    Returns:
        dict: Per-model results (auc, acc, cm, proba, y_test, clf)
              Each value is a tuple: (auc, acc, cm, proba, y_test, clf)
    """
    from sklearn.metrics import accuracy_score, confusion_matrix

    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    idx = np.random.RandomState(seed).permutation(len(y))
    n_tr = int(0.7 * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]

    out = {}
    for name in models:
        clf = (LogisticRegression(max_iter=2000, random_state=seed) if name == "LR"
               else RandomForestClassifier(n_estimators=200, random_state=seed))
        clf.fit(Xs[tr], y[tr])
        proba = clf.predict_proba(Xs[te])[:, 1]
        pred = (proba > 0.5).astype(int)
        out[name] = (safe_auc(y[te], proba), accuracy_score(y[te], pred),
                     confusion_matrix(y[te], pred), proba, y[te], clf)
    return out
