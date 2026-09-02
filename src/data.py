"""Data loading utilities for per-sample metrics and labels."""

import os
import pandas as pd

from .config import get_results_dir


def load_metrics(cfg, tag=None):
    """Load per-sample metrics CSV.

    Args:
        cfg: Config dict
        tag: Experiment tag (defaults to cfg's own tag)

    Returns:
        pd.DataFrame: Per-sample metrics with columns:
            sample_id, dataset, noise_type, category, loss_mean, ...
    """
    results_dir = get_results_dir(cfg, tag)
    path = os.path.join(results_dir, "per_sample_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics not found: {path}")
    return pd.read_csv(path)


def load_dataset(df, dataset_name):
    """Filter DataFrame to a specific dataset.

    Args:
        df: Full metrics DataFrame
        dataset_name: Dataset name (e.g., 'mixed', 'garbled')

    Returns:
        pd.DataFrame: Filtered to dataset_name rows
    """
    return df[df["dataset"] == dataset_name].copy()


def get_noise_mask(df):
    """Get boolean mask for noise samples (noise_type != 'none').

    Args:
        df: Metrics DataFrame with noise_type column

    Returns:
        pd.Series: Boolean mask, True for noise samples
    """
    return df["noise_type"] != "none"


def get_labels(df):
    """Get binary noise labels (0=clean, 1=noise).

    Args:
        df: Metrics DataFrame with noise_type column

    Returns:
        np.ndarray: Binary labels as int array
    """
    return get_noise_mask(df).astype(int).values


def filter_features(df, feature_list, drop_na=True):
    """Extract feature matrix from DataFrame.

    Args:
        df: Metrics DataFrame
        feature_list: List of feature column names
        drop_na: If True, drop rows with any NaN in feature columns

    Returns:
        tuple: (filtered_df, feature_matrix)
            - filtered_df: DataFrame after dropna (if requested)
            - feature_matrix: np.ndarray of shape (n_samples, n_features)
    """
    available = [f for f in feature_list if f in df.columns]
    if len(available) < len(feature_list):
        missing = set(feature_list) - set(available)
        print(f"Warning: {len(missing)} features not found: {missing}")

    if drop_na:
        filtered = df.dropna(subset=available).copy()
    else:
        filtered = df.copy()

    X = filtered[available].fillna(0).values
    return filtered, X


def get_noise_spec(df):
    """Infer noise types present in a DataFrame.

    Args:
        df: Metrics DataFrame with noise_type column

    Returns:
        dict: Maps dataset name -> list of noise_types present
    """
    spec = {}
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        noise_types = sorted(sub[sub["noise_type"] != "none"]["noise_type"].unique())
        if noise_types:
            spec[ds] = noise_types
    return spec
