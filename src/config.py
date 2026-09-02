"""Configuration loading and tag resolution."""

import os
import yaml


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml, relative to repo root or absolute

    Returns:
        dict: Configuration dictionary
    """
    if not os.path.isabs(config_path):
        # Assume it's relative to repo root
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(here)
        config_path = os.path.join(repo_root, config_path)

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_tag(cfg):
    """Extract experiment tag from config.

    Originally named _tag() in analyze_detection.py, imported by 6+ scripts.

    Args:
        cfg: Config dict with paths.experiment_tag

    Returns:
        str: Experiment tag (e.g., 'ratio10', 'extra10')
    """
    return cfg["paths"]["experiment_tag"]


def get_repo_root(cfg):
    """Get repository root path from config.

    Args:
        cfg: Config dict

    Returns:
        str: Absolute path to repo root
    """
    return cfg["paths"]["repo_root"]


def get_results_dir(cfg, tag=None):
    """Get results directory for a specific tag.

    Args:
        cfg: Config dict
        tag: Experiment tag (defaults to cfg's own tag)

    Returns:
        str: Path to results/{tag}/
    """
    if tag is None:
        tag = get_tag(cfg)
    return os.path.join(get_repo_root(cfg), "results", tag)
