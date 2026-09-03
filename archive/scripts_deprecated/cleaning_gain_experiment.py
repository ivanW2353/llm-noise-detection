"""Cleaning gain control experiment: does targeted cleaning beat random dropout?

Motivation (§4.5 in cross_experiment_synthesis, "biggest remaining gap"):
  dynanoise Phase 4 found that precise noise filtering (99.8% hit rate) improved
  MT-Bench by only +0.48, barely more than random 10% drop (+0.41). This repo
  has measured detection AUC and cleaning precision, but not whether the model
  actually gets better after cleaning.

Experiment design:
  1. Load a trained detector (RF from analyze_detection.py)
  2. Score all samples in the training set
  3. Create two cleaned datasets:
     - targeted: remove top-k% by noise score
     - random:   remove random k%
  4. Fine-tune the base model on each cleaned dataset
  5. Evaluate both on the validation benchmarks (MMLU, GSM8K, etc.)
  6. Compare: does targeted cleaning beat random?

Expected result: targeted ≈ random (ceiling effect), validating dynanoise's finding.

Usage:
  python scripts/cleaning_gain_experiment.py --tag ratio10 --noise-type unrelated --budget 0.10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.detection import load_detector

def load_training_data(cfg, tag):
    """Load the original training set with noise labels."""
    results_dir = Path(cfg["paths"]["repo_root"]) / "results" / tag
    metrics = pd.read_csv(results_dir / "per_sample_metrics.csv")

    # Load the actual training data
    train_path = Path(cfg["paths"]["train_data_dir"]) / tag / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    with open(train_path) as f:
        train_data = [json.loads(line) for line in f]

    # Merge metrics with training data
    assert len(train_data) == len(metrics), "Mismatch between data and metrics"
    for sample, row in zip(train_data, metrics.itertuples()):
        sample["sample_id"] = row.sample_id
        sample["dataset"] = row.dataset
        sample["noise_type"] = row.noise_type

    return train_data, metrics

def score_samples(cfg, tag, noise_type, metrics):
    """Load detector and score all samples."""
    results_dir = Path(cfg["paths"]["repo_root"]) / "results" / tag
    detector_path = results_dir / f"detector_{noise_type}.pkl"

    if not detector_path.exists():
        raise FileNotFoundError(
            f"Detector not found: {detector_path}\n"
            f"Run: python scripts/analyze_detection.py --tag {tag} first"
        )

    clf, scaler, features = load_detector(detector_path)

    # Score all samples
    subset = metrics[metrics["dataset"] == noise_type].copy()
    X = subset[features].values
    X_scaled = scaler.transform(X)

    # Higher score = more likely to be noise
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_scaled)[:, 1]
    else:
        scores = clf.decision_function(X_scaled)

    subset["noise_score"] = scores
    return subset

def create_cleaned_datasets(train_data, scores_df, budget, output_dir, seed=42):
    """Create two cleaned versions: targeted and random."""
    np.random.seed(seed)

    # Build a score lookup
    score_map = dict(zip(scores_df["sample_id"], scores_df["noise_score"]))

    # Assign scores to all training samples
    for sample in train_data:
        sample["noise_score"] = score_map.get(sample["sample_id"], 0.0)

    n_total = len(train_data)
    n_drop = int(budget * n_total)

    print(f"Total samples: {n_total}")
    print(f"Budget: {budget:.1%} → dropping {n_drop} samples")

    # Targeted: remove top-k% by score
    sorted_by_score = sorted(train_data, key=lambda x: x["noise_score"], reverse=True)
    targeted_keep = sorted_by_score[n_drop:]
    targeted_drop = sorted_by_score[:n_drop]

    # Random: remove random k%
    shuffled = train_data.copy()
    np.random.shuffle(shuffled)
    random_keep = shuffled[n_drop:]
    random_drop = shuffled[:n_drop]

    # Report drop precision
    targeted_noise_count = sum(1 for s in targeted_drop if s["noise_type"] != "none")
    random_noise_count = sum(1 for s in random_drop if s["noise_type"] != "none")

    print(f"\nTargeted drop: {targeted_noise_count}/{n_drop} are noise ({targeted_noise_count/n_drop:.1%})")
    print(f"Random drop:   {random_noise_count}/{n_drop} are noise ({random_noise_count/n_drop:.1%})")

    # Save cleaned datasets
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_jsonl(data, path):
        with open(path, "w") as f:
            for sample in data:
                # Remove metadata fields
                clean_sample = {
                    "instruction": sample["instruction"],
                    "response": sample["response"]
                }
                f.write(json.dumps(clean_sample, ensure_ascii=False) + "\n")

    save_jsonl(targeted_keep, output_dir / "train_targeted.jsonl")
    save_jsonl(random_keep, output_dir / "train_random.jsonl")

    # Save metadata
    metadata = {
        "budget": budget,
        "n_total": n_total,
        "n_drop": n_drop,
        "n_keep": n_total - n_drop,
        "targeted_precision": targeted_noise_count / n_drop,
        "random_precision": random_noise_count / n_drop,
        "seed": seed
    }

    with open(output_dir / "cleaning_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"  - train_targeted.jsonl ({len(targeted_keep)} samples)")
    print(f"  - train_random.jsonl ({len(random_keep)} samples)")
    print(f"  - cleaning_metadata.json")

    return metadata

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tag", required=True, help="experiment tag (e.g., ratio10)")
    ap.add_argument("--noise-type", required=True,
                    help="noise type to target (e.g., unrelated)")
    ap.add_argument("--budget", type=float, default=0.10,
                    help="fraction to drop (default: 0.10)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)

    print(f"=== Cleaning Gain Experiment ===")
    print(f"Tag: {args.tag}")
    print(f"Target noise: {args.noise_type}")
    print(f"Budget: {args.budget:.1%}")
    print()

    # Step 1: Load training data
    print("[1/3] Loading training data...")
    train_data, metrics = load_training_data(cfg, args.tag)
    print(f"  Loaded {len(train_data)} samples")

    # Step 2: Score samples
    print(f"\n[2/3] Scoring samples with trained detector...")
    scores_df = score_samples(cfg, args.tag, args.noise_type, metrics)
    print(f"  Scored {len(scores_df)} samples from '{args.noise_type}' dataset")

    # Step 3: Create cleaned datasets
    print(f"\n[3/3] Creating cleaned datasets...")
    output_dir = Path(cfg["paths"]["repo_root"]) / "data" / "cleaned" / args.tag / args.noise_type
    metadata = create_cleaned_datasets(
        train_data, scores_df, args.budget, output_dir, args.seed
    )

    print(f"\n{'='*60}")
    print("✅ Dataset preparation complete!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print(f"1. Train on targeted cleaned data:")
    print(f"   python scripts/train.py --data {output_dir}/train_targeted.jsonl \\")
    print(f"       --tag {args.tag}_targeted --base-model {cfg['model']['base_model']}")
    print(f"\n2. Train on random cleaned data:")
    print(f"   python scripts/train.py --data {output_dir}/train_random.jsonl \\")
    print(f"       --tag {args.tag}_random --base-model {cfg['model']['base_model']}")
    print(f"\n3. Evaluate both on benchmarks:")
    print(f"   python scripts/evaluate.py --tag {args.tag}_targeted")
    print(f"   python scripts/evaluate.py --tag {args.tag}_random")
    print(f"\n4. Compare results:")
    print(f"   python scripts/compare_cleaning_gains.py --tag {args.tag} \\")
    print(f"       --noise-type {args.noise_type}")

if __name__ == "__main__":
    main()
