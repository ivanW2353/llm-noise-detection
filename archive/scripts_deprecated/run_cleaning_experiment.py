"""Cleaning gain control experiment - Complete pipeline.

This script implements the full experiment to test whether targeted noise cleaning
produces better models than random data dropout (dynanoise's "ceiling" finding).

Pipeline:
  0. Train a detector on the noisy dataset
  1. Score all training samples
  2. Create two cleaned datasets (targeted vs random)
  3. Fine-tune base model on each cleaned dataset
  4. Evaluate both on validation benchmarks
  5. Compare results

Usage:
  # Run the full experiment in tmux
  python scripts/run_cleaning_experiment.py --tag ratio10 --noise-type unrelated
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.detection import fit_eval
from src.metrics import METRIC_ORDER

def train_detector(cfg, tag, noise_type, output_dir):
    """Train a detector for the specified noise type."""
    print(f"\n[Step 1/5] Training detector for {noise_type}...")

    # Load metrics
    results_dir = Path(cfg["paths"]["repo_root"]) / "results" / tag
    metrics_path = results_dir / "per_sample_metrics.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics not found: {metrics_path}\n"
            f"Run training and metric collection first."
        )

    metrics = pd.read_csv(metrics_path)

    # Prepare training data
    subset = metrics[metrics["dataset"] == noise_type].copy()
    y = (subset["noise_type"] != "none").astype(int).values
    X = subset[METRIC_ORDER].values

    if y.sum() < 10:
        raise ValueError(f"Not enough noise samples: {y.sum()}")

    # Train RF classifier with scaling
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")

    # Fit on full data
    clf.fit(X_scaled, y)

    result = {
        "clf": clf,
        "scaler": scaler,
        "features": METRIC_ORDER,
        "auc": cv_aucs.mean(),
        "auc_std": cv_aucs.std(),
    }

    # Save detector and scores
    detector_info = {
        "tag": tag,
        "noise_type": noise_type,
        "model": "RF",
        "features": result["features"],
        "auc": result["auc"],
        "scaler_mean": result["scaler"].mean_.tolist(),
        "scaler_scale": result["scaler"].scale_.tolist(),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "detector_info.json", "w") as f:
        json.dump(detector_info, f, indent=2)

    # Score all samples
    subset = metrics[metrics["dataset"] == noise_type].copy()
    X = subset[result["features"]].values
    X_scaled = result["scaler"].transform(X)
    scores = result["clf"].predict_proba(X_scaled)[:, 1]

    subset["noise_score"] = scores
    subset[["sample_id", "noise_type", "noise_score"]].to_csv(
        output_dir / "sample_scores.csv", index=False
    )

    print(f"  ✓ Detector AUC: {result['auc']:.4f}")
    print(f"  ✓ Scores saved to {output_dir}/sample_scores.csv")

    return result, subset

def create_cleaned_datasets(cfg, tag, noise_type, scores_df, budget, output_dir, seed=42):
    """Create targeted and random cleaned datasets."""
    print(f"\n[Step 2/5] Creating cleaned datasets (budget={budget:.0%})...")

    np.random.seed(seed)

    # Load original training data
    data_dir = Path(cfg["paths"]["repo_root"]) / "data" / tag / noise_type
    train_path = data_dir / "train.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    with open(train_path) as f:
        train_data = [json.loads(line) for line in f]

    # Build score lookup (sample_id is just the index)
    score_map = dict(enumerate(scores_df["noise_score"].values))

    # Assign scores
    for i, sample in enumerate(train_data):
        sample["noise_score"] = score_map.get(i, 0.0)
        sample["sample_id"] = i
        sample["noise_type"] = scores_df.iloc[i]["noise_type"] if i < len(scores_df) else "none"

    n_total = len(train_data)
    n_drop = int(budget * n_total)

    print(f"  Total samples: {n_total}")
    print(f"  Dropping: {n_drop} samples")

    # Targeted: remove top-k% by score
    sorted_by_score = sorted(train_data, key=lambda x: x["noise_score"], reverse=True)
    targeted_keep = sorted_by_score[n_drop:]
    targeted_drop = sorted_by_score[:n_drop]

    # Random: remove random k%
    shuffled = train_data.copy()
    np.random.shuffle(shuffled)
    random_keep = shuffled[n_drop:]
    random_drop = shuffled[:n_drop]

    # Report precision
    targeted_noise = sum(1 for s in targeted_drop if s["noise_type"] != "none")
    random_noise = sum(1 for s in random_drop if s["noise_type"] != "none")

    print(f"  Targeted precision: {targeted_noise}/{n_drop} = {targeted_noise/n_drop:.2%}")
    print(f"  Random precision:   {random_noise}/{n_drop} = {random_noise/n_drop:.2%}")

    # Save cleaned datasets
    output_dir = Path(output_dir)

    def save_jsonl(data, path):
        with open(path, "w") as f:
            for sample in data:
                # Keep the messages format but drop metadata
                clean_sample = {
                    "messages": sample["messages"]
                }
                f.write(json.dumps(clean_sample, ensure_ascii=False) + "\n")

    save_jsonl(targeted_keep, output_dir / "train_targeted.jsonl")
    save_jsonl(random_keep, output_dir / "train_random.jsonl")

    metadata = {
        "tag": tag,
        "noise_type": noise_type,
        "budget": budget,
        "n_total": n_total,
        "n_drop": n_drop,
        "n_keep": n_total - n_drop,
        "targeted_precision": targeted_noise / n_drop,
        "random_precision": random_noise / n_drop,
        "seed": seed
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved train_targeted.jsonl ({len(targeted_keep)} samples)")
    print(f"  ✓ Saved train_random.jsonl ({len(random_keep)} samples)")

    return metadata

def generate_training_commands(cfg, tag, noise_type, cleaned_dir):
    """Generate the training commands for both cleaned datasets."""
    print(f"\n[Step 3/5] Generating training commands...")

    base_model = cfg["paths"]["model"]

    # Use the same training config as the original runs
    lr = cfg["train"]["lr"]
    epochs = cfg["train"]["epochs"]
    micro_batch = cfg["train"]["micro_batch"]
    grad_accum = cfg["train"]["grad_accum"]
    lora_r = cfg["train"]["lora_r"]
    lora_alpha = cfg["train"]["lora_alpha"]

    repo_root = Path(cfg["paths"]["repo_root"])
    output_base = repo_root / "models" / "cleaning_exp"

    commands = {
        "targeted": f"""python scripts/train_sft.py \\
    --train-file {cleaned_dir}/train_targeted.jsonl \\
    --model {base_model} \\
    --output-dir {output_base}/{tag}_{noise_type}_targeted \\
    --epochs {epochs} \\
    --lr {lr} \\
    --micro-batch {micro_batch} \\
    --grad-accum {grad_accum} \\
    --lora-r {lora_r} \\
    --lora-alpha {lora_alpha}""",

        "random": f"""python scripts/train_sft.py \\
    --train-file {cleaned_dir}/train_random.jsonl \\
    --model {base_model} \\
    --output-dir {output_base}/{tag}_{noise_type}_random \\
    --epochs {epochs} \\
    --lr {lr} \\
    --micro-batch {micro_batch} \\
    --grad-accum {grad_accum} \\
    --lora-r {lora_r} \\
    --lora-alpha {lora_alpha}"""
    }

    with open(cleaned_dir / "training_commands.sh", "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Training commands for cleaning gain experiment\n\n")
        f.write("echo 'Training targeted cleaned model...'\n")
        f.write(commands["targeted"] + "\n\n")
        f.write("echo 'Training random cleaned model...'\n")
        f.write(commands["random"] + "\n")

    os.chmod(cleaned_dir / "training_commands.sh", 0o755)

    print(f"  ✓ Training commands saved to {cleaned_dir}/training_commands.sh")

    return commands

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tag", required=True, help="experiment tag")
    ap.add_argument("--noise-type", required=True, help="noise type to target")
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-detector", action="store_true",
                    help="skip detector training if scores already exist")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(cfg["paths"]["repo_root"])

    # Create experiment directory
    exp_dir = repo_root / "experiments" / "cleaning_gain" / f"{args.tag}_{args.noise_type}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("  Cleaning Gain Control Experiment")
    print("="*70)
    print(f"Tag:        {args.tag}")
    print(f"Noise type: {args.noise_type}")
    print(f"Budget:     {args.budget:.0%}")
    print(f"Output:     {exp_dir}")
    print("="*70)

    # Step 1: Train detector
    scores_path = exp_dir / "sample_scores.csv"
    if args.skip_detector and scores_path.exists():
        print(f"\n[Step 1/5] Using existing scores from {scores_path}")
        scores_df = pd.read_csv(scores_path)
    else:
        _, scores_df = train_detector(cfg, args.tag, args.noise_type, exp_dir)

    # Step 2: Create cleaned datasets
    metadata = create_cleaned_datasets(
        cfg, args.tag, args.noise_type, scores_df, args.budget, exp_dir, args.seed
    )

    # Step 3: Generate training commands
    commands = generate_training_commands(cfg, args.tag, args.noise_type, exp_dir)

    print(f"\n{'='*70}")
    print("✅ Dataset preparation complete!")
    print(f"{'='*70}")
    print(f"\nExperiment directory: {exp_dir}")
    print(f"\nNext: Run the training script in tmux:")
    print(f"  bash {exp_dir}/training_commands.sh")
    print(f"\nOr run steps manually:")
    print(f"  1. Train targeted: see {exp_dir}/training_commands.sh")
    print(f"  2. Train random: see {exp_dir}/training_commands.sh")
    print(f"  3. Evaluate both models on benchmarks")
    print(f"  4. Compare: python scripts/compare_cleaning_gains.py --exp-dir {exp_dir}")

if __name__ == "__main__":
    main()
