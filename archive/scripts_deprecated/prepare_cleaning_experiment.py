"""Prepare datasets for the cleaning-gain experiment.

Creates two cleaned variants of garbled_ratio10:
  1. garbled_clean_scored: removes top-10% by detection score
  2. garbled_clean_random: removes random 10%

Then trains both and compares validation metrics against the original garbled run.
This tests the "cleaning gains have a ceiling" hypothesis from dynanoise.
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../config.yaml")
    ap.add_argument("--tag", default="ratio10", help="source tag with detection scores")
    ap.add_argument("--dataset", default="garbled", help="noise type to clean")
    ap.add_argument("--budget", type=float, default=0.10, help="fraction to remove")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = yaml.safe_load(open(args.config))
    repo = cfg["paths"]["repo_root"]

    # Load original training data
    src_path = os.path.join(repo, "data", args.tag, args.dataset, "train.jsonl")
    print(f"Loading {src_path}")
    with open(src_path) as f:
        rows = [json.loads(line) for line in f]
    print(f"  {len(rows)} samples")

    # Load detection scores
    score_path = os.path.join(repo, "results", args.tag, "per_sample_metrics.csv")
    print(f"Loading scores from {score_path}")
    df = pd.read_csv(score_path)
    df = df[df["dataset"] == args.dataset].copy()
    print(f"  {len(df)} scored samples")

    # Load the RF detector's predictions
    det_path = os.path.join(repo, "results", args.tag, "detection_results.csv")
    if os.path.exists(det_path):
        det = pd.read_csv(det_path)
        det = det[det["dataset"] == args.dataset].copy()
        if "rf_score" in det.columns:
            df = df.merge(det[["sample_id", "rf_score"]], on="sample_id", how="left")
            print(f"  merged RF scores")

    # Map sample_id to row index
    id_to_idx = {r["sample_id"]: i for i, r in enumerate(rows)}
    df["row_idx"] = df["sample_id"].map(id_to_idx)
    df = df.dropna(subset=["row_idx"])
    df["row_idx"] = df["row_idx"].astype(int)

    n_remove = int(len(rows) * args.budget)
    print(f"\nRemoving {n_remove} samples ({args.budget:.0%})")

    # Variant 1: remove top-10% by RF score (or loss_mean as fallback)
    score_col = "rf_score" if "rf_score" in df.columns else "loss_mean"
    print(f"  using {score_col} for scored removal")
    scored_remove = df.nlargest(n_remove, score_col)["row_idx"].values
    scored_keep = [r for i, r in enumerate(rows) if i not in scored_remove]

    # Variant 2: remove random 10%
    remove_indices = set(random.sample(range(len(rows)), n_remove))
    random_keep = [r for i, r in enumerate(rows) if i not in remove_indices]

    # Save both
    out_tag = f"{args.tag}_clean"
    for suffix, data in [("scored", scored_keep), ("random", random_keep)]:
        out_dir = os.path.join(repo, "data", out_tag, f"{args.dataset}_{suffix}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "train.jsonl")
        with open(out_path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(data)} samples -> {out_path}")

    # Copy heldout.jsonl to the new tag
    heldout_src = os.path.join(repo, "data", args.tag, "heldout.jsonl")
    heldout_dst = os.path.join(repo, "data", out_tag, "heldout.jsonl")
    os.makedirs(os.path.dirname(heldout_dst), exist_ok=True)
    with open(heldout_src) as f:
        heldout = f.read()
    with open(heldout_dst, "w") as f:
        f.write(heldout)
    print(f"  copied heldout -> {heldout_dst}")

    print(f"\n✓ Ready to train with --tag {out_tag}")
    print(f"  python scripts/train.py --tag {out_tag} --dataset {args.dataset}_scored")
    print(f"  python scripts/train.py --tag {out_tag} --dataset {args.dataset}_random")


if __name__ == "__main__":
    main()
