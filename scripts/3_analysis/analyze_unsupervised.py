"""Unsupervised / label-free detection: what survives without known labels?

Every AUC in the reports so far comes from a SUPERVISED classifier (LR/RF)
trained on ground-truth noise labels. That is not the deployment setting — if
you already had the labels you would not need a detector. This script asks how
much of the measured detectability survives when no labels are available.

Four label-free scorers on the always-tracked TRAJ_METRICS (every sample, not
just the 1/8 diagnostic subsample):

  1. `zscore_max`  — bidirectional per-feature z-scores, score = max |z|.
     Directly implements the "explicit bidirectional signals" item that the
     cross-experiment synthesis (§4.4) lists as still open: taking |z| makes
     the scorer immune to the direction inversions (duplicate/template have
     LOWER loss than clean) that break any one-sided threshold rule.
  2. `zscore_mean` — mean |z| across features (less spiky, more robust).
  3. `iforest`     — IsolationForest, the standard unsupervised outlier model.
  4. `mahalanobis` — distance from the robust (median/MAD-scaled) centre,
     which unlike iforest accounts for feature correlation.

All four are fit on the *unlabeled* run and scored on the same rows; labels are
used ONLY to evaluate. A supervised RF (5-fold CV) is reported alongside as the
ceiling, plus precision@k at the true noise ratio and at 10%, since a
label-free scorer is only useful if it beats the random-drop baseline.

Note the built-in handicap: these scorers rank by *atypicality*, and noise is
10%/5% of the data, so the clean majority defines "normal" — but a 10% noise
population is large enough to shift the mean/covariance it is measured against,
which caps how well any single-population outlier model can do.

Usage:
  python scripts/3_analysis/analyze_unsupervised.py --tag ratio10
  python scripts/3_analysis/analyze_unsupervised.py --tags ratio10,ratio05,extra10
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.config import get_tag
from src.metrics import TRAJ_METRICS
from src.scorers import robust_z
from src.eval_utils import precision_at_k


def _tag(cfg):
    """Legacy helper, use get_tag() for new code."""
    return get_tag(cfg)


def unsupervised_scores(X, seed=0):
    Z = robust_z(X)
    out = {
        "zscore_max": np.abs(Z).max(axis=1),
        "zscore_mean": np.abs(Z).mean(axis=1),
    }
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)
    out["iforest"] = -iso.fit(Xs).score_samples(Xs)  # higher = more anomalous
    # Mahalanobis on the robust z-scores, shrunk covariance for stability
    C = np.cov(Z, rowvar=False)
    C += np.eye(C.shape[0]) * 1e-3 * np.trace(C) / C.shape[0]
    d = Z @ np.linalg.pinv(C)
    out["mahalanobis"] = np.sqrt(np.maximum((d * Z).sum(axis=1), 0))
    return out


def supervised_ceiling(X, y, seed=0):
    Xs = StandardScaler().fit_transform(X)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, y):
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(Xs[tr], y[tr])
        oof[te] = clf.predict_proba(Xs[te])[:, 1]
    return roc_auc_score(y, oof), oof


def main():
    # Go up 3 levels: scripts/3_analysis/analyze_unsupervised.py -> scripts/3_analysis -> scripts -> project_root
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(here))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(project_root, "config.yaml"))
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--tags", type=str, default=None, help="comma-separated")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    repo = cfg["paths"]["repo_root"]
    tags = ([t.strip() for t in args.tags.split(",")] if args.tags
            else [args.tag or _tag(cfg)])

    for tag in tags:
        tag_dir = os.path.join(repo, "results", tag) if tag else os.path.join(repo, "results")
        src = os.path.join(tag_dir, "per_sample_metrics.csv")
        if not os.path.exists(src):
            print(f"[{tag}] missing {src} — skip")
            continue
        df = pd.read_csv(src)
        feats = [m for m in TRAJ_METRICS if m in df.columns]
        rows = []
        for ds in sorted(df["dataset"].unique()):
            if ds == "clean":
                continue
            sub = df[df["dataset"] == ds].dropna(subset=feats)
            y = (sub["noise_type"] != "none").astype(int).values
            if len(set(y)) < 2 or y.sum() < 10:
                continue
            X = sub[feats].values.astype(float)
            sup_auc, sup_score = supervised_ceiling(X, y)
            scorers = unsupervised_scores(X)
            scorers["SUPERVISED_rf"] = sup_score
            base = y.mean()
            for name, s in scorers.items():
                auc = roc_auc_score(y, s)
                # a label-free scorer has no calibrated direction either: an
                # atypicality score could in principle anti-correlate
                p_base, r_base, _ = precision_at_k(y, s, base)
                p10, r10, _ = precision_at_k(y, s, 0.10)
                rows.append({
                    "tag": tag, "dataset": ds, "scorer": name,
                    "n": len(y), "n_noise": int(y.sum()),
                    "auc": round(auc, 4), "auc_dir": round(max(auc, 1 - auc), 4),
                    "p_at_base": round(p_base, 4), "recall_at_base": round(r_base, 4),
                    "p_at_10": round(p10, 4),
                    "random_p": round(base, 4),
                    "lift_at_10": round(p10 / base, 2) if base else None,
                    "frac_of_supervised": round(auc / sup_auc, 3) if sup_auc else None,
                })
            best = max((r for r in rows[-len(scorers):] if r["scorer"] != "SUPERVISED_rf"),
                       key=lambda r: r["auc"])
            print(f"  {tag:<9}{ds:<15}n={len(y):<6}noise={int(y.sum()):<5}"
                  f"best-unsup {best['scorer']:<12}{best['auc']:.3f} "
                  f"(P@10 {best['p_at_10']:.3f})  vs supervised RF {sup_auc:.3f}", flush=True)
        if not rows:
            print(f"[{tag}] nothing scored")
            continue
        tab = pd.DataFrame(rows)
        out = os.path.join(tag_dir, "unsupervised_detection.csv")
        tab.to_csv(out, index=False)
        print(f"\n=== [{tag}] label-free vs supervised ===")
        print(tab.pivot_table(index="dataset", columns="scorer",
                              values="auc").to_string())
        print(f"\n=== [{tag}] precision at a 10% cleaning budget ===")
        print(tab.pivot_table(index="dataset", columns="scorer",
                              values="p_at_10").to_string())
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
