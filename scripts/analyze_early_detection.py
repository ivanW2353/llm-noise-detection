"""Deployable-detector analysis: how early, on how many samples, at what precision.

The main pipeline's 40-dim classifier answers "is the signal there?". This
script answers "can you actually clean data with it?", along three axes the
40-dim number silently conflates:

  1. EPOCH BUDGET. Cleaning has to happen early (detectability decays with
     epochs, §3.3), but loss_std / loss_curvature / converge_epoch need all 5
     epochs. Here a detector is built from epoch 0 only, epochs 0-1, ... so
     the cost of stopping early is explicit.

  2. SAMPLE COVERAGE. The 40-dim feature table includes diagnostic and
     token-detail features that only exist for the 1/8 diagnostic subsample,
     so dropna() shrinks training to ~900 rows (~40-90 noise). The
     always-tracked trajectory features (TRAJ_METRICS) cover every sample.
     Both are reported so a low AUC can be attributed to weak features rather
     than to a small sample.

  3. PRECISION AT A CLEANING BUDGET. Real cleaning drops the top-k% scored
     samples, so precision@k and recall@k matter, not AUC. Reported at
     k = the true noise ratio and at 5 / 10 / 20%, with the random baseline.

Cross-validated (5-fold stratified) rather than a single 70/30 split: the
diagnostic-subsample settings put only ~10-25 noise samples in one test fold.

Usage:
  python scripts/analyze_early_detection.py --tag ratio10
  python scripts/analyze_early_detection.py --tag ratio10 --datasets keyword
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import get_tag, load_config
from src.metrics import METRIC_ORDER, TRAJ_METRICS
from src.eval_utils import precision_at_k


def _tag(cfg):
    """Legacy helper, use get_tag() for new code."""
    return get_tag(cfg)

# Features derivable from the first `n_ep` epochs alone. Anything needing a
# trajectory shape (std / curvature / slope / rank / convergence) is only
# available once enough epochs have run.
def epoch_features(df, n_ep):
    """Build the feature matrix visible after training `n_ep` epochs.

    Uses only per-epoch loss/grad/cos columns up to n_ep, plus text_nn_sim
    (data-side, available before training starts).
    """
    cols, out = [], pd.DataFrame(index=df.index)
    per_ep = {"loss": [], "grad_norm": [], "cos_ref": [], "cos_global": []}
    for base in per_ep:
        for e in range(n_ep):
            c = f"{base}_ep{e}"
            if c in df.columns and df[c].notna().any():
                per_ep[base].append(c)
    for base, cs in per_ep.items():
        if not cs:
            continue
        m = df[cs]
        out[f"{base}_mean_e{n_ep}"] = m.mean(axis=1)
        out[f"{base}_last_e{n_ep}"] = m[cs[-1]]
        cols += [f"{base}_mean_e{n_ep}", f"{base}_last_e{n_ep}"]
        if n_ep >= 2:
            out[f"{base}_std_e{n_ep}"] = m.std(axis=1)
            out[f"{base}_slope_e{n_ep}"] = m[cs[-1]] - m[cs[0]]
            cols += [f"{base}_std_e{n_ep}", f"{base}_slope_e{n_ep}"]
        if n_ep >= 3:
            xs = np.arange(len(cs), dtype=float)
            X = np.stack([np.ones_like(xs), xs, xs ** 2], axis=1)
            coef = m.values.astype(float) @ np.linalg.pinv(X).T
            out[f"{base}_curv_e{n_ep}"] = coef[:, 0]
            cols.append(f"{base}_curv_e{n_ep}")
    if "text_nn_sim" in df.columns:
        out["text_nn_sim"] = df["text_nn_sim"]
        cols.append("text_nn_sim")
    return out, cols


def cv_scores(X, y, seed=0, n_splits=5):
    """5-fold stratified CV; returns (auc, out-of-fold scores) per model."""
    Xs = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {"LR": np.zeros(len(y)), "RF": np.zeros(len(y))}
    for tr, te in skf.split(Xs, y):
        for name in oof:
            clf = (LogisticRegression(max_iter=2000) if name == "LR"
                   else RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1))
            clf.fit(Xs[tr], y[tr])
            oof[name][te] = clf.predict_proba(Xs[te])[:, 1]
    return {n: (roc_auc_score(y, s), s) for n, s in oof.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--datasets", type=str, default=None,
                    help="comma-separated (default: every trained noise dataset)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.tag:
        cfg["paths"]["experiment_tag"] = args.tag
    tag = _tag(cfg)
    repo = cfg["paths"]["repo_root"]
    tag_dir = os.path.join(repo, "results", tag) if tag else os.path.join(repo, "results")
    src = os.path.join(tag_dir, "per_sample_metrics.csv")
    if not os.path.exists(src):
        sys.exit(f"missing {src} — run analyze_detection.py --tag {tag} first")
    print(f"[{time.strftime('%H:%M:%S')}] reading {src} ...", flush=True)
    df = pd.read_csv(src)
    if args.datasets:
        want = [d.strip() for d in args.datasets.split(",")]
    else:
        want = [d for d in sorted(df["dataset"].unique()) if d != "clean"]
    print(f"datasets: {want}", flush=True)

    ep_cols = sorted([c for c in df.columns if c.startswith("loss_ep")],
                     key=lambda c: int(c.split("_ep")[1]))
    n_epochs = len(ep_cols)
    # per-epoch grad/cos columns are not in per_sample_metrics.csv (only the
    # aggregates are), so epoch-budget features come from loss + text_nn_sim.
    print(f"epochs available: {n_epochs}", flush=True)

    abl_rows, early_rows, prec_rows = [], [], []
    for ds in want:
        sub_all = df[df["dataset"] == ds]
        pos_types = sorted(set(sub_all["noise_type"]) - {"none"})
        if not pos_types:
            continue
        sub_all = sub_all[sub_all["noise_type"].isin(["none"] + pos_types)]
        diag_mask = sub_all[METRIC_ORDER].notna().all(axis=1)

        # ---- axis 1+2: feature set x sample coverage --------------------
        for feats, fname in [(METRIC_ORDER, "40dim"), (TRAJ_METRICS, "13traj")]:
            for mask, sname in [(diag_mask, "diag_subsample"),
                                (pd.Series(True, index=sub_all.index), "all_samples")]:
                s = sub_all[mask].dropna(subset=feats)
                y = (s["noise_type"] != "none").astype(int).values
                if len(set(y)) < 2 or y.sum() < 10:
                    continue
                fits = cv_scores(s[feats].values, y)
                row = {"dataset": ds, "features": fname, "samples": sname,
                       "n": len(y), "n_noise": int(y.sum()),
                       "lr_auc": round(fits["LR"][0], 4), "rf_auc": round(fits["RF"][0], 4)}
                abl_rows.append(row)
                print(f"  {ds:<15}{fname:<7}{sname:<16}n={len(y):<6}noise={int(y.sum()):<5}"
                      f"LR={row['lr_auc']:.3f} RF={row['rf_auc']:.3f}", flush=True)
                # ---- axis 3: precision at a cleaning budget -------------
                if fname == "13traj" and sname == "all_samples":
                    best = "RF" if fits["RF"][0] >= fits["LR"][0] else "LR"
                    score = fits[best][1]
                    base_rate = y.mean()
                    for k in sorted({round(base_rate, 4), 0.05, 0.10, 0.20}):
                        p, r, nd = precision_at_k(y, score, k)
                        prec_rows.append({"dataset": ds, "model": best,
                                          "k_frac": round(k, 4), "n_dropped": nd,
                                          "precision": round(p, 4), "recall": round(r, 4),
                                          "random_precision": round(base_rate, 4),
                                          "lift": round(p / base_rate, 2) if base_rate else None})

        # ---- axis 1: epoch budget (all samples, loss-trajectory only) ---
        s_all = sub_all
        for n_ep in range(1, n_epochs + 1):
            X, cols = epoch_features(s_all, n_ep)
            X = X.dropna()
            y = (s_all.loc[X.index, "noise_type"] != "none").astype(int).values
            if len(set(y)) < 2 or y.sum() < 10:
                continue
            fits = cv_scores(X.values, y)
            best = "RF" if fits["RF"][0] >= fits["LR"][0] else "LR"
            p_at_base, r_at_base, _ = precision_at_k(y, fits[best][1], y.mean())
            early_rows.append({"dataset": ds, "epochs_used": n_ep, "n_features": len(cols),
                               "n": len(y), "n_noise": int(y.sum()),
                               "lr_auc": round(fits["LR"][0], 4),
                               "rf_auc": round(fits["RF"][0], 4),
                               "precision_at_base_rate": round(p_at_base, 4),
                               "random_precision": round(y.mean(), 4)})
            print(f"  {ds:<15}epochs 0-{n_ep-1:<3}({len(cols):>2} feats) "
                  f"LR={fits['LR'][0]:.3f} RF={fits['RF'][0]:.3f} "
                  f"P@base={p_at_base:.3f}", flush=True)

    for rows, name, title in [
            (abl_rows, "detector_ablation", "feature set x sample coverage (5-fold CV)"),
            (early_rows, "detector_epoch_budget", "epoch budget: detector from first N epochs"),
            (prec_rows, "detector_precision_at_k", "precision at a cleaning budget")]:
        if not rows:
            continue
        tab = pd.DataFrame(rows)
        out = os.path.join(tag_dir, f"{name}.csv")
        tab.to_csv(out, index=False)
        print(f"\n=== {title} ===")
        print(tab.to_string(index=False))
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
