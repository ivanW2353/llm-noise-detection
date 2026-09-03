"""Detector transfer: does a detector trained on one setting work on another?

Every AUC reported so far trains and tests within the same run (same noise
ratio, same noise type). Deployment never looks like that — you train a
detector on whatever labeled noise you have and apply it to data whose
contamination rate and noise family are unknown. Two transfer axes:

  1. CROSS-RATIO. Train on ratio10's run, test on ratio05's run of the SAME
     noise type (and vice versa). The concern is prevalence shift: rank-based
     features (loss_rank) and anything the classifier calibrates against the
     contaminated majority will drift when the noise ratio halves.

  2. CROSS-TYPE. Train on noise type A, test on type B (all pairs within a
     tag). This is the practically important one: it says whether a detector
     generalizes to noise families it has never seen, or whether each type
     needs its own labeled examples. A high off-diagonal means the features
     capture "is anomalous" rather than "is this specific corruption".

Both use TRAJ_METRICS (available for every sample, not just the diagnostic
subsample) so the transfer numbers are not confounded by sample coverage, and
both report the within-run 5-fold CV AUC on the diagonal as the reference
ceiling. Features are standardized with the TRAINING run's scaler — refitting
on the target would leak target information and is not available at deploy
time either.

Usage:
  python scripts/3_analysis/analyze_transfer.py                      # both axes, all tags
  python scripts/3_analysis/analyze_transfer.py --tags ratio10,ratio05
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.config import get_tag, load_config
from src.metrics import TRAJ_METRICS
from src.eval_utils import precision_at_k


def _tag(cfg):
    """Legacy helper, use get_tag() for new code."""
    return get_tag(cfg)


def load(repo, tag):
    p = os.path.join(repo, "results", tag, "per_sample_metrics.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def xy(df, ds, feats):
    """Feature matrix + binary noise labels for one run."""
    sub = df[df["dataset"] == ds].dropna(subset=feats)
    y = (sub["noise_type"] != "none").astype(int).values
    return sub[feats].values.astype(float), y


def fit_transfer(Xtr, ytr, Xte, yte, seed=0):
    """Fit on source, score target. Scaler comes from the source only."""
    sc = StandardScaler().fit(Xtr)
    out = {}
    for name, clf in (("LR", LogisticRegression(max_iter=2000)),
                      ("RF", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1))):
        clf.fit(sc.transform(Xtr), ytr)
        s = clf.predict_proba(sc.transform(Xte))[:, 1]
        out[name] = (roc_auc_score(yte, s), s)
    return out


def cv_auc(X, y, seed=0):
    """Within-run 5-fold CV — the reference ceiling for the diagonal."""
    Xs = StandardScaler().fit_transform(X)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, y):
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(Xs[tr], y[tr])
        oof[te] = clf.predict_proba(Xs[te])[:, 1]
    return roc_auc_score(y, oof), oof


def main():
    # Go up 3 levels: scripts/3_analysis/analyze_transfer.py -> scripts/3_analysis -> scripts -> project_root
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(here))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(project_root, "config.yaml"))
    ap.add_argument("--tags", type=str, default="ratio10,ratio05,extra10")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    repo = cfg["paths"]["repo_root"]
    tags = [t.strip() for t in args.tags.split(",")]
    data = {t: d for t in tags if (d := load(repo, t)) is not None}
    if not data:
        sys.exit("no per_sample_metrics.csv found for any tag")
    any_df = next(iter(data.values()))
    feats = [m for m in TRAJ_METRICS if m in any_df.columns]
    print(f"tags: {list(data)}  features: {len(feats)}", flush=True)

    # ---- axis 1: cross-ratio ------------------------------------------
    ratio_rows = []
    ratio_tags = [t for t in ("ratio10", "ratio05") if t in data]
    if len(ratio_tags) == 2:
        a, b = ratio_tags
        shared = sorted(set(data[a]["dataset"]) & set(data[b]["dataset"]) - {"clean"})
        for ds in shared:
            Xa, ya = xy(data[a], ds, feats)
            Xb, yb = xy(data[b], ds, feats)
            if ya.sum() < 10 or yb.sum() < 10:
                continue
            for src, dst, Xs_, ys_, Xt, yt in ((a, b, Xa, ya, Xb, yb),
                                               (b, a, Xb, yb, Xa, ya)):
                fits = fit_transfer(Xs_, ys_, Xt, yt)
                own, _ = cv_auc(Xt, yt)
                best = max(fits, key=lambda k: fits[k][0])
                p10, _, _ = precision_at_k(yt, fits[best][1], 0.10)
                ratio_rows.append({
                    "dataset": ds, "train_tag": src, "test_tag": dst,
                    "n_train_noise": int(ys_.sum()), "n_test_noise": int(yt.sum()),
                    "lr_auc": round(fits["LR"][0], 4), "rf_auc": round(fits["RF"][0], 4),
                    "within_run_auc": round(own, 4),
                    "retention": round(max(fits["LR"][0], fits["RF"][0]) / own, 3) if own else None,
                    "p_at_10": round(p10, 4), "random_p": round(yt.mean(), 4)})
                print(f"  ratio {src}->{dst:<9}{ds:<15}"
                      f"LR={fits['LR'][0]:.3f} RF={fits['RF'][0]:.3f} "
                      f"(own {own:.3f}, retain {ratio_rows[-1]['retention']})", flush=True)

    # ---- axis 2: cross-type -------------------------------------------
    type_rows = []
    for tag, df in data.items():
        dss = [d for d in sorted(df["dataset"].unique()) if d not in ("clean", "mixed")]
        cache = {}
        for ds in dss:
            X, y = xy(df, ds, feats)
            if y.sum() >= 10:
                cache[ds] = (X, y)
        for dst, (Xt, yt) in cache.items():
            own, _ = cv_auc(Xt, yt)
            for src, (Xs_, ys_) in cache.items():
                if src == dst:
                    auc, s = own, None
                    p10 = np.nan
                else:
                    fits = fit_transfer(Xs_, ys_, Xt, yt)
                    best = max(fits, key=lambda k: fits[k][0])
                    auc, s = fits[best][0], fits[best][1]
                    p10, _, _ = precision_at_k(yt, s, 0.10)
                type_rows.append({
                    "tag": tag, "train_type": src, "test_type": dst,
                    "auc": round(auc, 4), "auc_dir": round(max(auc, 1 - auc), 4),
                    "within_run_auc": round(own, 4),
                    "retention": round(auc / own, 3) if own else None,
                    "p_at_10": None if np.isnan(p10) else round(p10, 4),
                    "random_p": round(yt.mean(), 4),
                    "is_diagonal": src == dst})

    for rows, name, title in (
            (ratio_rows, "transfer_cross_ratio", "cross-ratio transfer (train tag -> test tag)"),
            (type_rows, "transfer_cross_type", "cross-type transfer (train type -> test type)")):
        if not rows:
            continue
        tab = pd.DataFrame(rows)
        out = os.path.join(repo, "results", f"{name}.csv")
        tab.to_csv(out, index=False)
        print(f"\n=== {title} ===")
        if name == "transfer_cross_type":
            for tag, g in tab.groupby("tag"):
                print(f"\n[{tag}] AUC (rows = trained on, cols = tested on; diagonal = within-run CV)")
                print(g.pivot_table(index="train_type", columns="test_type",
                                    values="auc").round(3).to_string())
                off = g[~g["is_diagonal"]]
                print(f"  off-diagonal mean AUC {off['auc'].mean():.3f} "
                      f"(directional {off['auc_dir'].mean():.3f}), "
                      f"diagonal mean {g[g['is_diagonal']]['auc'].mean():.3f}")
        else:
            print(tab.to_string(index=False))
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
