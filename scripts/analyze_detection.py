"""Noise-vs-normal separation analysis on per-sample training metrics.

Combines per-sample metrics (loss / grad_norm / cos_sim_ref / cos_sim_global
and epoch-end token diagnostics) with noise labels from the datasets, then:

  1. compares metric distributions noise vs normal, per noise type
  2. univariate detection: AUC per metric per noise type
  3. multivariate detection: logistic regression / random forest + ROC +
     feature importance + confusion matrix
  4. PCA scatter of samples colored by noise type
  5. loss trajectories across epochs per noise type
  6. aggregates evaluation tables (results/eval_*.json)

Outputs everything into <repo>/results/.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, confusion_matrix, roc_auc_score, roc_curve,
                             accuracy_score)
from sklearn.preprocessing import StandardScaler

DATASETS = ["clean", "garbled", "duplicate", "unrelated", "keyword", "mixed"]
METRIC_ORDER = ["loss_mean", "loss_last", "loss_slope", "grad_norm_mean",
                "cos_ref_mean", "cos_global_mean", "max_token_loss", "frac_hard"]


def load_labels(cfg, dataset):
    path = os.path.join(cfg["paths"]["data_root"], "data", "train", dataset, "train.jsonl")
    recs = {}
    for l in open(path):
        r = json.loads(l)
        recs[r["sample_id"]] = (r["noise_label"], r["noise_type"])
    return recs


def load_run_metrics(cfg, dataset):
    mdir = os.path.join(cfg["paths"]["data_root"], "runs", dataset, "metrics")
    mp = os.path.join(mdir, "per_sample.jsonl")
    if not os.path.exists(mp) or os.path.getsize(mp) == 0:
        print(f"  skip {dataset}: no metrics yet")
        return pd.DataFrame()
    df = pd.read_json(mp, lines=True)
    if df.empty:
        return df
    df = df.dropna(subset=["loss"])
    pivot = df.pivot_table(index="sample_id", columns="epoch", values="loss",
                           aggfunc="first").add_prefix("loss_ep")
    loss_cols = [c for c in pivot.columns if c.startswith("loss_ep")]
    out = pd.DataFrame(index=pivot.index)
    out[loss_cols] = pivot[loss_cols]
    out["loss_mean"] = pivot[loss_cols].mean(axis=1)
    out["loss_last"] = pivot[loss_cols].iloc[:, -1]
    out["loss_slope"] = pivot[loss_cols].iloc[:, -1] - pivot[loss_cols].iloc[:, 0]
    for metric, cols in [("grad_norm", ["grad_norm"]), ("cos_ref", ["cos_sim_ref"]),
                         ("cos_global", ["cos_sim_global"])]:
        d = df[df[cols[0]].notna()].groupby("sample_id")[cols[0]].mean()
        out[metric + "_mean"] = d
    diag_files = sorted(glob.glob(os.path.join(mdir, "diag_epoch*.jsonl")))
    if diag_files:
        diag = pd.concat([pd.read_json(f, lines=True) for f in diag_files])
        diag = diag.groupby("sample_id")[["max_token_loss", "frac_hard", "mean_loss"]].mean()
        out = out.join(diag)
    return out


def build_table(cfg):
    all_rows = []
    for ds in DATASETS:
        metrics = load_run_metrics(cfg, ds)
        if metrics.empty:
            continue
        labels = load_labels(cfg, ds)
        for sid, row in metrics.iterrows():
            label, ntype = labels.get(int(sid), (0, "none"))
            all_rows.append({"dataset": ds, "sample_id": sid, "noise_label": label,
                             "noise_type": ntype, **row.to_dict()})
    return pd.DataFrame(all_rows)


def univariate_auc(df, dataset, pos_types, neg_types=None):
    sub = df[df["dataset"] == dataset]
    if neg_types is None:
        neg = sub[sub["noise_label"] == 0]
    else:
        neg = sub[sub["noise_type"].isin(neg_types)]
    pos = sub[sub["noise_type"].isin(pos_types)]
    res = {}
    for m in METRIC_ORDER:
        vals_pos, vals_neg = pos[m].dropna(), neg[m].dropna()
        if len(vals_pos) < 10 or len(vals_neg) < 10:
            res[m] = np.nan
            continue
        y = np.concatenate([np.ones(len(vals_pos)), np.zeros(len(vals_neg))])
        x = np.concatenate([vals_pos, vals_neg])
        # direction: higher metric -> more likely noise
        res[m] = roc_auc_score(y, x)
    return res


def main():
    cfg = yaml.safe_load(open("/root/noisedetect/config.yaml"))
    repo = cfg["paths"]["repo_root"]
    res_dir = os.path.join(repo, "results")
    os.makedirs(res_dir, exist_ok=True)
    print("building metric table ...")
    df = build_table(cfg)
    for m in METRIC_ORDER:
        if m not in df.columns:
            df[m] = np.nan
    df.to_csv(os.path.join(res_dir, "per_sample_metrics.csv"), index=False)
    print(f"table: {df.shape}")

    # ---- 1. univariate AUC table -------------------------------------------
    noise_types = ["garbled", "duplicate", "unrelated", "keyword", "mixed"]
    auc_rows = []
    for nt in noise_types:
        pos = [nt] if nt != "mixed" else ["garbled", "duplicate", "unrelated", "keyword"]
        ds = nt if nt != "mixed" else "mixed"
        res = univariate_auc(df, ds, pos)
        auc_rows.append({"noise_type": nt, **{k: round(v, 4) if v == v else None for k, v in res.items()}})
    auc_tab = pd.DataFrame(auc_rows)
    auc_tab.to_csv(os.path.join(res_dir, "auc_univariate.csv"), index=False)
    print("\n=== univariate AUC (noise vs normal, same run) ===")
    print(auc_tab.to_string(index=False))

    # ---- 2. best metric per noise type -------------------------------------
    best = auc_tab.set_index("noise_type").mean(axis=1).sort_values(ascending=False)
    print("\n=== mean AUC over metrics per noise type ===")
    print(best.round(4).to_string())

    # ---- 3. multivariate detection per noise type --------------------------
    mdir = os.path.join(res_dir, "models")
    os.makedirs(mdir, exist_ok=True)
    det_rows = []
    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
    for nt in noise_types:
        ds = nt if nt != "mixed" else "mixed"
        sub = df[df["dataset"] == ds].dropna(subset=METRIC_ORDER)
        if sub.empty:
            print(f"  skip {nt}: no data")
            continue
        sub["label"] = (sub["noise_type"] != "none").astype(int)
        X = sub[METRIC_ORDER].values
        y = sub["label"].values
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        rng = np.random.RandomState(0)
        idx = rng.permutation(len(y))
        n_tr = int(0.7 * len(y))
        tr, te = idx[:n_tr], idx[n_tr:]
        for name, clf in [("LR", LogisticRegression(max_iter=2000)),
                          ("RF", RandomForestClassifier(n_estimators=200, random_state=0))]:
            clf.fit(Xs[tr], y[tr])
            proba = clf.predict_proba(Xs[te])[:, 1]
            a = roc_auc_score(y[te], proba)
            acc = accuracy_score(y[te], (proba > 0.5).astype(int))
            cm = confusion_matrix(y[te], (proba > 0.5).astype(int))
            det_rows.append({"noise_type": nt, "model": name, "auc": round(a, 4),
                             "acc": round(acc, 4), "cm": str(cm.tolist()), "n_test": len(te)})
            if name == "RF":
                fpr, tpr, _ = roc_curve(y[te], proba)
                ax_roc.plot(fpr, tpr, label=f"{nt} (AUC={a:.3f})")
            if name == "LR":
                importances = pd.Series(clf.coef_[0], index=METRIC_ORDER).abs().sort_values(ascending=False)
                print(f"\n[LR] {nt}: top features: {dict(importances.head(3))}")
    det_tab = pd.DataFrame(det_rows)
    det_tab.to_csv(os.path.join(res_dir, "detection_multivariate.csv"), index=False)
    print("\n=== multivariate detection ===")
    print(det_tab.to_string(index=False))
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title("RF ROC: noise vs normal")
    ax_roc.legend(fontsize=7)
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(res_dir, "roc_multivariate.png"), dpi=150)

    # ---- 4. distribution comparison ------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, m in zip(axes.ravel(), METRIC_ORDER):
        data = []
        labels = []
        for nt in noise_types:
            ds = nt if nt != "mixed" else "mixed"
            sub = df[(df["dataset"] == ds)]
            pos = sub[(sub["noise_type"] != "none") & (sub["noise_type"].isin(
                ["garbled", "duplicate", "unrelated", "keyword"] if nt == "mixed" else [nt]))]
            neg = sub[sub["noise_label"] == 0]
            data.append(pos[m].dropna().values)
            data.append(neg[m].dropna().values)
            labels += [f"{nt}\nnoise", f"{nt}\nnormal"]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.tick_params(axis="x", labelsize=6)
        ax.set_title(m, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, "metric_distributions.png"), dpi=150)

    # ---- 5. loss trajectory ------------------------------------------------
    ep_cols = sorted([c for c in df.columns if c.startswith("loss_ep")],
                     key=lambda c: int(c.split("_ep")[1]))
    if ep_cols:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for nt in noise_types:
            ds = nt if nt != "mixed" else "mixed"
            sub = df[(df["dataset"] == ds)]
            pos = sub[(sub["noise_type"] != "none") & (sub["noise_type"].isin(
                ["garbled", "duplicate", "unrelated", "keyword"] if nt == "mixed" else [nt]))]
            if pos.empty:
                continue
            vals = pos[ep_cols].mean(axis=0)
            ax2.plot(range(len(ep_cols)), vals, marker="o", label=f"{nt} noise")
        clean = df[df["dataset"] == "clean"]
        if not clean.empty:
            ax2.plot(range(len(ep_cols)), clean[ep_cols].mean(axis=0),
                     marker="o", color="black", label="clean normal", linewidth=2)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("mean loss")
        ax2.legend()
        ax2.set_title("Loss trajectory by noise type")
        fig2.tight_layout()
        fig2.savefig(os.path.join(res_dir, "loss_trajectory.png"), dpi=150)

    # ---- 6. PCA scatter ------------------------------------------------------
    sub = df.dropna(subset=METRIC_ORDER)
    if not sub.empty:
        feats = StandardScaler().fit_transform(sub[METRIC_ORDER].values)
        pca = PCA(n_components=2).fit_transform(feats)
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        colors = {"none": "#bbbbbb", "garbled": "#e41a1c", "duplicate": "#377eb8",
                  "unrelated": "#4daf4a", "keyword": "#984ea3"}
        for nt, c in colors.items():
            m = sub["noise_type"] == nt
            if m.sum():
                ax3.scatter(pca[m, 0], pca[m, 1], s=4, c=c, label=nt, alpha=0.5)
        ax3.set_title("PCA of per-sample metrics")
        ax3.legend(markerscale=3, fontsize=8)
        fig3.tight_layout()
        fig3.savefig(os.path.join(res_dir, "pca_metrics.png"), dpi=150)

    # ---- 7. evaluation comparison table --------------------------------------
    ev_rows = []
    for ds in DATASETS + ["base"]:
        p = os.path.join(res_dir, f"eval_{ds}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            ev_rows.append({"model": ds, **{k: round(v, 4) for k, v in r.items()}})
    if ev_rows:
        ev_tab = pd.DataFrame(ev_rows)
        ev_tab.to_csv(os.path.join(res_dir, "eval_comparison.csv"), index=False)
        print("\n=== evaluation comparison ===")
        print(ev_tab.to_string(index=False))

    print("plots saved to", res_dir)


if __name__ == "__main__":
    main()
