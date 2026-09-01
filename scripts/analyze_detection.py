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

import argparse
import glob
import json
import math
import os
import sys
import time

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
METRIC_ORDER = ["loss_mean", "loss_last", "loss_std", "loss_slope", "converge_epoch",
                "loss_rank", "loss_curvature", "grad_norm_mean", "grad_norm_cv",
                "cos_ref_mean", "cos_ref_trend", "cos_global_mean", "update_contrib_mean",
                "max_token_loss", "frac_hard", "user_loss", "entropy",
                "token_loss_skew", "text_nn_sim"]


def _tag(cfg):
    return cfg["paths"].get("experiment_tag", "")


def load_labels(cfg, dataset):
    path = os.path.join(cfg["paths"]["data_root"], "data", _tag(cfg),
                        dataset, "train.jsonl")
    recs = {}
    for l in open(path):
        r = json.loads(l)
        recs[r["sample_id"]] = (r["noise_label"], r["noise_type"], r.get("category"))
    return recs


def load_run_metrics(cfg, dataset):
    mdir = os.path.join(cfg["paths"]["data_root"], "runs", _tag(cfg), dataset, "metrics")
    mp = os.path.join(mdir, "per_sample.jsonl")
    if not os.path.exists(mp) or os.path.getsize(mp) == 0:
        print(f"  skip {dataset}: no metrics yet")
        return pd.DataFrame()
    df = pd.read_json(mp, lines=True)
    if df.empty:
        return df
    df = df.dropna(subset=["loss"])
    piv = df.pivot_table(index="sample_id", columns="epoch", aggfunc="first")
    ep_cols = sorted(df["epoch"].unique())
    out = pd.DataFrame(index=piv.index)
    # per-epoch loss / grad_norm / cos_ref / cos_global
    for base, out_name in [("loss", "loss"), ("grad_norm", "grad_norm"),
                           ("cos_sim_ref", "cos_ref"), ("cos_sim_global", "cos_global")]:
        ep = {e: piv[(base, e)] for e in ep_cols if (base, e) in piv.columns}
        if not ep:
            continue
        m = pd.DataFrame(ep)
        out[f"{out_name}_mean"] = m.mean(axis=1)
        out[f"{out_name}_last"] = m.iloc[:, -1]
        out[f"{out_name}_std"] = m.std(axis=1)
        out[f"{out_name}_slope"] = m.iloc[:, -1] - m.iloc[:, 0]
        if out_name == "loss":
            out[[f"loss_ep{e}" for e in ep_cols]] = m
            out["loss_min"] = m.min(axis=1)
            out["converge_epoch"] = (m < 2.0).idxmax(axis=1)
            out.loc[(m >= 2.0).all(axis=1), "converge_epoch"] = len(ep_cols)  # never converged
            # curvature of the loss trajectory (quadratic fit coeff, vectorized)
            xs = np.arange(len(ep_cols))
            X = np.stack([np.ones_like(xs, dtype=float), xs.astype(float), xs.astype(float) ** 2], axis=1)
            pinvX = np.linalg.pinv(X)
            coeffs = m.values.astype(float) @ pinvX.T
            out["loss_curvature"] = coeffs[:, 0]
            out["loss_rank"] = m.rank(pct=True).mean(axis=1)
    if "grad_norm_mean" in out.columns and out["grad_norm_mean"].notna().any():
        out["grad_norm_cv"] = out["grad_norm_std"] / out["grad_norm_mean"].replace(0, np.nan)
    if "cos_ref_mean" in out.columns:
        out["cos_ref_trend"] = out["cos_ref_slope"]
    for m in ("update_contrib",):
        if m in df.columns:
            out[m + "_mean"] = df[df[m].notna()].groupby("sample_id")[m].mean()
    diag_files = sorted(glob.glob(os.path.join(mdir, "diag_epoch*.jsonl")))
    if diag_files:
        diag = pd.concat([pd.read_json(f, lines=True) for f in diag_files])
        cols = [c for c in ["max_token_loss", "frac_hard", "mean_loss", "user_loss",
                            "entropy", "token_loss_skew", "token_loss_kurt"]
                if c in diag.columns]
        diag = diag.groupby("sample_id")[cols].mean()
        out = out.join(diag)
    # user_loss was recorded as 0.0 by an early CE bug; use the post-hoc
    # recomputed values (final model) when available
    dfinal = os.path.join(mdir, "diag_final.jsonl")
    if os.path.exists(dfinal):
        d2 = pd.read_json(dfinal, lines=True)
        if "user_loss" in d2.columns:
            ul = d2.set_index("sample_id")["user_loss"]
            out["user_loss"] = ul
    return out


def load_text_features(cfg, dataset):
    """TF-IDF nearest-neighbor similarity: strong signal for duplicate (exact
    copies) and keyword (only a few words changed) noise."""
    path = os.path.join(cfg["paths"]["data_root"], "data", _tag(cfg),
                        dataset, "train.jsonl")
    texts, sids = [], []
    for l in open(path):
        r = json.loads(l)
        texts.append(r["messages"][0]["content"] + " " + r["messages"][1]["content"])
        sids.append(r["sample_id"])
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=10, sublinear_tf=True,
                          max_features=200_000)
    X = vec.fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(X)
    dist, _ = nn.kneighbors(X)
    sim = 1.0 - dist[:, 1]  # nearest NON-self neighbor similarity
    return dict(zip(sids, sim))


def load_tb_metrics(cfg, dataset):
    """Extract key scalar trajectories from the run's TensorBoard events."""
    import glob as _g
    tag = _tag(cfg)
    tb_dirs = _g.glob(os.path.join(cfg["paths"]["data_root"], "runs", tag, dataset, "tb", "*"))
    if not tb_dirs:
        return None
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(os.path.dirname(tb_dirs[0]))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    for want in ["eval/heldout_loss", "train/loss", "train/grad_norm",
                 "train/cos_ref", "train/cos_global", "train/update_contrib",
                 "lora_layer_gradnorm/layer0", "lora_layer_gradnorm/layer18",
                 "lora_layer_gradnorm/layer35", "diag/max_token_loss_mean",
                 "diag/frac_hard_mean"]:
        if want in tags:
            out[want] = pd.Series({e.step: e.value for e in ea.Scalars(want)})
    return out


def tb_dynamics(cfg, df=None):
    """Compare training-dynamics trajectories across runs (from TensorBoard)."""
    repo = cfg["paths"]["repo_root"]
    tag = _tag(cfg)
    res_dir = os.path.join(repo, "results")
    chart_dir = os.path.join(repo, "results", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    def res_name(name):
        return f"{name}_{tag}.csv" if tag else f"{name}.csv"
    def res_img(name):
        return f"{name}_{tag}.png" if tag else f"{name}.png"
    series = {}
    for ds in DATASETS:
        tb = load_tb_metrics(cfg, ds)
        if tb:
            series[ds] = tb
    if not series:
        print("  (no TensorBoard data yet)")
        return
    # 1. held-out loss trajectory (generalization damage timeline)
    rows = []
    for ds, tb in series.items():
        if "eval/heldout_loss" in tb:
            for step, v in tb["eval/heldout_loss"].items():
                rows.append({"dataset": ds, "step": step, "heldout_loss": v})
    if rows:
        tab = pd.DataFrame(rows).pivot(index="step", columns="dataset", values="heldout_loss")
        tab.to_csv(os.path.join(res_dir, res_name("tb_heldout_loss")))
        fig, ax = plt.subplots(figsize=(8, 5))
        for ds in tab.columns:
            ax.plot(tab.index, tab[ds], marker="o", ms=3, label=ds)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("held-out clean loss")
        ax.set_title("Held-out loss during training (generalization damage)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(chart_dir, res_img("tb_heldout_trajectory")), dpi=150)
        plt.close(fig)
        print("saved tb_heldout_*")
    # 2. per-layer grad norm comparison (final window value per run)
    rows = []
    for ds, tb in series.items():
        for tag_name, li in [("lora_layer_gradnorm/layer0", 0),
                             ("lora_layer_gradnorm/layer18", 18),
                             ("lora_layer_gradnorm/layer35", 35)]:
            if tag_name in tb:
                rows.append({"dataset": ds, "layer": li,
                             "final_grad_norm": tb[tag_name].iloc[-1]})
    if rows:
        tab = pd.DataFrame(rows).pivot(index="layer", columns="dataset", values="final_grad_norm")
        tab.to_csv(os.path.join(res_dir, res_name("tb_layer_gradnorm")))
        fig, ax = plt.subplots(figsize=(8, 5))
        tab.plot.bar(ax=ax)
        ax.set_ylabel("final layer grad norm (window avg)")
        ax.set_title("Per-layer LoRA gradient norms by run")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(chart_dir, res_img("tb_layer_gradnorm")), dpi=150)
        plt.close(fig)
        print("saved tb_layer_gradnorm_*")
    # 3. diagnostic metric trajectory per epoch
    rows = []
    for ds, tb in series.items():
        if "diag/max_token_loss_mean" in tb:
            for step, v in tb["diag/max_token_loss_mean"].items():
                rows.append({"dataset": ds, "step": step, "max_token_loss": v})
    if rows:
        tab = pd.DataFrame(rows).pivot(index="step", columns="dataset", values="max_token_loss")
        tab.to_csv(os.path.join(res_dir, res_name("tb_diag_trajectory")))


def build_table(cfg):
    all_rows = []
    for ds in DATASETS:
        metrics = load_run_metrics(cfg, ds)
        if metrics.empty:
            continue
        labels = load_labels(cfg, ds)
        try:
            text_sim = load_text_features(cfg, ds)
        except Exception as e:
            print(f"  warn text features {ds}: {e}")
            text_sim = {}
        for sid, row in metrics.iterrows():
            label, ntype, cat = labels.get(int(sid), (0, "none", None))
            all_rows.append({"dataset": ds, "sample_id": sid, "noise_label": label,
                             "noise_type": ntype, "category": cat,
                             "text_nn_sim": text_sim.get(int(sid)),
                             **row.to_dict()})
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    ap.add_argument("--tag", type=str, default=None, help="experiment tag (e.g. ratio20)")
    ap.add_argument("--datasets", type=str, default=None,
                    help="comma-separated dataset list (default: the 6 standard datasets)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.tag:
        cfg["paths"]["experiment_tag"] = args.tag
    tag = _tag(cfg)
    if args.datasets:
        globals()["DATASETS"] = [d.strip() for d in args.datasets.split(",")]
    else:
        # auto-detect trained datasets: one experiment, one analysis
        run_base = os.path.join(cfg["paths"]["data_root"], "runs", tag)
        trained = sorted(os.path.basename(os.path.dirname(d))
                         for d in glob.glob(os.path.join(run_base, "*", "summary.json")))
        if trained:
            globals()["DATASETS"] = trained
    repo = cfg["paths"]["repo_root"]
    res_dir = os.path.join(repo, "results")
    chart_dir = os.path.join(repo, "results", "charts")
    eval_dir = os.path.join(repo, "results", "eval")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    tag_dir = os.path.join(res_dir, tag) if tag else res_dir
    os.makedirs(tag_dir, exist_ok=True)
    def res_name(name):
        return os.path.join(tag_dir, f"{name}.csv") if tag else f"{name}.csv"
    def res_img(name):
        return f"{name}_{tag}.png" if tag else f"{name}.png"
    t0_ana = time.time()
    t_sec = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] building metric table ...", flush=True)
    df = build_table(cfg)
    for m in METRIC_ORDER:
        if m not in df.columns:
            df[m] = np.nan
    df.to_csv(os.path.join(res_dir, res_name("per_sample_metrics")), index=False)
    print(f"table: {df.shape}")

    # ---- 0. training-dynamics comparison (from TensorBoard events) ----------
    print("\n=== TensorBoard dynamics (heldout loss / layer norms / diag) ===")
    tb_dynamics(cfg)

    # ---- 1. univariate AUC table -------------------------------------------
    noise_types = ["garbled", "duplicate", "unrelated", "keyword", "mixed"]
    auc_rows = []
    for nt in noise_types:
        pos = [nt] if nt != "mixed" else ["garbled", "duplicate", "unrelated", "keyword"]
        ds = nt if nt != "mixed" else "mixed"
        res = univariate_auc(df, ds, pos)
        auc_rows.append({"noise_type": nt, **{k: round(v, 4) if v == v else None for k, v in res.items()}})
    auc_tab = pd.DataFrame(auc_rows)
    auc_tab.to_csv(os.path.join(res_dir, res_name("auc_univariate")), index=False)
    print(f"[{time.strftime('%H:%M:%S')}] section done in {time.time()-t_sec:.0f}s", flush=True)
    t_sec = time.time()
    print("\n=== univariate AUC (noise vs normal, same run) ===")
    print(auc_tab.to_string(index=False))
    avg_auc = auc_tab.set_index("noise_type")[METRIC_ORDER].mean(axis=1)
    metric_mean = auc_tab[METRIC_ORDER].mean(axis=0).sort_values(ascending=False)
    print(f"[{time.strftime('%H:%M:%S')}] section done in {time.time()-t_sec:.0f}s", flush=True)
    t_sec = time.time()
    print("\n=== best metrics by mean AUC ===")
    print(metric_mean.round(4).to_string())

    # ---- 2. best metric per noise type -------------------------------------
    best = avg_auc.sort_values(ascending=False)
    print(f"[{time.strftime('%H:%M:%S')}] section done in {time.time()-t_sec:.0f}s", flush=True)
    t_sec = time.time()
    print("\n=== mean AUC over metrics per noise type ===")
    print(best.round(4).to_string())

    # ---- 3. multivariate detection per noise type --------------------------
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
    det_tab.to_csv(os.path.join(res_dir, res_name("detection_multivariate")), index=False)
    print(f"[{time.strftime('%H:%M:%S')}] section done in {time.time()-t_sec:.0f}s", flush=True)
    t_sec = time.time()
    print("\n=== multivariate detection ===")
    print(det_tab.to_string(index=False))
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title("RF ROC: noise vs normal")
    ax_roc.legend(fontsize=7)
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(chart_dir, res_img("roc_multivariate")), dpi=150)

    # ---- 3.5 category-stratified detection (task-type transferability) ------
    cat_rows = []
    for cat in sorted(df["category"].dropna().unique()):
        sub = df[(df["category"] == cat)].dropna(subset=METRIC_ORDER)
        if sub.empty or sub["noise_label"].sum() < 10:
            continue
        sub["label"] = (sub["noise_type"] != "none").astype(int)
        X = sub[METRIC_ORDER].values
        y = sub["label"].values
        if len(set(y)) < 2:
            continue
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        rng = np.random.RandomState(0)
        idx = rng.permutation(len(y))
        n_tr = int(0.7 * len(y))
        tr, te = idx[:n_tr], idx[n_tr:]
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
        clf.fit(Xs[tr], y[tr])
        proba = clf.predict_proba(Xs[te])[:, 1]
        cat_rows.append({"category": cat, "n": len(y), "n_noise": int(y.sum()),
                         "rf_auc": round(roc_auc_score(y[te], proba), 4)})
    if cat_rows:
        cat_tab = pd.DataFrame(cat_rows).sort_values("rf_auc", ascending=False)
        cat_tab.to_csv(os.path.join(res_dir, res_name("auc_by_category")), index=False)
        print(f"[{time.strftime('%H:%M:%S')}] section done in {time.time()-t_sec:.0f}s", flush=True)
        t_sec = time.time()
        print("\n=== RF detection AUC by task category (transferability) ===")
        print(cat_tab.to_string(index=False))
    # noise-type × category AUC matrix
    mat_rows = []
    for cat in sorted(df["category"].dropna().unique()):
        for nt in ["garbled", "duplicate", "unrelated", "keyword"]:
            ds = nt
            sub = df[(df["category"] == cat) & (df["dataset"] == ds)].dropna(subset=METRIC_ORDER)
            if sub.empty or sub["noise_label"].sum() < 5:
                continue
            sub["label"] = (sub["noise_type"] != "none").astype(int)
            X = sub[METRIC_ORDER].values
            y = sub["label"].values
            if len(set(y)) < 2:
                continue
            sc = StandardScaler().fit(X)
            rng = np.random.RandomState(0)
            idx = rng.permutation(len(y))
            n_tr = int(0.7 * len(y))
            tr, te = idx[:n_tr], idx[n_tr:]
            clf = RandomForestClassifier(n_estimators=200, random_state=0)
            clf.fit(sc.transform(X)[tr], y[tr])
            proba = clf.predict_proba(sc.transform(X)[te])[:, 1]
            mat_rows.append({"category": cat, "noise_type": nt,
                             "rf_auc": round(roc_auc_score(y[te], proba), 4),
                             "n": len(y)})
    if mat_rows:
        mat_tab = pd.DataFrame(mat_rows).pivot(index="category", columns="noise_type", values="rf_auc")
        mat_tab.to_csv(os.path.join(res_dir, res_name("auc_category_x_noise")), index=False)
        print(f"[{time.strftime('%H:%M:%S')}] section done in {time.time()-t_sec:.0f}s", flush=True)
        t_sec = time.time()
        print("\n=== RF AUC: category x noise type ===")
        print(mat_tab.to_string())

    # ---- 4. distribution comparison (one figure per metric) ----------------
    for m in METRIC_ORDER:
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
        if all(len(d) == 0 for d in data):
            continue
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(m, fontsize=11)
        ax.set_ylabel(m)
        fig.tight_layout()
        dist_dir = os.path.join(chart_dir, "metric_dist")
        os.makedirs(dist_dir, exist_ok=True)
        fig.savefig(os.path.join(dist_dir, res_img(f"metric_dist_{m}")), dpi=150)
        plt.close(fig)

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
        fig2.savefig(os.path.join(chart_dir, res_img("loss_trajectory")), dpi=150)

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
        fig3.savefig(os.path.join(chart_dir, res_img("pca_metrics")), dpi=150)

    # ---- 7. evaluation comparison table --------------------------------------
    ev_rows = []
    for ds in DATASETS + ["base"]:
        p = os.path.join(eval_dir, f"eval_{tag}_{ds}.json" if tag else f"eval_{ds}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            row = {"model": ds}
            for k, v in r.items():
                row[k] = round(v["acc"], 4) if isinstance(v, dict) and "acc" in v else round(v, 4)
            ev_rows.append(row)
    if ev_rows:
        ev_tab = pd.DataFrame(ev_rows)
        ev_tab.to_csv(os.path.join(tag_dir, f"eval_comparison.csv"), index=False)
        print(f"[{time.strftime('%H:%M:%S')}] section done in {time.time()-t_sec:.0f}s", flush=True)
        t_sec = time.time()
        print("\n=== evaluation comparison ===")
        print(ev_tab.to_string(index=False))
        # per-group comparisons (MMLU subjects, HellaSwag activities,
        # TruthfulQA categories, BBH tasks)
        task_of = {"subjects": "mmlu", "activities": "hellaswag",
                   "categories": "truthfulqa", "per_task": "bbh"}
        for gkey, gname in [("subjects", "mmlu_subjects"), ("activities", "hellaswag_activities"),
                            ("categories", "truthfulqa_categories"), ("per_task", "bbh_tasks")]:
            subj_rows = []
            for ds in DATASETS + ["base"]:
                p = os.path.join(eval_dir, f"eval_{tag}_{ds}.json" if tag else f"eval_{ds}.json")
                if os.path.exists(p):
                    r = json.load(open(p))
                    task_r = r.get(task_of[gkey])
                    v = task_r.get(gkey) if isinstance(task_r, dict) else None
                    if v:
                        subj_rows.append({"model": ds, **{k: round(x, 4) for k, x in v.items()}})
            if subj_rows:
                subj_tab = pd.DataFrame(subj_rows)
                subj_tab.to_csv(os.path.join(tag_dir, f"eval_{gname}.csv"), index=False)
                print(f"\n=== {gname} ({len(subj_tab.columns)-1} groups) ===")
                print(subj_tab.to_string(index=False))

    print(f"[{time.strftime('%H:%M:%S')}] total analysis time {time.time()-t0_ana:.0f}s")
    print("plots saved to", res_dir)


if __name__ == "__main__":
    main()