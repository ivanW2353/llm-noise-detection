"""Noise-vs-normal separation analysis on per-sample training metrics.

Combines per-sample metrics (loss / grad_norm / cos_sim_ref / cos_sim_global
and epoch-end token diagnostics) with noise labels from the datasets, then:

  1. compares metric distributions noise vs normal, per noise type
  2. univariate detection: AUC per metric per noise type
  3. multivariate detection: logistic regression / random forest + ROC +
     feature importance + confusion matrix
  3b. dilution check: per-subtype detectability inside the `mixed` run vs the
     same noise type's own single-type run (trajectory features only, so all
     samples count, not just the 1/8 diagnostic subsample)
  4. PCA scatter of samples colored by noise type
  5. loss trajectories across epochs per noise type
  6. aggregates evaluation tables (results/eval_*.json)

Detection targets are derived from the data (`noise_spec`): every trained
noise dataset is scored individually, and `mixed` aggregates whatever subtypes
its own labels contain (4-way or 7-way) — no hardcoded noise-type list.

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
                "token_loss_skew", "text_nn_sim",
                # full-feature exploration additions (see analyze_all_features.py / §3.6)
                "mean_loss", "mean_loss_std", "mean_loss_curv",
                "frac_hard_std", "frac_hard_curv", "entropy_std", "entropy_curv",
                "max_token_loss_std", "max_token_loss_curv",
                "user_loss_std", "token_loss_skew_std", "token_loss_kurt",
                "token_loss_kurt_std", "token_loss_kurt_curv",
                "n_hard", "hard_loss_mean", "hard_loss_max",
                "hard_pos_peak", "hard_pos_std_mean", "hard_id_uniq", "hard_pos_jaccard"]

# Features available for EVERY training sample (per-epoch tracking, micro-batch 1)
# as opposed to the diagnostic/token features that only exist for the 1/8
# subsample. Needed for the mixed-run per-subtype analysis: a subtype has
# 200-730 samples in a mixed run, but only ~30-90 of them are in the diag
# subsample, so dropna over the full METRIC_ORDER would collapse it.
TRAJ_METRICS = ["loss_mean", "loss_last", "loss_std", "loss_slope", "converge_epoch",
                "loss_rank", "loss_curvature", "grad_norm_mean", "grad_norm_cv",
                "cos_ref_mean", "cos_ref_trend", "update_contrib_mean", "text_nn_sim"]


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
        diag_mean = diag.groupby("sample_id")[cols].mean()
        out = out.join(diag_mean)
        for c in cols:
            s = diag[c]
            std = s.groupby(level=0).std() if isinstance(s.index, pd.MultiIndex) else diag.groupby("sample_id")[c].std()
            out[f"{c}_std"] = std
            out[f"{c}_curv"] = diag.groupby("sample_id")[c].apply(
                lambda v: np.polyfit(np.arange(len(v)), v, 2)[0] if len(v) >= 3 else np.nan)
    # user_loss was recorded as 0.0 by an early CE bug; use the post-hoc
    # recomputed values (final model) when available
    dfinal = os.path.join(mdir, "diag_final.jsonl")
    if os.path.exists(dfinal):
        d2 = pd.read_json(dfinal, lines=True)
        if "user_loss" in d2.columns:
            ul = d2.set_index("sample_id")["user_loss"]
            out["user_loss"] = ul
    tok = load_token_features(cfg, dataset)
    if not tok.empty:
        out = out.join(tok)
    return out


def load_token_features(cfg, dataset):
    """Features from token_diag_epoch*.jsonl (top-k hard label tokens per sample).

    Each record: sample_id -> top_tokens [[pos, token_id, loss], ...] per epoch.
    Derived (per sample): hard-token count & loss stats, hard-position stats,
    unique hard-token-ids, adjacent-epoch position-overlap (Jaccard).
    """
    mdir = os.path.join(cfg["paths"]["data_root"], "runs", _tag(cfg), dataset, "metrics")
    rows = []
    for f in sorted(glob.glob(os.path.join(mdir, "token_diag_epoch*.jsonl"))):
        ep = int(os.path.basename(f).split("epoch")[-1].split(".")[0])
        for l in open(f):
            r = json.loads(l)
            r["_epoch"] = ep
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    feats = []
    for sid, g in pd.DataFrame(rows).groupby("sample_id"):
        g = g[g["top_tokens"].map(len) > 0]
        if g.empty:
            continue
        ts = [t for row in g["top_tokens"] for t in row]
        rec = {"sample_id": sid}
        rec["n_hard"] = float(np.mean([len(x) for x in g["top_tokens"]]))
        rec["hard_loss_mean"] = float(np.mean([t[2] for t in ts]))
        rec["hard_loss_max"] = float(np.mean([max(x[2] for x in row) for row in g["top_tokens"]]))
        rec["hard_pos_peak"] = float(np.mean([np.mean([t[0] for t in row]) for row in g["top_tokens"]]))
        rec["hard_pos_std_mean"] = float(np.mean([np.std([t[0] for t in row]) for row in g["top_tokens"]]))
        all_ids = set()
        for row in g["top_tokens"]:
            all_ids |= {t[1] for t in row}
        rec["hard_id_uniq"] = float(len(all_ids))
        pos_by_epoch = {}
        for ep, gg in g.groupby("_epoch"):
            pos_by_epoch[ep] = set(x for row in gg["top_tokens"] for x in [t[0] for t in row])
        eps = sorted(pos_by_epoch)
        jac = []
        for a, b in zip(eps, eps[1:]):
            pa, pb = pos_by_epoch[a], pos_by_epoch[b]
            jac.append(len(pa & pb) / max(1, len(pa | pb)))
        rec["hard_pos_jaccard"] = float(np.mean(jac)) if jac else np.nan
        feats.append(rec)
    return pd.DataFrame(feats).set_index("sample_id")


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
    res_dir = os.path.join(repo, "results", tag) if tag else os.path.join(repo, "results")
    chart_dir = os.path.join(repo, "results", "charts")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)
    def res_name(name):
        return f"{name}.csv"
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


def noise_spec(df):
    """(name, dataset, pos_types) per detection target, derived from the data.

    Any trained noise dataset gets its own single-type row (so extended noise
    types like template/truncation/near_duplicate are evaluated individually,
    not only inside `mixed`), and `mixed` aggregates whatever subtypes its own
    labels contain (4-way or 7-way).
    """
    spec = []
    for ds in sorted(df["dataset"].unique()):
        if ds in ("clean", "mixed"):
            continue
        types = sorted(set(df[df["dataset"] == ds]["noise_type"]) - {"none"})
        if types:
            spec.append((ds, ds, types))
    if "mixed" in set(df["dataset"]):
        sub_types = sorted(set(df[df["dataset"] == "mixed"]["noise_type"]) - {"none"})
        if sub_types:
            spec.append(("mixed", "mixed", sub_types))
    return spec


def fit_eval(X, y, seed=0, models=("LR", "RF")):
    """70/30 split fit; returns per-model (auc, acc, cm, proba, y_test, clf)."""
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    idx = np.random.RandomState(seed).permutation(len(y))
    n_tr = int(0.7 * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]
    out = {}
    for name in models:
        clf = (LogisticRegression(max_iter=2000) if name == "LR"
               else RandomForestClassifier(n_estimators=200, random_state=seed))
        clf.fit(Xs[tr], y[tr])
        proba = clf.predict_proba(Xs[te])[:, 1]
        pred = (proba > 0.5).astype(int)
        out[name] = (roc_auc_score(y[te], proba), accuracy_score(y[te], pred),
                     confusion_matrix(y[te], pred), proba, y[te], clf)
    return out


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
    spec = noise_spec(df)
    print(f"detection targets: {[(n, len(t)) for n, _, t in spec]}")
    auc_rows = []
    for name, ds, pos in spec:
        res = univariate_auc(df, ds, pos)
        auc_rows.append({"noise_type": name, **{k: round(v, 4) if v == v else None for k, v in res.items()}})
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
    for name, ds, pos in spec:
        sub = df[df["dataset"] == ds].dropna(subset=METRIC_ORDER)
        if sub.empty:
            print(f"  skip {name}: no data")
            continue
        sub = sub[sub["noise_type"].isin(["none"] + list(pos))]
        y = (sub["noise_type"] != "none").astype(int).values
        if len(set(y)) < 2 or y.sum() < 10:
            print(f"  skip {name}: {y.sum()} noise samples with full features")
            continue
        fits = fit_eval(sub[METRIC_ORDER].values, y)
        for mname, (a, acc, cm, proba, y_te, clf) in fits.items():
            det_rows.append({"noise_type": name, "model": mname, "auc": round(a, 4),
                             "acc": round(acc, 4), "cm": str(cm.tolist()), "n_test": len(y_te)})
            if mname == "RF":
                fpr, tpr, _ = roc_curve(y_te, proba)
                ax_roc.plot(fpr, tpr, label=f"{name} (AUC={a:.3f})")
            if mname == "LR":
                importances = pd.Series(clf.coef_[0], index=METRIC_ORDER).abs().sort_values(ascending=False)
                print(f"\n[LR] {name}: top features: {dict(importances.head(3))}")
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
        for name, ds, pos in spec:
            if name == "mixed":
                continue
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
            mat_rows.append({"category": cat, "noise_type": name,
                             "rf_auc": round(roc_auc_score(y[te], proba), 4),
                             "n": len(y)})
    if mat_rows:
        mat_tab = pd.DataFrame(mat_rows).pivot(index="category", columns="noise_type", values="rf_auc")
        mat_tab.to_csv(os.path.join(res_dir, res_name("auc_category_x_noise")))
        print(f"[{time.strftime('%H:%M:%S')}] section done in {time.time()-t_sec:.0f}s", flush=True)
        t_sec = time.time()
        print("\n=== RF AUC: category x noise type ===")
        print(mat_tab.to_string())

    # ---- 3.6 dilution: per-subtype detectability inside the mixed run ------
    # Does mixing noise types dilute each type's signal? The mixed run has
    # 200-730 samples per subtype for the always-tracked trajectory features
    # (TRAJ_METRICS), so each subtype can be scored against the SAME run's
    # normal samples and compared with its own single-type run.
    if "mixed" in set(df["dataset"]):
        dil_rows = []
        mx = df[df["dataset"] == "mixed"]
        mx_neg = mx[mx["noise_type"] == "none"]
        for nt in sorted(set(mx["noise_type"]) - {"none"}):
            pos = mx[mx["noise_type"] == nt]
            row = {"noise_type": nt, "n_noise_mixed": len(pos)}
            # univariate AUC of each trajectory feature, mixed run vs own run
            for m in TRAJ_METRICS:
                p, n = pos[m].dropna(), mx_neg[m].dropna()
                if len(p) < 10 or len(n) < 10:
                    row[f"mixed_{m}"] = None
                    continue
                row[f"mixed_{m}"] = round(roc_auc_score(
                    np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]), 4)
            own = df[(df["dataset"] == nt)] if nt in set(df["dataset"]) else None
            if own is not None and not own.empty:
                own_neg = own[own["noise_type"] == "none"]
                own_pos = own[own["noise_type"] == nt]
                for m in TRAJ_METRICS:
                    p, n = own_pos[m].dropna(), own_neg[m].dropna()
                    if len(p) < 10 or len(n) < 10:
                        row[f"own_{m}"] = None
                        continue
                    row[f"own_{m}"] = round(roc_auc_score(
                        np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]), 4)
            # multivariate RF/LR on the trajectory features only
            sub = mx[mx["noise_type"].isin(["none", nt])].dropna(subset=TRAJ_METRICS)
            y = (sub["noise_type"] != "none").astype(int).values
            if len(set(y)) == 2 and y.sum() >= 10:
                fits = fit_eval(sub[TRAJ_METRICS].values, y)
                row["mixed_rf_auc"] = round(fits["RF"][0], 4)
                row["mixed_lr_auc"] = round(fits["LR"][0], 4)
            if own is not None and not own.empty:
                sub = own[own["noise_type"].isin(["none", nt])].dropna(subset=TRAJ_METRICS)
                y = (sub["noise_type"] != "none").astype(int).values
                if len(set(y)) == 2 and y.sum() >= 10:
                    fits = fit_eval(sub[TRAJ_METRICS].values, y)
                    row["own_rf_auc"] = round(fits["RF"][0], 4)
                    row["own_lr_auc"] = round(fits["LR"][0], 4)
            dil_rows.append(row)
        if dil_rows:
            dil_tab = pd.DataFrame(dil_rows)
            dil_tab.to_csv(os.path.join(res_dir, res_name("mixed_subtype_dilution")), index=False)
            print(f"[{time.strftime('%H:%M:%S')}] section done in {time.time()-t_sec:.0f}s", flush=True)
            t_sec = time.time()
            print("\n=== dilution: subtype AUC inside mixed vs its own run "
                  "(trajectory features, all samples) ===")
            show = ["noise_type", "n_noise_mixed", "mixed_rf_auc", "own_rf_auc",
                    "mixed_lr_auc", "own_lr_auc",
                    "mixed_loss_curvature", "own_loss_curvature",
                    "mixed_text_nn_sim", "own_text_nn_sim"]
            print(dil_tab[[c for c in show if c in dil_tab.columns]].to_string(index=False))

    # ---- 4. distribution comparison (one figure per metric) ----------------
    for m in METRIC_ORDER:
        data = []
        labels = []
        for name, ds, pos_types in spec:
            sub = df[(df["dataset"] == ds)]
            pos = sub[sub["noise_type"].isin(pos_types)]
            neg = sub[sub["noise_label"] == 0]
            data.append(pos[m].dropna().values)
            data.append(neg[m].dropna().values)
            labels += [f"{name}\nnoise", f"{name}\nnormal"]
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
        for name, ds, pos_types in spec:
            sub = df[(df["dataset"] == ds)]
            pos = sub[sub["noise_type"].isin(pos_types)]
            if pos.empty:
                continue
            vals = pos[ep_cols].mean(axis=0)
            ax2.plot(range(len(ep_cols)), vals, marker="o", label=f"{name} noise")
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
                  "unrelated": "#4daf4a", "keyword": "#984ea3",
                  "template": "#ff7f00", "truncation": "#a65628",
                  "near_duplicate": "#f781bf"}
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