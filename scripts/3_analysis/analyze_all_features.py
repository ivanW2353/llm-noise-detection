"""Feature exploration: uses ALL collected per-sample data for noise detection.

Reads, per trained dataset:
  - runs/{tag}/{ds}/metrics/per_sample.jsonl      (per sample x epoch: loss/grad/cos/contrib)
  - runs/{tag}/{ds}/metrics/diag_epoch*.jsonl     (1/8 subsample: max_token_loss/frac_hard/
                                                   mean_loss/user_loss/entropy/skew/kurt)
  - runs/{tag}/{ds}/metrics/token_diag_epoch*.jsonl (top-k hard tokens: pos/token id/loss)
  - runs/{tag}/{ds}/metrics/layer_norms.jsonl     (per-step 36-layer grad/weight norms)
  - runs/{tag}/{ds}/tb/*                          (window aggregates from TensorBoard)

Builds one row per sample with feature columns; computes univariate AUC per noise
type (versus clean samples in the same run), comparing "existing 19-feature" AUCs
against newly derived features. Saves results/{tag}/feature_exploration.csv and
prints the best new features.

Usage: python scripts/3_analysis/analyze_all_features.py --tag ratio05
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.config import get_tag, load_config

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


def _tag(cfg):
    """Legacy helper, use get_tag() for new code."""
    return get_tag(cfg)


def load_labels(cfg, dataset):
    path = os.path.join(cfg["paths"]["data_root"], "data", _tag(cfg), dataset, "train.jsonl")
    recs = {}
    for l in open(path):
        r = json.loads(l)
        recs[r["sample_id"]] = (r["noise_label"], r["noise_type"])
    return recs


def load_per_sample(cfg, dataset):
    mp = os.path.join(cfg["paths"]["data_root"], "runs", _tag(cfg), dataset, "metrics", "per_sample.jsonl")
    df = pd.read_json(mp, lines=True)
    if df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["loss"])
    # basic pivoted stats (mirror analyze_detection)
    ep_cols = sorted(df["epoch"].unique())
    out = pd.DataFrame(index=df["sample_id"].unique())
    out.index.name = "sample_id"
    g = df.groupby("sample_id")["loss"]
    out["loss_mean"] = g.mean()
    out["loss_last"] = g.last()
    out["loss_std"] = g.std()
    out["loss_slope"] = g.last() - g.first()
    out["loss_curvature"] = df.groupby("sample_id")["loss"].apply(
        lambda v: np.polyfit(np.arange(len(v)), v, 2)[0] if len(v) >= 3 else np.nan)
    out["converge_epoch"] = df.groupby("sample_id")["loss"].apply(
        lambda v: int(next((i for i, x in enumerate(v.values) if x < 2.0), len(v))))
    out["grad_norm_mean"] = df.groupby("sample_id")["grad_norm"].mean()
    out["grad_norm_cv"] = df.groupby("sample_id")["grad_norm"].std() / df.groupby("sample_id")["grad_norm"].mean()
    for col in ["cos_sim_ref", "cos_sim_global"]:
        out[f"{col}_mean"] = df.groupby("sample_id")[col].mean()
        out[f"{col}_std"] = df.groupby("sample_id")[col].std()
    out["update_contrib_mean"] = df.groupby("sample_id")["update_contrib"].mean()
    out["tokens"] = df.groupby("sample_id")["tokens"].first()
    # per-window step mapping (window = step)
    out["first_step"] = df.groupby("sample_id")["step"].first()
    return out


def load_diag(cfg, dataset):
    mdir = os.path.join(cfg["paths"]["data_root"], "runs", _tag(cfg), dataset, "metrics")
    rows = []
    for f in sorted(glob.glob(os.path.join(mdir, "diag_epoch*.jsonl"))):
        for l in open(f):
            rows.append(json.loads(l))
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d = d.set_index("sample_id")
    out = pd.DataFrame(index=d.index.unique())
    for c in ["max_token_loss", "frac_hard", "mean_loss", "user_loss", "entropy",
              "token_loss_skew", "token_loss_kurt"]:
        if c in d.columns:
            s = d[c]
            out[c] = s.groupby(level=0).mean()
            out[f"{c}_std"] = s.groupby(level=0).std()          # cross-epoch volatility
            out[f"{c}_curv"] = d.groupby(level=0)[c].apply(
                lambda v: np.polyfit(np.arange(len(v)), v, 2)[0] if len(v) >= 3 else np.nan)
    return out


def load_token_diag(cfg, dataset):
    mdir = os.path.join(cfg["paths"]["data_root"], "runs", _tag(cfg), dataset, "metrics")
    rows = []
    for f in sorted(glob.glob(os.path.join(mdir, "token_diag_epoch*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            r["diag_epoch"] = int(os.path.basename(f).split("epoch")[-1].split(".")[0])
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    feats = []
    for sid, g in d.groupby("sample_id"):
        rec = {"sample_id": sid}
        g = g[g["top_tokens"].map(len) > 0]
        if len(g) == 0:
            continue
        g = g.copy()
        g["n_top"] = g["top_tokens"].map(len)
        g["top_losses"] = g["top_tokens"].map(lambda ts: [t[2] for t in ts])
        g["top_pos"] = g["top_tokens"].map(lambda ts: [t[0] for t in ts])
        g["top_ids"] = g["top_tokens"].map(lambda ts: [t[1] for t in ts])
        rec["n_hard"] = g["n_top"].mean()
        rec["hard_loss_mean"] = g["top_losses"].map(lambda v: float(np.mean(v))).mean()
        rec["hard_loss_max"] = g["top_losses"].map(lambda v: float(np.max(v))).mean()
        rec["hard_pos_std_mean"] = g["top_pos"].map(lambda v: float(np.std(v))).mean()
        rec["hard_pos_peak"] = g["top_pos"].map(lambda v: float(np.mean(v))).mean()
        # hard-token position stability across epochs (Jaccard)
        pos_by_epoch = {}
        for ep, gg in g.groupby("diag_epoch"):
            pos_by_epoch[ep] = set(x for v in gg["top_pos"] for x in v)
        ep_list = sorted(pos_by_epoch)
        jac = []
        for a, b in zip(ep_list, ep_list[1:]):
            pa, pb = pos_by_epoch[a], pos_by_epoch[b]
            jac.append(len(pa & pb) / max(1, len(pa | pb)))
        rec["hard_pos_jaccard"] = float(np.mean(jac)) if jac else np.nan
        # how many unique hard token ids across epochs (repetition signal)
        all_ids = set()
        for ts in g["top_ids"]:
            all_ids |= set(ts)
        rec["hard_id_uniq"] = len(all_ids)
        feats.append(rec)
    return pd.DataFrame(feats).set_index("sample_id")


def load_layer_feats(cfg, dataset):
    """Window-level (per optimizer step) layer gradient norms -> sample-context features."""
    lp = os.path.join(cfg["paths"]["data_root"], "runs", _tag(cfg), dataset, "metrics", "layer_norms.jsonl")
    if not os.path.exists(lp):
        return pd.DataFrame()
    rows = []
    for l in open(lp):
        r = json.loads(l)
        d = r.get("grad", {})
        if not d:
            continue
        v = np.array([float(x) for x in d.values()])
        rows.append({"step": r["step"], "ln_mean": v.mean(), "ln_std": v.std(), "ln_first10": v[:10].mean()})
    ln = pd.DataFrame(rows).set_index("step")
    return ln


def dtype_safe(df):
    return df.apply(pd.to_numeric, errors="coerce")


def auc(series_pos, series_neg):
    y = np.r_[np.ones(len(series_pos)), np.zeros(len(series_neg))]
    x = pd.concat([series_pos, series_neg]).values.astype(float)
    mask = ~np.isnan(x)
    if mask.sum() < 10 or len(set(y[mask])) < 2:
        return np.nan
    try:
        return roc_auc_score(y[mask], x[mask])
    except ValueError:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    ap.add_argument("--datasets", default=None,
                    help="comma-separated noise datasets (default: auto-detect trained)")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.tag:
        cfg["paths"]["experiment_tag"] = args.tag
    tag = _tag(cfg)
    repo = cfg["paths"]["repo_root"]

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        run_base = os.path.join(cfg["paths"]["data_root"], "runs", tag)
        trained = sorted(os.path.basename(os.path.dirname(d))
                         for d in glob.glob(os.path.join(run_base, "*", "summary.json")))
        datasets = [d for d in trained if d != "clean"]

    existing = {"loss_mean", "loss_last", "loss_std", "loss_slope", "loss_curvature",
                "converge_epoch", "grad_norm_mean", "grad_norm_cv", "cos_sim_ref_mean",
                "cos_sim_global_mean", "update_contrib_mean", "max_token_loss", "frac_hard",
                "user_loss", "entropy", "token_loss_skew", "text_nn_sim"}
    new_cols = []   # populated below

    results = []
    for ds in datasets:
        labels = load_labels(cfg, ds)
        base = load_per_sample(cfg, ds)
        diag = load_diag(cfg, ds)
        td = load_token_diag(cfg, ds)
        ln = load_layer_feats(cfg, ds)
        if ln is not None and not ln.empty:
            base = base.merge(ln, left_on="first_step", right_index=True, how="left",
                              suffixes=("", "_win"))
        df = base.join(diag, how="left").join(td, how="left")
        df = dtype_safe(df).reset_index()
        df["noise_type"] = df["sample_id"].map(lambda s: labels.get(int(s), ("?", "none"))[1])
        df["noise_label"] = df["sample_id"].map(lambda s: labels.get(int(s), (0, "none"))[0])

        pos = df[df["noise_type"] == ds]
        neg = df[df["noise_label"] == 0]
        for c in df.columns:
            if c in {"sample_id", "noise_type", "noise_label", "token_nn_sim"}:
                continue
            if c.endswith("_win"):
                continue  # step-level, not per-sample: reported separately
            a = auc(pos[c], neg[c])
            results.append({"dataset": ds, "feature": c,
                            "existing": int(c in existing), "auc": a})

    tab = pd.DataFrame(results)
    out_dir = os.path.join(repo, "results", tag)
    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(os.path.join(out_dir, "feature_exploration.csv"), index=False)

    print(f"\n=== new features AUC >= 0.6 (any noise type) {tag} ===")
    new = tab[(tab["existing"] == 0) & tab["auc"].notna() & (tab["auc"] >= 0.60)]
    new = new.sort_values("auc", ascending=False)
    if new.empty:
        print("  (none)")
    else:
        print(new.to_string(index=False))
    print(f"\n=== existing 19-feature top-5 (per dataset) ===")
    ex = tab[(tab["existing"] == 1) & tab["auc"].notna()]
    for ds, gg in ex.groupby("dataset"):
        gg = gg.sort_values("auc", ascending=False)
        print(f"  {ds}: " + ", ".join(f"{r.feature}={r.auc:.3f}" for r in gg.head(3).itertuples()))
    print(f"\nsaved -> {os.path.join(out_dir, 'feature_exploration.csv')} ({len(tab)} rows)")


if __name__ == "__main__":
    main()
