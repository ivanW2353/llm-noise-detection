"""True top-20% token-loss concentration, computed offline from stored data.

dynanoise's most model-stable signal is `token_loss_top20` — how much of a
sample's total loss sits in its hardest 20% of tokens. This repo never
computed it directly; `frac_hard` (fraction over an absolute threshold 4.0)
and `max_token_loss` are related but not the same thing: frac_hard is
threshold-based and therefore scale-dependent, while a concentration ratio is
scale-free.

No retraining or GPU needed. The diagnostic pass already stored, per sample
per diagnostic epoch:
  - `top_tokens`: the top-32 hardest label tokens as [pos, token_id, loss]
  - `mean_loss`:  mean CE over ALL label tokens
  - `tokens`:     the label-token count (from per_sample.jsonl)
so total loss = mean_loss * n_tokens, and the numerator of any top-k
concentration ratio is a prefix sum of the stored top-32 losses.

COVERAGE CAVEAT (reported per dataset): the exact top-20% needs
ceil(0.2 * n_tokens) <= 32 stored tokens, i.e. n_tokens <= 160. Longer samples
have their top-20% truncated at 32 tokens, which *understates* concentration
for exactly the long samples. Two mitigations are emitted:
  - `top20_share`     : truncated at 32 (biased low for long samples)
  - `top20_share_ok`  : same, NaN when the sample is too long (unbiased, fewer rows)
  - `top32_share`     : always exactly 32 tokens, so no truncation bias at all,
                        but the denominator's token count varies -> length-confounded
  - `top8_share`      : always exact (8 <= 32); the cleanest scale-free variant
Univariate AUC is reported for each so the truncation effect is visible rather
than assumed away.

Usage:
  python scripts/analyze_token_concentration.py --tag ratio10
  python scripts/analyze_token_concentration.py --tags ratio10,ratio05,extra10
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import get_tag, load_config


def _tag(cfg):
    """Legacy helper, use get_tag() for new code."""
    return get_tag(cfg)

# Reference signals already in the feature set, for a like-for-like comparison.
BASELINE_FEATS = ["frac_hard", "max_token_loss", "hard_loss_mean", "entropy"]


def concentration_feats(cfg, tag, dataset):
    """Per-sample top-k loss concentration from the stored top-32 token detail."""
    mdir = os.path.join(cfg["paths"]["data_root"], "runs", tag, dataset, "metrics")
    if not os.path.isdir(mdir):
        return pd.DataFrame()

    # label-token count per sample (constant across epochs)
    ps = os.path.join(mdir, "per_sample.jsonl")
    n_tok = {}
    if os.path.exists(ps):
        for l in open(ps):
            r = json.loads(l)
            if r.get("tokens"):
                n_tok.setdefault(r["sample_id"], r["tokens"])

    # mean_loss per sample per epoch -> total loss denominator
    mean_loss = {}
    for f in sorted(glob.glob(os.path.join(mdir, "diag_epoch*.jsonl"))):
        ep = int(os.path.basename(f).split("epoch")[-1].split(".")[0])
        for l in open(f):
            r = json.loads(l)
            if r.get("mean_loss") is not None:
                mean_loss[(r["sample_id"], ep)] = r["mean_loss"]

    per_epoch = []
    for f in sorted(glob.glob(os.path.join(mdir, "token_diag_epoch*.jsonl"))):
        ep = int(os.path.basename(f).split("epoch")[-1].split(".")[0])
        for l in open(f):
            r = json.loads(l)
            sid, tops = r["sample_id"], r.get("top_tokens") or []
            n = n_tok.get(sid)
            ml = mean_loss.get((sid, ep))
            if not tops or not n or ml is None or ml <= 0:
                continue
            losses = np.sort(np.array([t[2] for t in tops], dtype=float))[::-1]
            total = ml * n
            k20 = int(np.ceil(0.2 * n))
            rec = {"sample_id": sid, "diag_epoch": ep, "n_tokens": n,
                   "n_stored": len(losses),
                   "top20_exact": bool(k20 <= len(losses))}
            # top-20% share (truncated at whatever is stored)
            rec["top20_share"] = float(losses[:min(k20, len(losses))].sum() / total)
            rec["top20_share_ok"] = rec["top20_share"] if rec["top20_exact"] else np.nan
            # fixed-k variants: always exact
            for k in (8, 32):
                kk = min(k, len(losses))
                rec[f"top{k}_share"] = float(losses[:kk].sum() / total)
                rec[f"top{k}_loss_mean"] = float(losses[:kk].mean())
            # concentration shape, independent of the absolute loss level
            rec["top1_over_top8"] = float(losses[0] / max(losses[:min(8, len(losses))].sum(), 1e-9))
            # Gini over the stored hard tokens (how uneven the hard tail is)
            v = np.sort(losses)
            m = len(v)
            rec["hard_gini"] = float((2 * np.arange(1, m + 1) - m - 1).dot(v)
                                     / max(m * v.sum(), 1e-9))
            per_epoch.append(rec)

    if not per_epoch:
        return pd.DataFrame()
    d = pd.DataFrame(per_epoch)
    feat_cols = [c for c in d.columns
                 if c not in ("sample_id", "diag_epoch", "top20_exact", "n_stored")]
    g = d.groupby("sample_id")
    out = g[feat_cols].mean()
    # cross-epoch volatility of the concentration itself
    out["top20_share_std"] = g["top20_share"].std()
    out["top8_share_std"] = g["top8_share"].std()
    out["top20_exact_frac"] = g["top20_exact"].mean()
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(here), "config.yaml"))
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
        base = pd.read_csv(src)
        rows, cov = [], []
        for ds in sorted(base["dataset"].unique()):
            conc = concentration_feats(cfg, tag, ds)
            if conc.empty:
                continue
            sub = base[base["dataset"] == ds].set_index("sample_id")
            j = sub.join(conc, how="inner")
            pos_types = sorted(set(j["noise_type"]) - {"none"})
            if not pos_types:
                continue
            neg = j[j["noise_label"] == 0]
            cov.append({"tag": tag, "dataset": ds, "n_joined": len(j),
                        "median_n_tokens": float(j["n_tokens"].median()),
                        "frac_top20_exact": round(float(j["top20_exact_frac"].mean()), 4)})
            feats = [c for c in conc.columns if c != "top20_exact_frac"]
            for pt in pos_types + (["ALL"] if len(pos_types) > 1 else []):
                p = j if pt == "ALL" else j[j["noise_type"] == pt]
                p = p[p["noise_label"] == 1] if pt == "ALL" else p
                if len(p) < 10 or len(neg) < 10:
                    continue
                for m in feats + [f for f in BASELINE_FEATS if f in j.columns]:
                    vp, vn = p[m].dropna(), neg[m].dropna()
                    if len(vp) < 10 or len(vn) < 10:
                        continue
                    y = np.concatenate([np.ones(len(vp)), np.zeros(len(vn))])
                    auc = roc_auc_score(y, np.concatenate([vp, vn]))
                    rows.append({"tag": tag, "dataset": ds, "noise_type": pt,
                                 "feature": m, "n_pos": len(vp), "n_neg": len(vn),
                                 "auc": round(auc, 4),
                                 "auc_dir": round(max(auc, 1 - auc), 4),
                                 "is_new": m not in BASELINE_FEATS})
        if not rows:
            print(f"[{tag}] no token_diag data — skip")
            continue
        tab = pd.DataFrame(rows)
        out = os.path.join(tag_dir, "token_concentration.csv")
        tab.to_csv(out, index=False)
        print(f"\n=== [{tag}] coverage (top-20% is exact only when n_tokens <= 160) ===")
        print(pd.DataFrame(cov).to_string(index=False))
        print(f"\n=== [{tag}] best concentration feature vs best existing token feature ===")
        for (ds, nt), g in tab.groupby(["dataset", "noise_type"]):
            new = g[g["is_new"]].nlargest(1, "auc_dir")
            old = g[~g["is_new"]].nlargest(1, "auc_dir")
            if new.empty or old.empty:
                continue
            n, o = new.iloc[0], old.iloc[0]
            flag = "  <-- new wins" if n["auc_dir"] > o["auc_dir"] + 0.01 else ""
            print(f"  {ds:<15}{nt:<15}new {n['feature']:<20}{n['auc_dir']:.3f} "
                  f"(raw {n['auc']:.3f})  |  old {o['feature']:<16}{o['auc_dir']:.3f}{flag}")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
