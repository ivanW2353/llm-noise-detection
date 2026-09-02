"""Label-free detection of MEMORIZED noise via a signed 'hyper-typicality' rule.

§3.9 found that generic outlier detection fails on exactly the noise families
the three experiments care about most: duplicate (0.982 supervised -> 0.699
label-free) and template (0.988 -> 0.633, with P@10% 0.059, *below* random).
The mechanism is structural, not a tuning failure: IsolationForest and
Mahalanobis rank by *atypicality*, but memorized noise is not atypical — it is
**hyper-typical**. A fixed response template is learned perfectly, so its loss
and entropy sit far BELOW the clean population and it lands at the centre of
the distribution, where no single-population outlier model can reach it.

Both reports list the fix as future work: stop asking "is this sample an
outlier?" and ask "is this sample *too easy*?" — a SIGNED rule instead of a
two-sided |z|. This script measures three variants, all still label-free
(labels used only to evaluate afterwards):

  1. `memo_signed`   — mean of signed robust z-scores over learnability
     features, oriented so that "easier than typical" scores HIGH:
     low loss_mean, low loss_last, low loss_std, low loss_curvature,
     fast converge_epoch, low grad_norm_mean. No labels, and unlike |z| the
     direction is fixed a priori by the memorization hypothesis rather than
     fitted.
  2. `memo_plus_conc` — variant 1 plus the scale-free concentration term
     (top20_share, §3.12), which is the one template feature that does not
     invert. Only available for the 1/8 diagnostic subsample.
  3. `low_loss_only`  — the trivial one-feature baseline (-loss_mean), to show
     how much of the result is just "noise has low loss".

The direction is a *hypothesis about memorization*, not a fit to these labels,
so the same rule applies unchanged to every dataset. That is the honest test:
a rule that had to be re-signed per dataset would need labels and would defeat
the purpose. Consequently it is expected to score BELOW 0.5 on the
non-memorized families (garbled/unrelated are HARDER than clean) — that
asymmetry is the point, and it is reported rather than hidden by taking a
directional max.

Usage:
  python scripts/analyze_memorization_score.py --tags ratio10,ratio05,extra10
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import get_tag
from src.scorers import robust_z, memo_scores, MEMO_FEATS
from src.eval_utils import precision_at_k
from analyze_token_concentration import concentration_feats


def _tag(cfg):
    """Legacy helper, use get_tag() for new code."""
    return get_tag(cfg)


def two_tailed_precision(y, s, budget=0.10):
    """Precision of spending half the budget at each tail.

    The tempting fix when you don't know the sign: drop the most extreme
    samples from BOTH ends. For a SINGLE noise family it is strictly worse than
    the correctly-signed one-sided rule — one tail is pure clean data, so half
    the budget is wasted by construction. But on a MIXED run it wins, because
    several families populate opposite tails (memorized noise at the low-loss
    end, surface corruption at the high-loss end). Reported so the trade-off is
    measured rather than argued.
    """
    k = int(budget / 2 * len(y))
    if k < 1:
        return np.nan
    o = np.argsort(s)
    return float(y[np.r_[o[:k], o[-k:]]].mean())


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

    all_rows = []
    for tag in tags:
        tag_dir = os.path.join(repo, "results", tag)
        src = os.path.join(tag_dir, "per_sample_metrics.csv")
        if not os.path.exists(src):
            print(f"[{tag}] missing {src} — skip")
            continue
        df = pd.read_csv(src)

        rows = []
        for ds in sorted(df["dataset"].unique()):
            if ds == "clean":
                continue
            d = df[df["dataset"] == ds]
            variants = {"memo_signed": MEMO_FEATS,
                        "low_loss_only": {"loss_mean": -1}}
            # concentration (diagnostic subsample only) for variant 2
            c = concentration_feats(cfg, tag, ds)
            if not c.empty and "top20_share" in c.columns:
                d = d.merge(c[["top20_share"]].reset_index(), on="sample_id", how="left")
                variants["memo_plus_conc"] = {**MEMO_FEATS, "top20_share": +1}
                variants["conc_only"] = {"top20_share": +1}
            for name, fs in variants.items():
                sub, s, cols = memo_scores(d, fs)
                if sub is None:
                    continue
                y = (sub["noise_type"] != "none").astype(int).values
                if len(set(y)) < 2 or y.sum() < 10:
                    continue
                auc = roc_auc_score(y, s)
                base = y.mean()
                p10, r10, _ = precision_at_k(y, s, 0.10)
                pb, rb, _ = precision_at_k(y, s, base)
                rows.append({
                    "tag": tag, "dataset": ds, "scorer": name,
                    "n": len(y), "n_noise": int(y.sum()), "n_feats": len(cols),
                    "auc": round(auc, 4), "p_at_10": round(p10, 4),
                    "recall_at_10": round(r10, 4),
                    "p_at_base": round(pb, 4), "random_p": round(base, 4),
                    "lift_at_10": round(p10 / base, 2) if base else None,
                    "p_two_tailed_10": round(two_tailed_precision(y, s), 4)})
        if not rows:
            print(f"[{tag}] nothing scored")
            continue
        tab = pd.DataFrame(rows)
        out = os.path.join(tag_dir, "memorization_detection.csv")
        tab.to_csv(out, index=False)
        all_rows.append(tab)
        print(f"\n=== [{tag}] signed memorization score (label-free) ===")
        print(tab.pivot_table(index="dataset", columns="scorer",
                              values="auc").round(3).to_string())
        print(f"--- precision@10% (random = {tab['random_p'].iloc[0]}) ---")
        print(tab.pivot_table(index="dataset", columns="scorer",
                              values="p_at_10").round(3).to_string())
        print(f"saved -> {out}")

    if all_rows:
        a = pd.concat(all_rows)
        m = a[a["scorer"] == "memo_signed"]
        print("\n=== signed rule: AUC by dataset (>0.5 = memorized-type noise) ===")
        print(m[["tag", "dataset", "auc", "p_at_10", "random_p"]]
              .sort_values("auc", ascending=False).to_string(index=False))
        print("\n=== the two-tailed 'don't know the sign' fallback (5% each end) ===")
        print("(compare p_two_tailed_10 against p_at_10: it loses to a correctly-signed\n"
              " one-sided rule on single-type runs, but wins on mixed runs where\n"
              " several families populate opposite tails)")
        print(m[["tag", "dataset", "p_at_10", "p_two_tailed_10", "random_p"]]
              .sort_values("p_at_10", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
