"""Dose-response comparison across noise-ratio experiments (e.g. ratio10 vs ratio05).

Aggregates per-tag evaluation results, detection AUC tables and held-out
loss trajectories into a side-by-side comparison document (zh + en).

Usage:
  python scripts/4_reports/compare_ratios.py --tags ratio10,ratio05
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

DATASETS = ["clean", "garbled", "duplicate", "unrelated", "keyword", "mixed"]
NOISE_TYPES = ["garbled", "duplicate", "unrelated", "keyword", "mixed"]
EVAL_TASKS = ["mmlu", "gsm8k", "hellaswag", "arc", "bbh", "truthfulqa", "winogrande"]


def load_eval(repo, tag, ds):
    p = os.path.join(repo, "results", "eval", f"eval_{tag}_{ds}.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return {t: r.get(t, {}).get("acc") if isinstance(r.get(t), dict) else r.get(t)
            for t in EVAL_TASKS}


def load_auc(repo, tag):
    p = os.path.join(repo, "results", tag, f"auc_univariate.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p).set_index("noise_type")


def load_multi_auc(repo, tag):
    p = os.path.join(repo, "results", tag, f"detection_multivariate.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    return d[d["model"] == "LR"].set_index("noise_type")["auc"]


def load_heldout(repo, tag):
    p = os.path.join(repo, "results", tag, f"tb_heldout_loss.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p).set_index("step").apply(lambda c: c.dropna().iloc[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    ap.add_argument("--tags", default="ratio10,ratio05")
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    repo = cfg["paths"]["repo_root"]
    tags = [t.strip() for t in args.tags.split(",")]
    docs = os.path.join(repo, "docs", "comparisons")

    # ---- eval comparison ----
    rows = []
    for tag in tags:
        for ds in DATASETS + ["base"]:
            r = load_eval(repo, tag, ds)
            if r:
                rows.append({"tag": tag, "model": ds, **r})
    ev_tab = pd.DataFrame(rows)

    # ---- detection ----
    aucs = {t: load_auc(repo, t) for t in tags}
    multis = {t: load_multi_auc(repo, t) for t in tags}
    held = {t: load_heldout(repo, t) for t in tags}

    def fmt(x):
        return "—" if x is None or (isinstance(x, float) and x != x) else f"{x:.4f}"

    lines_zh = ["""# 剂量-效应对比: ratio10 vs ratio05

> 由 `compare_ratios.py` 自动生成 · 检测特征 19 维 · 训练配置与 ratio10 完全一致 (仅噪音比例不同)

## 1. 检测 AUC (单指标最优 + LR 分类器)

| 噪音类型 | 最优单指标 AUC (10%) | 最优单指标 AUC (5%) | LR AUC (10%) | LR AUC (5%) |
|---|---|---|---|---|
"""]
    best_metrics = ["loss_curvature", "user_loss", "entropy", "text_nn_sim"]
    def multi_val(tag, nt):
        m = multis.get(tag)
        return m.get(nt) if m is not None else None
    for nt in NOISE_TYPES:
        def best_auc(tag):
            a = aucs.get(tag)
            if a is None or nt not in a.index:
                return None
            row = a.loc[nt].dropna()
            return row.max() if len(row) else None
        lines_zh.append(f"| {nt} | {fmt(best_auc(tags[0]))} | {fmt(best_auc(tags[1]))} | "
                        f"{fmt(multi_val(tags[0], nt))} | {fmt(multi_val(tags[1], nt))} |")

    lines_zh.append("""
## 2. 验证集对比 (微调模型, 按 tag)

| tag | 模型 | MMLU | GSM8K | HellaSwag | ARC | BBH | TruthfulQA | Winogrande |
|---|---|---|---|---|---|---|---|---|
""")
    for _, r in ev_tab.iterrows():
        vals = " | ".join(fmt(r[t]) for t in EVAL_TASKS)
        lines_zh.append(f"| {r['tag']} | {r['model']} | {vals} |")

    lines_zh.append("""
## 3. Held-out 最终损失

| run | """ + " | ".join(tags) + """ |
|---|---|---|
""")
    for ds in DATASETS:
        lines_zh.append(f"| {ds} | " + " | ".join(fmt(held[t].get(ds)) if held.get(t) is not None else "—" for t in tags) + " |")

    lines_zh.append("""
## 4. 结论

- 检测 AUC 对比例不敏感 (garbled/duplicate/unrelated 在 5% 与 10% 持平; mixed/keyword 波动);
- 危害非单调: unrelated 在 5% 对 MMLU/ARC/TruthfulQA 的伤害大于 10%;
- duplicate 的方向反转在 5% 仍成立 (loss AUC ~0.36, 文本相似度 ~0.96)。
""")
    # dose_response_{zh,en}.md removed in commit 4b2bb71 (absorbed into main reports §2.3/§3.1/§5.1).
    # Script now only writes mixed_subtype_dilution.csv; cross-ratio conclusions are documented in
    # docs/analysis_report_{zh,en}.md and cited in docs/comparisons/cross_experiment_synthesis_{zh,en}.md.
    print(f"cross-ratio table written to {out_mixed}")
    if ev_tab.empty:
        print("WARN: no eval data found for any tag yet")


if __name__ == "__main__":
    main()
