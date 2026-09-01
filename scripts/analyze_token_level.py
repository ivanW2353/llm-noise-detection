"""Token-level noise analysis (offline, after training).

For a subsample of each dataset:
  - per-token loss over the full sequence (from a retained forward)
  - top-k hardest label tokens: their exact per-token LoRA gradient norm and
    cosine similarity with a clean reference direction (per-token backward)
  - garbled-localization check: are the top-loss tokens exactly the corrupted
    characters? (cross-reference with the clean dataset text)
  - token-level features per sample + detection AUC per noise type

Method C (exact): one backward per hard token, ~1-2 min per dataset.

Usage:
  python scripts/analyze_token_level.py                # all datasets
  python scripts/analyze_token_level.py --dataset garbled
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (MAX_LEN, compute_reference_direction, tokenize_rows)

TOP_K = 24
N_PER_TYPE = 60          # noise samples per type
N_NORMAL = 60            # normal samples per run


def load_model_and_ref(cfg, dataset, ref_rows):
    model = AutoModelForCausalLM.from_pretrained(
        cfg["paths"]["model"], dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map={"": 0})
    if dataset != "base":
        tag = cfg["paths"].get("experiment_tag", "")
        lora_path = os.path.join(cfg["paths"]["data_root"], "runs", tag, dataset, "lora")
        model = PeftModel.from_pretrained(model, lora_path)
    # PEFT loads adapters with requires_grad=False (inference mode);
    # re-enable so per-token backward has a graph
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
    model.eval()
    params, offsets, total, ref_buf = compute_reference_direction(
        model, None, ref_rows, cfg, batch_size=2)
    return model, params, offsets, ref_buf


def hard_tokens(model, row, params, offsets, ref_buf, k=TOP_K):
    """Per-token losses + exact per-token LoRA grad metrics for top-k tokens."""
    input_ids = row["input_ids"].unsqueeze(0).cuda()
    labels = row["labels"].unsqueeze(0).cuda()
    out = model(input_ids=input_ids, labels=labels)
    logits = out.logits[0][:-1]                 # [L-1, V] (shifted prediction)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
        reduction="none").view(-1)
    label_mask = (labels[0, 1:] != -100)
    toks = ce[label_mask]
    k = min(k, len(toks))
    v, idx = toks.topk(k)
    pos = label_mask.nonzero().flatten()[idx]
    buf = torch.zeros(offsets[-1][1], device="cuda")
    res = []
    for val, p in zip(v.tolist(), pos.tolist()):
        loss_t = ce[p]
        g = torch.autograd.grad(loss_t, [p2 for _, p2 in params], retain_graph=True)
        for (s, e), gg in zip(offsets, g):
            buf[s:e].copy_(gg.detach().reshape(-1))
        gn = torch.linalg.vector_norm(buf).item()
        dot = torch.dot(buf, ref_buf).item()
        res.append({"pos": int(p), "loss": float(val),
                    "grad_norm": gn, "cos_ref": dot / max(gn, 1e-12)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--tag", type=str, default=None, help="experiment tag (e.g. ratio05)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.tag:
        cfg["paths"]["experiment_tag"] = args.tag
    tag = cfg["paths"].get("experiment_tag", "")
    res_dir = os.path.join(cfg["paths"]["repo_root"], "results")
    chart_dir = os.path.join(res_dir, "charts")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["model"])
    if args.dataset:
        datasets = [args.dataset]
    else:
        # auto-detect trained noise datasets (exclude clean): one experiment, one analysis
        run_base = os.path.join(cfg["paths"]["data_root"], "runs", tag)
        trained = sorted(os.path.basename(os.path.dirname(d))
                         for d in glob.glob(os.path.join(run_base, "*", "summary.json")))
        datasets = [d for d in trained if d != "clean"] or ["garbled", "duplicate", "unrelated", "keyword"]

    # clean held-out rows (shared reference direction source)
    clean_path = os.path.join(cfg["paths"]["data_root"], "data", tag, "clean", "train.jsonl")
    clean_raw = [json.loads(l) for l in open(clean_path)]
    n_hold = cfg["train"]["ref_samples"] + cfg["train"]["heldout_samples"]
    ref_rows, _ = tokenize_rows(tokenizer, clean_raw[:n_hold], MAX_LEN)

    for ds in datasets:
        path = os.path.join(cfg["paths"]["data_root"], "data", tag, ds, "train.jsonl")
        raw = [json.loads(l) for l in open(path)]
        clean_by_id = {r["sample_id"]: r for r in clean_raw}
        labels = {r["sample_id"]: r["noise_type"] for r in raw}
        train_raw = raw[n_hold:]
        rows, _ = tokenize_rows(tokenizer, train_raw, MAX_LEN)
        sel = [r for r in rows if labels[r["sample_id"]] == ds][:N_PER_TYPE] + \
              [r for r in rows if labels[r["sample_id"]] == "none"][:N_NORMAL]
        print(f"[{ds}] analyzing {len(sel)} samples (top-k={TOP_K}) ...")
        model, params, offsets, ref_buf = load_model_and_ref(cfg, ds, ref_rows)
        results = []
        for k, row in enumerate(sel):
            sid = row["sample_id"]
            if k % 10 == 0:
                print(f"  [{time.strftime('%H:%M:%S')}] {ds}: {k}/{len(sel)} samples", flush=True)
            hard = hard_tokens(model, row, params, offsets, ref_buf)
            if not hard:
                continue
            losses = np.array([h["loss"] for h in hard])
            gns = np.array([h["grad_norm"] for h in hard])
            crs = np.array([h["cos_ref"] for h in hard])
            entry = {
                "sample_id": sid, "noise_type": labels[sid],
                "n_hard": len(hard),
                "hard_loss_mean": float(losses.mean()), "hard_loss_max": float(losses.max()),
                "hard_gradnorm_mean": float(gns.mean()), "hard_cos_ref_mean": float(crs.mean()),
                "hard_cos_ref_min": float(crs.min()),
                "pos_std": float(np.std([h["pos"] for h in hard])) if len(hard) > 1 else 0.0,
                "hard_positions": [h["pos"] for h in hard],
                "hard_losses": losses.tolist(),
            }
            if labels[sid] == "garbled":
                cl = clean_by_id.get(sid)
                if cl:
                    ids = row["input_ids"].tolist()
                    hard_pos = [h["pos"] + 1 for h in hard]
                    cids = tokenizer.apply_chat_template(cl["messages"], tokenize=True)
                    mismatch = sum(1 for p in hard_pos if p < len(cids) and ids[p] != cids[p])
                    entry["loc_mismatch_frac"] = mismatch / max(1, len(hard_pos))
            results.append(entry)
        if tag:
            tl_dir = os.path.join(res_dir, tag)
        else:
            tl_dir = os.path.join(res_dir, "token_level")
        os.makedirs(tl_dir, exist_ok=True)
        out_p = os.path.join(tl_dir, f"token_level_{ds}.jsonl" if tag else f"token_level_{ds}.jsonl")
        with open(out_p, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  saved -> {out_p} ({len(results)} samples)")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        examples = [r for r in results if r["noise_type"] == ds][:3]
        for ax, r in zip(axes, examples):
            ax.scatter(r["hard_positions"], r["hard_losses"], s=10)
            ax.set_title(f"{ds} sample {r['sample_id']} (n_hard={r['n_hard']})")
            ax.set_xlabel("token position")
            ax.set_ylabel("loss")
        fig.tight_layout()
        tc_dir = os.path.join(chart_dir, "token_curve")
        os.makedirs(tc_dir, exist_ok=True)
        fig.savefig(os.path.join(tc_dir, f"token_curve_{tag}_{ds}.png" if tag else f"token_curve_{ds}.png"), dpi=150)

    # detection AUC from token-level features
    from sklearn.metrics import roc_auc_score
    print("\n=== token-level feature AUC ===")
    for ds in datasets:
        p = os.path.join(res_dir, f"token_level_{tag}_{ds}.jsonl" if tag else f"token_level_{ds}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p)]
        pos = [r for r in rows if r["noise_type"] == ds]
        neg = [r for r in rows if r["noise_type"] == "none"]
        if len(pos) < 5 or len(neg) < 5:
            continue
        print(f"[{ds}] pos={len(pos)} neg={len(neg)}")
        for feat in ["hard_loss_mean", "hard_loss_max", "hard_gradnorm_mean",
                     "hard_cos_ref_mean", "pos_std"]:
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            x = np.array([r.get(feat, np.nan) for r in pos + neg], dtype=float)
            x = np.nan_to_num(x)
            try:
                print(f"  {feat:20s} AUC={roc_auc_score(y, x):.3f}")
            except ValueError:
                print(f"  {feat:20s} n/a")


if __name__ == "__main__":
    main()
