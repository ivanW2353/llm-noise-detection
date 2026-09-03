"""Recompute the diagnostic pass with each run's FINAL model.

The training-time diagnostic pass recorded user_loss as 0.0 (cross_entropy
returns 0 for ignore_index positions), so this re-runs the pass with the
fixed CE computation and saves correct values to <run>/metrics/diag_final.jsonl
(only user_loss is taken from it by the analysis; other metrics keep their
training-time values).

Usage:
  python scripts/2_train/recompute_diag.py            # all datasets
  python scripts/2_train/recompute_diag.py --dataset garbled
"""

import argparse
import json
import os
import sys
import time

import torch
import yaml
from transformers import AutoTokenizer
from peft import PeftModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from train import (MAX_LEN, build_model, diagnostic_pass, tokenize_rows)

DATASETS = ["clean", "garbled", "duplicate", "unrelated", "keyword", "mixed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--subsample", type=int, default=8)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tag = cfg["paths"].get("experiment_tag", "")
    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["model"])
    datasets = [args.dataset] if args.dataset else DATASETS

    for ds in datasets:
        run_dir = os.path.join(cfg["paths"]["data_root"], "runs", tag, ds)
        lora_path = os.path.join(run_dir, "lora")
        if not os.path.exists(lora_path):
            print(f"  skip {ds}: no lora")
            continue
        path = os.path.join(cfg["paths"]["data_root"], "data", tag, ds, "train.jsonl")
        rows = [json.loads(l) for l in open(path)]
        data, _ = tokenize_rows(tokenizer, rows[::args.subsample], MAX_LEN)
        model = build_model(cfg)
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        t0 = time.time()
        diag = diagnostic_pass(model, tokenizer, data,
                               thresh=cfg["train"].get("hard_threshold", 4.0))
        out_p = os.path.join(run_dir, "metrics", "diag_final.jsonl")
        with open(out_p, "w") as f:
            for d in diag:
                f.write(json.dumps({k: v for k, v in d.items() if k != "top_tokens"}) + "\n")
        print(f"  [{time.strftime('%H:%M:%S')}] {ds}: {len(diag)} samples in {time.time()-t0:.0f}s -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
