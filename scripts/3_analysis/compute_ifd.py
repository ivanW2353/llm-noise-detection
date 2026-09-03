"""Post-hoc IFD (Instruction-Following Difficulty) computation.

IFD(x) = L(A | Q) / L(A)   where L(A|Q) is the response loss given the
prompt, and L(A) is the response loss computed on the response alone
(no prompt). dynanoise found IFD is the only signal that reliably detects
template (consistent-pattern) noise (AUROC 0.90) and helps for
pseudo-quality noise (0.60).

Runs on each dataset's FINAL model over the same 1/8 diagnostic subsample,
saving results/<tag>/ifd_<dataset>.jsonl. Needs GPU.

Usage:
  python scripts/3_analysis/compute_ifd.py
  python scripts/3_analysis/compute_ifd.py --dataset shortcut --tag shortcut10
"""

import argparse
import glob
import json
import os
import sys
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "scripts", "2_train"))
sys.path.insert(0, project_root)
from train import MAX_LEN, tokenize_rows

DATASETS = ["clean", "garbled", "duplicate", "unrelated", "keyword",
            "template", "truncation", "near_duplicate", "mixed"]


@torch.no_grad()
def response_only_loss(model, tokenizer, assistant_text):
    """L(A): loss of the assistant response without any prompt."""
    msg = [{"role": "assistant", "content": assistant_text}]
    text = tokenizer.apply_chat_template(msg, tokenize=False)
    ids = tokenizer(text, add_special_tokens=False, truncation=True,
                    max_length=MAX_LEN)["input_ids"]
    out = model(input_ids=torch.tensor([ids]).cuda(), labels=torch.tensor([ids]).cuda())
    return out.loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    ap.add_argument("--tag", type=str, default=None, help="experiment tag (e.g. extra10)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--subsample", type=int, default=8)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.tag:
        cfg["paths"]["experiment_tag"] = args.tag
    tag = cfg["paths"].get("experiment_tag", "")
    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["model"])
    if args.dataset:
        datasets = [args.dataset]
    else:
        # auto-detect trained datasets: one experiment, one analysis
        run_base = os.path.join(cfg["paths"]["data_root"], "runs", tag)
        trained = sorted(os.path.basename(os.path.dirname(d))
                         for d in glob.glob(os.path.join(run_base, "*", "summary.json")))
        datasets = trained or DATASETS

    for ds in datasets:
        run_dir = os.path.join(cfg["paths"]["data_root"], "runs", tag, ds)
        if not os.path.exists(os.path.join(run_dir, "lora")):
            print(f"  skip {ds}: no lora")
            continue
        path = os.path.join(cfg["paths"]["data_root"], "data", tag, ds, "train.jsonl")
        raw = [json.loads(l) for l in open(path)]
        rows, _ = tokenize_rows(tokenizer, raw[::args.subsample], MAX_LEN)
        labels = {r["sample_id"]: (r["noise_label"], r["noise_type"]) for r in raw}
        raw_by_id = {r["sample_id"]: r for r in raw}
        model = AutoModelForCausalLM.from_pretrained(
            cfg["paths"]["model"], dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map={"": 0})
        model = PeftModel.from_pretrained(model, os.path.join(run_dir, "lora"))
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        model.eval()
        out_dir = os.path.join(cfg["paths"]["repo_root"], "results")
        if tag:
            out_dir = os.path.join(out_dir, tag)
            os.makedirs(out_dir, exist_ok=True)
        out_p = os.path.join(out_dir, f"ifd_{ds}.jsonl")
        n = 0
        t0 = time.time()
        with open(out_p, "w") as f:
            for k, r in enumerate(rows):
                sid = r["sample_id"]
                # L(A|Q) from a forward with labels = full sequence (label tokens)
                out = model(input_ids=r["input_ids"].unsqueeze(0).cuda(),
                            labels=r["labels"].unsqueeze(0).cuda())
                l_aq = out.loss.item()
                label, ntype = labels.get(sid, (0, "none"))
                asst = next(m["content"] for m in raw_by_id[sid]["messages"] if m["role"] == "assistant")
                l_a = response_only_loss(model, tokenizer, asst)
                f.write(json.dumps({
                    "sample_id": sid, "noise_label": label, "noise_type": ntype,
                    "L_AQ": l_aq, "L_A": l_a, "IFD": l_aq / max(l_a, 1e-9),
                }) + "\n")
                n += 1
                if k and k % 100 == 0:
                    print(f"  [{time.strftime('%H:%M:%S')}] {ds}: {k}/{len(rows)} samples "
                          f"({(time.time()-t0)/k:.2f}s/sample)", flush=True)
        print(f"  [{time.strftime('%H:%M:%S')}] {ds}: {n} samples in {time.time()-t0:.0f}s -> {out_p}")
        # a fresh 3B+LoRA is loaded per dataset; free it before the next one so
        # the loop fits on smaller cards (32GB) as well as the 96GB one
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
