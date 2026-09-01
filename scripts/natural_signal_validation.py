"""Natural-data signal validation (reproduces dynanoise Phase 6).

On lmsys-chat-1m (cached locally), use a trained model (default: clean run)
to compute per-sample loss dynamics signals and verify their internal
direction against the model's predicted difficulty:

  token_loss_top20  vs  loss_mu   -> expect negative Spearman (like -0.78)
  loss_cv           vs  loss_mu   -> direction from dynanoise (-0.20)

Signals are computed post-hoc on a sample (no training needed). Needs GPU.

Usage:
  python scripts/natural_signal_validation.py --n 20000 --model clean
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import MAX_LEN

TOKEN_TOP_K = 0.2


def load_model(cfg, dataset):
    model = AutoModelForCausalLM.from_pretrained(
        cfg["paths"]["model"], dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map={"": 0})
    if dataset != "base":
        tag = cfg["paths"].get("experiment_tag", "")
        lora_path = os.path.join(cfg["paths"]["data_root"], "runs", tag, dataset, "lora")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model


@torch.no_grad()
def signals_for_prompt(model, tokenizer, prompts, max_new=64):
    """Per-sample signals from an autoregressive rollout.

    Uses generate(..., output_scores=True) so per-token CE comes from the
    generation's own scores — NO second forward pass over the full sequence
    (~40% faster than the original implementation).

    loss_mu: mean next-token loss over the GENERATED continuation
    loss_cv: std/mean of per-token losses
    token_loss_top20: fraction of total loss carried by the hardest 20% tokens
    """
    tokenizer.padding_side = "left"
    texts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts]
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LEN - max_new)
    gen_out = model.generate(
        input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda(),
        max_new_tokens=max_new, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True, output_scores=True)
    gen = gen_out.sequences
    start = enc["input_ids"].shape[1]
    n_gen = min(max_new, gen.shape[1] - start, len(gen_out.scores))
    ce_cols = []
    for j in range(n_gen):
        s = gen_out.scores[j]                      # [B, V]
        tgt = gen[:, start + j]                    # [B]
        ce_cols.append(-torch.log_softmax(s, dim=-1)
                       .gather(1, tgt.unsqueeze(1)).squeeze(1))
    ce = torch.stack(ce_cols, dim=1)               # [B, n_gen]
    gen_part = gen[:, start:start + n_gen]
    mask = (gen_part != tokenizer.pad_token_id) & (gen_part != tokenizer.eos_token_id)
    res = []
    for i in range(gen.shape[0]):
        toks = ce[i][mask[i]].float().cpu().numpy()
        if len(toks) < 8:
            continue
        mu = float(toks.mean())
        cv = float(toks.std() / mu)
        n_top = max(1, int(round(len(toks) * TOKEN_TOP_K)))
        top20 = float(np.sort(toks)[-n_top:].sum() / toks.sum())
        res.append((mu, cv, top20))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"))
    ap.add_argument("--model", default="clean")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["model"])
    model = load_model(cfg, args.model)

    ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
    prompts, convs = [], []
    for r in ds.select(range(args.n)):
        c = r["conversation"]
        if len(c) < 1:
            continue
        u = c[0]["content"]
        if u and len(u) > 20:
            prompts.append(u)
            convs.append(c)
    print(f"scored {len(prompts)} prompts ...", flush=True)
    mus, cvs, tops = [], [], []
    for s in range(0, len(prompts), 8):
        for mu, cv, t20 in signals_for_prompt(model, tokenizer, prompts[s:s + 8],
                                              max_new=args.max_new):
            mus.append(mu); cvs.append(cv); tops.append(t20)
        if s and s % (8 * 100) == 0:
            print(f"  ... {s}/{len(prompts)} prompts done", flush=True)
    print(f"valid samples: {len(mus)}")
    for a, b, na, nb in [(tops, mus, "token_top20", "loss_mu"),
                         (cvs, mus, "loss_cv", "loss_mu"),
                         (tops, cvs, "token_top20", "loss_cv")]:
        r, p = spearmanr(a, b)
        print(f"Spearman({na}, {nb}) = {r:+.3f}  (p={p:.1e})")


if __name__ == "__main__":
    main()
