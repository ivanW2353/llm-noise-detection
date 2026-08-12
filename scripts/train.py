"""LoRA SFT training with per-sample metric tracking.

For every sample (micro-batch=1) we record:
  - loss                    : per-sample CE loss (masked to assistant tokens)
  - grad_norm               : L2 norm of that sample's LoRA gradient
  - cos_sim_ref             : cosine similarity with a reference gradient
                              direction computed once, before training, on
                              held-out clean samples (LESS-style influence)
  - cos_sim_global          : cosine similarity with the accumulated gradient
                              of the current accumulation window
  - tokens                  : number of label tokens

Metrics are appended to a JSONL file and aggregates go to TensorBoard.
One run per dataset; all runs share the same seed/order for comparability.

Usage:
  python scripts/train.py --dataset clean
  python scripts/train.py --dataset garbled --smoke   # quick sanity check
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import os
import re
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MAX_LEN = 1024
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        config["paths"]["model"],
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    model.config.use_cache = False
    lora_cfg = LoraConfig(
        r=config["train"]["lora_r"],
        lora_alpha=config["train"]["lora_alpha"],
        lora_dropout=config["train"]["lora_dropout"],
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def tokenize_rows(tokenizer, rows, max_len):
    data = []
    for r in rows:
        msg = r["messages"]
        user_ids = tokenizer.apply_chat_template(
            msg[:-1], tokenize=True, return_dict=True,
            add_generation_prompt=True, truncation=True, max_length=max_len)["input_ids"]
        full = tokenizer.apply_chat_template(
            msg, tokenize=True, return_dict=True,
            truncation=True, max_length=max_len)
        input_ids = full["input_ids"]
        n_user = len(user_ids)
        labels = input_ids[:]
        labels[:n_user] = [-100] * n_user
        data.append({
            "sample_id": r["sample_id"],
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "n_label_tokens": max_len - n_user,
        })
    return data, {d["sample_id"]: d["n_label_tokens"] for d in data}


def compute_reference_direction(model, tokenizer, ref_rows, cfg, batch_size=2):
    """Mean LoRA gradient direction over held-out clean samples (before training)."""
    model.train()
    for _, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            p.grad = None
    n = len(ref_rows)
    for s in range(0, n, batch_size):
        chunk = ref_rows[s:s + batch_size]
        loss = 0.0
        for row in chunk:
            out = model(input_ids=row["input_ids"].unsqueeze(0).cuda(),
                        labels=row["labels"].unsqueeze(0).cuda())
            loss = loss + out.loss
        (loss / n).backward()
    ref = {}
    total_sq = 0.0
    for name, p in model.named_parameters():
        if "lora_" in name:
            g = p.grad.detach().float()
            ref[name] = g
            total_sq += (g ** 2).sum().item()
    norm = math.sqrt(total_sq)
    for name in ref:
        ref[name] = ref[name] / norm
    for _, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            p.grad = None
    return ref


def layer_index(name):
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else -1


def window_layer_grad_norms(model, target_ids):
    """L2 norm of accumulated window grads per target layer (GPU->CPU sync)."""
    norms = {}
    for name, p in model.named_parameters():
        if "lora_" in name and p.grad is not None:
            li = layer_index(name)
            if li in target_ids:
                norms[li] = norms.get(li, 0.0) + (p.grad.float() ** 2).sum().item()
    return {li: math.sqrt(v) for li, v in norms.items()}


@torch.no_grad()
def log_histograms(writer, model, step, target_ids):
    """LoRA weight/grad histograms for representative layers (bucketed, cheap)."""
    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue
        li = layer_index(name)
        if li not in target_ids:
            continue
        tag = name.replace("base_model.model.", "").replace(".default.weight", "")
        writer.add_histogram(f"lora_w/{tag}", p.detach().float().cpu(), step)
        if p.grad is not None:
            writer.add_histogram(f"lora_g/{tag}", p.grad.detach().float().cpu(), step)


def sample_grad_metrics(model, ref, before):
    gsq = None
    dot_ref = None
    dot_b = None
    bsq = None
    for name, p in model.named_parameters():
        if "lora_" in name and p.requires_grad:
            b = before.get(name)
            d = p.grad.detach() if b is None else (p.grad.detach() - b)
            df = d.float()
            gsq = df.pow(2).sum() if gsq is None else gsq + df.pow(2).sum()
            dot_ref = (df * ref[name]).sum() if dot_ref is None else dot_ref + (df * ref[name]).sum()
            if b is not None:
                bf = b.float()
                dot_b = (df * bf).sum() if dot_b is None else dot_b + (df * bf).sum()
                bsq = bf.pow(2).sum() if bsq is None else bsq + bf.pow(2).sum()
    return gsq, dot_ref, dot_b, bsq


@torch.no_grad()
def eval_heldout(model, tokenizer, eval_rows, bs=8):
    model.eval()
    pad = tokenizer.pad_token_id
    total = 0.0
    n = len(eval_rows)
    for s in range(0, n, bs):
        chunk = eval_rows[s:s + bs]
        maxl = max(len(r["input_ids"]) for r in chunk)
        ids = torch.full((len(chunk), maxl), pad, dtype=torch.long)
        labels = torch.full((len(chunk), maxl), -100, dtype=torch.long)
        mask = torch.zeros((len(chunk), maxl), dtype=torch.long)
        for i, r in enumerate(chunk):
            L = len(r["input_ids"])
            ids[i, :L] = r["input_ids"]
            labels[i, :L] = r["labels"]
            mask[i, :L] = 1
        out = model(input_ids=ids.cuda(), labels=labels.cuda(),
                    attention_mask=mask.cuda())
        total += out.loss.item() * len(chunk)
    model.train()
    return total / n


@torch.no_grad()
def diagnostic_pass(model, tokenizer, rows, bs=8, thresh=4.0):
    """Per-token diagnostics on a subsample: mean/max token loss and
    fraction of 'hard' tokens (loss > thresh). Forward-only, ~30s/epoch."""
    model.eval()
    pad = tokenizer.pad_token_id
    out = []
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        maxl = max(len(r["input_ids"]) for r in chunk)
        ids = torch.full((len(chunk), maxl), pad, dtype=torch.long)
        labels = torch.full((len(chunk), maxl), -100, dtype=torch.long)
        mask = torch.zeros((len(chunk), maxl), dtype=torch.long)
        for i, r in enumerate(chunk):
            L = len(r["input_ids"])
            ids[i, :L] = r["input_ids"]
            labels[i, :L] = r["labels"]
            mask[i, :L] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
        B, L, V = logits.shape
        shift = logits[:, :-1].reshape(-1, V)
        tgt = labels[:, 1:].reshape(-1).cuda()
        lm = (labels[:, 1:] != -100) * mask[:, 1:]
        ce = F.cross_entropy(shift, tgt, reduction="none").view(B, L - 1)
        for i, r in enumerate(chunk):
            tok_mask = lm[i].bool()
            if not tok_mask.any():
                continue
            toks = ce[i][tok_mask].float()
            out.append({"sample_id": r["sample_id"],
                        "mean_loss": toks.mean().item(),
                        "max_token_loss": toks.max().item(),
                        "frac_hard": (toks > thresh).float().mean().item()})
    model.train()
    return out


def train(cfg, dataset, smoke=False):
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    data_root = cfg["paths"]["data_root"]
    n_ref = cfg["train"]["ref_samples"]
    n_held = cfg["train"]["heldout_samples"]

    path = os.path.join(data_root, "data", "train", dataset, "train.jsonl")
    rows = [json.loads(l) for l in open(path)]
    heldout_rows = rows[:n_ref + n_held]
    train_rows = rows[n_ref + n_held:]
    if smoke:
        train_rows = train_rows[:64]

    run_dir = os.path.join(data_root, "runs", dataset)
    metric_dir = os.path.join(run_dir, "metrics")
    os.makedirs(metric_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    metric_f = open(os.path.join(metric_dir, "per_sample.jsonl"), "w")

    print("loading model ...")
    model = build_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["model"])

    print("tokenizing ...")
    train_data, n_label_tokens = tokenize_rows(tokenizer, train_rows, MAX_LEN)
    held_data, _ = tokenize_rows(tokenizer, heldout_rows[:n_held], MAX_LEN)
    ref_data, _ = tokenize_rows(tokenizer, heldout_rows[n_held:], MAX_LEN)

    print("computing reference gradient direction ...")
    ref_dir = compute_reference_direction(model, tokenizer, ref_data, cfg)

    tcfg = cfg["train"]
    lr = tcfg["lr"]
    n_train = len(train_data)
    steps_per_epoch = math.ceil(n_train / tcfg["grad_accum"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    warmup_steps = int(total_steps * tcfg["warmup_ratio"])
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=tcfg["weight_decay"], betas=(0.9, 0.999))

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    global_step = 0
    t0 = time.time()
    tb_t0 = t0
    tb_sum = {"loss": 0.0, "grad_norm": 0.0, "cos_ref": 0.0, "cos_global": 0.0,
              "tokens": 0.0, "cnt": 0}
    n_layers = model.config.num_hidden_layers
    target_ids = {0, n_layers // 2, n_layers - 1}
    layer_norm_sum = {li: 0.0 for li in target_ids}
    layer_norm_cnt = 0

    def flush_window(acc, epoch, step):
        """Materialize per-sample metrics (one sync batch per window) and log."""
        losses = torch.stack([a[1] for a in acc])
        gsqs = torch.stack([a[2] for a in acc])
        dots_ref = torch.stack([a[3] for a in acc])
        gs = gsqs.sqrt()
        cos_refs = (dots_ref / gs).tolist()
        cos_globs = [None] * len(acc)
        if any(a[4] is not None for a in acc):
            had_b = [a[4] is not None for a in acc]
            dots_b = torch.stack([a[4] if a[4] is not None else torch.zeros((), device=gs.device) for a in acc])
            bsqs = torch.stack([a[5] if a[5] is not None else torch.ones((), device=gs.device) for a in acc])
            raw = (dots_b / (gs * bsqs.sqrt())).tolist()
            cos_globs = [raw[i] if had_b[i] else None for i in range(len(acc))]
        for a, l, gn, cr, cg in zip(acc, losses.tolist(), gs.tolist(), cos_refs, cos_globs):
            sid = a[0]
            metric_f.write(json.dumps({
                "step": step, "epoch": epoch, "sample_id": sid,
                "loss": l, "grad_norm": gn, "cos_sim_ref": cr,
                "cos_sim_global": cg,
                "tokens": max(1, n_label_tokens[sid]),
            }) + "\n")
            tb_sum["loss"] += l
            tb_sum["grad_norm"] += gn
            if cr is not None:
                tb_sum["cos_ref"] += cr
            if cg is not None:
                tb_sum["cos_global"] += cg
            tb_sum["tokens"] += max(1, n_label_tokens[sid])
            tb_sum["cnt"] += 1
        metric_f.flush()

    for epoch in range(tcfg["epochs"]):
        acc = []          # (sample_id, loss, gsq, dot_ref, dot_b, bsq) - GPU tensors
        for i, row in enumerate(train_data):
            if smoke and global_step >= 8:
                break
            input_ids = row["input_ids"].unsqueeze(0).cuda()
            labels = row["labels"].unsqueeze(0).cuda()
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss

            before = {}
            for name, p in model.named_parameters():
                if "lora_" in name and p.requires_grad and p.grad is not None:
                    before[name] = p.grad.detach().clone()

            loss.backward()

            gsq, dot_ref, dot_b, bsq = sample_grad_metrics(model, ref_dir, before)
            acc.append((row["sample_id"], loss, gsq, dot_ref, dot_b, bsq))

            # optimizer step at window boundary
            if (i + 1) % tcfg["grad_accum"] == 0:
                for li, v in window_layer_grad_norms(model, target_ids).items():
                    layer_norm_sum[li] += v
                layer_norm_cnt += 1
                if (global_step + 1) % tcfg["eval_steps"] == 0:
                    log_histograms(writer, model, global_step + 1, target_ids)
                for g in opt.param_groups:
                    g["lr"] = lr_at(global_step)
                opt.step()
                opt.zero_grad()
                flush_window(acc, epoch, global_step)
                acc = []
                global_step += 1
                if global_step % 50 == 0:
                    now = time.time()
                    for key in ("loss", "grad_norm", "cos_ref", "cos_global"):
                        writer.add_scalar(f"train/{key}", tb_sum[key] / max(1, tb_sum["cnt"]), global_step)
                    writer.add_scalar("train/lr", lr_at(global_step), global_step)
                    writer.add_scalar("train/tokens_per_sec", tb_sum["tokens"] / max(1.0, now - tb_t0), global_step)
                    writer.add_scalar("train/gpu_mem_GB", torch.cuda.memory_allocated() / 1e9, global_step)
                    if layer_norm_cnt:
                        for li in target_ids:
                            writer.add_scalar(f"lora_layer_gradnorm/layer{li}",
                                              layer_norm_sum[li] / layer_norm_cnt, global_step)
                    writer.flush()
                    tb_t0 = now
                    tb_sum = {"loss": 0.0, "grad_norm": 0.0, "cos_ref": 0.0, "cos_global": 0.0,
                              "tokens": 0.0, "cnt": 0}
                    layer_norm_sum = {li: 0.0 for li in target_ids}
                    layer_norm_cnt = 0
                    print(f"[{dataset}] epoch {epoch} step {global_step}/{total_steps} "
                          f"lr {lr_at(global_step):.2e} elapsed {time.time()-t0:.0f}s", flush=True)
                if global_step % tcfg["eval_steps"] == 0:
                    el = eval_heldout(model, tokenizer, held_data)
                    writer.add_scalar("eval/heldout_loss", el, global_step)
                    writer.flush()
                    print(f"  heldout eval loss: {el:.4f}", flush=True)

        # flush remaining window
        if acc:
            for g in opt.param_groups:
                g["lr"] = lr_at(global_step)
            opt.step()
            opt.zero_grad()
            flush_window(acc, epoch, global_step)
            acc = []
            global_step += 1

        # epoch-end diagnostic pass on a subsample (max/hard-token loss)
        diag_step = max(1, tcfg.get("diag_subsample", 8))
        diag_rows = train_data[::diag_step]
        diag = diagnostic_pass(model, tokenizer, diag_rows,
                               thresh=tcfg.get("hard_threshold", 4.0))
        with open(os.path.join(metric_dir, f"diag_epoch{epoch}.jsonl"), "w") as f:
            for d in diag:
                f.write(json.dumps(d) + "\n")
        if diag:
            writer.add_scalar("diag/max_token_loss_mean",
                              sum(d["max_token_loss"] for d in diag) / len(diag), global_step)
            writer.add_scalar("diag/frac_hard_mean",
                              sum(d["frac_hard"] for d in diag) / len(diag), global_step)
            writer.flush()
        print(f"  epoch {epoch} diagnostic pass: {len(diag)} samples", flush=True)

        if smoke:
            break

    metric_f.close()
    model.eval()
    model.save_pretrained(os.path.join(run_dir, "lora"))
    tokenizer.save_pretrained(os.path.join(run_dir, "lora"))
    writer.close()
    print(f"done {dataset}: {global_step} steps in {time.time()-t0:.0f}s -> {run_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/noisedetect/config.yaml")
    ap.add_argument("--dataset", required=True,
                    choices=["clean", "garbled", "duplicate", "unrelated", "keyword", "mixed"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    train(cfg, args.dataset, smoke=args.smoke)
