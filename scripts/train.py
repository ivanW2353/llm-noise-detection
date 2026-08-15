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
            add_generation_prompt=True)["input_ids"]
        full = tokenizer.apply_chat_template(msg, tokenize=True, return_dict=True)
        input_ids = full["input_ids"]
        n_user = len(user_ids)
        if len(input_ids) > max_len:
            # keep assistant response intact; truncate the USER prefix instead
            assistant_ids = input_ids[n_user:]
            if len(assistant_ids) >= max_len:
                input_ids, n_user = assistant_ids[:max_len], 0
            else:
                keep_user = max_len - len(assistant_ids)
                input_ids, n_user = user_ids[-keep_user:] + assistant_ids, keep_user
        labels = input_ids[:]
        labels[:n_user] = [-100] * n_user
        data.append({
            "sample_id": r["sample_id"],
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "n_label_tokens": len(input_ids) - n_user,
        })
    return data, {d["sample_id"]: d["n_label_tokens"] for d in data}


def fill_flat(params, offsets, buf):
    """Copy all LoRA grads into a preallocated float32 buffer (no per-step
    allocation churn). Returns buf, or None if no grads exist."""
    any_grad = False
    for (s, e), (_, p) in zip(offsets, params):
        g = p.grad
        if g is None:
            continue
        any_grad = True
        buf[s:e].copy_(g.reshape(-1))
    return buf if any_grad else None


def lora_params(model):
    return [(name, p) for name, p in model.named_parameters() if "lora_" in name]


def lora_offsets(params):
    offsets, total = [], 0
    for _, p in params:
        offsets.append((total, total + p.numel()))
        total += p.numel()
    return offsets, total


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
    params = lora_params(model)
    offsets, total = lora_offsets(params)
    flat = torch.zeros(total, device="cuda", dtype=torch.float32)
    fill_flat(params, offsets, flat)
    norm = torch.linalg.vector_norm(flat)
    ref_buf = flat / norm
    torch.cuda.empty_cache()
    for _, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            p.grad = None
    return params, offsets, total, ref_buf


def layer_index(name):
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else -1


def window_layer_grad_norms(model):
    """L2 norm of the accumulated window gradients per layer (all layers)."""
    norms = {}
    for name, p in model.named_parameters():
        if "lora_" in name and p.grad is not None:
            li = layer_index(name)
            if li >= 0:
                norms[li] = norms.get(li, 0.0) + (p.grad.float() ** 2).sum().item()
    return {li: math.sqrt(v) for li, v in norms.items()}


def window_layer_weight_norms(model):
    """L2 norm of LoRA weights per layer (all layers, after optimizer step)."""
    norms = {}
    for name, p in model.named_parameters():
        if "lora_" in name and p.requires_grad:
            li = layer_index(name)
            if li >= 0:
                norms[li] = norms.get(li, 0.0) + (p.detach().float() ** 2).sum().item()
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
def diagnostic_pass(model, tokenizer, rows, bs=8, thresh=4.0, top_k=32):
    """Per-token diagnostics on a subsample. Returns per-sample aggregates plus
    top-k hardest label tokens for token-level analysis:
      user_loss      : mean CE over the USER (prompt) tokens
      entropy        : mean next-token entropy over label tokens
      skew / kurt    : shape of the per-token loss distribution
      top_tokens     : [[pos, token_id, loss], ...] for the hardest tokens
    Forward-only, ~1 min/epoch."""
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
        # full-sequence next-token CE with REAL targets: labels==-100 positions
        # would otherwise come back as 0.0 from cross_entropy (ignore_index)
        ce = F.cross_entropy(shift, ids[:, 1:].reshape(-1).cuda(),
                             reduction="none").view(B, L - 1)
        att = mask[:, 1:]
        label_mask = (labels[:, 1:] != -100) * att
        user_mask = (labels[:, 1:] == -100) * att
        # next-token entropy over label tokens only (gather keeps it cheap)
        flat_lm = label_mask.reshape(-1).bool()
        ent_all = None
        if flat_lm.any():
            lse = torch.log_softmax(shift[flat_lm], dim=-1)
            ent_flat = torch.zeros(B * (L - 1), device=ce.device)
            ent_flat[flat_lm] = (-(torch.exp(lse) * lse).sum(-1)).float()
            ent_all = ent_flat.view(B, L - 1)
        shift_ids = ids[:, 1:]
        for i, r in enumerate(chunk):
            lm = label_mask[i].bool()
            if not lm.any():
                continue
            toks = ce[i][lm].float()
            pos = lm.nonzero().flatten()
            agg = {"sample_id": r["sample_id"],
                   "mean_loss": toks.mean().item(),
                   "max_token_loss": toks.max().item(),
                   "frac_hard": (toks > thresh).float().mean().item(),
                   "user_loss": None, "entropy": None,
                   "token_loss_skew": None, "token_loss_kurt": None,
                   "top_tokens": []}
            um = user_mask[i].bool()
            if um.any():
                agg["user_loss"] = ce[i][um].float().mean().item()
            if ent_all is not None:
                ent = ent_all[i][lm].float()
                agg["entropy"] = ent.mean().item()
            toks_np = toks.cpu().numpy()
            if len(toks_np) > 3:
                from scipy.stats import skew, kurtosis
                agg["token_loss_skew"] = float(skew(toks_np))
                agg["token_loss_kurt"] = float(kurtosis(toks_np))
            k = min(top_k, len(toks))
            v, idx = toks.topk(k)
            for val, ix in zip(v.tolist(), idx.tolist()):
                agg["top_tokens"].append([int(pos[ix].item()),
                                          int(shift_ids[i, pos[ix]].item()),
                                          float(val)])
            out.append(agg)
    model.train()
    return out


def train(cfg, dataset, smoke=False):
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    data_root = cfg["paths"]["data_root"]
    n_ref = cfg["train"]["ref_samples"]
    n_held = cfg["train"]["heldout_samples"]

    path = os.path.join(data_root, "data", cfg["paths"].get("experiment_tag", ""),
                        dataset, "train.jsonl")
    hold_path = os.path.join(data_root, "data", cfg["paths"].get("experiment_tag", ""),
                             "heldout.jsonl")
    rows = [json.loads(l) for l in open(path)]
    heldout_rows = [json.loads(l) for l in open(hold_path)]
    train_rows = rows
    if smoke:
        train_rows = train_rows[:64]

    tag = cfg["paths"].get("experiment_tag", "")
    # smoke runs NEVER touch the real run directories
    run_dir = os.path.join(data_root, "runs", tag, "_smoke" if smoke else "", dataset)
    metric_dir = os.path.join(run_dir, "metrics")
    os.makedirs(metric_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    metric_f = open(os.path.join(metric_dir, "per_sample.jsonl"), "w")
    ln_f = open(os.path.join(metric_dir, "layer_norms.jsonl"), "w")

    print("loading model ...")
    model = build_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["model"])

    print("tokenizing ...")
    train_data, n_label_tokens = tokenize_rows(tokenizer, train_rows, MAX_LEN)
    held_data, _ = tokenize_rows(tokenizer, heldout_rows[:n_held], MAX_LEN)
    ref_data, _ = tokenize_rows(tokenizer, heldout_rows[n_held:], MAX_LEN)

    print("computing reference gradient direction ...")
    params, offsets, n_lora, ref_buf = compute_reference_direction(model, tokenizer, ref_data, cfg)
    # update_contrib uses B-only offsets: lora_A gradients are ~0 while B is
    # still zero-initialized, so their Adam-normalized values would explode
    b_offsets = [(s, e) for (s, e), (name, _) in zip(offsets, params) if "lora_B" in name]
    total_b = sum(e - s for s, e in b_offsets)
    buf_before = torch.zeros(n_lora, device="cuda", dtype=torch.float32)
    buf_after = torch.zeros(n_lora, device="cuda", dtype=torch.float32)
    delta_buf = torch.zeros(n_lora, device="cuda", dtype=torch.float32)
    delta_b = torch.zeros(total_b, device="cuda", dtype=torch.float32)

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
              "update_contrib": 0.0, "tokens": 0.0, "cnt": 0}
    n_layers = model.config.num_hidden_layers
    target_ids = {0, n_layers // 2, n_layers - 1}
    layer_norm_sum = {li: 0.0 for li in target_ids}
    layer_norm_cnt = 0
    epoch_stats = []
    log_every = max(1, tcfg.get("log_every", 25))
    v_buf = torch.zeros(total_b, device="cuda", dtype=torch.float32)
    v_ready = False

    def flush_window(acc, epoch, step):
        """Materialize per-sample metrics (one sync batch per window) and log."""
        losses = torch.stack([a[1] for a in acc])
        gns = torch.stack([a[2] for a in acc])
        dots_ref = torch.stack([a[3] for a in acc])
        ups = torch.stack([a[6] if a[6] is not None else torch.zeros((), device=gns.device) for a in acc])
        had_up = [a[6] is not None for a in acc]
        cos_refs = (dots_ref / gns.clamp_min(1e-12)).tolist()
        cos_globs = [None] * len(acc)
        if any(a[4] is not None for a in acc):
            had_b = [a[4] is not None for a in acc]
            dots_b = torch.stack([a[4] if a[4] is not None else torch.zeros((), device=gns.device) for a in acc])
            bsqs = torch.stack([a[5] if a[5] is not None else torch.ones((), device=gns.device) for a in acc])
            raw = (dots_b / (gns * bsqs.sqrt()).clamp_min(1e-12)).tolist()
            cos_globs = [raw[i] if had_b[i] else None for i in range(len(acc))]
        for a, l, gn, cr, cg, up in zip(acc, losses.tolist(), gns.tolist(),
                                        cos_refs, cos_globs, ups.tolist()):
            sid = a[0]
            metric_f.write(json.dumps({
                "step": step, "epoch": epoch, "sample_id": sid,
                "loss": l, "grad_norm": gn, "cos_sim_ref": cr,
                "cos_sim_global": cg, "update_contrib": up,
                "tokens": max(1, n_label_tokens[sid]),
            }) + "\n")
            if l == l and gn == gn:  # skip NaN rows in aggregates
                tb_sum["loss"] += l
                tb_sum["grad_norm"] += gn
                if cr is not None and cr == cr:
                    tb_sum["cos_ref"] += cr
                if cg is not None and cg == cg:
                    tb_sum["cos_global"] += cg
                if up == up:
                    tb_sum["update_contrib"] += up
                tb_sum["tokens"] += max(1, n_label_tokens[sid])
                tb_sum["cnt"] += 1
        metric_f.flush()

    for epoch in range(tcfg["epochs"]):
        ep_t0 = time.time()
        acc = []          # (sample_id, loss, grad_norm, dot_ref, dot_b, bsq) - GPU scalars
        for i, row in enumerate(train_data):
            if smoke and global_step >= 8:
                break
            if row["n_label_tokens"] == 0:
                continue
            input_ids = row["input_ids"].unsqueeze(0).cuda()
            labels = row["labels"].unsqueeze(0).cuda()
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss

            # flat-vector metric capture: preallocated buffers, no allocation churn
            has_before = fill_flat(params, offsets, buf_before) is not None
            loss.backward()
            fill_flat(params, offsets, buf_after)
            if has_before:
                delta_buf.copy_(buf_after).sub_(buf_before)
                bsq = torch.dot(buf_before, buf_before)
                dot_b = torch.dot(delta_buf, buf_before)
            else:
                delta_buf.copy_(buf_after)
                bsq = dot_b = None
            g_norm = torch.linalg.vector_norm(delta_buf)
            dot_ref = torch.dot(delta_buf, ref_buf)
            cos_ref = dot_ref / g_norm
            # Adam-normalized update contribution over B params: ||grad/sqrt(v)||
            # (None for the first window, before any optimizer step)
            upd = None
            if v_ready:
                o = 0
                for (s, e) in b_offsets:
                    delta_b[o:o + e - s].copy_(delta_buf[s:e])
                    o += e - s
                # sample gradient norm relative to the running RMS gradient
                upd = torch.linalg.vector_norm(delta_b) / (
                    torch.linalg.vector_norm(v_buf.sqrt()) + 1e-8)
            acc.append((row["sample_id"], loss, g_norm, dot_ref, dot_b, bsq, upd))

            # optimizer step at window boundary
            if (i + 1) % tcfg["grad_accum"] == 0:
                all_norms = window_layer_grad_norms(model)
                for li in target_ids:
                    layer_norm_sum[li] += all_norms.get(li, 0.0)
                layer_norm_cnt += 1
                if (global_step + 1) % tcfg["eval_steps"] == 0:
                    log_histograms(writer, model, global_step + 1, target_ids)
                for g in opt.param_groups:
                    g["lr"] = lr_at(global_step)
                opt.step()
                # per-layer LoRA grad+weight norms for ALL layers (window-level)
                ln_f.write(json.dumps({"step": global_step + 1,
                                       "grad": all_norms,
                                       "weight": window_layer_weight_norms(model)}) + "\n")
                ln_f.flush()
                # snapshot Adam second-moment (B params) for update_contrib
                o = 0
                for (s, e), (name, p) in zip(offsets, params):
                    if "lora_B" not in name:
                        continue
                    st = opt.state.get(p)
                    if st and "exp_avg_sq" in st:
                        v_buf[o:o + e - s].copy_(st["exp_avg_sq"].reshape(-1))
                    o += e - s
                v_ready = True
                opt.zero_grad()
                flush_window(acc, epoch, global_step)
                acc = []
                global_step += 1
                if global_step % 50 == 0:
                    now = time.time()
                    cnt = max(1, tb_sum["cnt"])
                    for key in ("loss", "grad_norm", "cos_ref", "cos_global", "update_contrib"):
                        writer.add_scalar(f"train/{key}", tb_sum[key] / cnt, global_step)
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
                              "update_contrib": 0.0, "tokens": 0.0, "cnt": 0}
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

        # epoch summary from the full per-sample log
        ep_metrics = []
        with open(os.path.join(metric_dir, "per_sample.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r["epoch"] == epoch and r["loss"] is not None:
                    ep_metrics.append(r)
        n_ep = len(ep_metrics)
        if n_ep:
            def avg(key):
                vals = [r[key] for r in ep_metrics if r.get(key) is not None and r[key] == r[key]]
                return sum(vals) / len(vals) if vals else None
            loss_min = min(r["loss"] for r in ep_metrics)
            loss_max = max(r["loss"] for r in ep_metrics)
            print(f"  == epoch {epoch} summary == n={n_ep} "
                  f"loss mean {avg('loss'):.4f} [min {loss_min:.4f}, max {loss_max:.4f}] "
                  f"grad_norm {avg('grad_norm'):.3f} "
                  f"cos_ref {avg('cos_sim_ref'):.4f} cos_global {avg('cos_sim_global'):.4f} "
                  f"elapsed {time.time()-ep_t0:.0f}s", flush=True)
            epoch_stats.append({
                "epoch": epoch, "n": n_ep,
                "loss_mean": avg("loss"), "loss_min": loss_min, "loss_max": loss_max,
                "grad_norm_mean": avg("grad_norm"),
                "cos_ref_mean": avg("cos_sim_ref"), "cos_global_mean": avg("cos_sim_global"),
                "seconds": round(time.time() - ep_t0, 1),
            })

        # epoch-end diagnostic pass on a subsample (max/hard-token loss)
        diag_step = max(1, tcfg.get("diag_subsample", 8))
        diag_rows = train_data[::diag_step]
        diag = diagnostic_pass(model, tokenizer, diag_rows,
                               thresh=tcfg.get("hard_threshold", 4.0))
        with open(os.path.join(metric_dir, f"diag_epoch{epoch}.jsonl"), "w") as f:
            for d in diag:
                f.write(json.dumps({k: v for k, v in d.items() if k != "top_tokens"}) + "\n")
        with open(os.path.join(metric_dir, f"token_diag_epoch{epoch}.jsonl"), "w") as f:
            for d in diag:
                f.write(json.dumps({"sample_id": d["sample_id"],
                                    "top_tokens": d["top_tokens"]}) + "\n")
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
    ln_f.close()
    model.eval()
    model.save_pretrained(os.path.join(run_dir, "lora"))
    tokenizer.save_pretrained(os.path.join(run_dir, "lora"))
    writer.close()
    summary = {
        "dataset": dataset, "epochs": tcfg["epochs"], "total_steps": global_step,
        "n_train": n_train, "seconds": round(time.time() - t0, 1),
        "lora_params": n_lora, "epochs_detail": epoch_stats,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done {dataset}: {global_step} steps in {time.time()-t0:.0f}s -> {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/noisedetect/config.yaml")
    ap.add_argument("--dataset", required=True,
                    choices=["clean", "garbled", "duplicate", "unrelated", "keyword",
                             "template", "truncation", "near_duplicate", "mixed"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None, help="experiment tag (run dir suffix)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.tag:
        cfg["paths"]["experiment_tag"] = args.tag
    train(cfg, args.dataset, smoke=args.smoke)
