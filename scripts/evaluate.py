"""Few-shot evaluation of fine-tuned models on common validation sets.

Tasks (loglikelihood multiple-choice or CoT generation):
  mmlu (5-shot, per-subject), hellaswag (5-shot), arc (25-shot), winogrande (5-shot),
  truthfulqa (0-shot), gsm8k (5-shot CoT), bbh (3-shot CoT, 20/task)

Full results (incl. per-subject / per-task breakdowns) are saved after every
task, so evaluation is resumable; re-running skips completed models/tasks.

Usage:
  python scripts/evaluate.py --dataset clean
  python scripts/evaluate.py --dataset base          # base model, no LoRA
  python scripts/evaluate.py --dataset clean --tasks mmlu,gsm8k
  python scripts/evaluate.py --dataset clean --force # redo even if cached
  python scripts/evaluate.py --dataset clean --smoke # tiny sanity check
"""

import argparse
import glob
import json
import os
import re
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BBH_DIR = "/root/autodl-tmp/noisedetect/data/bbh"
MAX_LEN = 2048          # generation room (long CoT few-shot + output)
SCORE_MAX_LEN = 1024    # MC scoring cap: [B, L, V] logits are memory-heavy


# ----------------------------------------------------------------------------
# model helpers
# ----------------------------------------------------------------------------
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
def score_options(model, tokenizer, samples, bs=48):
    """samples: list of (prompt, options); returns per-sample nll list."""
    flat = [(p, " " + o) for p, opts in samples for o in opts]
    all_nll = []
    t0 = time.time()
    for s in range(0, len(flat), bs):
        chunk = flat[s:s + bs]
        if s % (bs * 500) == 0:
            rate = s / max(1, time.time() - t0)
            print(f"    [{time.strftime('%H:%M:%S')}] ... {s}/{len(flat)} options "
                  f"({rate:.0f} opts/s, ETA {max(0, (len(flat)-s)/max(rate, 1e-9))/60:.1f} min)", flush=True)
        pids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p, _ in chunk]
        cids = [tokenizer(c, add_special_tokens=False)["input_ids"] for _, c in chunk]
        maxl = min(SCORE_MAX_LEN, max(len(p) + len(c) for p, c in zip(pids, cids)))
        for i in range(len(pids)):  # keep tail (query + Answer:) within MAX_LEN
            if len(pids[i]) + len(cids[i]) > maxl:
                keep = max(1, maxl - len(cids[i]))
                pids[i] = pids[i][len(pids[i]) - keep:]
        ids = torch.full((len(chunk), maxl), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(chunk), maxl), -100, dtype=torch.long)
        mask = torch.zeros((len(chunk), maxl), dtype=torch.long)
        for i, (p, c) in enumerate(zip(pids, cids)):
            ids[i, :len(p)] = torch.tensor(p)
            labels[i, len(p):len(p) + len(c)] = torch.tensor(c)
            mask[i, :len(p) + len(c)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
        all_nll.extend(_per_row_nll(logits, labels.cuda(), mask.cuda()))
    nlls, idx = [], 0
    for _, opts in samples:
        nlls.append(all_nll[idx:idx + len(opts)])
        idx += len(opts)
    return nlls


def _per_row_nll(logits, labels, mask):
    B, L, V = logits.shape
    shift = logits[:, :-1].reshape(-1, V)
    tgt = labels[:, 1:].reshape(-1)
    m = (labels[:, 1:] != -100) * mask[:, 1:]
    ce = torch.nn.functional.cross_entropy(shift, tgt, reduction="none").view(B, L - 1)
    return (ce * m.float()).sum(1) / m.float().sum(1).clamp(min=1)


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens=256, bs=32):
    model.config.use_cache = True
    tokenizer.padding_side = "left"  # flash-attn generation is broken with right padding
    outs = []
    t0 = time.time()
    for s in range(0, len(prompts), bs):
        if s % (bs * 50) == 0:
            rate = s / max(1, time.time() - t0)
            print(f"    [{time.strftime('%H:%M:%S')}] ... generated {s}/{len(prompts)} "
                  f"({rate:.1f} prompts/s, ETA {max(0, (len(prompts)-s)/max(rate, 1e-9))/60:.1f} min)", flush=True)
        enc = tokenizer(prompts[s:s + bs], return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN - max_new_tokens)
        gen = model.generate(
            input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda(),
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gen = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    model.config.use_cache = False
    return outs


# ----------------------------------------------------------------------------
# task loaders -> (samples, answers, shot_desc, n)
#   MC tasks: samples = [(prompt, options)], answers = [correct_idx, ...]
#   gen tasks: samples = [prompt, ...], answers = [correct_str, ...]
# ----------------------------------------------------------------------------
def load_mmlu():
    dev = load_dataset("cais/mmlu", "all", split="dev")
    test = load_dataset("cais/mmlu", "all", split="test")
    by_subj = {}
    for r in dev:
        by_subj.setdefault(r["subject"], []).append(r)
    samples, answers, subjects = [], [], []
    for r in test:
        shots = by_subj[r["subject"]][:5]
        shot_txt = "".join(
            f"{s['question']}\nA. {s['choices'][0]}\nB. {s['choices'][1]}\n"
            f"C. {s['choices'][2]}\nD. {s['choices'][3]}\nAnswer: {chr(65 + s['answer'])}\n\n"
            for s in shots)
        prompt = (f"The following are multiple choice questions (with answers) "
                  f"about {r['subject']}.\n\n{shot_txt}{r['question']}\n"
                  f"A. {r['choices'][0]}\nB. {r['choices'][1]}\nC. {r['choices'][2]}\n"
                  f"D. {r['choices'][3]}\nAnswer:")
        samples.append((prompt, list("ABCD")))
        answers.append(r["answer"])
        subjects.append(r["subject"])
    return samples, answers, "5-shot", len(test), subjects


def load_hellaswag():
    tr = load_dataset("Rowan/hellaswag", split="train")
    val = load_dataset("Rowan/hellaswag", split="validation")
    shots = [f"{r['ctx']} {r['endings'][int(r['label'])]}" for r in tr.select(range(5))]
    shot_txt = "".join(s + "\n\n" for s in shots)
    samples, answers, activities = [], [], []
    for r in val:
        samples.append((shot_txt + r["ctx"], r["endings"]))
        answers.append(int(r["label"]))
        activities.append(r["activity_label"])
    return samples, answers, "5-shot", len(val), activities


def load_arc():
    tr = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    shot_txt = "".join(
        f"{s['question']}\n" + "".join(f"{l}. {t}\n" for t, l in zip(s["choices"]["text"], s["choices"]["label"]))
        + f"Answer: {s['answerKey']}\n\n" for s in tr.select(range(25)))
    samples, answers = [], []
    for r in test:
        prompt = f"{shot_txt}{r['question']}\n" + "".join(
            f"{l}. {t}\n" for t, l in zip(r["choices"]["text"], r["choices"]["label"])) + "Answer:"
        samples.append((prompt, r["choices"]["label"]))
        answers.append(r["choices"]["label"].index(r["answerKey"]))
    return samples, answers, "25-shot", len(test)


def load_winogrande():
    tr = load_dataset("allenai/winogrande", "winogrande_debiased", split="train")
    val = load_dataset("allenai/winogrande", "winogrande_debiased", split="validation")
    shot_txt = "".join(
        f"{s['sentence']} {s['option1']}\nAnswer: {s['option1'] if s['answer'] == '1' else s['option2']}\n\n"
        for s in tr.select(range(5)))
    samples, answers = [], []
    for r in val:
        samples.append((shot_txt + r["sentence"], [r["option1"], r["option2"]]))
        answers.append(int(r["answer"]) - 1)
    return samples, answers, "5-shot", len(val)


def load_truthfulqa():
    d = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    g = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    cat_map = {r["question"]: r["category"] for r in g}
    samples, answers, categories = [], [], []
    for r in d:
        mc = r["mc1_targets"]
        samples.append((r["question"] + "\nAnswer:", mc["choices"]))
        answers.append(mc["labels"].index(1))
        categories.append(cat_map.get(r["question"], "unknown"))
    return samples, answers, "0-shot", len(d), categories


def load_gsm8k():
    tr = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    shot_txt = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in tr.select(range(5)))
    samples, answers = [], []
    for r in test:
        samples.append(shot_txt + f"Question: {r['question']}\nAnswer:")
        answers.append(normalize_num(r["answer"].split("#### ")[1]))
    return samples, answers, "5-shot CoT", len(test)


def normalize_num(s):
    s = re.sub(r"[,$\s]", "", s).rstrip(".")
    return s


def parse_gsm8k(text):
    m = re.search(r"####\s*(-?\d+[\d,.]*)", text)
    if not m:
        m = re.search(r"\\boxed\{(-?\d+[\d,.]*)\}", text)
    if m:
        return normalize_num(m.group(1))
    nums = re.findall(r"-?\d+[\d,.]*", text)
    return normalize_num(nums[-1]) if nums else None


def chat_wrap(tokenizer, prompts):
    return [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True)
            for p in prompts]


def load_bbh(max_per_task=20):
    task_files = sorted(glob.glob(os.path.join(BBH_DIR, "test", "*.json")))
    samples, prompts, answers, task_names = [], [], [], []
    for f in task_files:
        task = os.path.basename(f).replace(".json", "")
        d = json.load(open(f))
        cot = open(os.path.join(BBH_DIR, "cot-prompts", f"{task}.txt")).read()
        ex = d["examples"]
        shot_txt = "".join(
            f"Q: {s['input']}\nA: Let's think step by step. {cot}\nSo the answer is {s['target']}.\n\n"
            for s in ex[:3])
        for s in ex[3:3 + max_per_task]:
            prompt = f"{shot_txt}Q: {s['input']}\nA: Let's think step by step."
            samples.append(prompt)
            answers.append(s["target"].lower())
            task_names.append(task)
            prompts.append(prompt)
    return samples, answers, task_names, len(task_files)


def parse_bbh(text):
    m = re.findall(r"the answer is ([A-Za-z0-9][^\n.]*)", text.lower())
    if m:
        return m[-1].strip()
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-1].strip().lower() if lines else ""


TASKS = {
    "mmlu": load_mmlu, "hellaswag": load_hellaswag, "arc": load_arc,
    "winogrande": load_winogrande, "truthfulqa": load_truthfulqa,
    "gsm8k": load_gsm8k, "bbh": load_bbh,
}


# ----------------------------------------------------------------------------
def run_task(model, tokenizer, task, smoke=False):
    if task == "gsm8k":
        samples, answers, _, n = load_gsm8k()
        if smoke:
            samples, answers, n = samples[:50], answers[:50], 50
        gens = generate(model, tokenizer, chat_wrap(tokenizer, samples), max_new_tokens=512)
        preds = [parse_gsm8k(t) for t in gens]
        raw = [{"qid": i, "correct": 1 if p == a else 0, "answer": a,
                "pred": p, "gen_len": len(t.split())} for i, (p, a, t) in enumerate(zip(preds, answers, gens))]
        acc = sum(1 for p, a in zip(preds, answers) if p == a) / n
        return {"acc": acc, "n": n, "raw": raw}
    if task == "bbh":
        samples, answers, task_names, n_tasks = load_bbh(max_per_task=2 if smoke else 20)
        gens = generate(model, tokenizer, chat_wrap(tokenizer, samples), max_new_tokens=128)
        per_task = {}
        raw = []
        for i, (task_name, g, a) in enumerate(zip(task_names, gens, answers)):
            ok = parse_bbh(g) == a
            per_task.setdefault(task_name, []).append(ok)
            raw.append({"qid": i, "task": task_name, "correct": 1 if ok else 0,
                        "target": a, "gen_len": len(g.split())})
        acc = sum(sum(v) / len(v) for v in per_task.values()) / n_tasks
        return {"acc": acc, "n": n_tasks * 20,
                "per_task": {k: sum(v) / len(v) for k, v in per_task.items()},
                "raw": raw}
    unpacked = TASKS[task]()
    samples, answers, n = unpacked[0], unpacked[1], unpacked[3]
    groups = unpacked[4] if len(unpacked) > 4 else None
    group_key = {"mmlu": "subjects", "hellaswag": "activities",
                 "truthfulqa": "categories"}.get(task, "groups")
    if smoke:
        samples, answers, n = samples[:200], answers[:200], 200
        if groups:
            groups = groups[:200]
    nlls = score_options(model, tokenizer, samples)
    correct = []
    raw = []
    for k, (nll, a, (p, opts)) in enumerate(zip(nlls, answers, samples)):
        nll_f = [float(x) for x in nll]
        idx = nll_f.index(min(nll_f))
        correct.append(1 if idx == a else 0)
        best, second = sorted(nll_f)[0], sorted(nll_f)[1]
        rec = {"qid": k, "correct": 1 if idx == a else 0,
               "margin": round(second - best, 4), "chosen": idx, "answer": a}
        if groups:
            rec["group"] = groups[k]
        raw.append(rec)
    res = {"acc": sum(correct) / n, "n": n, "raw": raw}
    if groups:
        per = {}
        for grp, c in zip(groups, correct):
            per.setdefault(grp, []).append(c)
        res[group_key] = {k: sum(v) / len(v) for k, v in sorted(per.items())}
    return res


def evaluate(cfg, dataset, tasks, smoke=False, force=False):
    repo = cfg["paths"]["repo_root"]
    eval_dir = os.path.join(repo, "results", "eval")
    os.makedirs(eval_dir, exist_ok=True)
    tag = cfg["paths"].get("experiment_tag", "")
    name = f"eval_{tag}_{dataset}.json" if tag else f"eval_{dataset}.json"
    out_path = os.path.join(eval_dir, name)
    results = {}
    if os.path.exists(out_path) and not force:
        results = json.load(open(out_path))
    remaining = [t for t in tasks if t not in results or force]
    if not remaining:
        print(f"[{dataset}] all tasks done, skip (use --force to redo)")
        return
    print(f"[{time.strftime('%F %T')}] loading model [{dataset}] ...", flush=True)
    t_load = time.time()
    model = load_model(cfg, dataset)
    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["model"])
    print(f"[{time.strftime('%F %T')}] model loaded in {time.time()-t_load:.0f}s", flush=True)
    for task in tasks:
        if smoke and task not in ("mmlu", "gsm8k"):
            continue
        if task in results and not force:
            print(f"  {task}: cached")
            continue
        t_task = time.time()
        r = run_task(model, tokenizer, task, smoke=smoke)
        raw = r.pop("raw", None)
        results[task] = r
        # incremental save: interruption loses at most the current task
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"  [{time.strftime('%H:%M:%S')}] {task} took {time.time()-t_task:.0f}s", flush=True)
        if raw is not None:
            raw_path = os.path.join(eval_dir,
                                    f"eval_raw_{tag}_{dataset}.jsonl" if tag else f"eval_raw_{dataset}.jsonl")
            with open(raw_path, "a") as f:
                for rec in raw:
                    rec["task"] = task
                    f.write(json.dumps(rec) + "\n")
        print(f"  {task}: {r['acc']:.4f} (n={r['n']})", flush=True)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/noisedetect/config.yaml")
    ap.add_argument("--dataset", default="clean")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--tag", type=str, default=None, help="experiment tag (run dir suffix)")
    ap.add_argument("--force", action="store_true", help="re-evaluate even if results exist")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.tag:
        cfg["paths"]["experiment_tag"] = args.tag
    tasks = args.tasks.split(",") if args.tasks else cfg["eval"]["tasks"]
    evaluate(cfg, args.dataset, tasks, smoke=args.smoke, force=args.force)
