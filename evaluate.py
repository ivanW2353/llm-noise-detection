"""Few-shot evaluation of fine-tuned models on standard benchmarks.

Tasks (loglikelihood multiple-choice or CoT generation):
  mmlu (5-shot, per-subject), hellaswag (5-shot), arc (25-shot), winogrande (5-shot),
  truthfulqa (0-shot), gsm8k (5-shot CoT), bbh (3-shot CoT, 20/task)

Results are saved after every task, so evaluation is resumable; re-running
skips tasks already cached in results/eval/eval_{tag}_{dataset}.json unless
--force. Mock models get a zero-cost stub result per task (no model loaded).
"""
import glob
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

MAX_LEN = 2048          # generation room (long CoT few-shot + output)
SCORE_MAX_LEN = 1024    # MC scoring cap: [B, L, V] logits are memory-heavy


# ----------------------------------------------------------------------------
# model helpers
# ----------------------------------------------------------------------------
def _load_model(settings, dataset):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = settings.section('paths').get('model')
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map={'': 0})
    if dataset != 'base':
        from peft import PeftModel
        lora_path = settings.runs_dir() / dataset / 'lora'
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _per_row_nll(logits, labels, mask):
    import torch
    B, L, V = logits.shape
    shift = logits[:, :-1].reshape(-1, V)
    tgt = labels[:, 1:].reshape(-1)
    m = (labels[:, 1:] != -100) * mask[:, 1:]
    ce = torch.nn.functional.cross_entropy(shift, tgt, reduction='none').view(B, L - 1)
    return (ce * m.float()).sum(1) / m.float().sum(1).clamp(min=1)


def _score_options(model, tokenizer, samples, bs=8):
    """samples: list of (prompt, options); returns per-sample nll list."""
    import torch
    flat = [(p, ' ' + o) for p, opts in samples for o in opts]
    all_nll = []
    t0 = time.time()
    for s in range(0, len(flat), bs):
        chunk = flat[s:s + bs]
        if s % (bs * 500) == 0:
            rate = s / max(1, time.time() - t0)
            print(f"    [{time.strftime('%H:%M:%S')}] ... {s}/{len(flat)} options "
                  f"({rate:.0f} opts/s, ETA {max(0, (len(flat)-s)/max(rate, 1e-9))/60:.1f} min)", flush=True)
        pids = [tokenizer(p, add_special_tokens=False)['input_ids'] for p, _ in chunk]
        cids = [tokenizer(c, add_special_tokens=False)['input_ids'] for _, c in chunk]
        maxl = min(SCORE_MAX_LEN, max(len(p) + len(c) for p, c in zip(pids, cids)))
        for i in range(len(pids)):  # keep tail (query + Answer:) within max_len
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
        with torch.no_grad():
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
            all_nll.extend(_per_row_nll(logits, labels.cuda(), mask.cuda()).tolist())
    nlls, idx = [], 0
    for _, opts in samples:
        nlls.append(all_nll[idx:idx + len(opts)])
        idx += len(opts)
    return nlls


def _generate(model, tokenizer, prompts, max_new_tokens=256, bs=32):
    import torch
    model.config.use_cache = True
    tokenizer.padding_side = 'left'  # flash-attn generation is broken with right padding
    outs = []
    t0 = time.time()
    for s in range(0, len(prompts), bs):
        if s % (bs * 50) == 0:
            rate = s / max(1, time.time() - t0)
            print(f"    [{time.strftime('%H:%M:%S')}] ... generated {s}/{len(prompts)} "
                  f"({rate:.1f} prompts/s, ETA {max(0, (len(prompts)-s)/max(rate, 1e-9))/60:.1f} min)", flush=True)
        enc = tokenizer(prompts[s:s + bs], return_tensors='pt', padding=True,
                        truncation=True, max_length=MAX_LEN - max_new_tokens)
        with torch.no_grad():
            gen = model.generate(
                input_ids=enc['input_ids'].cuda(), attention_mask=enc['attention_mask'].cuda(),
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gen = gen[:, enc['input_ids'].shape[1]:]
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    model.config.use_cache = False
    return outs


# ----------------------------------------------------------------------------
# TriviaQA official EM/F1 (github.com/mandarjoshi90/triviaqa evaluation script)
# ----------------------------------------------------------------------------
def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + ''.join(["'", '’', '´', '`']))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def _exact_match_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _f1_score(prediction, ground_truth):
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def _parse_qa_answer(text):
    text = text.strip()
    return text.splitlines()[0].strip() if text else ''


def _load_qa_correctness(seed=42, max_samples=2000):
    """Official TriviaQA rc.nocontext validation split (never used by our data pipeline,
    which trains on the official train split -- see AGENTS.md). Subsampled for eval-time
    cost since generation is far slower per-sample than MC scoring."""
    import random
    from datasets import load_dataset
    tr = load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='train')
    val = load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='validation')
    shot_txt = ''.join(
        f"Question: {s['question']}\nAnswer: {s['answer']['value']}\n\n" for s in tr.select(range(5)))
    idx = list(range(len(val)))
    random.Random(seed).shuffle(idx)
    idx = idx[:max_samples]
    samples, answers = [], []
    for i in idx:
        r = val[i]
        samples.append(shot_txt + f"Question: {r['question']}\nAnswer:")
        answers.append(r['answer']['aliases'])
    return samples, answers, '5-shot', len(idx)


# ----------------------------------------------------------------------------
# task loaders -> (samples, answers, shot_desc, n[, groups])
#   MC tasks: samples = [(prompt, options)], answers = [correct_idx, ...]
#   gen tasks: samples = [prompt, ...], answers = [correct_str, ...]
# ----------------------------------------------------------------------------
def _load_mmlu():
    from datasets import load_dataset
    dev = load_dataset('cais/mmlu', 'all', split='dev')
    test = load_dataset('cais/mmlu', 'all', split='test')
    by_subj = {}
    for r in dev:
        by_subj.setdefault(r['subject'], []).append(r)
    samples, answers, subjects = [], [], []
    for r in test:
        shots = by_subj[r['subject']][:5]
        shot_txt = ''.join(
            f"{s['question']}\nA. {s['choices'][0]}\nB. {s['choices'][1]}\n"
            f"C. {s['choices'][2]}\nD. {s['choices'][3]}\nAnswer: {chr(65 + s['answer'])}\n\n"
            for s in shots)
        prompt = (f"The following are multiple choice questions (with answers) "
                  f"about {r['subject']}.\n\n{shot_txt}{r['question']}\n"
                  f"A. {r['choices'][0]}\nB. {r['choices'][1]}\nC. {r['choices'][2]}\n"
                  f"D. {r['choices'][3]}\nAnswer:")
        samples.append((prompt, list('ABCD')))
        answers.append(r['answer'])
        subjects.append(r['subject'])
    return samples, answers, '5-shot', len(test), subjects


def _load_hellaswag():
    from datasets import load_dataset
    tr = load_dataset('Rowan/hellaswag', split='train')
    val = load_dataset('Rowan/hellaswag', split='validation')
    shots = [f"{r['ctx']} {r['endings'][int(r['label'])]}" for r in tr.select(range(5))]
    shot_txt = ''.join(s + '\n\n' for s in shots)
    samples, answers, activities = [], [], []
    for r in val:
        samples.append((shot_txt + r['ctx'], r['endings']))
        answers.append(int(r['label']))
        activities.append(r['activity_label'])
    return samples, answers, '5-shot', len(val), activities


def _load_arc():
    from datasets import load_dataset
    tr = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train')
    test = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
    shot_txt = ''.join(
        f"{s['question']}\n" + ''.join(f"{l}. {t}\n" for t, l in zip(s['choices']['text'], s['choices']['label']))
        + f"Answer: {s['answerKey']}\n\n" for s in tr.select(range(25)))
    samples, answers = [], []
    for r in test:
        prompt = f"{shot_txt}{r['question']}\n" + ''.join(
            f"{l}. {t}\n" for t, l in zip(r['choices']['text'], r['choices']['label'])) + 'Answer:'
        samples.append((prompt, r['choices']['label']))
        answers.append(r['choices']['label'].index(r['answerKey']))
    return samples, answers, '25-shot', len(test)


def _load_winogrande():
    from datasets import load_dataset
    tr = load_dataset('allenai/winogrande', 'winogrande_debiased', split='train')
    val = load_dataset('allenai/winogrande', 'winogrande_debiased', split='validation')
    shot_txt = ''.join(
        f"{s['sentence']} {s['option1']}\nAnswer: {s['option1'] if s['answer'] == '1' else s['option2']}\n\n"
        for s in tr.select(range(5)))
    samples, answers = [], []
    for r in val:
        samples.append((shot_txt + r['sentence'], [r['option1'], r['option2']]))
        answers.append(int(r['answer']) - 1)
    return samples, answers, '5-shot', len(val)


def _load_truthfulqa():
    from datasets import load_dataset
    d = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')
    g = load_dataset('truthfulqa/truthful_qa', 'generation', split='validation')
    cat_map = {r['question']: r['category'] for r in g}
    samples, answers, categories = [], [], []
    for r in d:
        mc = r['mc1_targets']
        samples.append((r['question'] + '\nAnswer:', mc['choices']))
        answers.append(mc['labels'].index(1))
        categories.append(cat_map.get(r['question'], 'unknown'))
    return samples, answers, '0-shot', len(d), categories


def _normalize_num(s):
    return re.sub(r'[,$\s]', '', s).rstrip('.')


def _load_gsm8k():
    from datasets import load_dataset
    tr = load_dataset('openai/gsm8k', 'main', split='train')
    test = load_dataset('openai/gsm8k', 'main', split='test')
    shot_txt = ''.join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in tr.select(range(5)))
    samples, answers = [], []
    for r in test:
        samples.append(shot_txt + f"Question: {r['question']}\nAnswer:")
        answers.append(_normalize_num(r['answer'].split('#### ')[1]))
    return samples, answers, '5-shot CoT', len(test)


def _parse_gsm8k(text):
    m = re.search(r'####\s*(-?\d+[\d,.]*)', text)
    if not m:
        m = re.search(r'\\boxed\{(-?\d+[\d,.]*)\}', text)
    if m:
        return _normalize_num(m.group(1))
    nums = re.findall(r'-?\d+[\d,.]*', text)
    return _normalize_num(nums[-1]) if nums else None


def _chat_wrap(tokenizer, prompts):
    return [tokenizer.apply_chat_template([{'role': 'user', 'content': p}],
                                          tokenize=False, add_generation_prompt=True)
            for p in prompts]


def _load_bbh(bbh_dir, max_per_task=20):
    task_files = sorted(glob.glob(os.path.join(bbh_dir, 'test', '*.json')))
    samples, answers, task_names = [], [], []
    for f in task_files:
        task = os.path.basename(f).replace('.json', '')
        d = json.load(open(f))
        cot = open(os.path.join(bbh_dir, 'cot-prompts', f'{task}.txt')).read()
        ex = d['examples']
        shot_txt = ''.join(
            f"Q: {s['input']}\nA: Let's think step by step. {cot}\nSo the answer is {s['target']}.\n\n"
            for s in ex[:3])
        for s in ex[3:3 + max_per_task]:
            prompt = f"{shot_txt}Q: {s['input']}\nA: Let's think step by step."
            samples.append(prompt)
            answers.append(s['target'].lower())
            task_names.append(task)
    return samples, answers, task_names, len(task_files)


def _parse_bbh(text):
    m = re.findall(r'the answer is ([A-Za-z0-9][^\n.]*)', text.lower())
    if m:
        return m[-1].strip()
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-1].strip().lower() if lines else ''


TASKS = {
    'mmlu': _load_mmlu, 'hellaswag': _load_hellaswag, 'arc': _load_arc,
    'winogrande': _load_winogrande, 'truthfulqa': _load_truthfulqa,
    'gsm8k': _load_gsm8k,
}


def _run_task(model, tokenizer, task, bbh_dir=None, smoke=False):
    if task == 'qa_correctness':
        samples, aliases, _, n = _load_qa_correctness(max_samples=50 if smoke else 2000)
        if smoke:
            samples, aliases, n = samples[:50], aliases[:50], 50
        gens = _generate(model, tokenizer, _chat_wrap(tokenizer, samples), max_new_tokens=32)
        preds = [_parse_qa_answer(t) for t in gens]
        em = [_metric_max_over_ground_truths(_exact_match_score, p, a) for p, a in zip(preds, aliases)]
        f1 = [_metric_max_over_ground_truths(_f1_score, p, a) for p, a in zip(preds, aliases)]
        raw = [{'qid': i, 'em': int(e), 'f1': round(f, 4), 'aliases': a, 'pred': p}
               for i, (e, f, a, p) in enumerate(zip(em, f1, aliases, preds))]
        return {'acc': sum(em) / n, 'em': sum(em) / n, 'f1': sum(f1) / n, 'n': n, 'raw': raw}
    if task == 'gsm8k':
        samples, answers, _, n = _load_gsm8k()
        if smoke:
            samples, answers, n = samples[:50], answers[:50], 50
        gens = _generate(model, tokenizer, _chat_wrap(tokenizer, samples), max_new_tokens=512)
        preds = [_parse_gsm8k(t) for t in gens]
        raw = [{'qid': i, 'correct': 1 if p == a else 0, 'answer': a,
                'pred': p, 'gen_len': len(t.split())} for i, (p, a, t) in enumerate(zip(preds, answers, gens))]
        acc = sum(1 for p, a in zip(preds, answers) if p == a) / n
        return {'acc': acc, 'n': n, 'raw': raw}
    if task == 'bbh':
        samples, answers, task_names, n_tasks = _load_bbh(bbh_dir, max_per_task=2 if smoke else 20)
        gens = _generate(model, tokenizer, _chat_wrap(tokenizer, samples), max_new_tokens=128)
        per_task = {}
        raw = []
        for i, (task_name, g, a) in enumerate(zip(task_names, gens, answers)):
            ok = _parse_bbh(g) == a
            per_task.setdefault(task_name, []).append(ok)
            raw.append({'qid': i, 'task': task_name, 'correct': 1 if ok else 0,
                        'target': a, 'gen_len': len(g.split())})
        acc = sum(sum(v) / len(v) for v in per_task.values()) / n_tasks
        return {'acc': acc, 'n': n_tasks * 20,
                'per_task': {k: sum(v) / len(v) for k, v in per_task.items()},
                'raw': raw}
    unpacked = TASKS[task]()
    samples, answers, n = unpacked[0], unpacked[1], unpacked[3]
    groups = unpacked[4] if len(unpacked) > 4 else None
    group_key = {'mmlu': 'subjects', 'hellaswag': 'activities',
                 'truthfulqa': 'categories'}.get(task, 'groups')
    if smoke:
        samples, answers, n = samples[:200], answers[:200], 200
        if groups:
            groups = groups[:200]
    nlls = _score_options(model, tokenizer, samples)
    correct = []
    raw = []
    for k, (nll, a, (p, opts)) in enumerate(zip(nlls, answers, samples)):
        nll_f = [float(x) for x in nll]
        idx = nll_f.index(min(nll_f))
        correct.append(1 if idx == a else 0)
        best, second = sorted(nll_f)[0], sorted(nll_f)[1]
        rec = {'qid': k, 'correct': 1 if idx == a else 0,
               'margin': round(second - best, 4), 'chosen': idx, 'answer': a}
        if groups:
            rec['group'] = groups[k]
        raw.append(rec)
    res = {'acc': sum(correct) / n, 'n': n, 'raw': raw}
    if groups:
        per = {}
        for grp, c in zip(groups, correct):
            per.setdefault(grp, []).append(c)
        res[group_key] = {k: sum(v) / len(v) for k, v in sorted(per.items())}
    return res


class Evaluator:
    def __init__(self, settings, model):
        self.settings, self.model = settings, model

    def run(self, dataset, tasks, force=False, smoke=False):
        if getattr(self.model, 'name', 'mock') == 'mock':
            return self._run_mock(dataset, tasks, force)
        return self._run_benchmarks(dataset, tasks, force, smoke)

    def _run_mock(self, dataset, tasks, force):
        p = self.settings.root / 'results' / 'eval' / f'eval_{self.settings.tag}_{dataset}.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        result = {} if force or not p.exists() else json.loads(p.read_text())
        for task in tasks:
            result.setdefault(task, {'accuracy': 0.0, 'n': 0, 'model': getattr(self.model, 'name', 'unknown')})
        p.write_text(json.dumps(result, indent=2))
        return result

    def _run_benchmarks(self, dataset, tasks, force, smoke):
        eval_dir = self.settings.root / 'results' / 'eval'
        eval_dir.mkdir(parents=True, exist_ok=True)
        tag = self.settings.tag
        out_path = eval_dir / f'eval_{tag}_{dataset}.json'
        results = json.loads(out_path.read_text()) if out_path.exists() and not force else {}
        remaining = [t for t in tasks if t not in results or force]
        if not remaining:
            print(f'[{dataset}] all tasks done, skip (use --force to redo)')
            return results
        print(f"[{time.strftime('%F %T')}] loading model [{dataset}] ...", flush=True)
        t_load = time.time()
        model, tokenizer = _load_model(self.settings, dataset)
        print(f"[{time.strftime('%F %T')}] model loaded in {time.time()-t_load:.0f}s", flush=True)
        bbh_dir = self.settings.data_root / 'datasets' / 'benchmarks' / 'bbh'
        for task in tasks:
            if smoke and task not in ('mmlu', 'gsm8k', 'qa_correctness'):
                continue
            if task in results and not force:
                print(f'  {task}: cached')
                continue
            t_task = time.time()
            r = _run_task(model, tokenizer, task, bbh_dir=str(bbh_dir), smoke=smoke)
            raw = r.pop('raw', None)
            results[task] = r
            out_path.write_text(json.dumps(results, indent=2))  # incremental save
            print(f"  [{time.strftime('%H:%M:%S')}] {task} took {time.time()-t_task:.0f}s", flush=True)
            if raw is not None:
                raw_path = eval_dir / f'eval_raw_{tag}_{dataset}.jsonl'
                with raw_path.open('a') as f:
                    for rec in raw:
                        rec['task'] = task
                        f.write(json.dumps(rec) + '\n')
            print(f"  {task}: {r['acc']:.4f} (n={r['n']})", flush=True)
        print(f'saved -> {out_path}')
        return results
