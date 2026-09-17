"""Builds a natural-noise dataset from OASST, where "noise" is a human quality
judgement rather than a programmatic perturbation.

Every dataset elsewhere in this project injects noise with `data.py::apply`, so
the perturbation is known and the ground truth is exact. That is what makes
per-type detection difficulty measurable, and also what limits the conclusions:
wild noise (a one-character reply, a language switch mid-answer, a
machine-translated passage) may look nothing like the 7 injected types in
either text distribution or training dynamics.

OASST is used here because it is the rare corpus that is BOTH genuinely wild
(real volunteers writing real replies) and labelled by hand: every message
carries per-annotator ratings for quality, spam, lang_mismatch, toxicity and
more. Thresholding the mean `quality` rating turns those ratings into the
noise/clean split the evaluation code already expects, so AUC / P@10% /
removal-precision all keep working without pretending the noise was injected.

Consequences of using a rating as the label, which the report must state:

- The label is a human judgement, not ground truth about a perturbation. Low
  quality mixes several mechanisms at once (too short, off-topic, disfluent,
  wrong) instead of isolating one, so per-mechanism difficulty is no longer
  separable the way `data.py`'s types are.
- Ratings are noisy themselves: most messages have only 3 annotators, so a
  rating near the threshold is not a reliable label. `--min-annotators` and the
  margin band below exist to trade coverage for label reliability.
- Quality correlates with response length, and length is trivially visible to
  the detector. `summarize()` reports the correlation so the report can say how
  much of any detection result is just "short replies are rated worse".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LABEL_FIELDS = ('quality', 'spam', 'lang_mismatch', 'not_appropriate', 'toxicity')


def _label(row: dict, name: str) -> tuple[float | None, int]:
    labels = row.get('labels')
    if not labels or name not in labels['name']:
        return None, 0
    i = labels['name'].index(name)
    return labels['value'][i], labels['count'][i]


def load_pairs(dataset: str = 'OpenAssistant/oasst2', split: str = 'train',
               lang: str = 'en') -> list[dict]:
    """Single-turn (prompter -> assistant) pairs carrying a quality rating."""
    from datasets import load_dataset

    rows = load_dataset(dataset, split=split)
    by_id = {r['message_id']: r for r in rows}
    pairs = []
    for r in rows:
        if r['role'] != 'assistant' or r['deleted'] or not r['parent_id']:
            continue
        if lang and r['lang'] != lang:
            continue
        parent = by_id.get(r['parent_id'])
        if not parent or parent['role'] != 'prompter' or not parent['text'].strip():
            continue
        quality, n_annot = _label(r, 'quality')
        if quality is None or not r['text'].strip():
            continue
        pairs.append({
            'message_id': r['message_id'],
            'prompt': parent['text'],
            'response': r['text'],
            'quality': float(quality),
            'n_annotators': int(n_annot),
            'rank': r['rank'],
            **{f: (_label(r, f)[0] or 0.0) for f in LABEL_FIELDS if f != 'quality'},
        })
    return pairs


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for i, p in enumerate(rows):
            f.write(json.dumps({
                'sample_id': str(i),
                'messages': [{'role': 'user', 'content': p['prompt']},
                             {'role': 'assistant', 'content': p['response']}],
                'noise_type': p['noise_type'],
                'quality': p['quality'],
                'n_annotators': p['n_annotators'],
                'source_message_id': p['message_id'],
            }, ensure_ascii=False) + '\n')


def build(tag_dir: str | Path, dataset: str = 'OpenAssistant/oasst2', lang: str = 'en',
          noise_threshold: float = 0.30, clean_threshold: float = 0.50,
          min_annotators: int = 3, max_samples: int | None = None,
          n_holdout: int = 400, seed: int = 42) -> dict:
    """Lay out a tag directory the training pipeline can consume directly:
    `{tag_dir}/heldout.jsonl` plus `{tag_dir}/wild/train.jsonl`.

    Samples between the two thresholds are DROPPED rather than assigned to
    either side: a rating of 0.4 from 3 annotators is not evidence of much, and
    keeping that band would put the least reliable labels exactly where the
    detector is scored. The gap is the main knob trading dataset size for label
    trustworthiness.

    The held-out split is carved from the CLEAN samples only. model.py averages
    those gradients into the reference direction that `cos_sim_ref` is measured
    against, so letting known-bad replies into it would blunt the very contrast
    the feature exists to expose.
    """
    if not noise_threshold < clean_threshold:
        raise ValueError('noise_threshold must be below clean_threshold')
    pairs = load_pairs(dataset, lang=lang)
    kept = [p for p in pairs if p['n_annotators'] >= min_annotators]
    labelled = []
    for p in kept:
        if p['quality'] < noise_threshold:
            p = {**p, 'noise_type': 'low_quality'}
        elif p['quality'] >= clean_threshold:
            p = {**p, 'noise_type': 'none'}
        else:
            continue
        labelled.append(p)

    rng = np.random.default_rng(seed)
    rng.shuffle(labelled)

    clean_idx = [i for i, p in enumerate(labelled) if p['noise_type'] == 'none']
    if len(clean_idx) < n_holdout:
        raise ValueError(f'need {n_holdout} clean samples for the held-out split, '
                         f'have {len(clean_idx)}')
    holdout_ids = set(clean_idx[:n_holdout])
    heldout = [labelled[i] for i in sorted(holdout_ids)]
    train = [p for i, p in enumerate(labelled) if i not in holdout_ids]
    if max_samples is not None:
        train = train[:max_samples]

    tag_dir = Path(tag_dir)
    (tag_dir / 'wild').mkdir(parents=True, exist_ok=True)
    _write_rows(tag_dir / 'heldout.jsonl', heldout)
    _write_rows(tag_dir / 'wild' / 'train.jsonl', train)

    meta = summarize(train, pairs, kept, dataset, lang, noise_threshold,
                     clean_threshold, min_annotators, seed)
    meta['n_heldout'] = len(heldout)
    meta['heldout_note'] = 'drawn from clean samples only (reference gradient must not include known-bad replies)'
    json.dump(meta, (tag_dir / 'wild' / 'metadata.json').open('w'), indent=2, ensure_ascii=False)
    return meta


def summarize(labelled: list[dict], pairs: list[dict], kept: list[dict], dataset: str,
              lang: str, noise_threshold: float, clean_threshold: float,
              min_annotators: int, seed: int) -> dict:
    noise = [p for p in labelled if p['noise_type'] == 'low_quality']
    resp_len = np.array([len(p['response']) for p in labelled], dtype=float)
    is_noise = np.array([p['noise_type'] == 'low_quality' for p in labelled], dtype=float)
    # A detector could get credit for merely noticing that short replies are
    # rated worse; the report needs this number to discount that.
    corr = float(np.corrcoef(resp_len, is_noise)[0, 1]) if len(labelled) > 2 else float('nan')
    return {
        'source': dataset, 'lang': lang, 'seed': seed,
        'label': 'human quality rating (OASST `labels.quality`), not an injected perturbation',
        'noise_threshold': noise_threshold, 'clean_threshold': clean_threshold,
        'min_annotators': min_annotators,
        'n_pairs_available': len(pairs),
        'n_after_annotator_filter': len(kept),
        'n_dropped_ambiguous_band': len(kept) - len(labelled),
        'n_total': len(labelled),
        'n_noise': len(noise),
        'noise_ratio': len(noise) / len(labelled) if labelled else 0.0,
        'response_len_median': float(np.median(resp_len)) if len(resp_len) else None,
        'response_len_median_noise': (float(np.median([len(p['response']) for p in noise]))
                                      if noise else None),
        'corr_response_len_vs_noise': corr,
    }
