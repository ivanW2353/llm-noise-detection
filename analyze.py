"""Post-hoc analysis of training, token, unsupervised and transfer metrics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score


def auc(y, scores):
    y = np.asarray(y); scores = np.asarray(scores)
    if len(np.unique(y)) < 2: return float('nan')
    value = roc_auc_score(y, scores)
    return float(max(value, 1 - value))


def summarize(frame, features):
    rows=[]
    for feature in features:
        if feature in frame:
            values=frame[feature].fillna(frame[feature].median())
            rows.append({'feature':feature,'auc':auc(frame.noise_type.ne('none'),values)})
    return rows


def training_metrics(root: str | Path, tag: str) -> pd.DataFrame:
    rows=[]; base=Path(root)/'runs'/tag
    for dataset in sorted(base.iterdir()) if base.exists() else []:
        path=dataset/'metrics'/'per_sample.jsonl'
        if not path.exists(): continue
        records=[json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
        frame=pd.DataFrame(records)
        if frame.empty or 'epoch' not in frame: continue
        numeric=[c for c in ['loss','grad_norm','cos_sim_ref','cos_sim_global','update_contrib','tokens'] if c in frame]
        grouped=frame.groupby('epoch')[numeric].agg(['mean','std','count'])
        for epoch, values in grouped.iterrows():
            row={'tag':tag,'dataset':dataset.name,'epoch':int(epoch)}
            for feature in numeric:
                row[f'{feature}_mean']=float(values[(feature,'mean')])
                row[f'{feature}_std']=float(values[(feature,'std')]) if pd.notna(values[(feature,'std')]) else 0.0
                row[f'{feature}_n']=int(values[(feature,'count')])
            rows.append(row)
    return pd.DataFrame(rows)


def token_metrics(path: str | Path) -> pd.DataFrame:
    records=[json.loads(line) for line in Path(path).open(encoding='utf-8') if line.strip()]
    frame=pd.DataFrame(records)
    if frame.empty: return frame
    numeric=[c for c in ['n_hard','hard_loss_mean','hard_loss_max','hard_gradnorm_mean','hard_cos_ref_mean','pos_std','loc_mismatch_frac'] if c in frame]
    group='noise_type' if 'noise_type' in frame else None
    if not group: return frame[numeric]
    result=frame.groupby(group)[numeric].agg(['mean','std','count']).reset_index()
    result.columns=[('_'.join(str(x) for x in c if x) if isinstance(c,tuple) else str(c)) for c in result.columns]
    return result


def token_metrics_for_tag(root: str | Path, tag: str, dataset: str | None = None) -> pd.DataFrame:
    """Aggregate token-level files for one tag, optionally selecting a dataset."""
    base = Path(root) / 'results' / tag
    paths = [base / f'token_level_{dataset}.jsonl'] if dataset else sorted(base.glob('token_level_*.jsonl'))
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = token_metrics(path)
        if frame.empty:
            continue
        frame.insert(0, 'dataset', path.stem.removeprefix('token_level_'))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def unsupervised_metrics(frame: pd.DataFrame, features=None, seed=42) -> pd.DataFrame:
    excluded={'sample_id','dataset','noise_type','category','noise_label'}
    features=list(features or [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])])
    features=[c for c in features if c in frame]
    if not features: raise ValueError('No numeric features available')
    clean=frame.dropna(subset=features).reset_index(drop=True).copy(); x=clean[features].to_numpy(float); med=np.median(x,axis=0); mad=np.median(np.abs(x-med),axis=0); mad=np.where(mad==0,1,mad); z=(x-med)/(1.4826*mad)
    scores={'zscore_max':np.max(np.abs(z),axis=1),'zscore_mean':np.mean(np.abs(z),axis=1)}
    if len(x)>=10:
        scores['iforest']=-IsolationForest(contamination='auto',random_state=seed).fit(x).score_samples(x)
    labels=clean.noise_type.fillna('none').ne('none').to_numpy(int); output=[]
    for method, values in scores.items():
        for dataset, indexes in clean.groupby('dataset').groups.items():
            idx=np.asarray(list(indexes)); y=labels[idx]; score=values[idx]; k=max(1,int(round(.1*len(idx)))); order=np.argsort(score)[-k:]
            output.append({'dataset':dataset,'method':method,'n':len(idx),'n_noise':int(y.sum()),'auc':auc(y,score),'p_at_10':float(y[order].mean()),'random_p':float(y.mean())})
    return pd.DataFrame(output)


def transfer_metrics(path: str | Path, tags=None) -> pd.DataFrame:
    frame=pd.read_csv(path)
    if tags and 'train_tag' in frame: frame=frame[frame.train_tag.isin(tags) | frame.test_tag.isin(tags)]
    if tags and 'tag' in frame and 'train_tag' not in frame: frame=frame[frame.tag.isin(tags)]
    return frame.reset_index(drop=True)
