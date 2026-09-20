"""Assembling the per-sample metrics table from on-disk run/data artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics_common import DIAG_COLS


def _load_run_metrics(metrics_dir, max_epoch=None):
    mp=metrics_dir/'per_sample.jsonl'
    if not mp.exists() or mp.stat().st_size==0: return pd.DataFrame()
    df=pd.read_json(mp,lines=True).dropna(subset=['loss'])
    if max_epoch is not None: df=df[df['epoch']<=max_epoch]
    if df.empty: return df
    df['sample_id']=df['sample_id'].astype(str)
    piv=df.pivot_table(index='sample_id',columns='epoch',aggfunc='first')
    ep_cols=sorted(df['epoch'].unique())
    out=pd.DataFrame(index=piv.index)
    for base,name in [('loss','loss'),('grad_norm','grad_norm'),('cos_sim_ref','cos_ref'),('cos_sim_global','cos_global')]:
        ep={e:piv[(base,e)] for e in ep_cols if (base,e) in piv.columns}
        if not ep: continue
        m=pd.DataFrame(ep)
        out[f'{name}_mean']=m.mean(axis=1); out[f'{name}_last']=m.iloc[:,-1]
        out[f'{name}_std']=m.std(axis=1); out[f'{name}_slope']=m.iloc[:,-1]-m.iloc[:,0]
        if name=='loss':
            out['loss_min']=m.min(axis=1)
            out['converge_epoch']=(m<2.0).idxmax(axis=1)
            out.loc[(m>=2.0).all(axis=1),'converge_epoch']=len(ep_cols)
            xs=np.arange(len(ep_cols)); X=np.stack([np.ones_like(xs,dtype=float),xs.astype(float),xs.astype(float)**2],axis=1)
            coeffs=m.values.astype(float)@np.linalg.pinv(X).T
            out['loss_curvature']=coeffs[:,0]; out['loss_rank']=m.rank(pct=True).mean(axis=1)
    if 'grad_norm_mean' in out and out['grad_norm_mean'].notna().any():
        out['grad_norm_cv']=out['grad_norm_std']/out['grad_norm_mean'].replace(0,np.nan)
    if 'update_contrib' in df:
        out['update_contrib_mean']=df[df['update_contrib'].notna()].groupby('sample_id')['update_contrib'].mean()
    diag_files=[f for f in sorted(metrics_dir.glob('diag_epoch*.jsonl')) if max_epoch is None or int(f.stem.split('epoch')[-1])<=max_epoch]
    if diag_files:
        diag=pd.concat([pd.read_json(f,lines=True) for f in diag_files])
        diag['sample_id']=diag['sample_id'].astype(str)
        cols=[c for c in DIAG_COLS if c in diag]
        out=out.join(diag.groupby('sample_id')[cols].mean())
    token=_load_token_features(metrics_dir,max_epoch)
    if not token.empty: out=out.join(token)
    return out


def _load_token_features(metrics_dir, max_epoch=None):
    """Aggregates token_diag_epoch*.jsonl (top-k hard label tokens per sample)."""
    rows=[]
    for f in sorted(metrics_dir.glob('token_diag_epoch*.jsonl')):
        epoch=int(f.stem.split('epoch')[-1])
        if max_epoch is not None and epoch>max_epoch: continue
        for line in f.open(encoding='utf-8'):
            r=json.loads(line); r['_epoch']=epoch; rows.append(r)
    if not rows: return pd.DataFrame()
    feats=[]
    for sid,g in pd.DataFrame(rows).groupby('sample_id'):
        g=g[g['top_tokens'].map(len)>0]
        if g.empty: continue
        ts=[t for row in g['top_tokens'] for t in row]
        rec={'sample_id':sid,
             'n_hard':float(np.mean([len(x) for x in g['top_tokens']])),
             'hard_loss_mean':float(np.mean([t[2] for t in ts])),
             'hard_loss_max':float(np.mean([max(x[2] for x in row) for row in g['top_tokens']])),
             'hard_pos_peak':float(np.mean([np.mean([t[0] for t in row]) for row in g['top_tokens']])),
             'hard_pos_std_mean':float(np.mean([np.std([t[0] for t in row]) for row in g['top_tokens']]))}
        all_ids=set()
        for row in g['top_tokens']: all_ids|={t[1] for t in row}
        rec['hard_id_uniq']=float(len(all_ids))
        pos_by_epoch={ep:set(x for row in gg['top_tokens'] for x in [t[0] for t in row]) for ep,gg in g.groupby('_epoch')}
        eps=sorted(pos_by_epoch); jac=[]
        for a,b in zip(eps,eps[1:]):
            pa,pb=pos_by_epoch[a],pos_by_epoch[b]; jac.append(len(pa&pb)/max(1,len(pa|pb)))
        rec['hard_pos_jaccard']=float(np.mean(jac)) if jac else np.nan
        feats.append(rec)
    return pd.DataFrame(feats).set_index('sample_id')


def build_table(root, tag, datasets=None, max_epoch=None):
    """Assemble the per-sample metrics table for one tag by joining, per trained
    dataset, the training-trajectory metrics, the diagnostic-layer aggregates,
    the token-level hard-token stats and the data-side text_nn_sim.

    max_epoch truncates the trajectory to epochs <= max_epoch (0-indexed) before
    aggregating — used to simulate "early" detection from a partially-trained run
    without needing to actually stop training early."""
    import textsim
    root=Path(root); run_base=root/'runs'/tag; data_base=root/'data'/tag
    if datasets is None:
        # Only runs with a matching data/{tag}/{ds}/train.jsonl: cleaning-loop
        # retrains were fed via --train-file, so they have trajectories but no
        # label file of their own, and their noise labels live in the set they
        # were built from.
        datasets=sorted(p.parent.name for p in run_base.glob('*/summary.json')
                        if (data_base/p.parent.name/'train.jsonl').exists())
    all_rows=[]
    for ds in datasets:
        metrics=_load_run_metrics(run_base/ds/'metrics', max_epoch)
        if metrics.empty: continue
        label_path=data_base/ds/'train.jsonl'
        labels={}
        for line in label_path.open(encoding='utf-8'):
            r=json.loads(line); labels[str(r['sample_id'])]=(r.get('noise_type','none'),r.get('category'))
        text_sim=textsim.text_nn_sim(label_path)
        for sid,row in metrics.iterrows():
            ntype,cat=labels.get(str(sid),('none',None))
            all_rows.append({'dataset':ds,'sample_id':sid,'noise_type':ntype,'category':cat,
                             'text_nn_sim':text_sim.get(str(sid)),**row.to_dict()})
    return pd.DataFrame(all_rows)


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
