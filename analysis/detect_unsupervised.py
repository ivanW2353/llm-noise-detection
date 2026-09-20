"""Label-free / near-label-free detection scoring."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from .metrics_common import auc, _zs, MEMO_FEATS, precision_at_k
from .metrics_table import build_table


def unsupervised_metrics(frame: pd.DataFrame, features=None, seed=42) -> pd.DataFrame:
    """Label-free outlier scoring, fit independently per dataset (never pooled
    across datasets — a global fit lets extreme-magnitude datasets like garbled
    shift the outlier baseline for low-variance ones like template)."""
    excluded={'sample_id','dataset','noise_type','category','noise_label'}
    features=list(features or [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])])
    features=[c for c in features if c in frame]
    if not features: raise ValueError('No numeric features available')
    clean=frame.dropna(subset=features).reset_index(drop=True).copy()
    labels=clean.noise_type.fillna('none').ne('none').to_numpy(int)
    output=[]
    for dataset, indexes in clean.groupby('dataset').groups.items():
        idx=np.asarray(list(indexes)); y=labels[idx]
        x=clean.loc[idx,features].to_numpy(float)
        med=np.median(x,axis=0); mad=np.median(np.abs(x-med),axis=0); mad=np.where(mad==0,1,mad); z=(x-med)/(1.4826*mad)
        scores={'zscore_max':np.max(np.abs(z),axis=1),'zscore_mean':np.mean(np.abs(z),axis=1)}
        if len(x)>=10:
            xs=StandardScaler().fit_transform(x)
            scores['iforest']=-IsolationForest(n_estimators=300,random_state=seed,n_jobs=-1).fit(xs).score_samples(xs)
        k=max(1,int(round(.1*len(idx))))
        for method, score in scores.items():
            order=np.argsort(score)[-k:]
            output.append({'dataset':dataset,'method':method,'n':len(idx),'n_noise':int(y.sum()),'auc':auc(y,score),'p_at_10':float(y[order].mean()),'random_p':float(y.mean())})
    return pd.DataFrame(output)


def early_detection_sweep(root: str | Path, tag: str, n_epochs: int = 5) -> dict[str, pd.DataFrame]:
    """Every existing detection feature needs the full trajectory (converge_epoch,
    loss_slope, ...) to compute, so it only has forensic value after training
    finishes. This truncates build_table() to epochs [0..k] for each k and reruns
    unsupervised_metrics/memorization_score, to see how fast per-type detection
    AUC approaches its full-trajectory ceiling — if 1-2 epochs already gets close,
    noisy samples could be flagged and dropped mid-training instead of post-hoc."""
    excluded={'sample_id','dataset','noise_type','category','noise_label'}
    unsup_rows,memo_rows=[],[]
    for cutoff in range(n_epochs):
        frame=build_table(root,tag,max_epoch=cutoff)
        # a single-epoch trajectory has no spread: pandas .std() over 1 column is
        # NaN (ddof=1), which would otherwise wipe out every row via dropna below.
        # 0 spread is the correct value for a lone observation, not a missing one.
        for c in [c for c in frame.columns if c.endswith('_std')]: frame[c]=frame[c].fillna(0)
        numeric=[c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
        # Restrict to features with full coverage at EVERY cutoff (core loss/grad_norm/
        # cos_ref trajectory stats): the diagnostic/token-hard columns only cover a
        # ~900-row subsample AND are undefined with <2 epochs seen (e.g. hard_pos_jaccard
        # needs a pair of epochs), so using analyze.py's default "all numeric columns"
        # would compare different populations across cutoffs instead of the same
        # detection task with less training history.
        core=[c for c in numeric if frame[c].notna().all()]
        u=unsupervised_metrics(frame,features=core); u.insert(0,'max_epoch',cutoff); unsup_rows.append(u)
        m=memorization_score(frame); m.insert(0,'max_epoch',cutoff); memo_rows.append(m)
    return {'unsupervised':pd.concat(unsup_rows,ignore_index=True),'memorization':pd.concat(memo_rows,ignore_index=True)}


def memorization_score(frame: pd.DataFrame, feats_sign=None) -> pd.DataFrame:
    """Reproduction of the signed memorization rule on the current per-sample
    table. Expect AUC below 0.5 on non-memorized families (garbled/unrelated
    get HARDER under this rule, not easier) — that asymmetry is the point,
    not a bug, since the direction is never re-fit per dataset."""
    feats_sign=feats_sign or MEMO_FEATS
    variants={'memo_signed':feats_sign,'low_loss_only':{'loss_mean':-1}}
    rows=[]
    for dataset in sorted(frame.dataset.unique()):
        if dataset=='clean': continue
        sub_ds=frame[frame.dataset==dataset]
        for name,fs in variants.items():
            cols=[c for c in fs if c in sub_ds.columns]
            sub=sub_ds.dropna(subset=cols)
            if sub.empty or not cols: continue
            x=sub[cols].to_numpy(float)
            med=np.median(x,axis=0); mad=np.median(np.abs(x-med),axis=0); mad=np.where(mad==0,1,mad)
            z=(x-med)/(1.4826*mad)
            signs=np.array([fs[c] for c in cols],dtype=float)
            score=(z*signs).mean(axis=1)
            y=(sub.noise_type!='none').astype(int).to_numpy()
            if len(set(y))<2 or y.sum()<10: continue
            rows.append({'dataset':dataset,'scorer':name,'n':len(y),'n_noise':int(y.sum()),
                        'auc':roc_auc_score(y,score),'p_at_10':precision_at_k(y,score,0.10),
                        'random_p':float(y.mean())})
    return pd.DataFrame(rows)


def precision_lift_table(path: str | Path) -> pd.DataFrame:
    """P@10% cleaning-precision vs. random baseline, per dataset/method — the
    lift a practitioner actually gets from spending a 10% review budget on an
    unsupervised score, as opposed to the AUC that scores the whole ranking
    (a high AUC can still leave the top-10% slice barely better than random,
    e.g. keyword/near_duplicate below)."""
    frame=pd.read_csv(path)
    frame=frame[frame.dataset!='clean'].copy()
    frame['lift']=frame.p_at_10/frame.random_p.replace(0,np.nan)
    return frame.sort_values(['dataset','lift'],ascending=[True,False]).reset_index(drop=True)


def pooled_scorer_compare(frame: pd.DataFrame, dataset='mixed', seed=42) -> pd.DataFrame:
    """Compares cleaning_loop.py's three scorers (iforest/memo_signed/pooled),
    plus the raw text_nn_sim |z| leg on its own for attribution, on one
    dataset — overall removal quality (AUC/P@10%) and a per-type slice (type vs.
    clean) showing which corruptions each leg is blind to. Reuses
    cleaning_loop.py's internal _score_* helpers directly rather than
    reimplementing them, so this can never silently drift from what `cli.py
    clean` actually does."""
    import cleaning_loop as cl
    excluded={'sample_id','dataset','noise_type','category','noise_label'}
    sub=frame[frame.dataset==dataset].reset_index(drop=True)
    numeric=[c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
    features=[c for c in numeric if sub[c].notna().all()]
    scorers={
        'iforest':lambda: cl._score_iforest(sub[features].to_numpy(float),seed),
        'memo_signed':lambda: cl._score_memo_signed(sub,features)[0],
        'text_abs_z':lambda: np.abs(_zs(sub['text_nn_sim'].to_numpy(float))),
        'pooled':lambda: cl._score_pooled(sub,features,seed)[0],
    }
    noise_type=sub.noise_type.to_numpy(); y=(noise_type!='none').astype(int)
    types=[t for t in sorted(set(noise_type.tolist())) if t!='none']
    rows=[]
    for name,fn in scorers.items():
        score=fn()
        rows.append({'scorer':name,'scope':'overall','target_type':'any','n':len(y),'n_noise':int(y.sum()),
                    'auc':roc_auc_score(y,score),'p_at_10':precision_at_k(y,score,0.10),'random_p':float(y.mean())})
        for t in types:
            keep=(noise_type==t)|(noise_type=='none')
            y_t=(noise_type[keep]==t).astype(int)
            rows.append({'scorer':name,'scope':'per_type','target_type':t,'n':int(keep.sum()),'n_noise':int(y_t.sum()),
                        'auc':roc_auc_score(y_t,score[keep]),'p_at_10':precision_at_k(y_t,score[keep],0.10),
                        'random_p':float(y_t.mean())})
    return pd.DataFrame(rows)
