"""Cross-condition transfer/generalization evaluation (cross-type, cross-ratio, transfer to mixed)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .metrics_common import _zs, _cv_auc, _fit_transfer, precision_at_k


def transfer_metrics(path: str | Path, tags=None) -> pd.DataFrame:
    frame=pd.read_csv(path)
    if tags and 'train_tag' in frame: frame=frame[frame.train_tag.isin(tags) | frame.test_tag.isin(tags)]
    if tags and 'tag' in frame and 'train_tag' not in frame: frame=frame[frame.tag.isin(tags)]
    return frame.reset_index(drop=True)


def cross_type_transfer(frame: pd.DataFrame, features=None, seed=0) -> pd.DataFrame:
    """Does a detector trained on one noise type work on another? For each
    ordered pair of types, fit on the source (LR + RF, best of the two) and
    score the target; the diagonal is the within-type 5-fold CV AUC, the
    reference ceiling. A high off-diagonal means the features capture 'is
    anomalous' rather than 'is this specific corruption' — i.e. detectors do
    or don't generalize across noise families."""
    excluded={'sample_id','dataset','noise_type','category','noise_label'}
    features=list(features or [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])])
    features=[c for c in features if c in frame]

    cache={}
    for ds in sorted(frame.dataset.unique()):
        if ds in ('clean','mixed'): continue
        sub=frame[frame.dataset==ds].dropna(subset=features)
        y=(sub.noise_type!='none').astype(int).to_numpy()
        if y.sum()>=10: cache[ds]=(sub[features].to_numpy(float),y)
    own={ds:_cv_auc(x,y,seed) for ds,(x,y) in cache.items()}

    rows=[]
    for dst,(xt,yt) in cache.items():
        for src,(xs_,ys_) in cache.items():
            if src==dst:
                a,p10=own[dst],float('nan')
            else:
                a,score=_fit_transfer(xs_,ys_,xt,yt,seed); p10=precision_at_k(yt,score,0.10)
            rows.append({'train_type':src,'test_type':dst,'auc':a,'auc_dir':max(a,1-a),
                        'within_type_auc':own[dst],'retention':(a/own[dst] if own[dst] else None),
                        'p_at_10':p10,'random_p':float(yt.mean()),'is_diagonal':src==dst})
    return pd.DataFrame(rows)


def cross_ratio_transfer(frame_a: pd.DataFrame, frame_b: pd.DataFrame, tag_a='ratio10', tag_b='ratio5', features=None, seed=0) -> pd.DataFrame:
    """Does a detector trained at one noise ratio transfer to another, for the
    same noise type? Pairs samples by matching `dataset` (noise type) across
    the two per-sample tables — same method as cross_type_transfer (LR + RF,
    best of the two, scaler fit on the source only) but holding the type fixed
    and varying the ratio instead. The diagonal is the within-ratio 5-fold CV
    AUC for that type. A drop off-diagonal means the detector overfit the
    noise density it was trained at rather than learning a ratio-invariant
    signal."""
    excluded={'sample_id','dataset','noise_type','category','noise_label'}
    shared_cols=set(frame_a.columns)&set(frame_b.columns)
    features=list(features or [c for c in shared_cols if c not in excluded
                  and pd.api.types.is_numeric_dtype(frame_a[c]) and pd.api.types.is_numeric_dtype(frame_b[c])])

    def build(frame):
        cache={}
        for ds in sorted(frame.dataset.unique()):
            if ds in ('clean','mixed'): continue
            sub=frame[frame.dataset==ds].dropna(subset=features)
            y=(sub.noise_type!='none').astype(int).to_numpy()
            if y.sum()>=10: cache[ds]=(sub[features].to_numpy(float),y)
        return cache

    cache_a=build(frame_a); cache_b=build(frame_b)
    shared=sorted(set(cache_a)&set(cache_b))
    own_a={ds:_cv_auc(*cache_a[ds],seed) for ds in shared}
    own_b={ds:_cv_auc(*cache_b[ds],seed) for ds in shared}

    rows=[]
    for ds in shared:
        xa,ya=cache_a[ds]; xb,yb=cache_b[ds]
        rows.append({'noise_type':ds,'train_tag':tag_a,'test_tag':tag_a,'auc':own_a[ds],
                    'within_tag_auc':own_a[ds],'retention':1.0,'p_at_10':float('nan'),
                    'random_p':float(ya.mean()),'is_diagonal':True})
        rows.append({'noise_type':ds,'train_tag':tag_b,'test_tag':tag_b,'auc':own_b[ds],
                    'within_tag_auc':own_b[ds],'retention':1.0,'p_at_10':float('nan'),
                    'random_p':float(yb.mean()),'is_diagonal':True})
        a_to_b,score_b=_fit_transfer(xa,ya,xb,yb,seed)
        rows.append({'noise_type':ds,'train_tag':tag_a,'test_tag':tag_b,'auc':a_to_b,
                    'within_tag_auc':own_b[ds],'retention':(a_to_b/own_b[ds] if own_b[ds] else None),
                    'p_at_10':precision_at_k(yb,score_b,0.10),'random_p':float(yb.mean()),'is_diagonal':False})
        b_to_a,score_a=_fit_transfer(xb,yb,xa,ya,seed)
        rows.append({'noise_type':ds,'train_tag':tag_b,'test_tag':tag_a,'auc':b_to_a,
                    'within_tag_auc':own_a[ds],'retention':(b_to_a/own_a[ds] if own_a[ds] else None),
                    'p_at_10':precision_at_k(ya,score_a,0.10),'random_p':float(ya.mean()),'is_diagonal':False})
    return pd.DataFrame(rows)


def transfer_to_mixed(frame: pd.DataFrame, seed=0) -> pd.DataFrame:
    """Evaluates each single-type detector against the `mixed` dataset — the
    question cross_type_transfer() skips by excluding mixed entirely. Reports,
    per source-type detector: overall AUC/P@10% on mixed (deployment number),
    and own-type-only AUC (own noise type vs. clean within mixed, separating
    "blind to other types" from "population-shift breakdown"). Also reports a
    calibration-free text_nn_sim z-score control, an ensemble_all7 max-pooled
    score, and ensemble_loo (leave-one-type-out ensembles, the actual "unseen
    noise type" question) plus text_nn_sim on the same held-out slices. Runs
    both the full_diag (all numeric columns) and full_coverage conditions."""
    from metrics_common import FULL_COVERAGE_FEATS
    excluded={'sample_id','dataset','noise_type','category','noise_label'}
    all_numeric=[c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
    rows=[]
    rows+=_transfer_to_mixed_condition(frame,all_numeric,'full_diag',seed)
    rows+=_transfer_to_mixed_condition(frame,[c for c in FULL_COVERAGE_FEATS if c in frame.columns],'full_coverage',seed)
    return pd.DataFrame(rows)


def _transfer_to_mixed_condition(frame, features, condition, seed=0):
    mixed=frame[frame.dataset=='mixed'].dropna(subset=features)
    y_mixed=(mixed.noise_type!='none').astype(int).to_numpy()
    x_mixed=mixed[features].to_numpy(float)
    mixed_types=mixed.noise_type.to_numpy()
    sources={}
    for ds in sorted(frame.dataset.unique()):
        if ds in ('clean','mixed'): continue
        sub=frame[frame.dataset==ds].dropna(subset=features)
        y=(sub.noise_type!='none').astype(int).to_numpy()
        if y.sum()>=10: sources[ds]=(sub[features].to_numpy(float),y)
    rows=[]; scores={}
    for src,(xs_,ys_) in sources.items():
        a,score=_fit_transfer(xs_,ys_,x_mixed,y_mixed,seed)
        scores[src]=score
        rows.append({'condition':condition,'detector':src,'scope':'overall','target_type':'any',
                    'n':len(y_mixed),'n_noise':int(y_mixed.sum()),'auc':a,
                    'p_at_10':precision_at_k(y_mixed,score,0.10),'random_p':float(y_mixed.mean())})
        keep=(mixed_types==src)|(mixed_types=='none')
        y_own=(mixed_types[keep]==src).astype(int)
        rows.append({'condition':condition,'detector':src,'scope':'own_type_only','target_type':src,
                    'n':int(keep.sum()),'n_noise':int(y_own.sum()),'auc':roc_auc_score(y_own,score[keep]),
                    'p_at_10':precision_at_k(y_own,score[keep],0.10),'random_p':float(y_own.mean())})
    text=mixed['text_nn_sim'].to_numpy(float)
    zscore=(text-text.mean())/(text.std()+1e-12)
    rows.append({'condition':condition,'detector':'text_nn_sim_zscore','scope':'overall','target_type':'any',
                'n':len(y_mixed),'n_noise':int(y_mixed.sum()),'auc':roc_auc_score(y_mixed,zscore),
                'p_at_10':precision_at_k(y_mixed,zscore,0.10),'random_p':float(y_mixed.mean())})
    stacked=np.vstack([_zs(s) for s in scores.values()]).max(axis=0)
    rows.append({'condition':condition,'detector':'ensemble_all7','scope':'overall','target_type':'any',
                'n':len(y_mixed),'n_noise':int(y_mixed.sum()),'auc':roc_auc_score(y_mixed,stacked),
                'p_at_10':precision_at_k(y_mixed,stacked,0.10),'random_p':float(y_mixed.mean())})
    for held in scores:
        pool=[_zs(s) for name,s in scores.items() if name!=held]
        ens=np.vstack(pool).max(axis=0)
        keep=(mixed_types==held)|(mixed_types=='none')
        y_held=(mixed_types[keep]==held).astype(int)
        rows.append({'condition':condition,'detector':'ensemble_loo','scope':'held_out_type','target_type':held,
                    'n':int(keep.sum()),'n_noise':int(y_held.sum()),'auc':roc_auc_score(y_held,ens[keep]),
                    'p_at_10':precision_at_k(y_held,ens[keep],0.10),'random_p':float(y_held.mean())})
        rows.append({'condition':condition,'detector':'text_nn_sim_zscore','scope':'held_out_type','target_type':held,
                    'n':int(keep.sum()),'n_noise':int(y_held.sum()),'auc':roc_auc_score(y_held,zscore[keep]),
                    'p_at_10':precision_at_k(y_held,zscore[keep],0.10),'random_p':float(y_held.mean())})
    return rows
