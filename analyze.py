"""Post-hoc analysis of training, token, unsupervised and transfer metrics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


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


# Diagnostic-layer columns worth keeping: the *_std/*_curv second-order variants of
# these were checked against results/*/feature_exploration.csv and land at AUC
# 0.43-0.60 (near-random) for every noise type, so they are not recomputed here.
DIAG_COLS=['max_token_loss','frac_hard','user_loss','entropy','token_loss_skew','token_loss_kurt']
TOKEN_COLS=['n_hard','hard_loss_mean','hard_loss_max','hard_pos_peak','hard_pos_std_mean','hard_id_uniq','hard_pos_jaccard']

# The 19 trajectory+text features with full coverage across the entire corpus
# (unlike DIAG_COLS/TOKEN_COLS, which only populate a ~900-row/dataset
# diagnostic subsample) — the pool cleaning_loop.py and every full-corpus
# methodology function below actually score against.
FULL_COVERAGE_FEATS=['text_nn_sim','loss_mean','loss_last','loss_std','loss_slope','loss_min',
                     'converge_epoch','loss_curvature','loss_rank','grad_norm_mean','grad_norm_last',
                     'grad_norm_std','grad_norm_slope','cos_ref_mean','cos_ref_last','cos_ref_std',
                     'cos_ref_slope','grad_norm_cv','update_contrib_mean']

# Which raw signal each full-coverage feature is derived from — used to test
# whether redundancy follows family boundaries (it should: four views of one
# loss curve are not four independent measurements).
FEATURE_FAMILY={
    'text_nn_sim':'text',
    'loss_mean':'loss','loss_last':'loss','loss_std':'loss','loss_slope':'loss',
    'loss_min':'loss','converge_epoch':'loss','loss_curvature':'loss','loss_rank':'loss',
    'grad_norm_mean':'grad','grad_norm_last':'grad','grad_norm_std':'grad',
    'grad_norm_slope':'grad','grad_norm_cv':'grad',
    'cos_ref_mean':'cos','cos_ref_last':'cos','cos_ref_std':'cos','cos_ref_slope':'cos',
    'update_contrib_mean':'update',
}


def _zs(v: np.ndarray) -> np.ndarray:
    return (v-v.mean())/(v.std()+1e-12)


def _cv_auc(x, y, seed=0):
    """Supervised out-of-fold AUC: StratifiedKFold(5) + RandomForestClassifier."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    xs=StandardScaler().fit_transform(x); oof=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(xs,y):
        clf=RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=-1)
        clf.fit(xs[tr],y[tr]); oof[te]=clf.predict_proba(xs[te])[:,1]
    return roc_auc_score(y,oof)


def _if_auc(x, y, seed=42):
    """Label-free AUC: per-fit IsolationForest score_samples on standardized features."""
    xs=StandardScaler().fit_transform(x)
    score=-IsolationForest(n_estimators=300,random_state=seed,n_jobs=-1).fit(xs).score_samples(xs)
    return roc_auc_score(y,score)


def _fit_transfer(xtr, ytr, xte, yte, seed=0):
    """LR + RF, best of the two by AUC, scaler fit on the source only."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    sc=StandardScaler().fit(xtr); best_auc,best_score=0.0,None
    for clf in (LogisticRegression(max_iter=2000),RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=-1)):
        clf.fit(sc.transform(xtr),ytr); score=clf.predict_proba(sc.transform(xte))[:,1]; a=roc_auc_score(yte,score)
        if a>best_auc: best_auc,best_score=a,score
    return best_auc,best_score

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


def transfer_metrics(path: str | Path, tags=None) -> pd.DataFrame:
    frame=pd.read_csv(path)
    if tags and 'train_tag' in frame: frame=frame[frame.train_tag.isin(tags) | frame.test_tag.isin(tags)]
    if tags and 'tag' in frame and 'train_tag' not in frame: frame=frame[frame.tag.isin(tags)]
    return frame.reset_index(drop=True)


def precision_at_k(y_true, scores, budget):
    y_true=np.asarray(y_true); scores=np.asarray(scores)
    k=int(round(budget*len(y_true)))
    if k==0: return float('nan')
    return float(y_true[np.argsort(scores)[-k:]].mean())


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


# Signed hyper-typicality features for MEMORIZED noise (duplicate/template):
# such samples are not atypical, they are *hyper-typical* (low loss, fast
# convergence), so an unsigned |z| outlier score misses them entirely. Sign
# is fixed a priori by the memorization hypothesis, not fitted per dataset —
# that is what keeps this label-free.
MEMO_FEATS={'loss_mean':-1,'loss_last':-1,'loss_std':-1,'loss_curvature':-1,'converge_epoch':-1,'grad_norm_mean':-1}

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


def cross_type_transfer(frame: pd.DataFrame, features=None, seed=0) -> pd.DataFrame:
    """Does a detector trained on one noise type work on another? For each
    ordered pair of types, fit on the source (LR + RF, best of the two) and
    score the target; the diagonal is the within-type 5-fold CV AUC, the
    reference ceiling. A high off-diagonal means the features capture 'is
    anomalous' rather than 'is this specific corruption' — i.e. detectors do
    or don't generalize across noise families."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    excluded={'sample_id','dataset','noise_type','category','noise_label'}
    features=list(features or [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])])
    features=[c for c in features if c in frame]

    def cv_auc(x,y):
        xs=StandardScaler().fit_transform(x); oof=np.zeros(len(y))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(xs,y):
            clf=RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=-1)
            clf.fit(xs[tr],y[tr]); oof[te]=clf.predict_proba(xs[te])[:,1]
        return roc_auc_score(y,oof)

    def fit_transfer(xtr,ytr,xte,yte):
        sc=StandardScaler().fit(xtr); best_auc,best_score=0.0,None
        for clf in (LogisticRegression(max_iter=2000),RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=-1)):
            clf.fit(sc.transform(xtr),ytr); score=clf.predict_proba(sc.transform(xte))[:,1]; a=roc_auc_score(yte,score)
            if a>best_auc: best_auc,best_score=a,score
        return best_auc,best_score

    cache={}
    for ds in sorted(frame.dataset.unique()):
        if ds in ('clean','mixed'): continue
        sub=frame[frame.dataset==ds].dropna(subset=features)
        y=(sub.noise_type!='none').astype(int).to_numpy()
        if y.sum()>=10: cache[ds]=(sub[features].to_numpy(float),y)
    own={ds:cv_auc(x,y) for ds,(x,y) in cache.items()}

    rows=[]
    for dst,(xt,yt) in cache.items():
        for src,(xs_,ys_) in cache.items():
            if src==dst:
                a,p10=own[dst],float('nan')
            else:
                a,score=fit_transfer(xs_,ys_,xt,yt); p10=precision_at_k(yt,score,0.10)
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    excluded={'sample_id','dataset','noise_type','category','noise_label'}
    shared_cols=set(frame_a.columns)&set(frame_b.columns)
    features=list(features or [c for c in shared_cols if c not in excluded
                  and pd.api.types.is_numeric_dtype(frame_a[c]) and pd.api.types.is_numeric_dtype(frame_b[c])])

    def cv_auc(x,y):
        xs=StandardScaler().fit_transform(x); oof=np.zeros(len(y))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(xs,y):
            clf=RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=-1)
            clf.fit(xs[tr],y[tr]); oof[te]=clf.predict_proba(xs[te])[:,1]
        return roc_auc_score(y,oof)

    def fit_transfer(xtr,ytr,xte,yte):
        sc=StandardScaler().fit(xtr); best_auc,best_score=0.0,None
        for clf in (LogisticRegression(max_iter=2000),RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=-1)):
            clf.fit(sc.transform(xtr),ytr); score=clf.predict_proba(sc.transform(xte))[:,1]; a=roc_auc_score(yte,score)
            if a>best_auc: best_auc,best_score=a,score
        return best_auc,best_score

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
    own_a={ds:cv_auc(*cache_a[ds]) for ds in shared}
    own_b={ds:cv_auc(*cache_b[ds]) for ds in shared}

    rows=[]
    for ds in shared:
        xa,ya=cache_a[ds]; xb,yb=cache_b[ds]
        rows.append({'noise_type':ds,'train_tag':tag_a,'test_tag':tag_a,'auc':own_a[ds],
                    'within_tag_auc':own_a[ds],'retention':1.0,'p_at_10':float('nan'),
                    'random_p':float(ya.mean()),'is_diagonal':True})
        rows.append({'noise_type':ds,'train_tag':tag_b,'test_tag':tag_b,'auc':own_b[ds],
                    'within_tag_auc':own_b[ds],'retention':1.0,'p_at_10':float('nan'),
                    'random_p':float(yb.mean()),'is_diagonal':True})
        a_to_b,score_b=fit_transfer(xa,ya,xb,yb)
        rows.append({'noise_type':ds,'train_tag':tag_a,'test_tag':tag_b,'auc':a_to_b,
                    'within_tag_auc':own_b[ds],'retention':(a_to_b/own_b[ds] if own_b[ds] else None),
                    'p_at_10':precision_at_k(yb,score_b,0.10),'random_p':float(yb.mean()),'is_diagonal':False})
        b_to_a,score_a=fit_transfer(xb,yb,xa,ya)
        rows.append({'noise_type':ds,'train_tag':tag_b,'test_tag':tag_a,'auc':b_to_a,
                    'within_tag_auc':own_a[ds],'retention':(b_to_a/own_a[ds] if own_a[ds] else None),
                    'p_at_10':precision_at_k(ya,score_a,0.10),'random_p':float(ya.mean()),'is_diagonal':False})
    return pd.DataFrame(rows)


def feature_attribution(frame: pd.DataFrame, features=None, seed=0, n_repeats=20) -> pd.DataFrame:
    """cross_type_transfer() already shows the per-type RF detector's AUC; this
    asks not 'can we detect' but 'what is the detector looking at'. Fits one RF
    per dataset under the same 5-fold CV split as cross_type_transfer and
    averages sklearn's permutation_importance (AUC drop when a feature is
    shuffled on the held-out fold) across folds — more trustworthy than the
    RF's built-in impurity-based feature_importances_, which is biased toward
    high-cardinality/continuous features regardless of whether they are
    actually predictive, and would not transfer the CV-fold-averaging discipline
    already used everywhere else in this module."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.inspection import permutation_importance
    excluded={'sample_id','dataset','noise_type','category','noise_label'}
    features=list(features or [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])])
    features=[c for c in features if c in frame]

    rows=[]
    for ds in sorted(frame.dataset.unique()):
        if ds in ('clean','mixed'): continue
        sub=frame[frame.dataset==ds].dropna(subset=features)
        y=(sub.noise_type!='none').astype(int).to_numpy()
        if y.sum()<10: continue
        x=sub[features].to_numpy(float)
        xs=StandardScaler().fit_transform(x)
        fold_importances=[]
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(xs,y):
            clf=RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=-1)
            clf.fit(xs[tr],y[tr])
            r=permutation_importance(clf,xs[te],y[te],n_repeats=n_repeats,random_state=seed,scoring='roc_auc',n_jobs=-1)
            fold_importances.append(r.importances_mean)
        mean_imp=np.mean(fold_importances,axis=0); std_imp=np.std(fold_importances,axis=0)
        for feat,imp,sd in zip(features,mean_imp,std_imp):
            rows.append({'dataset':ds,'feature':feat,'importance':float(imp),'importance_std':float(sd),
                        'n':len(y),'n_noise':int(y.sum())})
    result=pd.DataFrame(rows)
    if result.empty: return result
    result['rank']=result.groupby('dataset')['importance'].rank(ascending=False,method='first').astype(int)
    return result.sort_values(['dataset','rank']).reset_index(drop=True)


def _vif(x: np.ndarray, names: list[str]) -> pd.Series:
    """Variance inflation factor: 1/(1-R^2) of regressing each column on the
    rest. Above ~10 conventionally signals near-collinearity."""
    out={}
    for j,name in enumerate(names):
        others=np.delete(x,j,axis=1)
        a=np.column_stack([np.ones(len(others)),others])
        coef,*_=np.linalg.lstsq(a,x[:,j],rcond=None)
        resid=x[:,j]-a@coef
        ss_tot=float(((x[:,j]-x[:,j].mean())**2).sum())
        r2=1-float((resid**2).sum())/ss_tot if ss_tot>0 else 0.0
        out[name]=1/(1-r2) if r2<1-1e-12 else np.inf
    return pd.Series(out)


def feature_correlation(frame: pd.DataFrame, features=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How much do the full-coverage features overlap? Measured three ways per
    dataset and pooled: Spearman correlation between every pair, effective
    dimensionality via PCA (components to reach 90%/99% variance), and VIF per
    feature. Backs the "highly correlated" claim behind RF's single-feature-drop
    insensitivity and IF's dimensional-dilution improvement (feature_attribution
    / single_feature_ablation). Returns (per-dataset summary, pooled pairwise
    correlations) — two tables, since the pairwise detail is only meaningful
    pooled across the whole corpus."""
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    feats=list(features or [c for c in FULL_COVERAGE_FEATS if c in frame.columns])
    rows,pairs=[],[]
    datasets=[d for d in sorted(frame.dataset.unique()) if d!='clean']
    for ds in datasets+['__pooled__']:
        sub=frame if ds=='__pooled__' else frame[frame.dataset==ds]
        sub=sub.dropna(subset=feats)
        if len(sub)<50: continue
        x=StandardScaler().fit_transform(sub[feats].to_numpy(float))
        rho,_=spearmanr(x); abs_rho=np.abs(rho); off=abs_rho[np.triu_indices_from(abs_rho,k=1)]
        p=PCA().fit(x); cum=np.cumsum(p.explained_variance_ratio_)
        k90=int(np.searchsorted(cum,0.90)+1); k99=int(np.searchsorted(cum,0.99)+1)
        ev=p.explained_variance_ratio_; eff=float(1.0/(ev**2).sum())
        v=_vif(x,feats)
        rows.append({'dataset':ds,'n':len(sub),'n_features':len(feats),
                    'mean_abs_spearman':float(off.mean()),
                    'frac_pairs_over_0.7':float((off>0.7).mean()),
                    'frac_pairs_over_0.9':float((off>0.9).mean()),
                    'pca_k_for_90pct':k90,'pca_k_for_99pct':k99,
                    'effective_dims':eff,'pc1_variance':float(ev[0]),
                    'median_vif':float(v.replace(np.inf,np.nan).median()),
                    'n_vif_over_10':int((v>10).sum())})
        if ds=='__pooled__':
            for i in range(len(feats)):
                for j in range(i+1,len(feats)):
                    pairs.append({'a':feats[i],'b':feats[j],
                                 'family_a':FEATURE_FAMILY.get(feats[i],'?'),
                                 'family_b':FEATURE_FAMILY.get(feats[j],'?'),
                                 'same_family':FEATURE_FAMILY.get(feats[i])==FEATURE_FAMILY.get(feats[j]),
                                 'abs_spearman':float(abs_rho[i,j])})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def minimal_feature_set(frame: pd.DataFrame, features=None, max_k=6, seed=0, if_seed=42) -> pd.DataFrame:
    """Greedy forward feature selection per dataset/route: starting from
    nothing, repeatedly add whichever remaining feature most improves AUC, up
    to max_k. Answers "what is the minimal set to collect", as opposed to
    single_feature_ablation's "what is redundant to drop" — a group of
    features can each be individually removable while jointly necessary.
    Selection uses the same labels it is scored on, so the reported AUC per k
    is optimistic (an upper bound on what a small set CAN carry, not a
    label-free recipe for picking one)."""
    feats=list(features or [c for c in FULL_COVERAGE_FEATS if c in frame.columns])
    rows=[]
    for ds in sorted(frame.dataset.unique()):
        if ds=='clean': continue
        sub=frame[frame.dataset==ds].dropna(subset=feats).reset_index(drop=True)
        y=sub.noise_type.fillna('none').ne('none').astype(int).to_numpy()
        if y.sum()<10: continue
        x=sub[feats].to_numpy(float)
        full={'rf':_cv_auc(x,y,seed),'iforest':_if_auc(x,y,if_seed)}
        routes=(('rf',lambda xx,yy:_cv_auc(xx,yy,seed),False),('iforest',lambda xx,yy:_if_auc(xx,yy,if_seed),True))
        for route,scorer,directed in routes:
            chosen=[]
            for step in range(min(max_k,len(feats))):
                best=None
                for j in range(len(feats)):
                    if j in chosen: continue
                    a=scorer(x[:,chosen+[j]],y)
                    key=max(a,1-a) if directed else a
                    if best is None or key>best[0]: best=(key,a,j)
                key,raw,j=best; chosen.append(j)
                rows.append({'dataset':ds,'route':route,'n':len(y),'n_noise':int(y.sum()),
                            'full_auc':full[route],'full_auc_dir':max(full[route],1-full[route]),
                            'k':step+1,'added':feats[j],'auc':raw,'auc_dir':max(raw,1-raw),
                            'features':'+'.join(feats[c] for c in chosen)})
    return pd.DataFrame(rows)


def single_feature_ablation(frame: pd.DataFrame, features=None, seed=0, if_seed=42) -> pd.DataFrame:
    """Single-feature leave-one-out ablation across both detection routes ('rf':
    supervised StratifiedKFold(5)+RandomForestClassifier; 'iforest': label-free
    per-dataset IsolationForest). delta = ablated_auc - full_auc: negative means
    dropping the feature hurt (it carried signal), positive means dropping it
    helped (it was diluting an undirected score)."""
    feats=list(features or [c for c in FULL_COVERAGE_FEATS if c in frame.columns])
    rows=[]
    for ds in sorted(frame.dataset.unique()):
        if ds=='clean': continue
        sub=frame[frame.dataset==ds].dropna(subset=feats).reset_index(drop=True)
        y=sub.noise_type.fillna('none').ne('none').astype(int).to_numpy()
        if y.sum()<10: continue
        x=sub[feats].to_numpy(float)
        full={'rf':_cv_auc(x,y,seed),'iforest':_if_auc(x,y,if_seed)}
        for route,base in full.items():
            rows.append({'dataset':ds,'route':route,'dropped':'(none)','n':len(y),
                        'n_noise':int(y.sum()),'auc':base,'full_auc':base,'delta':0.0})
        for i,f in enumerate(feats):
            xr=np.delete(x,i,axis=1)
            for route,fn in (('rf',lambda xx,yy:_cv_auc(xx,yy,seed)),('iforest',lambda xx,yy:_if_auc(xx,yy,if_seed))):
                a=fn(xr,y)
                rows.append({'dataset':ds,'route':route,'dropped':f,'n':len(y),'n_noise':int(y.sum()),
                            'auc':a,'full_auc':full[route],'delta':a-full[route]})
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


def feature_group_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    """Reruns unsupervised_metrics() under restricted feature-group conditions
    (full_diag / full_coverage / no_text / text_only / token_only) to quantify
    how much of the reported label-free AUC survives when token-level
    diagnostics (subsample-only, unavailable to cleaning_loop.py's full-dataset
    scoring) or text_nn_sim are removed. All conditions score the same
    diagnostic-subsample population, so AUCs are directly comparable across
    conditions for a given dataset."""
    no_text=[c for c in FULL_COVERAGE_FEATS if c!='text_nn_sim']
    conditions={'full_diag':None,'full_coverage':FULL_COVERAGE_FEATS,'no_text':no_text,
               'text_only':['text_nn_sim'],'token_only':DIAG_COLS+TOKEN_COLS}
    rows=[]
    for cond,feats in conditions.items():
        out=unsupervised_metrics(frame,features=feats)
        out.insert(0,'condition',cond)
        rows.append(out)
    table=pd.concat(rows,ignore_index=True)
    return table[table.dataset!='clean'].reset_index(drop=True)


def length_confound(frame: pd.DataFrame, root: str | Path, tag: str, dataset: str = 'wild',
                    features=None, seed: int = 42, n_bins: int = 10) -> pd.DataFrame:
    """Wild-noise length confound: OASST `quality` correlates with response
    length (see wild_data.py's docstring/summarize()), and length is trivially
    visible to every feature derived from the loss curve (a short reply gets
    fewer label tokens, a different loss trajectory shape, etc.), so any
    wild-noise AUC needs to state how much of it survives once length is
    controlled for. Reports, for the pooled cleaning_loop.py detector score
    and for each individual FULL_COVERAGE_FEATS column, four numbers side by
    side:
      raw_auc        - AUC of the score/feature against the noise label, as-is.
      length_auc     - AUC of response length alone (the confound's own ceiling,
                       same for every row since it does not depend on `signal`).
      residual_auc   - AUC after linearly regressing the score on length and
                       scoring the residual.
      stratified_auc - AUC computed within length-decile bins and averaged
                       (weighted by bin size), a nonparametric correction that
                       does not assume the score-vs-length relationship is linear.
    A raw_auc close to length_auc, with residual_auc/stratified_auc near 0.5,
    means the signal is mostly rediscovering "short replies are rated worse"
    rather than detecting anything about training dynamics."""
    import cleaning_loop as cl
    from data import read as read_rows
    root = Path(root)
    sub = frame[frame.dataset == dataset].reset_index(drop=True).copy()
    resp_len = {r.id: len(r.messages[-1].get('content', '')) if r.messages else 0
               for r in read_rows(root / 'data' / tag / dataset / 'train.jsonl')}
    sub['response_len'] = sub.sample_id.astype(str).map(resp_len)
    sub = sub.dropna(subset=['response_len']).reset_index(drop=True)
    y = sub.noise_type.fillna('none').ne('none').astype(int).to_numpy()
    length = sub['response_len'].to_numpy(float)

    feats = list(features or [c for c in FULL_COVERAGE_FEATS if c in sub.columns])
    core = [c for c in feats if sub[c].notna().all()]
    scores = {f: sub[f].to_numpy(float) for f in core}
    if core:
        scores = {'pooled_detector': cl._score_pooled(sub, core, seed)[0], **scores}

    def residual_auc(score):
        a = np.column_stack([np.ones_like(length), length])
        coef, *_ = np.linalg.lstsq(a, score, rcond=None)
        return auc(y, score - a @ coef)

    def stratified_auc(score):
        order = np.argsort(length); bins = np.array_split(order, n_bins)
        aucs, weights = [], []
        for b in bins:
            yb = y[b]
            if len(set(yb)) < 2: continue
            aucs.append(auc(yb, score[b])); weights.append(len(b))
        return float(np.average(aucs, weights=weights)) if aucs else float('nan')

    length_auc_value = auc(y, length)
    rows = []
    for name, score in scores.items():
        rows.append({'dataset': dataset, 'signal': name, 'n': len(y), 'n_noise': int(y.sum()),
                    'raw_auc': auc(y, score), 'length_auc': length_auc_value,
                    'residual_auc': residual_auc(score), 'stratified_auc': stratified_auc(score)})
    return pd.DataFrame(rows)


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
