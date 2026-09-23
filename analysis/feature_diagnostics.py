"""Feature-level diagnostics: attribution, correlation/redundancy, minimal sets, ablations."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics_common import auc, _cv_auc, _if_auc, FULL_COVERAGE_FEATS, FEATURE_FAMILY, DIAG_COLS, TOKEN_COLS
from .detect_unsupervised import unsupervised_metrics


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
    from sklearn.preprocessing import StandardScaler
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
    from sklearn.preprocessing import StandardScaler
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


def _if_score(x: np.ndarray, seed: int = 42) -> np.ndarray:
    """Label-free IsolationForest anomaly score (higher = more anomalous),
    the same fit `_if_auc` uses internally but returning the score itself
    instead of collapsing it against labels."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    xs = StandardScaler().fit_transform(x)
    return -IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(xs).score_samples(xs)


def _bimodality_dip(score: np.ndarray) -> float:
    """Label-free bimodality proxy: Hartigan & Hartigan's dip statistic for
    unimodality (higher = more evidence of a second mode). Unlike a
    top-fraction-vs-rest gap, the dip test is sensitive to the actual *shape*
    of the distribution rather than just how extreme its tail is, so it
    should not fire on every feature set's long right tail the way a raw
    gap does — a feature set where noise forms a genuinely separate cluster
    should score higher than one where the IsolationForest score is merely
    skewed."""
    import diptest
    return float(diptest.diptest(np.sort(score), sort_x=False)[0])


def label_free_feature_set(frame: pd.DataFrame, features=None, max_k=6, seed=42) -> pd.DataFrame:
    """Same greedy-forward search as minimal_feature_set(), but the per-step
    selection criterion never touches noise_type. minimal_feature_set() picks
    the feature that most improves AUC against the label; this picks the
    feature that most improves `_bimodality_dip()` — Hartigan's dip
    statistic on the IsolationForest score distribution. Labels are used
    only afterward, to report the resulting AUC for each step so it can be
    compared against minimal_feature_set()'s label-driven result — never for
    selection itself. This is the label-free-realizable counterpart to
    minimal_feature_set()'s 'iforest' route; there is no counterpart to its
    'rf' route, since a label-free recipe has no supervised classifier to
    run.

    v1 used a top-10%-vs-rest score gap instead of the dip statistic and it
    failed: the gap barely moved across candidate feature sets (any
    IsolationForest score has a long right tail regardless of whether that
    tail is real noise), so the greedy search was effectively picking at
    random — auc_dir landed near 0.50-0.60 for 5 of 8 dolly-ratio10 datasets
    vs. 0.65-0.87 for the label-driven baseline. The dip statistic is tried
    here because it is sensitive to distribution *shape*, not tail extremity.

    `converge_epoch` is excluded from the candidate pool: it is the one
    FULL_COVERAGE_FEATS column with only ~6 distinct values (all others have
    tens of thousands), and on IsolationForest scores this coarse a dip test
    fires on the discreteness itself, not on real bimodal separation — on
    clean-only dolly-ratio10 data (no injected noise, so no genuine structure
    to find) it alone scores dip=0.10 vs 0.01-0.03 for every continuous
    feature. Left in, it wins every greedy first step and actively hurts
    `template` (auc_dir 0.505, i.e. random) by crowding out the
    loss_last/loss_min/cos_ref_slope combination that actually detects it
    (auc_dir 0.775 once converge_epoch is excluded)."""
    feats = list(features or [c for c in FULL_COVERAGE_FEATS if c in frame.columns])
    feats = [c for c in feats if c != 'converge_epoch']
    rows = []
    for ds in sorted(frame.dataset.unique()):
        if ds == 'clean':
            continue
        sub = frame[frame.dataset == ds].dropna(subset=feats).reset_index(drop=True)
        y = sub.noise_type.fillna('none').ne('none').astype(int).to_numpy()
        if y.sum() < 10:
            continue
        x = sub[feats].to_numpy(float)
        full_score = _if_score(x, seed)
        full_dip = _bimodality_dip(full_score)
        full_auc = auc(y, full_score)
        chosen = []
        for step in range(min(max_k, len(feats))):
            best = None
            for j in range(len(feats)):
                if j in chosen:
                    continue
                score = _if_score(x[:, chosen + [j]], seed)
                dip = _bimodality_dip(score)
                if best is None or dip > best[0]:
                    best = (dip, j, score)
            dip, j, score = best
            chosen.append(j)
            rows.append({'dataset': ds, 'n': len(y), 'n_noise': int(y.sum()),
                        'full_dip': full_dip, 'full_auc_dir': max(full_auc, 1 - full_auc),
                        'k': step + 1, 'added': feats[j], 'dip': dip,
                        'auc_dir': max(auc(y, score), 1 - auc(y, score)),
                        'features': '+'.join(feats[c] for c in chosen)})
    return pd.DataFrame(rows)


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
               for r in read_rows(root / 'datasets' / tag / dataset / 'train.jsonl')}
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
