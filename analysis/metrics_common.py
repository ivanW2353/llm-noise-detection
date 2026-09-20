"""Shared low-level primitives: AUC helpers, CV/label-free scoring routines, and the
feature-name constants every metrics/detection module in this package builds on."""
from __future__ import annotations
import numpy as np
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


def precision_at_k(y_true, scores, budget):
    y_true=np.asarray(y_true); scores=np.asarray(scores)
    k=int(round(budget*len(y_true)))
    if k==0: return float('nan')
    return float(y_true[np.argsort(scores)[-k:]].mean())


# Signed hyper-typicality features for MEMORIZED noise (duplicate/template):
# such samples are not atypical, they are *hyper-typical* (low loss, fast
# convergence), so an unsigned |z| outlier score misses them entirely. Sign
# is fixed a priori by the memorization hypothesis, not fitted per dataset —
# that is what keeps this label-free.
MEMO_FEATS={'loss_mean':-1,'loss_last':-1,'loss_std':-1,'loss_curvature':-1,'converge_epoch':-1,'grad_norm_mean':-1}
