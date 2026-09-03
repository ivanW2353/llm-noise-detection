import numpy as np
from sklearn.metrics import roc_auc_score
def auc(y,scores):
    return float('nan') if len(set(y))<2 else float(max(roc_auc_score(y,scores),1-roc_auc_score(y,scores)))
def summarize(frame,features):
    out=[]
    for f in features:
        if f in frame: out.append({'feature':f,'auc':auc(frame.noise_type.ne('none'),frame[f].fillna(frame[f].median()))})
    return out
