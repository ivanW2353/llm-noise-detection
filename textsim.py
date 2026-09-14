from pathlib import Path
from data import Jsonl

def nn_similarity(rows):
    """TF-IDF nearest-neighbor cosine similarity per sample (excludes self)."""
    rows=list(rows)
    texts=[(r.messages[0].get('content','') if r.messages else '')+' '+(r.messages[-1].get('content','') if r.messages else '') for r in rows]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=min(10,len(texts)),sublinear_tf=True,max_features=200_000)
    x=vec.fit_transform(texts)
    nn=NearestNeighbors(n_neighbors=2,metric='cosine').fit(x)
    dist,_=nn.kneighbors(x)
    sim=1.0-dist[:,1]
    return {r.id:float(v) for r,v in zip(rows,sim)}

def text_nn_sim(path):
    return nn_similarity(Jsonl(path).rows())
