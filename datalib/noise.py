import re
import numpy as np
from .sample import Sample

GARBLED_CHARS = list(
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇ"
    "１２３４５６７８９０！＠＃＄％＾＆＊（）"
    "ъѫѭӓԉ҈ԆӜԒԨԬ"
    "のをんアイウエオカキクケコ"
    "㊙㊗㋡㍿ⅫⅪⅧ↯⌘℮№℗"
    "乱码测试无关联内容"
    "¤€£¥§¶©®°±×÷"
)
TEMPLATE_ANSWER = 'The answer to this question is 42.'
PERSON_NAMES = ['Jonathan Miller','Amanda Chen','Robert Blackwell','Sofia Reyes','David Okafor','Emily Zhang','Marcus Johnson','Lena Petrova','Oliver Hughes','Priya Sharma','Daniel Kowalski','Hannah Fischer','Thomas Nguyen','Isabella Rossi','Samuel Osei','Grace Kim','Lucas Moreau','Ava Patel','Nathan Brooks','Zoe Lindqvist']
ORGS = ['Acme Corporation','Global Dynamics','Vertex Industries','Northbridge Group','Helios Technologies','Crestline Partners','Falcon Systems','Meridian Health','Oakwood Holdings','Zenith Motors']
CITIES = ['Springfield','Riverdale','Fairview','Lakewood','Cedar Falls','Kingston','Ashford','Brookhaven','Maple Grove','Harbor City']
_SYNONYMS = {'great':['excellent','outstanding','superb'],'important':['significant','crucial','essential'],'good':['fine','great','solid'],'big':['large','huge','massive'],'small':['little','tiny','compact'],'fast':['quick','rapid','swift'],'use':['utilize','employ','apply'],'make':['create','produce','build'],'show':['demonstrate','display','reveal'],'start':['begin','commence','initiate']}

def _synonym(word,rng):
    if word in _SYNONYMS: return rng.choice(_SYNONYMS[word])
    try:
        from nltk.corpus import wordnet as wn
        syns=wn.synsets(word)
        if syns:
            lemmas=[l for s in syns[:2] for l in s.lemma_names() if '_' not in l and l.lower()!=word]
            if lemmas: return rng.choice(lemmas)
    except Exception: pass
    return word

def corrupt_text(text,rng,replace_prob=.12,insert_prob=.03,swap_prob=.02):
    """Character-level mojibake corruption. Keeps whitespace for structure."""
    chars=list(text); out=[]; i=0
    while i<len(chars):
        ch=chars[i]
        if ch.isspace(): out.append(ch); i+=1; continue
        r=rng.random()
        if r<replace_prob: out.append(rng.choice(GARBLED_CHARS))
        elif r<replace_prob+insert_prob: out.append(ch); out.append(rng.choice(GARBLED_CHARS))
        elif r<replace_prob+insert_prob+swap_prob and i+1<len(chars):
            nxt=chars[i+1]; out.append(nxt); out.append(ch); i+=1
        else: out.append(ch)
        i+=1
    return ''.join(out)

def replace_keywords(text,rng):
    """Replace only key words: numbers/dates and capitalized proper nouns. Grammar preserved."""
    def repl_num(m):
        digits=m.group(0)
        if re.fullmatch(r'\d{4}',digits): return str(int(rng.integers(1950,2024)))
        return ''.join(str(int(rng.integers(0,10))) for _ in digits)
    text=re.sub(r'\d+',repl_num,text)
    text=re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b',lambda m:rng.choice(PERSON_NAMES+ORGS+CITIES),text,count=4)
    return text

def paraphrase(text,rng,word_prob=.15,swap_prob=.35):
    """Light paraphrase: semantics preserved, surface wording changed."""
    sentences=re.split(r'(?<=[.!?])\s+',text.strip())
    if len(sentences)>=2:
        for k in range(len(sentences)-1):
            if rng.random()<swap_prob: sentences[k],sentences[k+1]=sentences[k+1],sentences[k]
    out_sents=[]
    for sent in sentences:
        tokens=re.split(r'(\s+)',sent)
        for i,tok in enumerate(tokens):
            word=tok.strip('.,!?;:\'"()')
            if word.isalpha() and len(word)>3 and rng.random()<word_prob:
                syn=_synonym(word.lower(),rng)
                if syn!=word.lower(): tokens[i]=syn+tok[len(word):] if tok.startswith(word) else syn
        out_sents.append(''.join(tokens))
    return ' '.join(out_sents)

def _edit(s,kind,fn):
    m=[dict(x) for x in s.messages]; text=m[-1].get('content','') if m else ''
    if m: m[-1]['content']=fn(text)
    return Sample(s.id,m,kind,s.meta)

def _edit_both(s,kind,fn):
    m=[dict(x) for x in s.messages]
    if len(m)>=1: m[0]['content']=fn(m[0].get('content',''))
    if len(m)>=1: m[-1]['content']=fn(m[-1].get('content',''))
    return Sample(s.id,m,kind,s.meta)

def _unrelated(s,rng,rows):
    category=(s.meta or {}).get('category')
    pool=[r for r in rows if r is not s and (r.meta or {}).get('category')!=category]
    if not pool: pool=[r for r in rows if r is not s] or [s]
    other=pool[int(rng.integers(0,len(pool)))]
    m=[dict(x) for x in s.messages]
    if m and other.messages: m[-1]['content']=other.messages[-1].get('content','')
    return Sample(s.id,m,'unrelated',s.meta)

def _duplicate_copy(s,suffix):
    return Sample(f'{s.id}_dup{suffix}',[dict(x) for x in s.messages],'duplicate',s.meta)

def _wrong_answer(text,rng):
    """Whole-answer replacement with a plausible-but-wrong short entity/number, unlike
    `keyword` which scatters replacements through the existing text."""
    if re.search(r'\d',text): return str(int(rng.integers(0,10000)))
    return str(rng.choice(PERSON_NAMES+ORGS+CITIES))

REFUSAL_PHRASES = ['Unknown', "I don't know", 'Not sure', 'No idea', 'Unclear', 'Cannot say', 'N/A']

def _refusal(text, rng):
    """Short-phrase non-answer, sized to match QA-style single/few-word answers
    (unlike a full refusal sentence, which would make response length alone a
    trivial detector — see length_confound's framing for why that matters)."""
    return str(rng.choice(REFUSAL_PHRASES))

def _confusable_wrong_batch(rows, ids):
    """Swap in the answer from the most similar *other* question (TF-IDF
    cosine nearest neighbor over question text), producing a same-topic
    plausible-but-wrong error rather than `wrong_answer`'s fully random
    person/org/city swap. Unlike dolly's `_unrelated()`, this can't group by
    `(s.meta or {}).get('category')` -- triviaqa-ratio10 carries no
    category/meta field -- so nearest-neighbor question similarity stands in
    for topic grouping. Special-cased in apply() (like 'duplicate') rather
    than following the per-sample fn(s,rng,rows) TRANSFORMS shape, since
    fitting a fresh TF-IDF index per corrupted sample would be O(n^2)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    questions=[r.messages[0].get('content','') if r.messages else '' for r in rows]
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=min(5,len(questions)),sublinear_tf=True,max_features=100_000)
    x=vec.fit_transform(questions)
    k=min(10,len(rows))
    nn=NearestNeighbors(n_neighbors=k,metric='cosine').fit(x)
    order=sorted(ids)
    _,idx=nn.kneighbors(x[order])
    out=list(rows)
    for row_pos,neighbors in zip(order,idx):
        src=rows[row_pos]
        src_ans=(src.messages[-1].get('content','') if src.messages else '').strip().lower()
        other=next((rows[j] for j in neighbors if j!=row_pos and (rows[j].messages[-1].get('content','') if rows[j].messages else '').strip().lower()!=src_ans),None)
        if other is None: continue
        m=[dict(d) for d in src.messages]
        if m and other.messages: m[-1]['content']=other.messages[-1].get('content','')
        out[row_pos]=Sample(src.id,m,'confusable_wrong',src.meta)
    return out

_FINAL_NUM_RE = re.compile(r'####\s*(-?[\d,]+(?:\.\d+)?)')

def _wrong_final_answer(text,rng):
    """Corrupt only the gsm8k `#### N` final number, leaving the reasoning chain intact —
    a silent-error stress test (chain looks fine, final value is wrong)."""
    m=_FINAL_NUM_RE.search(text)
    if not m: return text
    orig=m.group(1); wrong=orig
    try:
        val=float(orig.replace(',',''))
        delta=int(rng.integers(1,10))*rng.choice([-1,1])
        wrong=str(int(val)+delta) if float(val).is_integer() else str(round(val+delta,2))
    except ValueError: pass
    if wrong==orig: wrong=str(int(rng.integers(0,1000)))
    return text[:m.start(1)]+wrong+text[m.end(1):]

TRANSFORMS = {
    'garbled': lambda s,rng,rows: _edit_both(s,'garbled',lambda t: corrupt_text(t,rng)),
    'keyword': lambda s,rng,rows: _edit_both(s,'keyword',lambda t: replace_keywords(t,rng)),
    'template': lambda s,rng,rows: _edit(s,'template',lambda _: TEMPLATE_ANSWER),
    'truncation': lambda s,rng,rows: _edit(s,'truncation',lambda t: t[:max(1,int(len(t)*.5))]),
    'near_duplicate': lambda s,rng,rows: _edit(s,'near_duplicate',lambda t: paraphrase(t,rng)),
    'unrelated': _unrelated,
    'wrong_answer': lambda s,rng,rows: _edit(s,'wrong_answer',lambda t: _wrong_answer(t,rng)),
    'wrong_final_answer': lambda s,rng,rows: _edit(s,'wrong_final_answer',lambda t: _wrong_final_answer(t,rng)),
    'refusal': lambda s,rng,rows: _edit(s,'refusal',lambda t: _refusal(t,rng)),
}
NOISE_TYPES = list(TRANSFORMS)+['duplicate','confusable_wrong']

def apply(rows,kind,ratio,seed,mixed_types=None):
    if not 0<=ratio<=1: raise ValueError(f'ratio must be in [0,1], got {ratio}')
    rows=list(rows)
    if kind=='mixed':
        types=mixed_types or NOISE_TYPES
        out=list(rows)
        for offset,child in enumerate(types): out=apply(out,child,ratio/len(types),seed+offset,mixed_types=mixed_types)
        return out
    rng=np.random.default_rng(seed); n=int(len(rows)*ratio); ids={int(i) for i in rng.choice(len(rows),min(n,len(rows)),replace=False)}
    if kind=='duplicate':
        return rows+[_duplicate_copy(rows[i],k) for k,i in enumerate(sorted(ids))]
    if kind=='confusable_wrong':
        return _confusable_wrong_batch(rows,ids)
    fn=TRANSFORMS.get(kind)
    if fn is None: raise ValueError(f'Unknown noise kind: {kind}')
    return [fn(r,rng,rows) if i in ids else r for i,r in enumerate(rows)]
