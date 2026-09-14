from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Iterable
@dataclass
class Sample:
    id: str
    messages: list[dict[str,str]]
    noise: str='none'
    meta: dict[str,Any]|None=None
class Provider(Protocol):
    def rows(self) -> Iterable[Sample]: ...


import json
from pathlib import Path
from data import Sample
def read(path, max_samples=None):
    with Path(path).open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if not line.strip(): continue
            if max_samples is not None and i >= max_samples: break
            r=json.loads(line); messages=r.get('messages') or [{'role':'user','content':r.get('instruction','')},{'role':'assistant','content':r.get('response','')}]
            yield Sample(str(r.get('sample_id',i)),messages,r.get('noise_type','none'),r)
def write(rows,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps({'sample_id':r.id,'messages':r.messages,'noise_type':r.noise},ensure_ascii=False)+'\n')
class Jsonl:
    def __init__(self,path): self.path=Path(path)
    def rows(self): return read(self.path)


def split_holdout(rows, n_holdout, seed):
    """Shuffle deterministically and carve off a disjoint held-out prefix.
    Returns (heldout_rows, train_rows), both reindexed to sequential ids."""
    import random
    rows=list(rows); order=list(range(len(rows))); random.Random(seed).shuffle(order)
    ordered=[rows[i] for i in order]
    heldout,train=ordered[:n_holdout],ordered[n_holdout:]
    return reindex(heldout),reindex(train)

def reindex(rows):
    return [Sample(str(i),r.messages,r.noise,r.meta) for i,r in enumerate(rows)]

def load_rows(source, split='train', max_samples=None):
    """Load local JSONL or a Hugging Face dataset (hf://org/name)."""
    source = str(source)
    if not source.startswith('hf://'):
        return list(read(source, max_samples=max_samples))
    from datasets import load_dataset
    dataset = load_dataset(source.removeprefix('hf://'), split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    rows = []
    for index, row in enumerate(dataset):
        messages = row.get('messages')
        if not messages:
            prompt = row.get('prompt', row.get('instruction', row.get('question', '')))
            response = row.get('response', row.get('output', row.get('answer', '')))
            messages = [{'role': 'user', 'content': str(prompt)}, {'role': 'assistant', 'content': str(response)}]
        rows.append(Sample(str(row.get('sample_id', row.get('id', index))), messages, row.get('noise_type', 'none'), row))
    return rows


import re
import numpy as np
from data import Sample

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

TRANSFORMS = {
    'garbled': lambda s,rng,rows: _edit_both(s,'garbled',lambda t: corrupt_text(t,rng)),
    'keyword': lambda s,rng,rows: _edit_both(s,'keyword',lambda t: replace_keywords(t,rng)),
    'template': lambda s,rng,rows: _edit(s,'template',lambda _: TEMPLATE_ANSWER),
    'truncation': lambda s,rng,rows: _edit(s,'truncation',lambda t: t[:max(1,int(len(t)*.5))]),
    'near_duplicate': lambda s,rng,rows: _edit(s,'near_duplicate',lambda t: paraphrase(t,rng)),
    'unrelated': _unrelated,
}
NOISE_TYPES = list(TRANSFORMS)+['duplicate']

def apply(rows,kind,ratio,seed,mixed_types=None):
    rows=list(rows)
    if kind=='mixed':
        types=mixed_types or NOISE_TYPES
        out=list(rows)
        for offset,child in enumerate(types): out=apply(out,child,ratio/len(types),seed+offset,mixed_types=mixed_types)
        return out
    rng=np.random.default_rng(seed); n=int(len(rows)*ratio); ids={int(i) for i in rng.choice(len(rows),min(n,len(rows)),replace=False)}
    if kind=='duplicate':
        return rows+[_duplicate_copy(rows[i],k) for k,i in enumerate(sorted(ids))]
    fn=TRANSFORMS.get(kind)
    if fn is None: raise ValueError(f'Unknown noise kind: {kind}')
    return [fn(r,rng,rows) if i in ids else r for i,r in enumerate(rows)]


import json
from pathlib import Path
REQUIRED={'sample_id','messages','noise_type'}
def validate(path):
    errors=[]
    for n,line in enumerate(Path(path).open(encoding='utf-8'),1):
        try: row=json.loads(line)
        except json.JSONDecodeError as e: errors.append(f'{n}: invalid JSON ({e.msg})'); continue
        legacy = ('instruction' in row and 'response' in row) or ('messages' in row)
        missing=REQUIRED-set(row)
        if legacy: missing -= {'sample_id','messages','noise_type'}
        if missing: errors.append(f'{n}: missing {sorted(missing)}')
        if not legacy and not isinstance(row.get('messages'),list): errors.append(f'{n}: messages must be a list')
    return errors
