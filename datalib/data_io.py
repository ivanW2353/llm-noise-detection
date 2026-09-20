import json
from pathlib import Path
from .sample import Sample

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

def load_rows(source, split='train', extra_splits=None, max_samples=None):
    """Load local JSONL or a Hugging Face dataset (hf://org/name or hf://org/name#config).
    `extra_splits`, if given, are additional HF splits merged into the same pool (e.g. official
    train+validation), each row tagged with its own split index so ids stay unique pre-reindex."""
    source = str(source)
    if not source.startswith('hf://'):
        return list(read(source, max_samples=max_samples))
    from datasets import load_dataset
    path = source.removeprefix('hf://')
    config = None
    if '#' in path: path, config = path.split('#', 1)
    splits = [split] + list(extra_splits or [])
    rows = []
    for sp in splits:
        dataset = load_dataset(path, name=config, split=sp) if config else load_dataset(path, split=sp)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        for index, row in enumerate(dataset):
            messages = row.get('messages')
            if not messages:
                answers = row.get('answers')
                answer = row.get('answer')
                if isinstance(answers, dict):
                    prompt = f"{row.get('context','')}\n\n{row.get('question','')}".strip()
                    response = (answers.get('text') or [''])[0]
                elif isinstance(answer, dict):
                    prompt = row.get('question', '')
                    response = answer.get('value') or (answer.get('aliases') or [''])[0]
                else:
                    prompt = row.get('prompt', row.get('instruction', row.get('question', '')))
                    response = row.get('response', row.get('output', row.get('answer', '')))
                messages = [{'role': 'user', 'content': str(prompt)}, {'role': 'assistant', 'content': str(response)}]
            rows.append(Sample(str(row.get('sample_id', row.get('id', f'{sp}_{index}'))), messages, row.get('noise_type', 'none'), row))
    return rows

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
