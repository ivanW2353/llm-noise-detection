from typing import Protocol, Any
class Model(Protocol):
    def fit(self, rows, output, **kwargs)->dict[str,Any]: ...
    def predict(self, prompts, **kwargs): ...


import json
from pathlib import Path
class Mock:
    name='mock'
    def fit(self,rows,output,**kwargs):
        rows=list(rows); p=Path(output); p.mkdir(parents=True,exist_ok=True); result={'model':self.name,'samples':len(rows),'status':'ok'}; (p/'summary.json').write_text(json.dumps(result,indent=2)); return result
    def predict(self,prompts,**kwargs): return ['']*len(prompts)


from lora.lora_train import LoRA

def create(name,settings):
    if name=='mock': return Mock()
    if name in ('hf-lora','lora'): return LoRA(settings)
    raise ValueError(f'Unknown model: {name}')
