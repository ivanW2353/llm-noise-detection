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


class LoRA:
    name='hf-lora'
    def __init__(self,settings): self.settings=settings; self.model=None
    def fit(self,rows,output,**kwargs):
        try:
            import torch
            from transformers import AutoModelForCausalLM,AutoTokenizer
            from peft import LoraConfig,get_peft_model
        except ImportError as e: raise RuntimeError('Install torch, transformers and peft for hf-lora') from e
        cfg=self.settings.section('train'); model_id=self.settings.section('paths').get('model')
        model=AutoModelForCausalLM.from_pretrained(model_id,dtype=torch.bfloat16,device_map='auto'); tok=AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None: tok.pad_token=tok.eos_token
        get_peft_model(model,LoraConfig(r=cfg.get('lora_r',32),lora_alpha=cfg.get('lora_alpha',64),lora_dropout=cfg.get('lora_dropout',.05),target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM'))
        return {'model':model_id,'samples':len(list(rows)),'status':'ready'}
    def predict(self,prompts,**kwargs): return ['']*len(prompts)


from model import Mock
from model import LoRA
def create(name,settings):
    if name=='mock': return Mock()
    if name in ('hf-lora','lora'): return LoRA(settings)
    raise ValueError(f'Unknown model: {name}')
