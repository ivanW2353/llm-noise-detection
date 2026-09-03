import json
from pathlib import Path
class Evaluator:
    def __init__(self,settings,model): self.settings,self.model=settings,model
    def run(self,dataset,tasks,force=False):
        p=self.settings.root/'results'/'eval'/f'eval_{self.settings.tag}_{dataset}.json'; p.parent.mkdir(parents=True,exist_ok=True)
        result={} if force or not p.exists() else json.loads(p.read_text())
        for task in tasks: result.setdefault(task,{'accuracy':0.0,'n':0,'model':getattr(self.model,'name','unknown')})
        p.write_text(json.dumps(result,indent=2)); return result
