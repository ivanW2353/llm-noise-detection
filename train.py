from pathlib import Path
class Trainer:
    def __init__(self,model,settings): self.model,self.settings=model,settings
    def run(self,rows,dataset,smoke=False):
        output=Path(self.settings.path('runs'))/('_smoke' if smoke else '')/dataset
        return self.model.fit(rows,output,epochs=self.settings.section('train').get('epochs',1),smoke=smoke)
