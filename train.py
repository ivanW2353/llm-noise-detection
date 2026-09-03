from pathlib import Path
class Trainer:
    def __init__(self,model,settings): self.model,self.settings=model,settings
    def run(self,rows,dataset): return self.model.fit(rows,Path(self.settings.path('runs'))/dataset,epochs=self.settings.section('train').get('epochs',1))
