import argparse
from pathlib import Path
from settings import load
from data import Jsonl,read,write,apply
from model import create
from train import Trainer
from evaluate import Evaluator

def parser():
 p=argparse.ArgumentParser(prog='noisedetect',description='LLM noise experiment runner'); sub=p.add_subparsers(dest='command',required=True)
 d=sub.add_parser('data',help='build tagged datasets'); d.add_argument('--config',default='config.yaml'); d.add_argument('--tag'); d.add_argument('--source',required=True); d.add_argument('--datasets',default='clean,garbled,duplicate,unrelated,keyword,mixed'); d.add_argument('--ratio',type=float)
 t=sub.add_parser('train',help='train one dataset'); t.add_argument('--config',default='config.yaml'); t.add_argument('--tag'); t.add_argument('--dataset',required=True); t.add_argument('--train-file'); t.add_argument('--model',default='mock',choices=['mock','hf-lora']); t.add_argument('--smoke',action='store_true')
 e=sub.add_parser('evaluate',help='evaluate configured tasks'); e.add_argument('--config',default='config.yaml'); e.add_argument('--tag'); e.add_argument('--dataset',required=True); e.add_argument('--model',default='mock',choices=['mock','hf-lora']); e.add_argument('--tasks'); e.add_argument('--force',action='store_true')
 a=sub.add_parser('analyze',help='run metric analysis'); a.add_argument('--config',default='config.yaml'); a.add_argument('--tag'); a.add_argument('--input',required=True); a.add_argument('--output')
 return p

def main(argv=None):
 a=parser().parse_args(argv); s=load(a.config,a.tag)
 if a.command=='data':
  ratio=a.ratio if a.ratio is not None else s.section('noise').get('ratio',.1); source=Jsonl(a.source); names=a.datasets.split(','); base=list(source.rows());
  for n in names: write(apply(base,n,ratio,s.section('noise').get('seed',42)) if n!='clean' else base,s.data_dir()/n/'train.jsonl')
  return 0
 if a.command=='train':
  rows=Jsonl(a.train_file or (s.data_dir()/a.dataset/'train.jsonl')).rows(); result=Trainer(create(a.model,s),s).run(rows,a.dataset); print(result); return 0
 if a.command=='evaluate':
  tasks=a.tasks.split(',') if a.tasks else s.section('eval').get('tasks',[]); print(Evaluator(s,create(a.model,s)).run(a.dataset,tasks,a.force)); return 0
 if a.command=='analyze':
  import pandas as pd
  from analyze import summarize
  frame=pd.read_csv(a.input); features=[x for x in frame.columns if x not in ('sample_id','dataset','noise_type','category') and pd.api.types.is_numeric_dtype(frame[x])]; out=summarize(frame,features); pd.DataFrame(out).to_csv(a.output or s.results_dir()/'detection.csv',index=False); return 0
if __name__=='__main__': raise SystemExit(main())
