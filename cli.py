import argparse
from pathlib import Path
from settings import load
from data import Jsonl,read,write,apply,load_rows,split_holdout,split_fractions
from model import create
from train import Trainer
from evaluate import Evaluator

def parser():
 p=argparse.ArgumentParser(prog='noisedetect',description='LLM noise experiment runner'); sub=p.add_subparsers(dest='command',required=True)
 d=sub.add_parser('data',help='build tagged datasets'); d.add_argument('--config',default='config.yaml'); d.add_argument('--tag'); d.add_argument('--source',required=True); d.add_argument('--split'); d.add_argument('--extra-split',dest='extra_splits',action='append',help='additional HF split merged into the same pool before val/test/train are cut (repeatable)'); d.add_argument('--val-frac',type=float,default=0.0); d.add_argument('--test-frac',type=float,default=0.0); d.add_argument('--datasets',default='clean,garbled,duplicate,unrelated,keyword,mixed'); d.add_argument('--ratio',type=float); d.add_argument('--mixed-types',help='comma-separated noise types composing "mixed" (default: noise.types in config)')
 t=sub.add_parser('train',help='train one dataset'); t.add_argument('--config',default='config.yaml'); t.add_argument('--tag'); t.add_argument('--dataset',required=True); t.add_argument('--train-file'); t.add_argument('--model',default='mock',choices=['mock','hf-lora']); t.add_argument('--smoke',action='store_true')
 e=sub.add_parser('evaluate',help='evaluate configured tasks'); e.add_argument('--config',default='config.yaml'); e.add_argument('--tag'); e.add_argument('--dataset',required=True); e.add_argument('--model',default='mock',choices=['mock','hf-lora']); e.add_argument('--tasks'); e.add_argument('--force',action='store_true')
 a=sub.add_parser('analyze',help='run metric analysis'); a.add_argument('--config',default='config.yaml'); a.add_argument('--tag'); a.add_argument('--tags'); a.add_argument('--kind',choices=['features','training','token','unsupervised','transfer','cross_type','cross_ratio','precision_lift','memorization','early_unsupervised','early_memorization','feature_attribution','feature_correlation','minimal_feature_set','label_free_feature_set','single_feature_ablation','transfer_to_mixed','feature_group_ablation','pooled_scorer_compare','length_confound','external_baselines'],default='features'); a.add_argument('--dataset'); a.add_argument('--input'); a.add_argument('--output')
 c=sub.add_parser('clean',help='build a label-free cleaned training set (targeted + random-drop control)'); c.add_argument('--config',default='config.yaml'); c.add_argument('--tag'); c.add_argument('--dataset',required=True); c.add_argument('--budget',type=float,default=0.10); c.add_argument('--method',default='iforest',choices=['iforest','memo_signed','pooled'])
 return p

def main(argv=None):
 a=parser().parse_args(argv); s=load(a.config,a.tag)
 if a.command=='data':
  ratio=a.ratio if a.ratio is not None else s.section('noise').get('ratio',.1); names=a.datasets.split(','); seed=s.section('noise').get('seed',42)
  base=load_rows(a.source, split=a.split or s.section('data').get('split','train'), extra_splits=a.extra_splits, max_samples=s.section('data').get('max_samples'))
  n_holdout=s.section('train').get('ref_samples',200)+s.section('train').get('heldout_samples',200)
  heldout,pool=split_holdout(base,n_holdout,seed); write(heldout,s.data_dir()/'heldout.jsonl')
  val_frac=a.val_frac or 0.0; test_frac=a.test_frac or 0.0
  if val_frac or test_frac:
   train_frac=1.0-val_frac-test_frac
   if train_frac<=0: raise ValueError(f'val_frac+test_frac must be <1, got {val_frac}+{test_frac}')
   val_pool,test_pool,train_pool=split_fractions(pool,[val_frac,test_frac,train_frac],seed)
   if val_frac: write(val_pool,s.data_dir()/'val.jsonl')
   if test_frac: write(test_pool,s.data_dir()/'test.jsonl')
  else:
   train_pool=pool
  mixed_types=a.mixed_types.split(',') if a.mixed_types else s.section('noise').get('types')
  for n in names: write(apply(train_pool,n,ratio,seed,mixed_types=mixed_types) if n!='clean' else train_pool,s.data_dir()/n/'train.jsonl')
  return 0
 if a.command=='train':
  rows=Jsonl(a.train_file or (s.data_dir()/a.dataset/'train.jsonl')).rows(); result=Trainer(create(a.model,s),s).run(rows,a.dataset,smoke=a.smoke); print(result); return 0
 if a.command=='evaluate':
  tasks=a.tasks.split(',') if a.tasks else s.section('eval').get('tasks',[]); print(Evaluator(s,create(a.model,s)).run(a.dataset,tasks,a.force)); return 0
 if a.command=='clean':
  from cleaning_loop import build
  print(build(s.root,a.tag or s.tag,a.dataset,a.budget,method=a.method)); return 0
 if a.command=='analyze':
  import pandas as pd
  from analyze import summarize, build_table, training_metrics, token_metrics, token_metrics_for_tag, unsupervised_metrics, transfer_metrics, cross_type_transfer, cross_ratio_transfer, precision_lift_table, memorization_score, early_detection_sweep, feature_attribution, feature_correlation, minimal_feature_set, label_free_feature_set, single_feature_ablation, transfer_to_mixed, feature_group_ablation, pooled_scorer_compare, length_confound
  tags=a.tags.split(',') if a.tags else [s.tag]
  if a.kind=='features':
   if a.input: frame=pd.read_csv(a.input)
   else:
    frame=build_table(s.root,a.tag or s.tag,[a.dataset] if a.dataset else None)
    s.results_dir().mkdir(parents=True,exist_ok=True)
    frame.to_csv(s.results_dir()/'per_sample_metrics.csv',index=False)
   features=[x for x in frame.columns if x not in ('sample_id','dataset','noise_type','category') and pd.api.types.is_numeric_dtype(frame[x])]; out=pd.DataFrame(summarize(frame,features))
  elif a.kind=='training': out=training_metrics(s.data_root,a.tag or s.tag)
  elif a.kind=='token':
   if a.input:
    out=token_metrics(a.input)
   elif a.dataset:
    out=token_metrics(s.root/'results'/ (a.tag or s.tag) / f'token_level_{a.dataset}.jsonl')
   else:
    out=token_metrics_for_tag(s.root, a.tag or s.tag)
  elif a.kind=='unsupervised':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out=unsupervised_metrics(pd.read_csv(path))
  elif a.kind=='cross_type':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out=cross_type_transfer(pd.read_csv(path))
  elif a.kind=='cross_ratio':
   tag_a,tag_b=tags[0],tags[1]
   frame_a=pd.read_csv(s.root/'results'/tag_a/'per_sample_metrics.csv'); frame_b=pd.read_csv(s.root/'results'/tag_b/'per_sample_metrics.csv')
   out=cross_ratio_transfer(frame_a,frame_b,tag_a,tag_b)
  elif a.kind=='precision_lift':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'unsupervised.csv'); out=precision_lift_table(path)
  elif a.kind=='memorization':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out=memorization_score(pd.read_csv(path))
  elif a.kind in ('early_unsupervised','early_memorization'):
   sweep=early_detection_sweep(s.root,a.tag or s.tag); out=sweep['unsupervised' if a.kind=='early_unsupervised' else 'memorization']
  elif a.kind=='feature_attribution':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out=feature_attribution(pd.read_csv(path))
  elif a.kind=='feature_correlation':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out,pairs=feature_correlation(pd.read_csv(path))
   pairs_path=(Path(a.output).with_name(Path(a.output).stem+'_pairs.csv') if a.output else s.results_dir()/'feature_correlation_pairs.csv')
   pairs_path.parent.mkdir(parents=True,exist_ok=True); pairs.to_csv(pairs_path,index=False); print(pairs_path)
  elif a.kind=='minimal_feature_set':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out=minimal_feature_set(pd.read_csv(path))
  elif a.kind=='label_free_feature_set':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out=label_free_feature_set(pd.read_csv(path))
  elif a.kind=='single_feature_ablation':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out=single_feature_ablation(pd.read_csv(path))
  elif a.kind=='transfer_to_mixed':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out=transfer_to_mixed(pd.read_csv(path))
  elif a.kind=='feature_group_ablation':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out=feature_group_ablation(pd.read_csv(path))
  elif a.kind=='pooled_scorer_compare':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out=pooled_scorer_compare(pd.read_csv(path),dataset=a.dataset or 'mixed')
  elif a.kind=='length_confound':
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out=length_confound(pd.read_csv(path),s.root,a.tag or s.tag,dataset=a.dataset or 'wild')
  elif a.kind=='external_baselines':
   from baselines import external_baselines
   path=a.input or (s.root/'results'/ (a.tag or s.tag) / 'per_sample_metrics.csv'); out=external_baselines(pd.read_csv(path),s.root,a.tag or s.tag)
  else:
   path=a.input or (s.root/'results'/'transfer_cross_ratio.csv'); out=transfer_metrics(path,tags)
  output=Path(a.output) if a.output else s.results_dir() / f'{a.kind}.csv'; output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(output,index=False); print(output); return 0
if __name__=='__main__': raise SystemExit(main())
