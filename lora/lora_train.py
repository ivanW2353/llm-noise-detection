import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from datalib import Jsonl
from .lora_internals import (
    TARGET_MODULES, _tokenize_rows, _fill_flat, _lora_params, _lora_offsets,
    _compute_reference_direction, _window_layer_grad_norms, _window_layer_weight_norms,
    _log_histograms, _eval_heldout, _diagnostic_pass,
)

@dataclass
class _TrainState:
    """Mutable state shared across the phases fit() is decomposed into (config, model/tokenizer,
    gradient buffers, optimizer, checkpoint/IO handles, running counters). Fields are filled in
    stages by _setup_run/_load_or_init_reference/_open_run_io rather than all at construction time."""
    tcfg: Any=None; model_id: Any=None; smoke: Any=None; max_len: Any=None
    run_dir: Any=None; metric_dir: Any=None; ckpt_dir: Any=None; resume: Any=None
    heldout_rows: Any=None; n_ref: Any=None; n_held: Any=None
    model: Any=None; tokenizer: Any=None
    train_data: Any=None; n_label_tokens: Any=None; held_data: Any=None; ref_data: Any=None
    params: Any=None; offsets: Any=None; n_lora: Any=None; b_offsets: Any=None; total_b: Any=None
    buf_before: Any=None; buf_after: Any=None; delta_buf: Any=None; delta_b: Any=None
    lr: Any=None; n_train: Any=None; epochs: Any=None; grad_accum: Any=None
    steps_per_epoch: Any=None; total_steps: Any=None; warmup_steps: Any=None
    opt: Any=None; lr_at: Any=None
    start_epoch: Any=None; ref_buf: Any=None; v_buf: Any=None; v_ready: Any=None
    global_step: Any=None; epoch_stats: Any=None
    writer: Any=None; metric_f: Any=None; ln_f: Any=None
    n_layers: Any=None; target_ids: Any=None; layer_norm_sum: Any=None; layer_norm_cnt: Any=None
    eval_steps: Any=None; log_every: Any=None; diag_step: Any=None; hard_threshold: Any=None
    t0: Any=None; tb_t0: Any=None; tb_sum: Any=None

def _setup_run(model_obj,rows,output,**kwargs):
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from peft import LoraConfig,get_peft_model,PeftModel
    except ImportError as e: raise RuntimeError('Install torch, transformers and peft for hf-lora') from e

    settings=model_obj.settings
    tcfg=settings.section('train'); model_id=settings.section('paths').get('model')
    smoke=kwargs.get('smoke',False)
    seed=tcfg.get('seed',42); torch.manual_seed(seed)
    max_len=tcfg.get('max_len',1024)
    run_dir=Path(output); metric_dir=run_dir/'metrics'; metric_dir.mkdir(parents=True,exist_ok=True)
    ckpt_dir=run_dir/'checkpoint'
    resume=(ckpt_dir/'state.pt').exists()

    train_rows=list(rows)
    if smoke: train_rows=train_rows[:64]
    heldout_rows=list(Jsonl(settings.data_dir()/'heldout.jsonl').rows())
    n_ref=tcfg.get('ref_samples',200); n_held=tcfg.get('heldout_samples',200)

    model=AutoModelForCausalLM.from_pretrained(model_id,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':0})
    model.config.use_cache=False
    if resume:
        model=PeftModel.from_pretrained(model,ckpt_dir/'adapter',is_trainable=True)
    else:
        model=get_peft_model(model,LoraConfig(r=tcfg.get('lora_r',32),lora_alpha=tcfg.get('lora_alpha',64),lora_dropout=tcfg.get('lora_dropout',.05),target_modules=TARGET_MODULES,bias='none',task_type='CAUSAL_LM'))
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token

    train_data,n_label_tokens=_tokenize_rows(tokenizer,train_rows,max_len)
    held_data,_=_tokenize_rows(tokenizer,heldout_rows[:n_held],max_len)
    ref_data,_=_tokenize_rows(tokenizer,heldout_rows[n_held:n_held+n_ref],max_len)

    params=_lora_params(model); offsets,n_lora=_lora_offsets(params)
    b_offsets=[(s,e) for (s,e),(name,_) in zip(offsets,params) if 'lora_B' in name]
    total_b=sum(e-s for s,e in b_offsets)
    buf_before=torch.zeros(n_lora,device='cuda',dtype=torch.float32)
    buf_after=torch.zeros(n_lora,device='cuda',dtype=torch.float32)
    delta_buf=torch.zeros(n_lora,device='cuda',dtype=torch.float32)
    delta_b=torch.zeros(total_b,device='cuda',dtype=torch.float32)

    lr=tcfg.get('lr',2e-4); n_train=len(train_data)
    epochs=kwargs.get('epochs',tcfg.get('epochs',1))
    grad_accum=tcfg.get('grad_accum',16)
    steps_per_epoch=math.ceil(n_train/grad_accum); total_steps=steps_per_epoch*epochs
    warmup_steps=int(total_steps*tcfg.get('warmup_ratio',.03))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=tcfg.get('weight_decay',.01),betas=(0.9,0.999))

    def lr_at(step):
        if step<warmup_steps: return lr*(step+1)/max(1,warmup_steps)
        p=(step-warmup_steps)/max(1,total_steps-warmup_steps); return lr*0.5*(1+math.cos(math.pi*p))

    return _TrainState(
        tcfg=tcfg,model_id=model_id,smoke=smoke,max_len=max_len,
        run_dir=run_dir,metric_dir=metric_dir,ckpt_dir=ckpt_dir,resume=resume,
        heldout_rows=heldout_rows,n_ref=n_ref,n_held=n_held,
        model=model,tokenizer=tokenizer,
        train_data=train_data,n_label_tokens=n_label_tokens,held_data=held_data,ref_data=ref_data,
        params=params,offsets=offsets,n_lora=n_lora,b_offsets=b_offsets,total_b=total_b,
        buf_before=buf_before,buf_after=buf_after,delta_buf=delta_buf,delta_b=delta_b,
        lr=lr,n_train=n_train,epochs=epochs,grad_accum=grad_accum,
        steps_per_epoch=steps_per_epoch,total_steps=total_steps,warmup_steps=warmup_steps,
        opt=opt,lr_at=lr_at,
    )

def _load_or_init_reference(state):
    start_epoch=0
    if state.resume:
        ckpt=torch.load(state.ckpt_dir/'state.pt',map_location='cuda')
        ref_buf=ckpt['ref_buf'].to('cuda')
        v_buf=ckpt['v_buf'].to('cuda'); v_ready=ckpt['v_ready']
        epoch_stats=ckpt['epoch_stats']; global_step=ckpt['global_step']
        start_epoch=ckpt['epoch_done']+1
        state.opt.load_state_dict(torch.load(state.ckpt_dir/'optimizer.pt',map_location='cuda'))
        print(f'resuming from epoch {start_epoch}, step {global_step}')
    else:
        params,offsets,n_lora,ref_buf=_compute_reference_direction(state.model,state.ref_data)
        state.params,state.offsets,state.n_lora=params,offsets,n_lora
        v_buf=torch.zeros(state.total_b,device='cuda',dtype=torch.float32); v_ready=False
        global_step=0; epoch_stats=[]

    if state.resume:
        # checkpoint boundaries are per-epoch, but the kill can land mid-epoch after
        # some steps already flushed past the checkpoint's epoch_done — drop those
        # stale partial-epoch/step rows so resume doesn't duplicate them.
        per_sample_path=state.metric_dir/'per_sample.jsonl'
        kept=[l for l in per_sample_path.open() if json.loads(l)['epoch']<=start_epoch-1]
        per_sample_path.write_text(''.join(kept))
        ln_path=state.metric_dir/'layer_norms.jsonl'
        kept=[l for l in ln_path.open() if json.loads(l)['step']<=global_step]
        ln_path.write_text(''.join(kept))

    state.ref_buf=ref_buf; state.v_buf=v_buf; state.v_ready=v_ready
    state.global_step=global_step; state.epoch_stats=epoch_stats; state.start_epoch=start_epoch

def _open_run_io(state):
    from torch.utils.tensorboard import SummaryWriter
    state.writer=SummaryWriter(state.run_dir/'tb')
    state.metric_f=(state.metric_dir/'per_sample.jsonl').open('a' if state.resume else 'w')
    state.ln_f=(state.metric_dir/'layer_norms.jsonl').open('a' if state.resume else 'w')
    model=state.model
    n_layers=model.base_model.model.config.num_hidden_layers if hasattr(model,'base_model') else model.config.num_hidden_layers
    state.n_layers=n_layers
    state.target_ids={0,n_layers//2,n_layers-1}
    state.layer_norm_sum={li:0.0 for li in state.target_ids}; state.layer_norm_cnt=0
    state.eval_steps=state.tcfg.get('eval_steps',200); state.log_every=max(1,state.tcfg.get('log_every',25))
    state.diag_step=max(1,state.tcfg.get('diag_subsample',8)); state.hard_threshold=state.tcfg.get('hard_threshold',4.0)
    state.t0=time.time(); state.tb_t0=state.t0
    state.tb_sum={'loss':0.0,'grad_norm':0.0,'cos_ref':0.0,'cos_global':0.0,'update_contrib':0.0,'tokens':0.0,'cnt':0}

def _save_checkpoint(state,epoch_done):
    tmp_dir=state.run_dir/'checkpoint.tmp'
    if tmp_dir.exists():
        import shutil; shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    state.model.save_pretrained(tmp_dir/'adapter')
    torch.save(state.opt.state_dict(),tmp_dir/'optimizer.pt')
    torch.save({'epoch_done':epoch_done,'global_step':state.global_step,'v_ready':state.v_ready,
                'v_buf':state.v_buf.cpu(),'ref_buf':state.ref_buf.cpu(),'epoch_stats':state.epoch_stats},tmp_dir/'state.pt')
    if state.ckpt_dir.exists():
        import shutil; shutil.rmtree(state.ckpt_dir)
    tmp_dir.rename(state.ckpt_dir)

def _flush_window(state,acc,epoch,step):
    losses=torch.stack([a[1] for a in acc]); gns=torch.stack([a[2] for a in acc]); dots_ref=torch.stack([a[3] for a in acc])
    ups=torch.stack([a[6] if a[6] is not None else torch.zeros((),device=gns.device) for a in acc])
    cos_refs=(dots_ref/gns.clamp_min(1e-12)).tolist()
    cos_globs=[None]*len(acc)
    if any(a[4] is not None for a in acc):
        had_b=[a[4] is not None for a in acc]
        dots_b=torch.stack([a[4] if a[4] is not None else torch.zeros((),device=gns.device) for a in acc])
        bsqs=torch.stack([a[5] if a[5] is not None else torch.ones((),device=gns.device) for a in acc])
        raw=(dots_b/(gns*bsqs.sqrt()).clamp_min(1e-12)).tolist()
        cos_globs=[raw[i] if had_b[i] else None for i in range(len(acc))]
    for a,l,gn,cr,cg,up in zip(acc,losses.tolist(),gns.tolist(),cos_refs,cos_globs,ups.tolist()):
        sid=a[0]
        state.metric_f.write(json.dumps({'step':step,'epoch':epoch,'sample_id':sid,'loss':l,'grad_norm':gn,
                                    'cos_sim_ref':cr,'cos_sim_global':cg,'update_contrib':up,
                                    'tokens':max(1,state.n_label_tokens[sid])})+'\n')
        if l==l and gn==gn:
            state.tb_sum['loss']+=l; state.tb_sum['grad_norm']+=gn
            if cr is not None and cr==cr: state.tb_sum['cos_ref']+=cr
            if cg is not None and cg==cg: state.tb_sum['cos_global']+=cg
            if up==up: state.tb_sum['update_contrib']+=up
            state.tb_sum['tokens']+=max(1,state.n_label_tokens[sid]); state.tb_sum['cnt']+=1
    state.metric_f.flush()

def _run_epoch(state,epoch):
    acc=[]
    for i,row in enumerate(state.train_data):
        if state.smoke and state.global_step>=8: break
        if row['n_label_tokens']==0: continue
        out=state.model(input_ids=row['input_ids'].unsqueeze(0).cuda(),labels=row['labels'].unsqueeze(0).cuda())
        loss=out.loss
        has_before=_fill_flat(state.params,state.offsets,state.buf_before) is not None
        loss.backward()
        _fill_flat(state.params,state.offsets,state.buf_after)
        if has_before:
            state.delta_buf.copy_(state.buf_after).sub_(state.buf_before)
            bsq=torch.dot(state.buf_before,state.buf_before); dot_b=torch.dot(state.delta_buf,state.buf_before)
        else:
            state.delta_buf.copy_(state.buf_after); bsq=dot_b=None
        g_norm=torch.linalg.vector_norm(state.delta_buf)
        dot_ref=torch.dot(state.delta_buf,state.ref_buf); cos_ref=dot_ref/g_norm
        upd=None
        if state.v_ready:
            o=0
            for s,e in state.b_offsets: state.delta_b[o:o+e-s].copy_(state.delta_buf[s:e]); o+=e-s
            upd=torch.linalg.vector_norm(state.delta_b)/(torch.linalg.vector_norm(state.v_buf.sqrt())+1e-8)
        acc.append((row['sample_id'],loss,g_norm,dot_ref,dot_b,bsq,upd))

        if (i+1)%state.grad_accum==0:
            all_norms=_window_layer_grad_norms(state.model)
            for li in state.target_ids: state.layer_norm_sum[li]+=all_norms.get(li,0.0)
            state.layer_norm_cnt+=1
            if (state.global_step+1)%state.eval_steps==0: _log_histograms(state.writer,state.model,state.global_step+1,state.target_ids)
            for g in state.opt.param_groups: g['lr']=state.lr_at(state.global_step)
            state.opt.step()
            state.ln_f.write(json.dumps({'step':state.global_step+1,'grad':all_norms,'weight':_window_layer_weight_norms(state.model)})+'\n'); state.ln_f.flush()
            o=0
            for (s,e),(name,p) in zip(state.offsets,state.params):
                if 'lora_B' not in name: continue
                st=state.opt.state.get(p)
                if st and 'exp_avg_sq' in st: state.v_buf[o:o+e-s].copy_(st['exp_avg_sq'].reshape(-1))
                o+=e-s
            state.v_ready=True
            state.opt.zero_grad()
            _flush_window(state,acc,epoch,state.global_step); acc=[]
            state.global_step+=1
            if state.global_step%state.log_every==0:
                now=time.time(); cnt=max(1,state.tb_sum['cnt'])
                for key in ('loss','grad_norm','cos_ref','cos_global','update_contrib'):
                    state.writer.add_scalar(f'train/{key}',state.tb_sum[key]/cnt,state.global_step)
                state.writer.add_scalar('train/lr',state.lr_at(state.global_step),state.global_step)
                state.writer.add_scalar('train/tokens_per_sec',state.tb_sum['tokens']/max(1.0,now-state.tb_t0),state.global_step)
                state.writer.add_scalar('train/gpu_mem_GB',torch.cuda.memory_allocated()/1e9,state.global_step)
                if state.layer_norm_cnt:
                    for li in state.target_ids: state.writer.add_scalar(f'lora_layer_gradnorm/layer{li}',state.layer_norm_sum[li]/state.layer_norm_cnt,state.global_step)
                state.writer.flush(); state.tb_t0=now
                state.tb_sum={'loss':0.0,'grad_norm':0.0,'cos_ref':0.0,'cos_global':0.0,'update_contrib':0.0,'tokens':0.0,'cnt':0}
                state.layer_norm_sum={li:0.0 for li in state.target_ids}; state.layer_norm_cnt=0
            if state.global_step%state.eval_steps==0:
                el=_eval_heldout(state.model,state.held_data); state.writer.add_scalar('eval/heldout_loss',el,state.global_step); state.writer.flush()

    if acc:
        for g in state.opt.param_groups: g['lr']=state.lr_at(state.global_step)
        state.opt.step(); state.opt.zero_grad(); _flush_window(state,acc,epoch,state.global_step); acc=[]; state.global_step+=1

def _finalize_epoch(state,epoch):
    ep_metrics=[]
    with (state.metric_dir/'per_sample.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['epoch']==epoch and r['loss'] is not None: ep_metrics.append(r)
    n_ep=len(ep_metrics)
    if n_ep:
        def avg(key):
            vals=[r[key] for r in ep_metrics if r.get(key) is not None and r[key]==r[key]]
            return sum(vals)/len(vals) if vals else None
        state.epoch_stats.append({'epoch':epoch,'n':n_ep,'loss_mean':avg('loss'),
                             'loss_min':min(r['loss'] for r in ep_metrics),'loss_max':max(r['loss'] for r in ep_metrics),
                             'grad_norm_mean':avg('grad_norm'),'cos_ref_mean':avg('cos_sim_ref'),'cos_global_mean':avg('cos_sim_global')})

    diag_rows=state.train_data[::state.diag_step]
    pad=state.tokenizer.pad_token_id
    diag=_diagnostic_pass(state.model,diag_rows,pad,thresh=state.hard_threshold)
    with (state.metric_dir/f'diag_epoch{epoch}.jsonl').open('w') as f:
        for d in diag: f.write(json.dumps({k:v for k,v in d.items() if k!='top_tokens'})+'\n')
    with (state.metric_dir/f'token_diag_epoch{epoch}.jsonl').open('w') as f:
        for d in diag: f.write(json.dumps({'sample_id':d['sample_id'],'top_tokens':d['top_tokens']})+'\n')
    if diag:
        state.writer.add_scalar('diag/max_token_loss_mean',sum(d['max_token_loss'] for d in diag)/len(diag),state.global_step)
        state.writer.add_scalar('diag/frac_hard_mean',sum(d['frac_hard'] for d in diag)/len(diag),state.global_step)
        state.writer.flush()

    if not state.smoke: _save_checkpoint(state,epoch)

def _finalize_run(state):
    state.metric_f.close(); state.ln_f.close()
    state.model.eval()
    state.model.save_pretrained(state.run_dir/'lora'); state.tokenizer.save_pretrained(state.run_dir/'lora')
    state.writer.close()
    if state.ckpt_dir.exists():
        import shutil; shutil.rmtree(state.ckpt_dir)
    summary={'model':state.model_id,'samples':state.n_train,'epochs':state.epochs,'total_steps':state.global_step,
             'seconds':round(time.time()-state.t0,1),'lora_params':state.n_lora,'epochs_detail':state.epoch_stats,'status':'ok'}
    (state.run_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary

class LoRA:
    name='hf-lora'
    def __init__(self,settings): self.settings=settings; self.model=None

    def fit(self,rows,output,**kwargs):
        state=_setup_run(self,rows,output,**kwargs)
        _load_or_init_reference(state)
        _open_run_io(state)
        for epoch in range(state.start_epoch,state.epochs):
            _run_epoch(state,epoch)
            _finalize_epoch(state,epoch)
            if state.smoke: break
        return _finalize_run(state)

    def predict(self,prompts,**kwargs): return ['']*len(prompts)
