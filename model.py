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


import math
import re
import time
import torch
import torch.nn.functional as F
from data import Jsonl

TARGET_MODULES=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']

def _tokenize_rows(tokenizer,rows,max_len):
    data=[]
    for r in rows:
        msg=r.messages
        user_ids=tokenizer.apply_chat_template(msg[:-1],tokenize=True,return_dict=True,add_generation_prompt=True)['input_ids']
        full=tokenizer.apply_chat_template(msg,tokenize=True,return_dict=True)
        input_ids=full['input_ids']; n_user=len(user_ids)
        if len(input_ids)>max_len:
            assistant_ids=input_ids[n_user:]
            if len(assistant_ids)>=max_len: input_ids,n_user=assistant_ids[:max_len],0
            else:
                keep_user=max_len-len(assistant_ids); input_ids,n_user=user_ids[-keep_user:]+assistant_ids,keep_user
        labels=input_ids[:]; labels[:n_user]=[-100]*n_user
        data.append({'sample_id':r.id,'input_ids':torch.tensor(input_ids),'labels':torch.tensor(labels),'n_label_tokens':len(input_ids)-n_user})
    return data,{d['sample_id']:d['n_label_tokens'] for d in data}

def _fill_flat(params,offsets,buf):
    """Copy all LoRA grads into a preallocated float32 buffer. Returns buf, or None if no grads exist."""
    any_grad=False
    for (s,e),(_,p) in zip(offsets,params):
        g=p.grad
        if g is None: continue
        any_grad=True; buf[s:e].copy_(g.reshape(-1))
    return buf if any_grad else None

def _lora_params(model): return [(name,p) for name,p in model.named_parameters() if 'lora_' in name]

def _lora_offsets(params):
    offsets,total=[],0
    for _,p in params: offsets.append((total,total+p.numel())); total+=p.numel()
    return offsets,total

def _compute_reference_direction(model,ref_rows,batch_size=2):
    """Mean LoRA gradient direction over held-out clean samples (before training)."""
    model.train()
    for _,p in model.named_parameters():
        if p.requires_grad and p.grad is not None: p.grad=None
    n=len(ref_rows)
    for s in range(0,n,batch_size):
        chunk=ref_rows[s:s+batch_size]; loss=0.0
        for row in chunk:
            out=model(input_ids=row['input_ids'].unsqueeze(0).cuda(),labels=row['labels'].unsqueeze(0).cuda())
            loss=loss+out.loss
        (loss/n).backward()
    params=_lora_params(model); offsets,total=_lora_offsets(params)
    flat=torch.zeros(total,device='cuda',dtype=torch.float32); _fill_flat(params,offsets,flat)
    ref_buf=flat/torch.linalg.vector_norm(flat)
    torch.cuda.empty_cache()
    for _,p in model.named_parameters():
        if p.requires_grad and p.grad is not None: p.grad=None
    return params,offsets,total,ref_buf

def _layer_index(name):
    m=re.search(r'layers\.(\d+)',name); return int(m.group(1)) if m else -1

def _window_layer_grad_norms(model):
    norms={}
    for name,p in model.named_parameters():
        if 'lora_' in name and p.grad is not None:
            li=_layer_index(name)
            if li>=0: norms[li]=norms.get(li,0.0)+(p.grad.float()**2).sum().item()
    return {li:math.sqrt(v) for li,v in norms.items()}

def _window_layer_weight_norms(model):
    norms={}
    for name,p in model.named_parameters():
        if 'lora_' in name and p.requires_grad:
            li=_layer_index(name)
            if li>=0: norms[li]=norms.get(li,0.0)+(p.detach().float()**2).sum().item()
    return {li:math.sqrt(v) for li,v in norms.items()}

@torch.no_grad()
def _log_histograms(writer,model,step,target_ids):
    for name,p in model.named_parameters():
        if 'lora_' not in name: continue
        li=_layer_index(name)
        if li not in target_ids: continue
        tag=name.replace('base_model.model.','').replace('.default.weight','')
        writer.add_histogram(f'lora_w/{tag}',p.detach().float().cpu(),step)
        if p.grad is not None: writer.add_histogram(f'lora_g/{tag}',p.grad.detach().float().cpu(),step)

@torch.no_grad()
def _eval_heldout(model,eval_rows,bs=8):
    model.eval(); pad=model.config.pad_token_id or 0
    total=0.0; n=len(eval_rows)
    for s in range(0,n,bs):
        chunk=eval_rows[s:s+bs]; maxl=max(len(r['input_ids']) for r in chunk)
        ids=torch.full((len(chunk),maxl),pad,dtype=torch.long); labels=torch.full((len(chunk),maxl),-100,dtype=torch.long); mask=torch.zeros((len(chunk),maxl),dtype=torch.long)
        for i,r in enumerate(chunk):
            L=len(r['input_ids']); ids[i,:L]=r['input_ids']; labels[i,:L]=r['labels']; mask[i,:L]=1
        out=model(input_ids=ids.cuda(),labels=labels.cuda(),attention_mask=mask.cuda())
        total+=out.loss.item()*len(chunk)
    model.train(); return total/n

@torch.no_grad()
def _diagnostic_pass(model,rows,pad,bs=8,thresh=4.0,top_k=32):
    """Per-token diagnostics on a subsample. Returns per-sample aggregates plus
    top-k hardest label tokens for token-level analysis:
      user_loss  : mean CE over the USER (prompt) tokens
      entropy    : mean next-token entropy over label tokens
      skew/kurt  : shape of the per-token loss distribution
      top_tokens : [[pos, token_id, loss], ...] for the hardest tokens
    """
    model.eval(); out=[]
    for s in range(0,len(rows),bs):
        chunk=rows[s:s+bs]; maxl=max(len(r['input_ids']) for r in chunk)
        ids=torch.full((len(chunk),maxl),pad,dtype=torch.long); labels=torch.full((len(chunk),maxl),-100,dtype=torch.long); mask=torch.zeros((len(chunk),maxl),dtype=torch.long)
        for i,r in enumerate(chunk):
            L=len(r['input_ids']); ids[i,:L]=r['input_ids']; labels[i,:L]=r['labels']; mask[i,:L]=1
        logits=model(input_ids=ids.cuda(),attention_mask=mask.cuda()).logits
        B,L,V=logits.shape
        shift=logits[:,:-1].reshape(-1,V)
        ce=F.cross_entropy(shift,ids[:,1:].reshape(-1).cuda(),reduction='none').view(B,L-1)
        att=mask[:,1:]
        label_mask=(labels[:,1:]!=-100)*att
        user_mask=(labels[:,1:]==-100)*att
        flat_lm=label_mask.reshape(-1).bool()
        ent_all=None
        if flat_lm.any():
            lse=torch.log_softmax(shift[flat_lm],dim=-1)
            ent_flat=torch.zeros(B*(L-1),device=ce.device)
            ent_flat[flat_lm]=(-(torch.exp(lse)*lse).sum(-1)).float()
            ent_all=ent_flat.view(B,L-1)
        shift_ids=ids[:,1:]
        for i,r in enumerate(chunk):
            lm=label_mask[i].bool()
            if not lm.any(): continue
            toks=ce[i][lm].float(); pos=lm.nonzero().flatten()
            agg={'sample_id':r['sample_id'],'mean_loss':toks.mean().item(),'max_token_loss':toks.max().item(),
                 'frac_hard':(toks>thresh).float().mean().item(),'user_loss':None,'entropy':None,
                 'token_loss_skew':None,'token_loss_kurt':None,'top_tokens':[]}
            um=user_mask[i].bool()
            if um.any(): agg['user_loss']=ce[i][um].float().mean().item()
            if ent_all is not None:
                ent=ent_all[i][lm].float(); agg['entropy']=ent.mean().item()
            toks_np=toks.cpu().numpy()
            if len(toks_np)>3:
                from scipy.stats import skew,kurtosis
                agg['token_loss_skew']=float(skew(toks_np)); agg['token_loss_kurt']=float(kurtosis(toks_np))
            k=min(top_k,len(toks)); v,idx=toks.topk(k)
            for val,ix in zip(v.tolist(),idx.tolist()):
                agg['top_tokens'].append([int(pos[ix].item()),int(shift_ids[i,pos[ix]].item()),float(val)])
            out.append(agg)
    model.train(); return out


class LoRA:
    name='hf-lora'
    def __init__(self,settings): self.settings=settings; self.model=None

    def fit(self,rows,output,**kwargs):
        try:
            from transformers import AutoModelForCausalLM,AutoTokenizer
            from peft import LoraConfig,get_peft_model
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e: raise RuntimeError('Install torch, transformers and peft for hf-lora') from e

        tcfg=self.settings.section('train'); model_id=self.settings.section('paths').get('model')
        smoke=kwargs.get('smoke',False)
        seed=tcfg.get('seed',42); torch.manual_seed(seed)
        max_len=tcfg.get('max_len',1024)
        run_dir=Path(output); metric_dir=run_dir/'metrics'; metric_dir.mkdir(parents=True,exist_ok=True)

        train_rows=list(rows)
        if smoke: train_rows=train_rows[:64]
        heldout_rows=list(Jsonl(self.settings.data_dir()/'heldout.jsonl').rows())
        n_ref=tcfg.get('ref_samples',200); n_held=tcfg.get('heldout_samples',200)

        model=AutoModelForCausalLM.from_pretrained(model_id,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':0})
        model.config.use_cache=False
        model=get_peft_model(model,LoraConfig(r=tcfg.get('lora_r',32),lora_alpha=tcfg.get('lora_alpha',64),lora_dropout=tcfg.get('lora_dropout',.05),target_modules=TARGET_MODULES,bias='none',task_type='CAUSAL_LM'))
        tokenizer=AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token

        train_data,n_label_tokens=_tokenize_rows(tokenizer,train_rows,max_len)
        held_data,_=_tokenize_rows(tokenizer,heldout_rows[:n_held],max_len)
        ref_data,_=_tokenize_rows(tokenizer,heldout_rows[n_held:n_held+n_ref],max_len)

        params,offsets,n_lora,ref_buf=_compute_reference_direction(model,ref_data)
        b_offsets=[(s,e) for (s,e),(name,_) in zip(offsets,params) if 'lora_B' in name]
        total_b=sum(e-s for s,e in b_offsets)
        buf_before=torch.zeros(n_lora,device='cuda',dtype=torch.float32)
        buf_after=torch.zeros(n_lora,device='cuda',dtype=torch.float32)
        delta_buf=torch.zeros(n_lora,device='cuda',dtype=torch.float32)
        delta_b=torch.zeros(total_b,device='cuda',dtype=torch.float32)
        v_buf=torch.zeros(total_b,device='cuda',dtype=torch.float32); v_ready=False

        lr=tcfg.get('lr',2e-4); n_train=len(train_data)
        epochs=kwargs.get('epochs',tcfg.get('epochs',1))
        grad_accum=tcfg.get('grad_accum',16)
        steps_per_epoch=math.ceil(n_train/grad_accum); total_steps=steps_per_epoch*epochs
        warmup_steps=int(total_steps*tcfg.get('warmup_ratio',.03))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=tcfg.get('weight_decay',.01),betas=(0.9,0.999))

        def lr_at(step):
            if step<warmup_steps: return lr*(step+1)/max(1,warmup_steps)
            p=(step-warmup_steps)/max(1,total_steps-warmup_steps); return lr*0.5*(1+math.cos(math.pi*p))

        writer=SummaryWriter(run_dir/'tb')
        metric_f=(metric_dir/'per_sample.jsonl').open('w')
        ln_f=(metric_dir/'layer_norms.jsonl').open('w')
        n_layers=model.base_model.model.config.num_hidden_layers if hasattr(model,'base_model') else model.config.num_hidden_layers
        target_ids={0,n_layers//2,n_layers-1}
        layer_norm_sum={li:0.0 for li in target_ids}; layer_norm_cnt=0
        eval_steps=tcfg.get('eval_steps',200); log_every=max(1,tcfg.get('log_every',25))
        diag_step=max(1,tcfg.get('diag_subsample',8)); hard_threshold=tcfg.get('hard_threshold',4.0)
        global_step=0; t0=time.time(); tb_t0=t0
        tb_sum={'loss':0.0,'grad_norm':0.0,'cos_ref':0.0,'cos_global':0.0,'update_contrib':0.0,'tokens':0.0,'cnt':0}
        epoch_stats=[]

        def flush_window(acc,epoch,step):
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
                metric_f.write(json.dumps({'step':step,'epoch':epoch,'sample_id':sid,'loss':l,'grad_norm':gn,
                                            'cos_sim_ref':cr,'cos_sim_global':cg,'update_contrib':up,
                                            'tokens':max(1,n_label_tokens[sid])})+'\n')
                if l==l and gn==gn:
                    tb_sum['loss']+=l; tb_sum['grad_norm']+=gn
                    if cr is not None and cr==cr: tb_sum['cos_ref']+=cr
                    if cg is not None and cg==cg: tb_sum['cos_global']+=cg
                    if up==up: tb_sum['update_contrib']+=up
                    tb_sum['tokens']+=max(1,n_label_tokens[sid]); tb_sum['cnt']+=1
            metric_f.flush()

        for epoch in range(epochs):
            acc=[]
            for i,row in enumerate(train_data):
                if smoke and global_step>=8: break
                if row['n_label_tokens']==0: continue
                out=model(input_ids=row['input_ids'].unsqueeze(0).cuda(),labels=row['labels'].unsqueeze(0).cuda())
                loss=out.loss
                has_before=_fill_flat(params,offsets,buf_before) is not None
                loss.backward()
                _fill_flat(params,offsets,buf_after)
                if has_before:
                    delta_buf.copy_(buf_after).sub_(buf_before)
                    bsq=torch.dot(buf_before,buf_before); dot_b=torch.dot(delta_buf,buf_before)
                else:
                    delta_buf.copy_(buf_after); bsq=dot_b=None
                g_norm=torch.linalg.vector_norm(delta_buf)
                dot_ref=torch.dot(delta_buf,ref_buf); cos_ref=dot_ref/g_norm
                upd=None
                if v_ready:
                    o=0
                    for s,e in b_offsets: delta_b[o:o+e-s].copy_(delta_buf[s:e]); o+=e-s
                    upd=torch.linalg.vector_norm(delta_b)/(torch.linalg.vector_norm(v_buf.sqrt())+1e-8)
                acc.append((row['sample_id'],loss,g_norm,dot_ref,dot_b,bsq,upd))

                if (i+1)%grad_accum==0:
                    all_norms=_window_layer_grad_norms(model)
                    for li in target_ids: layer_norm_sum[li]+=all_norms.get(li,0.0)
                    layer_norm_cnt+=1
                    if (global_step+1)%eval_steps==0: _log_histograms(writer,model,global_step+1,target_ids)
                    for g in opt.param_groups: g['lr']=lr_at(global_step)
                    opt.step()
                    ln_f.write(json.dumps({'step':global_step+1,'grad':all_norms,'weight':_window_layer_weight_norms(model)})+'\n'); ln_f.flush()
                    o=0
                    for (s,e),(name,p) in zip(offsets,params):
                        if 'lora_B' not in name: continue
                        st=opt.state.get(p)
                        if st and 'exp_avg_sq' in st: v_buf[o:o+e-s].copy_(st['exp_avg_sq'].reshape(-1))
                        o+=e-s
                    v_ready=True
                    opt.zero_grad()
                    flush_window(acc,epoch,global_step); acc=[]
                    global_step+=1
                    if global_step%log_every==0:
                        now=time.time(); cnt=max(1,tb_sum['cnt'])
                        for key in ('loss','grad_norm','cos_ref','cos_global','update_contrib'):
                            writer.add_scalar(f'train/{key}',tb_sum[key]/cnt,global_step)
                        writer.add_scalar('train/lr',lr_at(global_step),global_step)
                        writer.add_scalar('train/tokens_per_sec',tb_sum['tokens']/max(1.0,now-tb_t0),global_step)
                        writer.add_scalar('train/gpu_mem_GB',torch.cuda.memory_allocated()/1e9,global_step)
                        if layer_norm_cnt:
                            for li in target_ids: writer.add_scalar(f'lora_layer_gradnorm/layer{li}',layer_norm_sum[li]/layer_norm_cnt,global_step)
                        writer.flush(); tb_t0=now
                        tb_sum={'loss':0.0,'grad_norm':0.0,'cos_ref':0.0,'cos_global':0.0,'update_contrib':0.0,'tokens':0.0,'cnt':0}
                        layer_norm_sum={li:0.0 for li in target_ids}; layer_norm_cnt=0
                    if global_step%eval_steps==0:
                        el=_eval_heldout(model,held_data); writer.add_scalar('eval/heldout_loss',el,global_step); writer.flush()

            if acc:
                for g in opt.param_groups: g['lr']=lr_at(global_step)
                opt.step(); opt.zero_grad(); flush_window(acc,epoch,global_step); acc=[]; global_step+=1

            ep_metrics=[]
            with (metric_dir/'per_sample.jsonl').open() as f:
                for line in f:
                    r=json.loads(line)
                    if r['epoch']==epoch and r['loss'] is not None: ep_metrics.append(r)
            n_ep=len(ep_metrics)
            if n_ep:
                def avg(key):
                    vals=[r[key] for r in ep_metrics if r.get(key) is not None and r[key]==r[key]]
                    return sum(vals)/len(vals) if vals else None
                epoch_stats.append({'epoch':epoch,'n':n_ep,'loss_mean':avg('loss'),
                                     'loss_min':min(r['loss'] for r in ep_metrics),'loss_max':max(r['loss'] for r in ep_metrics),
                                     'grad_norm_mean':avg('grad_norm'),'cos_ref_mean':avg('cos_sim_ref'),'cos_global_mean':avg('cos_sim_global')})

            diag_rows=train_data[::diag_step]
            pad=tokenizer.pad_token_id
            diag=_diagnostic_pass(model,diag_rows,pad,thresh=hard_threshold)
            with (metric_dir/f'diag_epoch{epoch}.jsonl').open('w') as f:
                for d in diag: f.write(json.dumps({k:v for k,v in d.items() if k!='top_tokens'})+'\n')
            with (metric_dir/f'token_diag_epoch{epoch}.jsonl').open('w') as f:
                for d in diag: f.write(json.dumps({'sample_id':d['sample_id'],'top_tokens':d['top_tokens']})+'\n')
            if diag:
                writer.add_scalar('diag/max_token_loss_mean',sum(d['max_token_loss'] for d in diag)/len(diag),global_step)
                writer.add_scalar('diag/frac_hard_mean',sum(d['frac_hard'] for d in diag)/len(diag),global_step)
                writer.flush()

            if smoke: break

        metric_f.close(); ln_f.close()
        model.eval()
        model.save_pretrained(run_dir/'lora'); tokenizer.save_pretrained(run_dir/'lora')
        writer.close()
        summary={'model':model_id,'samples':n_train,'epochs':epochs,'total_steps':global_step,
                 'seconds':round(time.time()-t0,1),'lora_params':n_lora,'epochs_detail':epoch_stats,'status':'ok'}
        (run_dir/'summary.json').write_text(json.dumps(summary,indent=2))
        return summary

    def predict(self,prompts,**kwargs): return ['']*len(prompts)


from model import Mock
from model import LoRA
def create(name,settings):
    if name=='mock': return Mock()
    if name in ('hf-lora','lora'): return LoRA(settings)
    raise ValueError(f'Unknown model: {name}')
