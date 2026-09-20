import math
import re
import torch
import torch.nn.functional as F

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
