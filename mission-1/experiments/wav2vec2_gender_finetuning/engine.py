"""Bounded-memory exact call-mean gradient, fixed frozen lower backbone."""
import contextlib
import hashlib
import math
import os
import random
import time
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from common import MODEL, REVISION, require, save


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(r):
    random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch']); torch.cuda.set_rng_state_all(r['cuda'])


def tensor_hash(named):
    h=hashlib.sha256()
    for name,p in named:
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def frozen_hash(model):
    return tensor_hash((n,p) for n,p in model.named_parameters() if not p.requires_grad)


def group_hashes(model):
    groups={}
    for n,p in model.named_parameters():
        if p.requires_grad:
            key='.'.join(n.split('.')[:4]) if n.startswith('backbone.encoder.layers.') else 'head'
            groups.setdefault(key,[]).append((n,p))
    return {k:tensor_hash(v) for k,v in groups.items()}


class Classifier(torch.nn.Module):
    def __init__(self, layers, cfg, device='cuda'):
        super().__init__()
        require(layers in (0,2,4),'Only preregistered layer counts')
        self.device=torch.device(device)
        cache='/content/gender_model_hf_cache'
        self.extractor=Wav2Vec2FeatureExtractor.from_pretrained(MODEL,revision=REVISION,cache_dir=cache)
        self.backbone=Wav2Vec2Model.from_pretrained(MODEL,revision=REVISION,cache_dir=cache,attn_implementation='eager')
        require(len(self.backbone.encoder.layers)==12 and self.backbone.config.hidden_size==768,'Checkpoint architecture')
        require(self.extractor.sampling_rate==16000 and self.extractor.do_normalize,'Extractor normalization')
        self.backbone.requires_grad_(False)
        if layers:
            for layer in self.backbone.encoder.layers[-layers:]:
                layer.requires_grad_(True)
        self.head=torch.nn.Linear(768,2)
        self.layers=layers
        require(not cfg['gradient_checkpointing'],'Two-pass streaming replaces gradient checkpointing in this version')
        self.precision=cfg['precision']; require(self.precision in ('bf16','fp32'),'Supported precision')
        self.to(self.device)
        # eval disables dropout, layerdrop and SpecAugment in BOTH passes, but NOT autograd.
        # Required for the two-pass chain rule to represent the same function.
        self.eval()

    def amp(self):
        return torch.autocast(device_type=self.device.type,dtype=torch.bfloat16) if self.precision=='bf16' else contextlib.nullcontext()

    def encode(self, chunk):
        with self.amp():
            h=self.backbone(input_values=chunk[None].to(self.device)).last_hidden_state
        require(bool(torch.isfinite(h).all()),'Nonfinite hidden state')
        return h.float().mean(dim=1)[0],h.shape[1]

    def logits(self,z):
        # FP32 head and loss; bf16 confined to backbone forward.
        return self.head(z.float())


def audio_chunks(job, data_root, extractor):
    start=time.perf_counter()
    wav=Path(data_root)/job['wav_relative_path']
    audio,sr=sf.read(wav,dtype='float32',always_2d=False)
    require(sr==8000 and audio.ndim==1 and np.isfinite(audio).all(),'Expected finite 8kHz mono WAV')
    result=[]; source_samples=0
    for a,b in job['intervals']:
        lo,hi=round(a*sr),round(b*sr)
        require(0<=lo<hi<=len(audio),'Caller bounds')
        source_samples+=hi-lo
        x=resample_poly(audio[lo:hi],2,1,window=('kaiser',5.0)).astype(np.float32)
        if len(x)<400:
            x=np.pad(x,(0,400-len(x)))
        x=extractor(x,sampling_rate=16000,return_tensors='pt').input_values[0]
        # Entire segment normalized BEFORE equal splitting, same as frozen baseline.
        chunks=list(torch.tensor_split(x,math.ceil(len(x)/240000)))
        require(all(len(c)>=400 for c in chunks),'Chunk too short')
        result.append(chunks)
    return result,source_samples,time.perf_counter()-start


def call_embedding(model, segments):
    means=[]; weights=[]
    with torch.no_grad():
        for chunks in segments:
            encoded=[model.encode(c) for c in chunks]
            frames=sum(n for _,n in encoded)
            means.append(sum(z*(n/frames) for z,n in encoded))
            weights.append([n/frames/len(segments) for _,n in encoded])
        z=torch.stack(means).mean(0)
    require(bool(torch.isfinite(z).all()),'Nonfinite embedding')
    return z,weights


def backward_call(model, segments, target, divisor, scaler):
    z,weights=call_embedding(model,segments)
    leaf=z.detach().requires_grad_(True)
    loss=torch.nn.functional.cross_entropy(model.logits(leaf)[None],torch.tensor([target],device=model.device))/divisor
    require(bool(torch.isfinite(loss)),'Nonfinite loss')
    scaler.scale(loss).backward()
    if model.layers:
        # leaf.grad already contains loss/divisor and optional scaler factor.
        for chunks,ww in zip(segments,weights):
            for chunk,w in zip(chunks,ww):
                part,_=model.encode(chunk)
                part.backward(leaf.grad*w)
    return float(loss.detach())*divisor


def optimizer_for(model,cfg):
    backbone=[p for p in model.backbone.parameters() if p.requires_grad]
    head=list(model.head.parameters())
    groups=[dict(params=head,lr=cfg['head_lr'],name='head')]
    if backbone:
        groups.append(dict(params=backbone,lr=cfg['backbone_lr'],name='backbone'))
    opt=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    require(len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad},'Optimizer parameter mismatch')
    require(not any(p.requires_grad for p in model.backbone.feature_extractor.parameters()),'Conv must freeze')
    audit={n:dict(numel=p.numel(),trainable=p.requires_grad) for n,p in model.named_parameters()}
    # Constant LR deliberately chosen so refit horizon cannot change the LR schedule.
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1.0)
    scaler=torch.amp.GradScaler('cuda',enabled=False) # bf16/FP32 need no loss scaling; state still checkpointed.
    return opt,scheduler,scaler,audit


def checkpoint(path, obj):
    start=time.perf_counter(); tmp=Path(str(path)+'.tmp')
    with tmp.open('wb') as f:
        torch.save(obj,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    return time.perf_counter()-start


def train(model,jobs,cfg,out,identity,epochs,dev_callback=None,max_steps=None):
    """Cursor is next call in a deterministic epoch permutation; commits only at optimizer boundaries."""
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    opt,sched,scaler,audit=optimizer_for(model,cfg)
    save(out/'parameters.json',dict(parameters=audit,groups=[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups]))
    frozen=frozen_hash(model); initial=group_hashes(model)
    epoch=cursor=step=0; history=[]; timing=dict(read_preprocess=0.,forward_backward=0.,optimizer=0.,checkpoint=0.,dev=0.,calls=0,steps=0)
    path=out/'last.pt'
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=False)
        require(state['identity']==identity,'Resume identity differs')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng']); epoch,cursor,step=state['epoch'],state['cursor'],state['step']
        history=state['history']; timing=state['timing']; initial=state['initial']
        require(state['frozen']==frozen_hash(model)==frozen,'Frozen parameters changed')
    def commit():
        nonlocal timing
        elapsed=checkpoint(path,dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),epoch=epoch,cursor=cursor,step=step,history=history,timing=timing.copy(),initial=initial,frozen=frozen,sampler=dict(seed=cfg['seed'],epoch=epoch,next_call=cursor)))
        timing['checkpoint']+=elapsed
        save(out/'progress.json',dict(epoch=epoch,next_call=cursor,step=step,timing=timing))
    while epoch<epochs and (max_steps is None or step<max_steps):
        order=list(range(len(jobs))); random.Random(cfg['seed']+epoch).shuffle(order)
        while cursor<len(order) and (max_steps is None or step<max_steps):
            batch=order[cursor:cursor+cfg['accumulation_calls']]
            opt.zero_grad(set_to_none=True)
            for i in batch:
                segments,_,elapsed=audio_chunks(jobs[i],cfg['data_root'],model.extractor)
                timing['read_preprocess']+=elapsed
                torch.cuda.synchronize(); start=time.perf_counter()
                loss=backward_call(model,segments,int(jobs[i]['gender']=='M'),len(batch),scaler)
                torch.cuda.synchronize(); timing['forward_backward']+=time.perf_counter()-start
                timing['calls']+=1
            torch.cuda.synchronize(); optimizer_start=time.perf_counter()
            scaler.unscale_(opt)
            params=[p for p in model.parameters() if p.requires_grad]
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),'Missing/nonfinite gradients')
            norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            scaler.step(opt); scaler.update(); sched.step()
            require(all(bool(torch.isfinite(p).all()) for p in params),'Nonfinite parameter update')
            torch.cuda.synchronize(); timing['optimizer']+=time.perf_counter()-optimizer_start
            cursor+=len(batch); step+=1; timing['steps']+=1
            if step==1:
                after=group_hashes(model)
                require(all(initial[k]!=after[k] for k in initial),'Trainable head/layer did not update')
                require(frozen_hash(model)==frozen,'Frozen layer changed on first step')
                save(out/'update_audit.json',dict(initial=initial,after_first_step=after,frozen_sha256=frozen,passed=True))
            if step%cfg['checkpoint_steps']==0:
                commit()
            if step%10==0:
                print(f'{out.name} epoch={epoch+1} calls={cursor}/{len(order)} step={step} loss={loss:.5f} grad={float(norm):.4f}',flush=True)
        if cursor==len(order):
            require(frozen_hash(model)==frozen,'Frozen layer changed')
            if dev_callback:
                start=time.perf_counter(); score=dev_callback(model,epoch+1); timing['dev']+=time.perf_counter()-start
                history.append(dict(epoch=epoch+1,**score))
            epoch+=1; cursor=0
        commit()
    require(frozen_hash(model)==frozen,'Final frozen integrity')
    final=group_hashes(model)
    require(all(initial[k]!=final[k] for k in initial),'Trainable groups unchanged')
    result=dict(status='complete',identity=identity,epochs=epoch,steps=step,history=history,timing=timing,
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),frozen_sha256=frozen,
                trainable_groups_changed=True)
    save(out/'training.json',result)
    return result


def evaluate(model,jobs,cfg,out,identity):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    from common import read,validate,metric
    rr=[]; expected={j['call_id'] for j in jobs}
    require({p.stem for p in out.glob('*.json')}<=expected,'Unexpected evaluation files')
    for j in jobs:
        p=out/(j['call_id']+'.json')
        if p.exists():
            r=read(p); validate([r],[j],identity)
        else:
            start=time.perf_counter()
            segments,samples,read_s=audio_chunks(j,cfg['data_root'],model.extractor)
            with torch.inference_mode():
                z,_=call_embedding(model,segments)
                probs=model.logits(z).softmax(-1).cpu().tolist()
            r=dict(call_id=j['call_id'],gender=j['gender'],prediction='M' if probs[1]>probs[0] else 'F',probabilities_F_M=probs,
                   identity=identity,segments=len(segments),source_samples=samples,read_preprocess_seconds=read_s,elapsed_seconds=time.perf_counter()-start)
            validate([r],[j],identity); save(p,r)
        rr.append(r)
        if len(rr)%100==0: print(f'Evaluate {out.name}: {len(rr)}/{len(jobs)}',flush=True)
    validate(rr,jobs,identity)
    return metric(rr),rr
