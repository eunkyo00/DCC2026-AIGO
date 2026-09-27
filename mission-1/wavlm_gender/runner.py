"""T4/L4 bounded experiment. CPU-only result checks live in verify.py."""
import argparse
import contextlib
import io
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
import warnings
import zipfile
from common import *
from portable import validate_paths, require_dataset_check


def gpu_idle():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    require(not out, 'Other GPU processes detected; stop other experiment first:\n'+out)


def setup(config):
    import importlib.metadata as im
    import torch
    import torchaudio
    require(torch.cuda.is_available(), 'CUDA GPU required')
    profile=gpu_profile(torch.cuda.get_device_name(0), config['precision'])
    require(config.get('expected_gpu', torch.cuda.get_device_name(0)) == torch.cuda.get_device_name(0), 'GPU changed since notebook setup; use a new matching output')
    require(torch.__version__.split('+')[0]==torchaudio.__version__.split('+')[0], 'torch/torchaudio version mismatch')
    if '+' in torch.__version__:
        require(torch.__version__.split('+')[1]==torchaudio.__version__.split('+')[-1], 'CUDA wheel mismatch')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    validate_paths(config)
    require_dataset_check(config)
    require(Path(config['data_root']).is_dir(),'Data root absent')
    m,jobs=bundle()
    packages={n:im.version(n) for n in ('torch','torchaudio','numpy','scipy','transformers','tokenizers','huggingface-hub','scikit-learn','soundfile')}
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if '==' in line:
            name,version=line.split('==')
            require(im.version(name)==version,f'Package mismatch: {name}')
    ident=dict(bundle_sha256=sha(ROOT/'bundle_manifest.json'), source_identity_sha256=None, input_provenance='verified bundled metadata; fresh teammate execution',
               model=MODEL,revision=REVISION,config=config,python=platform.python_version(),packages=packages,
               gpu=torch.cuda.get_device_name(0),gpu_profile=profile,cuda=torch.version.cuda,source_hashes=SOURCE_HASHES,
               inner_split_sha256=sha(ROOT/'data/inner_split.csv'),precision=config['precision'],
               window_seconds=WINDOW,layers=LAYERS,normalization='checkpoint processor on each real window, then pad to 400',
               pooling='real-sample-weighted first/second moments across all balanced caller windows',
               class_order=['F','M'],tf32=False,backbone_dropout=False,spec_augment=False)
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'identity.json').exists():require(read(out/'identity.json')==ident,'Identity changed; use a NEW experiment output')
    else:save(out/'identity.json',ident)
    return out,ident,jobs


def seed_all(seed=SEED):
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def memory(out):
    import psutil
    import torch
    return dict(rss_GiB=psutil.Process().memory_info().rss/2**30,
                available_RAM_GiB=psutil.virtual_memory().available/2**30,
                peak_VRAM_GiB=torch.cuda.max_memory_allocated()/2**30,
                reserved_VRAM_GiB=torch.cuda.max_memory_reserved()/2**30,
                drive_free_GiB=shutil.disk_usage(out).free/2**30)


def representative(jobs,n):
    # Deterministic length quantiles within each sex; Train only.
    result=[]
    for g in ('F','M'):
        jj=sorted([j for j in jobs if j['gender']==g],key=lambda j:(samples(j),j['call_id']))
        result.extend(jj[min(len(jj)-1,int((k+.5)*len(jj)/(n//2)))] for k in range(n//2))
    return result


def preflight(out,ident,jobs):
    import numpy as np
    import torch
    from audio_model import Encoder,load_wave
    require(shutil.disk_usage(ident['config']['hf_cache_dir']).free>3*2**30,'Need >=3 GiB local disk for pinned HF weights and temporary files')
    require(shutil.disk_usage(out).free>8*2**30,'Need >=8 GiB free output storage for pooled cache/checkpoints/temp files')
    import psutil
    require(psutil.virtual_memory().available>5*2**30,'Need >=5 GiB available RAM (LR copies included)')
    seed_all()
    model_start=time.perf_counter();enc=Encoder(ident['precision']);model_load_seconds=time.perf_counter()-model_start
    save(out/'initial_parameters.json',enc.parameter_record())
    # Hash actually resolved initial model files, not only repo revision.
    from huggingface_hub import snapshot_download
    snap=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=['config.json','preprocessor_config.json','*.safetensors','pytorch_model.bin'],local_files_only=True))
    weights={p.name:sha(p) for p in snap.iterdir() if p.is_file() and p.suffix in ('.bin','.safetensors','.json')}
    require(any(n.endswith(('.bin','.safetensors')) for n in weights),'Initial weights absent')
    train=[j for j in jobs if j['partition']=='train']
    errors=[]
    for j in representative(train,6):
        x=load_wave(j,ident['config']['data_root'])[:WINDOW*16000]
        enc.precision='fp32';a=enc.pooled(x).cpu().numpy()
        enc.precision=ident['precision'];b=enc.pooled(x).cpu().numpy()
        rel=float(np.linalg.norm(a-b)/max(1e-12,np.linalg.norm(a)))
        require(np.isfinite(b).all() and rel<.05,'Mixed precision numeric check failed; use new FP32 config/output')
        errors.append(dict(call_id=j['call_id'],relative_l2=rel,max_abs=float(np.abs(a-b).max())))
    save(out/'preflight.json',dict(identity=digest(ident),initial_files=weights,numeric_reference='fp32',numeric_candidate=ident['precision'],numeric_checks=errors,model_load_seconds=model_load_seconds,memory=memory(out),status='pass'))
    print(read(out/'preflight.json'),flush=True)


def audit(out,ident,jobs):
    import soundfile as sf
    from audio_model import load_wave,crop
    dst=out/'audio_review';dst.mkdir(exist_ok=True)
    by={j['call_id']:j for j in jobs if j['partition']=='train'}
    for r in rows(ROOT/'data/train_audio_review.csv'):
        j=by[r['call_id']];x=load_wave(j,ident['config']['data_root'])
        c,start=crop(x,ident['config']['crop_seconds'],SEED)
        sf.write(dst/(j['call_id']+'_caller.wav'),x,16000)
        sf.write(dst/(j['call_id']+'_crop.wav'),c,16000)
        append(dst/'clips.jsonl',dict(call_id=j['call_id'],crop_start_sample=start,crop_samples=len(c),gender=j['gender']))
    if not (dst/'review.csv').exists():shutil.copyfile(ROOT/'data/train_audio_review.csv',dst/'review.csv')
    print('Listen to caller AND crop; fill review.csv: review_status=reviewed, actual_speaker_matches_label=yes/no/uncertain, notes. No labels are changed.')


def audit_gate(out):
    p=out/'audio_review/review.csv';require(p.exists(),'Run audit and review Train audio first')
    rr=rows(p);expected={r['call_id'] for r in rows(ROOT/'data/train_audio_review.csv')}
    require(len(rr)==len(expected) and {r['call_id'] for r in rr}==expected,'Audio review coverage')
    require(all(r['review_status']=='reviewed' and r['actual_speaker_matches_label'] in ('yes','no','uncertain') for r in rr),'Pending audio review')
    require(all(r['actual_speaker_matches_label']=='yes' or r['notes'].strip() for r in rr),'Document mismatch/uncertainty')
    # No dropping or relabelling. Existence of risk is recorded and may justify stopping adaptation.
    return dict(sha256=sha(p),counts=dict(__import__('collections').Counter(r['actual_speaker_matches_label'] for r in rr)))


def cache_path(out,j):return out/'pooled'/j['call_id'][:2]/(j['call_id']+'.npz')


def cached(out,j,ih):
    import numpy as np
    p=cache_path(out,j)
    require(read(p.with_suffix('.hash.json'))['sha256']==sha(p),'Pooled cache hash mismatch')
    with np.load(p,allow_pickle=False) as d:
        require(str(d['identity'])==ih and str(d['call_id'])==j['call_id'] and int(d['samples'])==samples(j),'Cache identity/audio mismatch')
        a=d['features']
        require(a.shape==(4,2,1024) and a.dtype==np.float32 and np.isfinite(a).all(),'Invalid pooled features')
        return a.copy()


def extract(out,ident,jobs,partition):
    import numpy as np
    from audio_model import Encoder,load_wave
    ih=digest(ident);enc=None;t=time.perf_counter();done=0
    pending=[]
    for j in jobs:
        if j['partition']!=partition:continue
        if cache_path(out,j).exists():cached(out,j,ih)
        else:pending.append(j)
    for j in pending:
        p=cache_path(out,j)
        try:
            if enc is None:enc=Encoder(ident['precision'])
            start=time.perf_counter();x=load_wave(j,ident['config']['data_root'])
            a=enc.pooled(x).cpu().numpy();require(np.isfinite(a).all(),'Non-finite features')
            p.parent.mkdir(exist_ok=True,parents=True)
            buf=io.BytesIO();np.savez(buf,features=a,identity=ih,call_id=j['call_id'],samples=samples(j))
            atomic(p,buf.getvalue());save(p.with_suffix('.hash.json'),dict(sha256=sha(p)));cached(out,j,ih)
            append(out/'extraction.jsonl',dict(call_id=j['call_id'],partition=partition,identity=ih,status='complete',sha256=sha(p),seconds=time.perf_counter()-start))
            done+=1
            if done%10==0:
                remaining=len(pending)-done
                progress=dict(stage='extract',completed_session=done,remaining=remaining,ETA_seconds=(time.perf_counter()-t)/done*remaining,memory=memory(out))
                save(out/'progress.json',progress);print(progress,flush=True)
        except BaseException as e:
            append(out/'failures.jsonl',dict(stage='extract',call_id=j['call_id'],identity=ih,error=repr(e),time=time.time()));raise


def feature_matrix(out,ident,jobs,kind):
    import numpy as np
    # Last layer mean or 4 layer mean+std concat. No frame cache.
    a=np.stack([cached(out,j,digest(ident)) for j in jobs])
    return a[:,-1,0,:].copy() if kind=='last_mean' else a.reshape(len(a),-1)


def frozen_select(out,ident,jobs):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    inner=inner_assign(jobs);train=[j for j in jobs if j['partition']=='train']
    ix=np.array([inner[j['call_id']]=='inner_train' for j in train]);y=np.array([j['gender']=='M' for j in train],dtype=int)
    scores=[]
    for kind in ('last_mean','multi_mean_std'):
        X=feature_matrix(out,ident,train,kind)
        for C in (.1,1.):
            t=time.perf_counter();clf=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000,random_state=SEED))
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);clf.fit(X[ix],y[ix])
            pp=clf.predict_proba(X[~ix]);pred=pp.argmax(1)
            rr=[dict(call_id=j['call_id'],gender=j['gender'],prediction='FM'[int(p)],probabilities_F_M=prob.tolist()) for j,p,prob in zip([j for j in train if inner[j['call_id']]=='dev'],pred,pp)]
            scores.append(dict(kind=kind,C=C,metrics=metrics(rr),fit_seconds=time.perf_counter()-t))
            save(out/f'dev_lr_{kind}_{C}.json',rr)
        del X
    # Exact correct count; tie -> last_mean, smaller C (enumeration order).
    best=max(scores,key=lambda s:s['metrics']['correct'])
    save(out/'frozen_selection.json',dict(identity=digest(ident),candidates=scores,selected=best,
         adaptation_eligible=best['metrics']['accuracy']<.99,rule='accuracy, then last_mean, then smaller C'))
    print(read(out/'frozen_selection.json'),flush=True)


def make_train(mode,precision):
    import torch
    from audio_model import Encoder,Head
    seed_all();enc=Encoder(precision,mode);head=Head().cuda()
    opt=torch.optim.AdamW([{'params':list(head.parameters()),'lr':3e-4},
          {'params':[p for p in enc.parameters() if p.requires_grad],'lr':1e-5}],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=False) # BF16/FP32 do not use FP16 scaling
    return enc,head,opt,scaler


def checkpoint(path,enc,head,opt,sched,scaler,state,ih):
    import numpy as np
    import torch
    state=dict(state)
    record=dict(identity=ih,model={n:p.detach().cpu() for n,p in enc.named_parameters() if p.requires_grad},head=head.state_dict(),
          optimizer=opt.state_dict(),scheduler=sched.state_dict(),amp_scaler=scaler.state_dict(),state=state,
          rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
          initial_model=MODEL,revision=REVISION,model_storage='trainable delta; all frozen parameters reconstructed from pinned revision')
    tmp=path.with_suffix('.tmp')
    with open(tmp,'wb') as f:
        torch.save(record,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def restore(path,enc,head,opt=None,sched=None,scaler=None,ih=None):
    import numpy as np
    import torch
    d=torch.load(path,map_location='cpu',weights_only=False) # own trusted checkpoint only
    require(d['identity']==ih and d['revision']==REVISION,'Checkpoint identity mismatch')
    params=dict(enc.named_parameters())
    require(set(d['model'])=={n for n,p in params.items() if p.requires_grad},'Trainable parameter set mismatch')
    with torch.no_grad():
        for n,p in d['model'].items():params[n].copy_(p)
    head.load_state_dict(d['head'],strict=True)
    if opt is not None:
        opt.load_state_dict(d['optimizer']);sched.load_state_dict(d['scheduler']);scaler.load_state_dict(d['amp_scaler'])
        random.setstate(d['rng']['python']);np.random.set_state(d['rng']['numpy']);torch.set_rng_state(d['rng']['torch']);torch.cuda.set_rng_state_all(d['rng']['cuda'])
    return d['state']


def predict_neural(enc,head,x):
    import torch
    # Pool entire call before head; exact same call representation used by frozen extractor.
    with torch.no_grad():return head(enc.pooled(x)).float().softmax(-1).cpu().tolist()


def evaluate_neural(out,ident,jobs,enc,head,tag,model_hash):
    from audio_model import load_wave
    path=out/(tag+'.jsonl');ih=digest(dict(run=digest(ident),model=model_hash,tag=tag))
    rr=journal(path);done=check_predictions(rr,jobs,ih,complete=False)
    session_start=time.perf_counter();session_done=0
    for j in jobs:
        if j['call_id'] in done:continue
        try:
            t=time.perf_counter();x=load_wave(j,ident['config']['data_root']);pp=predict_neural(enc,head,x)
            r=dict(call_id=j['call_id'],gender=j['gender'],prediction='M' if pp[1]>pp[0] else 'F',probabilities_F_M=pp,
                   samples=len(x),windows=math.ceil(len(x)/(WINDOW*16000)),identity=ih,seconds=time.perf_counter()-t)
            check_predictions([r],[j],ih);append(path,r);rr.append(r);session_done+=1
            if session_done%10==0:
                progress=dict(stage=tag,completed=len(rr),total=len(jobs),remaining_seconds=(time.perf_counter()-session_start)/session_done*(len(jobs)-len(rr)))
                save(out/'evaluation_progress.json',progress);print(progress,flush=True)
        except BaseException as e:
            append(out/'failures.jsonl',dict(stage=tag,call_id=j['call_id'],identity=ih,error=repr(e),time=time.time()));raise
    check_predictions(rr,jobs,ih)
    return rr,ih


def training(out,ident,jobs,mode,final_epochs=None):
    import numpy as np
    import torch
    from audio_model import load_wave,crop,crop_stats
    inner=inner_assign(jobs);final=final_epochs is not None
    train=[j for j in jobs if j['partition']=='train' and (final or inner[j['call_id']]=='inner_train')]
    dev=[j for j in jobs if j['partition']=='train' and inner[j['call_id']]=='dev']
    epochs=final_epochs if final else 4
    accum=ident['config']['accumulation'];steps=math.ceil(len(train)/accum)
    dest=out/('final_'+mode if final else 'develop_'+mode);dest.mkdir(exist_ok=True)
    enc,head,opt,scaler=make_train(mode,ident['precision'])
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:max(0.,1-step/(4*steps)))
    head_initial=hashlib.sha256(b''.join(n.encode()+p.detach().cpu().numpy().tobytes() for n,p in sorted(head.named_parameters()))).hexdigest()
    ih=digest(ident);save(dest/'parameters.json',dict(backbone=enc.parameter_record(),head_initial_sha256=head_initial,head_seed=SEED,head_trainable=sum(p.numel() for p in head.parameters()),
         epochs=epochs,scheduler_horizon_epochs=4,accumulation=accum,microbatch=1,gradient_checkpointing=False,scaler_enabled=False))
    state=dict(epoch=0,next_batch=0,step=0,sampler_seed=SEED,order_sha256=None)
    latest=dest/'latest.pt'
    if latest.exists():
        state=restore(latest,enc,head,opt,sched,scaler,ih)
        attempted=journal(dest/'steps.jsonl')
        last=max([r['step'] for r in attempted]+[0])
        replay=max(0,last-state['step'])
        msg=dict(event='resume',committed_step=state['step'],previous_attempted_step=last,replayed_uncommitted_steps=replay,
                 guarantee='Only committed optimizer boundaries are durable; replay explicitly recorded',time=time.time())
        append(dest/'resume.jsonl',msg);print(msg,flush=True)
    elif (dest/'steps.jsonl').exists():
        last=max([r['step'] for r in journal(dest/'steps.jsonl')]+[0])
        msg=dict(event='resume_before_first_checkpoint',committed_step=0,previous_attempted_step=last,replayed_uncommitted_steps=last,time=time.time())
        append(dest/'resume.jsonl',msg);print(msg,flush=True)
    for epoch in range(state['epoch'],epochs):
        order=np.random.default_rng(SEED+epoch).permutation(len(train)).tolist()
        order_hash=digest([train[i]['call_id'] for i in order])
        if state['epoch']==epoch and state['next_batch']:
            require(state['order_sha256']==order_hash,'Sampler changed')
        start_batch=state['next_batch'] if state['epoch']==epoch else 0
        for batch in range(start_batch,steps):
            tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            indices=order[batch*accum:(batch+1)*accum]
            append(dest/'steps.jsonl',dict(event='attempt',epoch=epoch,batch=batch,step=state['step']+1))
            loss_total=0.
            for i in indices:
                j=train[i];x=load_wave(j,ident['config']['data_root'])
                crop_seed=int(hashlib.sha256(f'{SEED}|{epoch}|{j["call_id"]}'.encode()).hexdigest()[:16],16)
                x,_=crop(x,ident['config']['crop_seconds'],crop_seed)
                logits=head(crop_stats(enc(x)))
                loss=torch.nn.functional.cross_entropy(logits[None],torch.tensor([int(j['gender']=='M')],device='cuda'))/len(indices)
                require(torch.isfinite(loss).item(),'Nonfinite training loss')
                loss.backward();loss_total+=loss.item()
            params=[p for group in opt.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            opt.step();sched.step()
            state=dict(epoch=epoch,next_batch=batch+1,step=state['step']+1,sampler_seed=SEED,order_sha256=order_hash)
            if state['step']%100==0 or batch+1==steps:
                checkpoint(latest,enc,head,opt,sched,scaler,state,ih)
            torch.cuda.synchronize()
            append(dest/'timing.jsonl',dict(step=state['step'],loss=loss_total,seconds=time.perf_counter()-tick))
            if state['step']%10==0:
                timing=journal(dest/'timing.jsonl')[-100:];avg=sum(r['seconds'] for r in timing)/len(timing)
                progress=dict(stage=dest.name,epoch=epoch+1,step=state['step'],epoch_remaining_seconds=(steps-batch-1)*avg,
                              training_remaining_seconds=((epochs-epoch-1)*steps+steps-batch-1)*avg,memory=memory(out))
                save(out/'progress.json',progress);print(progress,flush=True)
        # Keep end-of-epoch latest until dev predictions and score are durably committed.
        if not final:
            model_hash=sha(latest)
            rr,_=evaluate_neural(dest,ident,dev,enc,head,f'dev_epoch_{epoch+1}',model_hash)
            metric=metrics(rr);result=dest/'selection.json'
            previous=read(result) if result.exists() else None
            if previous is None or metric['correct']>previous['metrics']['correct']:
                shutil.copyfile(latest,dest/'best.pt')
                save(result,dict(epoch=epoch+1,metrics=metric,checkpoint_sha256=sha(dest/'best.pt'),identity=ih))
            save(dest/f'epoch_{epoch+1}.json',dict(epoch=epoch+1,metrics=metric,checkpoint_sha256=model_hash))
        state=dict(epoch=epoch+1,next_batch=0,step=state['step'],sampler_seed=SEED,order_sha256=None)
        checkpoint(latest,enc,head,opt,sched,scaler,state,ih)
    save(dest/'completion.json',dict(identity=ih,checkpoint_sha256=sha(latest),epochs=epochs,steps=state['step'],status='complete'))


def benchmark(out,ident,jobs):
    import torch
    from audio_model import load_wave,crop,crop_stats
    train=[j for j in jobs if j['partition']=='train'];sample=representative(train,120)
    report={};accum=ident['config']['accumulation']
    for mode in ('frozen','top2'):
        model_start=time.perf_counter()
        enc,head,opt,scaler=make_train(mode,ident['precision']);model_load_seconds=time.perf_counter()-model_start
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        times=[];read_times=[];fb_times=[]
        torch.cuda.reset_peak_memory_stats()
        # Exactly 100 optimizer steps after 2 warmup steps. Includes real WAV reads/resampling.
        for step in range(102):
            start=time.perf_counter();opt.zero_grad(set_to_none=True);io_sec=0.;fb_sec=0.
            for k in range(accum):
                j=sample[(step*accum+k)%len(sample)];t=time.perf_counter()
                x=load_wave(j,ident['config']['data_root']);x,_=crop(x,ident['config']['crop_seconds'],SEED+step)
                io_sec+=time.perf_counter()-t;t=time.perf_counter()
                logits=head(crop_stats(enc(x)))
                loss=torch.nn.functional.cross_entropy(logits[None],torch.tensor([j['gender']=='M'],device='cuda',dtype=torch.long))/accum
                require(torch.isfinite(loss).item(),'Benchmark loss nonfinite');loss.backward();torch.cuda.synchronize();fb_sec+=time.perf_counter()-t
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],1.,error_if_nonfinite=True)
            opt.step();sched.step();torch.cuda.synchronize()
            if step>=2:times.append(time.perf_counter()-start);read_times.append(io_sec);fb_times.append(fb_sec)
            if step%10==0:print('benchmark',mode,step,'/102',flush=True)
        cp=out/('benchmark_'+mode+'.pt');start=time.perf_counter()
        checkpoint(cp,enc,head,opt,sched,scaler,dict(benchmark_only=True,steps=102),digest(ident))
        save_seconds=time.perf_counter()-start;cp_size=cp.stat().st_size;cp.unlink()
        ev=representative(train,60);start=time.perf_counter()
        sample_file=out/('benchmark_eval_'+mode+'.jsonl')
        if sample_file.exists():sample_file.unlink() # benchmark is disposable, never a Validation artifact
        for j in ev:
            pp=predict_neural(enc,head,load_wave(j,ident['config']['data_root']))
            append(sample_file,dict(call_id=j['call_id'],probabilities_F_M=pp))
        eval_seconds=time.perf_counter()-start
        avg=sum(times)/len(times);inner=inner_assign(jobs);ninner=sum(v=='inner_train' for v in inner.values());ndev=len(inner)-ninner
        epoch_train=math.ceil(ninner/accum)*avg+math.ceil(math.ceil(ninner/accum)/100)*save_seconds
        dev_eta=eval_seconds/len(ev)*ndev
        report[mode]=dict(model_load_seconds=model_load_seconds,backbone_total_parameters=sum(p.numel() for p in enc.parameters()),backbone_trainable_parameters=sum(p.numel() for p in enc.parameters() if p.requires_grad),optimizer_steps=100,microbatch=1,accumulation=accum,crop_seconds=ident['config']['crop_seconds'],
          step_seconds=avg,data_read_resample_seconds=sum(read_times)/100,forward_backward_seconds=sum(fb_times)/100,
          checkpoint_seconds=save_seconds,checkpoint_GiB=cp_size/2**30,evaluation_sample_calls=60,evaluation_wall_seconds=eval_seconds,
          epoch_train_seconds=epoch_train,epoch_with_dev_seconds=epoch_train+dev_eta,development_4_epochs_seconds=4*(epoch_train+dev_eta),
          final_refit_per_epoch_seconds=math.ceil(22388/accum)*avg+math.ceil(math.ceil(22388/accum)/100)*save_seconds,
          validation_5597_seconds=eval_seconds/60*5597,memory=memory(out),
          sampled_ids=[j['call_id'] for j in sample],evaluation_ids=[j['call_id'] for j in ev])
        del enc,head,opt,sched,scaler;torch.cuda.empty_cache()
    report['identity']=digest(ident)
    report['gpu']=ident['gpu']
    report['precision']=ident['precision']
    report['projection_note']='Measured current-GPU wall time, not a promise; length/sex stratified sample; includes storage reads and fsynced evaluation rows; repeated reads may be warm. GPU model load excluded; extraction and LR costs separate. Extraction projection omits per-NPZ/sidecar output-storage metadata overhead; actual extraction progress updates include it. The max-epoch plan is not a runtime upper bound.'
    report['frozen_extraction_all_train_seconds']=report['frozen']['evaluation_wall_seconds']/60*22388
    report['max_epoch_plan_seconds_excluding_model_load_LR_cache_overhead']=report['frozen_extraction_all_train_seconds']+sum(report[m]['development_4_epochs_seconds']+4*report[m]['final_refit_per_epoch_seconds']+report[m]['validation_5597_seconds'] for m in ('frozen','top2'))
    save(out/'benchmark.json',report);print(json.dumps(report,indent=2),flush=True)


def lock_selection(out,ident,jobs,adapt):
    frozen=read(out/'frozen_selection.json');require(frozen['identity']==digest(ident),'Selection identity')
    selection=dict(identity=digest(ident),frozen=frozen['selected'],adaptation=None,
                   refit='Reset pinned backbone and seed42 head; fit ALL 22388 Train. LR fit once; neural chosen epoch count, no dev early stopping; 4-epoch scheduler horizon.',
                   decision='F/M probability argmax; no Validation tuning',locked_at=time.time())
    if adapt:
        for mode in ('frozen','top2'):
            d=out/('develop_'+mode)
            c=read(d/'completion.json')
            require(c['status']=='complete' and c['epochs']==4 and c['identity']==digest(ident),'Finish all four development epochs before selection')
            require(sha(d/'best.pt')==read(d/'selection.json')['checkpoint_sha256'],'Best development checkpoint changed')
        control=read(out/'develop_frozen/selection.json');top=read(out/'develop_top2/selection.json')
        require(control['identity']==top['identity']==digest(ident),'Neural identity mismatch')
        selection['paired_dev']=dict(frozen_neural=control,top2=top)
        if top['metrics']['accuracy']>=control['metrics']['accuracy']+.002 and top['metrics']['accuracy']>=frozen['selected']['metrics']['accuracy']:
            selection['adaptation']=dict(mode='top2',epochs=top['epoch'])
        selection['adaptation_rule']='Top2 at least +0.2 percentage points over paired frozen head, and not worse than LR; otherwise no adapted Validation evaluation.'
    else:selection['adaptation_skipped']='Explicit bounded frozen-only run; no claim about fine-tuning benefit.'
    p=out/'selection.lock.json'
    require(not p.exists(),'Selection already locked; do not overwrite')
    save(p,selection)


def final_fit(out,ident,jobs):
    import numpy as np
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    selection=read(out/'selection.lock.json');require(selection['identity']==digest(ident),'Selection identity')
    require(not (out/'validation_started.json').exists(),'Validation already started; final fit cannot be rerun')
    train=[j for j in jobs if j['partition']=='train'];s=selection['frozen']
    p=out/'final_lr.joblib'
    if not p.exists():
        X=feature_matrix(out,ident,train,s['kind']);y=np.array([int(j['gender']=='M') for j in train])
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=s['C'],max_iter=2000,random_state=SEED))
        with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);clf.fit(X,y)
        buf=io.BytesIO();joblib.dump(clf,buf);atomic(p,buf.getvalue())
    models=dict(frozen_lr=dict(path=p.name,sha256=sha(p),kind=s['kind']))
    if selection['adaptation']:
        training(out,ident,jobs,'top2',selection['adaptation']['epochs'])
        p=out/'final_top2/latest.pt';models['top2']=dict(path=str(p.relative_to(out)),sha256=sha(p))
    save(out/'final_models.json',dict(identity=digest(ident),selection_sha256=sha(out/'selection.lock.json'),models=models))


def validation(out,ident,jobs):
    import joblib
    from audio_model import Encoder,Head
    selected=read(out/'final_models.json');require(selected['identity']==digest(ident) and selected['selection_sha256']==sha(out/'selection.lock.json'),'Final lock mismatch')
    marker=dict(identity=digest(ident),final_models_sha256=sha(out/'final_models.json'))
    if (out/'validation_started.json').exists():require(read(out/'validation_started.json')==marker,'Validation model changed')
    else:save(out/'validation_started.json',marker)
    val=[j for j in jobs if j['partition']=='internal_validation'];results={}
    for name,m in selected['models'].items():
        require(sha(out/m['path'])==m['sha256'],'Final checkpoint changed')
        if name=='frozen_lr':
            extract(out,ident,jobs,'internal_validation')
            clf=joblib.load(out/m['path']);path=out/'validation_frozen_lr.jsonl'
            ih=digest(dict(run=digest(ident),model=m['sha256'],tag='validation_frozen_lr'))
            rr=journal(path);done=check_predictions(rr,val,ih,False)
            for j in val:
                if j['call_id'] in done:continue
                try:
                    X=feature_matrix(out,ident,[j],m['kind']);pp=clf.predict_proba(X)[0].tolist()
                    r=dict(call_id=j['call_id'],gender=j['gender'],prediction='M' if pp[1]>pp[0] else 'F',probabilities_F_M=pp,
                           samples=samples(j),windows=math.ceil(samples(j)/(WINDOW*16000)),identity=ih)
                    check_predictions([r],[j],ih);append(path,r);rr.append(r)
                except BaseException as e:
                    append(out/'failures.jsonl',dict(stage='validation_frozen_lr',call_id=j['call_id'],identity=ih,error=repr(e),time=time.time()));raise
            check_predictions(rr,val,ih)
        else:
            enc=Encoder(ident['precision'],'top2');head=Head().cuda();restore(out/m['path'],enc,head,ih=digest(ident))
            rr,ih=evaluate_neural(out,ident,val,enc,head,'validation_top2',m['sha256'])
            del enc,head
        results[name]=dict(metrics=metrics(rr),prediction_identity=ih,path='validation_'+name+'.jsonl',sha256=sha(out/('validation_'+name+'.jsonl')))
    # Every intended model must have all 5597 rows before any completion/export.
    save(out/'completion.json',dict(status='complete',identity=digest(ident),models=results,scope='Previously observed Internal Validation, not independent test'))


def export(out):
    names=['dataset_check.json','identity.json','preflight.json','initial_parameters.json','benchmark.json','frozen_selection.json','selection.lock.json','final_models.json','validation_started.json','completion.json','events.jsonl']
    names += ['validation_'+n+'.jsonl' for n in read(out/'completion.json')['models']]
    if (out/'failures.jsonl').exists():names.append('failures.jsonl')
    for pattern in ('develop_*/*.json','develop_*/*.jsonl','final_top2/*.json','final_top2/*.jsonl'):
        for p in out.glob(pattern):names.append(str(p.relative_to(out)))
    if (out/'audit_decision.json').exists():names.append('audit_decision.json')
    if (out/'audio_review/review.csv').exists():names.append('audio_review/review.csv')
    hashes={n:sha(out/n) for n in names};save(out/'export_manifest.json',hashes)
    with zipfile.ZipFile(out/'wavlm_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in names+['export_manifest.json']:z.write(out/n,n)
    print('Complete export:',out/'wavlm_results.zip',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','audit','benchmark','extract','frozen-select','develop-frozen','develop-top2','lock-frozen','lock-adapted','final-fit','validation'])
    p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    config=read(args.config)
    paths=validate_paths(config)
    paths['hf_cache_dir'].mkdir(parents=True,exist_ok=True)
    require(paths['gpu_lock_path'].parent.is_dir(), 'Create/select the shared host lock parent first')
    require(config['crop_seconds'] in (8,12,16) and config['precision'] in ('fp32','bf16') and config['accumulation']>=1,'Unsupported config')
    out=Path(config['output_dir']);out.mkdir(exist_ok=True,parents=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ['HF_HOME']=config['hf_cache_dir']
    def stop(*_):raise KeyboardInterrupt('Terminated')
    signal.signal(signal.SIGTERM,stop)
    with lock(config['gpu_lock_path']),lock(out/'.run.lock'):
        gpu_idle();out,ident,jobs=setup(config)
        append(out/'events.jsonl',dict(stage=args.stage,status='started',time=time.time()))
        try:
            if args.stage=='preflight':
                preflight(out,ident,jobs)
                append(out/'events.jsonl',dict(stage=args.stage,status='complete',time=time.time()))
                return
            require(read(out/'preflight.json')['identity']==digest(ident),'Run matching preflight')
            if args.stage=='audit':audit(out,ident,jobs)
            elif args.stage=='benchmark':benchmark(out,ident,jobs)
            else:
                require(read(out/'benchmark.json')['identity']==digest(ident),'Run matching benchmark first')
                if args.stage in ('extract','frozen-select','develop-frozen','develop-top2','lock-frozen','lock-adapted'):
                    require(not (out/'selection.lock.json').exists(),'Selection locked; development closed')
                if args.stage=='extract':extract(out,ident,jobs,'train')
                elif args.stage=='frozen-select':frozen_select(out,ident,jobs)
                elif args.stage.startswith('develop-'):
                    require(read(out/'frozen_selection.json')['adaptation_eligible'],'Frozen dev >=99%; adaptation gate closed')
                    save(out/'audit_decision.json',audit_gate(out))
                    training(out,ident,jobs,'top2' if args.stage.endswith('top2') else 'frozen')
                elif args.stage.startswith('lock-'):lock_selection(out,ident,jobs,args.stage=='lock-adapted')
                elif args.stage=='final-fit':final_fit(out,ident,jobs)
                elif args.stage=='validation':validation(out,ident,jobs)
            append(out/'events.jsonl',dict(stage=args.stage,status='complete',time=time.time()))
            if args.stage=='validation':export(out)
        except BaseException as e:
            append(out/'events.jsonl',dict(stage=args.stage,status='failed',time=time.time()))
            append(out/'failures.jsonl',dict(stage=args.stage,identity=digest(ident),error=repr(e),time=time.time()))
            raise


if __name__=='__main__':main()
