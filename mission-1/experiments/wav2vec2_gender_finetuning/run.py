"""L4 stages: benchmark -> search -> lock -> refit -> evaluate. Never uses Validation to select."""
import argparse
import contextlib
import gc
import importlib.metadata
import math
import os
import platform
import signal
import subprocess
import time
import traceback
import uuid
from common import *


def representative(jobs,n):
    result=[]
    for gender in ('F','M'):
        group=sorted((j for j in jobs if j['gender']==gender),key=lambda j:(sum(b-a for a,b in j['intervals']),j['call_id']))
        k=n//2
        result.extend(group[round(i*(len(group)-1)/(k-1))] for i in range(k))
    return result


@contextlib.contextmanager
def locks(out):
    import fcntl
    local=open('/tmp/aigo_gpu_training.lock','a+')
    fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # nvidia-smi also detects WavLM workers that do not implement this lock.
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    require(not [p for p in pids if p.strip().isdigit() and int(p)!=os.getpid()], 'GPU occupied: stop WavLM/other CUDA workers and clear notebook CUDA tensors first')
    active=out/'ACTIVE_SESSION.lock'
    with active.open('x') as f:
        json.dump(dict(pid=os.getpid(),started=time.time()),f)
    try: yield
    finally:
        active.unlink(missing_ok=True); local.close()


def select_plan(out,cfg,identity):
    require(not (out/'selection.json').exists(),'Selection already locked; do not tune after Validation')
    pilot=read(out/'benchmark.json'); require(pilot['identity']==identity and pilot['status']=='complete','Benchmark identity')
    eligible=[x['layers'] for x in pilot['candidates'] if x['eligible']]
    require(0 in eligible and any(k>0 for k in eligible),'Control and fine-tuning candidates required')
    choices=[]
    for k in eligible:
        report=read(out/f'search_top{k}'/'training.json')
        require(report['identity']==digest([identity,'search',k]) and report['epochs']==cfg['epochs'],'Incomplete search')
        best=min(report['history'],key=lambda x:(-x['correct'],x['epoch']))
        choices.append(dict(layers=k,epochs=best['epoch'],dev_correct=best['correct'],dev_n=best['n'],training_sha256=sha(out/f'search_top{k}'/'training.json')))
    control=next(c for c in choices if c['layers']==0)
    fine=min((c for c in choices if c['layers']>0),key=lambda c:(-c['dev_correct'],c['layers'],c['epochs']))
    plan=dict(identity=identity,status='locked',control=control,finetuned=fine,candidates=choices,
              policy='Fresh pretrained initialization + same head seed; refit ALL 22388 Train calls for selected full epochs. Constant per-group LR. No early stopping or model choice on Internal Validation.',
              evaluation='Both preregistered arms once; no post-Validation tuning; not an independent test.')
    save(out/'selection.json',plan); print(json.dumps(plan,indent=2))


def execute(stage,cfg,out,identity,jobs):
    from engine import Classifier,seed_all,train,evaluate
    import torch
    inner=read(ROOT/'data/inner_split.json')
    fit=[j for j in jobs if inner.get(j['call_id'])=='inner_train']
    dev=[j for j in jobs if inner.get(j['call_id'])=='dev']
    if stage=='lock':
        select_plan(out,cfg,identity); return
    def fresh(k):
        gc.collect(); torch.cuda.empty_cache(); seed_all(cfg['seed'])
        torch.cuda.reset_peak_memory_stats()
        return Classifier(k,cfg)
    if stage=='benchmark':
        require(not (out/'selection.json').exists(),'Selection locked')
        require(100<=cfg['benchmark_steps']<=200,'Benchmark must measure 100-200 optimizer steps')
        candidates=[]
        sample=representative(fit,cfg['benchmark_steps']*cfg['accumulation_calls'])
        eval_sample=representative(dev,40)
        save(out/'benchmark_sample_ids.json',dict(train=[j['call_id'] for j in sample],dev=[j['call_id'] for j in eval_sample]))
        for k in cfg['layers']:
            model=fresh(k); folder=out/f'benchmark_top{k}'; sid=digest([identity,'benchmark',k])
            try:
                result=train(model,sample,cfg,folder,sid,1,max_steps=cfg['benchmark_steps'])
                torch.cuda.synchronize(); t=time.perf_counter()
                _,rr=evaluate(model,eval_sample,cfg,folder/('eval_sample-'+uuid.uuid4().hex),digest([sid,sha(folder/'last.pt')]))
                torch.cuda.synchronize(); eval_s=time.perf_counter()-t
                eval_read=sum(r['read_preprocess_seconds'] for r in rr)
                t=result['timing']; per_step=(t['read_preprocess']+t['forward_backward']+t['optimizer']+t['checkpoint'])/result['steps']
                peak=max(result['peak_reserved'],torch.cuda.max_memory_reserved())
                eligible=peak<torch.cuda.get_device_properties(0).total_memory*cfg['memory_fraction_limit'] and per_step<=cfg['max_step_seconds']
                estimate=dict(train_read_seconds_per_step=t['read_preprocess']/result['steps'],forward_backward_seconds_per_step=t['forward_backward']/result['steps'],
                    optimizer_seconds_per_step=t['optimizer']/result['steps'],checkpoint_seconds_per_step=t['checkpoint']/result['steps'],eval_read_seconds_per_call=eval_read/len(rr),eval_compute_write_seconds_per_call=(eval_s-eval_read)/len(rr),
                    search_training_hours=per_step*math.ceil(len(fit)/cfg['accumulation_calls'])*cfg['epochs']/3600,
                    search_dev_hours=eval_s/len(rr)*len(dev)*cfg['epochs']/3600,
                    refit_hours_per_epoch=per_step*math.ceil(22388/cfg['accumulation_calls'])/3600,
                    final_evaluation_hours=eval_s/len(rr)*5597/3600)
                candidates.append(dict(layers=k,eligible=eligible,peak_reserved=peak,seconds_per_step=per_step,estimate=estimate))
            except torch.cuda.OutOfMemoryError as e:
                save(folder/'oom.json',dict(error=str(e),status='ineligible'))
                candidates.append(dict(layers=k,eligible=False,reason='OOM'))
            finally:
                del model; gc.collect(); torch.cuda.empty_cache()
        save(out/'benchmark.json',dict(status='complete',identity=identity,candidates=candidates,note='Measured sample estimate, not a promised duration; Drive cold-read variance and interruptions can dominate.'))
        print(json.dumps(candidates,indent=2)); return
    if stage=='search':
        require(not (out/'selection.json').exists(),'Selection locked; search prohibited')
        pilot=read(out/'benchmark.json'); require(pilot['identity']==identity,'Benchmark mismatch')
        eligible=[c['layers'] for c in pilot['candidates'] if c['eligible']]
        require(0 in eligible and any(k>0 for k in eligible),'No feasible controlled comparison')
        for k in eligible:
            model=fresh(k); folder=out/f'search_top{k}'; sid=digest([identity,'search',k])
            def callback(m,epoch):
                # Includes exact weights so interrupted runs cannot silently reuse predictions from another model.
                from engine import tensor_hash
                eid=digest([sid,epoch,tensor_hash(m.named_parameters())])
                score,_=evaluate(m,dev,cfg,folder/f'dev_epoch{epoch}',eid)
                return score
            train(model,fit,cfg,folder,sid,cfg['epochs'],callback)
            del model
        return
    selection=read(out/'selection.json')
    require(selection['identity']==identity and selection['status']=='locked','Selection not locked')
    for arm in ('control','finetuned'):
        choice=selection[arm]; k=choice['layers']; folder=out/f'final_{arm}'
        sid=digest([identity,'refit',arm,digest(selection)])
        model=fresh(k)
        if stage=='refit':
            train(model,[j for j in jobs if j['partition']=='train'],cfg,folder,sid,choice['epochs'])
        else:
            report=read(folder/'training.json')
            require(report['identity']==sid and report['status']=='complete' and report['epochs']==choice['epochs'],'Incomplete refit')
            state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            require(state['identity']==sid and state['epoch']==choice['epochs'] and state['cursor']==0,'Checkpoint not final')
            model.load_state_dict(state['model'])
            eid=digest([sid,sha(folder/'last.pt')])
            save(folder/'evaluation_identity.json',dict(identity=eid,run_identity=identity,refit_identity=sid,checkpoint_sha256=sha(folder/'last.pt'),selection_sha256=sha(out/'selection.json')))
            val=[j for j in jobs if j['partition']=='internal_validation']
            score,rr=evaluate(model,val,cfg,folder/'validation_calls',eid)
            # Summary is committed only after exact coverage succeeds.
            historical=list((out/'events').glob('*.json'))
            save(folder/'evaluation.json',dict(status='complete',identity=eid,metrics=score,expected=5597,completed=len(rr),pending=0,failed=0,
                 historical_failure_events=[p.name for p in historical if read(p)['status']!='complete']))
        del model


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True)
    p.add_argument('stage',choices=['benchmark','search','lock','refit','evaluate'])
    a=p.parse_args(); cfg=read(a.config); manifest,jobs=bundle()
    require(cfg['layers']==[0,2,4] and cfg['seed']==42 and cfg['epochs']==4,'Preregistered search changed')
    require(cfg['accumulation_calls']>0 and cfg['checkpoint_steps']>0,'Invalid step configuration')
    out=Path(cfg['output']).resolve(); out.mkdir(parents=True,exist_ok=True)
    require(out.is_relative_to(Path('/content/drive')) and 'wav2vec2_finetuning' in out.name,'Dedicated Drive output required')
    require(Path(cfg['data_root']).is_dir(),'Original WAV root missing')
    def stopped(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stopped)
    with locks(out):
        import torch
        require(torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(0),'NVIDIA L4 required')
        if cfg['precision']=='bf16': require(torch.cuda.is_bf16_supported(),'bf16 unsupported')
        environment=dict(python=platform.python_version(),versions={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','scipy','soundfile','safetensors')},cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),cudnn=torch.backends.cudnn.version(),driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip())
        identity=dict(config=cfg,environment=environment,bundle_manifest_sha256=sha(ROOT/'bundle_manifest.json'),split=read(ROOT/'data/split_identity.json'),model=MODEL,revision=REVISION)
        ip=out/'identity.json'
        if ip.exists(): require(read(ip)==identity,'Output identity differs; preserve old run, use new dedicated output before any Validation')
        else: save(ip,identity)
        ih=digest(identity); started=time.time(); event=out/'events'/f'{a.stage}-{uuid.uuid4().hex}.json'
        save(event,dict(stage=a.stage,status='running',started=started,identity=ih))
        try:
            execute(a.stage,cfg,out,ih,jobs)
        except BaseException as e:
            save(event,dict(stage=a.stage,status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',started=started,ended=time.time(),identity=ih,error=repr(e),traceback=traceback.format_exc()))
            raise
        save(event,dict(stage=a.stage,status='complete',started=started,ended=time.time(),identity=ih))

if __name__=='__main__': main()
