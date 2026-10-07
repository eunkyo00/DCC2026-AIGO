"""Strict local ZIP verification and paired comparison, standard library only."""
import argparse
import tempfile
import zipfile
from common import *


def verify(folder, output):
    folder=Path(folder);output=Path(output)
    manifest,jobs=bundle()
    hh=read(folder/'export_manifest.json')
    for n,h in hh.items():
        require((folder/n).resolve().is_relative_to(folder.resolve()),'Unsafe export path')
        require(sha(folder/n)==h,'Export file hash mismatch: '+n)
    required={'identity.json','preflight.json','initial_parameters.json','benchmark.json','frozen_selection.json','selection.lock.json','final_models.json','validation_started.json','completion.json','events.jsonl'}
    require(required<=set(hh),'Missing provenance')
    ident=read(folder/'identity.json');ih=digest(ident)
    require(ident['bundle_sha256']==sha(ROOT/'bundle_manifest.json'),'Unknown code bundle')
    require(ident['model']==MODEL and ident['revision']==REVISION and ident['source_hashes']==SOURCE_HASHES,'Initial model/source mismatch')
    require(ident['inner_split_sha256']==sha(ROOT/'data/inner_split.csv'),'Inner split mismatch')
    require(ident['class_order']==['F','M'] and ident['layers']==LAYERS and ident['window_seconds']==WINDOW,'Pipeline mismatch')
    pre=read(folder/'preflight.json');require(pre['identity']==ih and pre['status']=='pass' and pre['initial_files'],'Invalid preflight')
    bench=read(folder/'benchmark.json');require(bench['identity']==ih,'Benchmark identity')
    for m in ('frozen','top2'):
        require(bench[m]['optimizer_steps']==100 and bench[m]['evaluation_sample_calls']==60,'Incomplete benchmark')
    lock_=read(folder/'selection.lock.json');models=read(folder/'final_models.json');complete=read(folder/'completion.json')
    require(lock_['identity']==models['identity']==complete['identity']==ih,'Selection/result identity')
    require(models['selection_sha256']==sha(folder/'selection.lock.json'),'Selection changed')
    if 'paired_dev' in lock_:
        for mode in ('frozen','top2'):
            d=folder/('develop_'+mode)
            c=read(d/'completion.json')
            require(c['identity']==ih and c['status']=='complete' and c['epochs']==4,'Incomplete development')
            assignment=inner_assign(jobs)
            dev=[j for j in jobs if assignment.get(j['call_id'])=='dev']
            for epoch in range(1,5):
                e=read(d/f'epoch_{epoch}.json')
                pi=digest(dict(run=ih,model=e['checkpoint_sha256'],tag=f'dev_epoch_{epoch}'))
                rr=journal(d/f'dev_epoch_{epoch}.jsonl')
                check_predictions(rr,dev,pi)
                require(metrics(rr)==e['metrics'],'Dev metrics mismatch')
    require(read(folder/'validation_started.json')==dict(identity=ih,final_models_sha256=sha(folder/'final_models.json')),'Final model changed')
    expected={'frozen_lr'} | ({'top2'} if lock_['adaptation'] else set())
    require(set(models['models'])==set(complete['models'])==expected and complete['status']=='complete','Model coverage')
    events=journal(folder/'events.jsonl');latest={}
    for r in events:latest[r['stage']]=r['status']
    require(latest.get('validation')=='complete' and all(s=='complete' for s in latest.values()),'Unresolved stage failure/interruption')
    failures=journal(folder/'failures.jsonl')
    val=[j for j in jobs if j['partition']=='internal_validation'];require(len(val)==5597,'Expected 5597')
    basefile=ROOT.parent/'gender_model_comparison/validation_results/call_comparison.csv'
    require(sha(basefile)==manifest['baseline_comparison_sha256'],'Baseline hash changed')
    base=rows(basefile);by={r['call_id']:r for r in base}
    require(len(base)==len(by)==5597 and set(by)=={j['call_id'] for j in val},'Baseline exact coverage')
    # Actual baseline file is locally trusted and independently checked against verified counts.
    for key,n in [('mfcc_prediction',5321),('wav2vec2_prediction',5443),('ecapa_prediction',5419)]:
        require(sum(r[key]==r['gender'] for r in base)==n,'Baseline score changed')
    output.mkdir(exist_ok=True,parents=True);summary={}
    comparison=[]
    for name,record in complete['models'].items():
        m=models['models'][name]
        pred_identity=digest(dict(run=ih,model=m['sha256'],tag='validation_'+name))
        require(record['prediction_identity']==pred_identity,'Prediction/model identity mismatch')
        require(record['path'] in hh and sha(folder/record['path'])==record['sha256'],'Prediction hash')
        rr=journal(folder/record['path']);preds=check_predictions(rr,val,pred_identity)
        metric=metrics(rr);require(metric==record['metrics'],'Reported metric mismatch')
        overlaps={}
        for baseline in ('mfcc','wav2vec2','ecapa'):
            counts=dict(corrected=0,regressed=0,both_wrong=0,both_correct=0)
            for r in rr:
                b=by[r['call_id']];require(b['gender']==r['gender'],'Baseline label mismatch')
                old=b[baseline+'_prediction']==r['gender'];new=r['prediction']==r['gender']
                counts['both_correct' if old and new else 'regressed' if old else 'corrected' if new else 'both_wrong']+=1
            overlaps[baseline]=counts
        for r in rr:comparison.append(dict(model=name,**r,**{k:by[r['call_id']][k+'_prediction'] for k in ('mfcc','wav2vec2','ecapa')}))
        summary[name]=dict(metrics=metric,versus_baselines=overlaps,reached_99=metric['correct']>=5542)
    result=dict(status='locally_verified',models=summary,baseline_sha256=sha(basefile),identity=ih,
                historical_failures=len(failures),unresolved_failures=0,scope='Previously observed Internal Validation; not independent test; official Validation unused')
    save(output/'metrics.json',result)
    with open(output/'call_comparison.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    lines=['# WavLM · Internal Validation 검증 결과','',result['scope'],'',
           '| 모델 | 정답 | 오답 | Overall | Male | Female |','|---|---:|---:|---:|---:|---:|']
    for name,r in summary.items():
        m=r['metrics'];lines.append(f"| {name} | {m['correct']} | {m['wrong']} | {m['accuracy']:.6%} | {m['gender_accuracy']['M']:.6%} | {m['gender_accuracy']['F']:.6%} |")
    for name,r in summary.items():
        m=r['metrics']
        lines+=['',f"{name} confusion (true/pred M,F): `{m['confusion_matrix_M_F']}`.",'']
        for b,o in r['versus_baselines'].items():lines.append(f"- {b}: corrected {o['corrected']}, regressed {o['regressed']}, both_wrong {o['both_wrong']}, both_correct {o['both_correct']}")
    lines+=['','99%는 5,542 정답 이상. 결과에 맞춘 재튜닝은 하지 않는다.',
            f'과거 실패 기록 {len(failures)}건, 미해결 실패 0건. 실제 L4 시간은 반환 benchmark.json 및 실행 기록 참조.']
    atomic(output/'REPORT.md',('\n'.join(lines)+'\n').encode());return result


def main():
    p=argparse.ArgumentParser();p.add_argument('zip',type=Path);p.add_argument('--output',type=Path,default=ROOT/'outputs/local_verified');a=p.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        with zipfile.ZipFile(a.zip) as z:
            require(len(z.namelist())==len(set(z.namelist())),'Duplicate ZIP paths')
            require(sum(i.file_size for i in z.infolist())<100*1024**2,'Unexpectedly large result ZIP')
            require(all((root/n).resolve().is_relative_to(root.resolve()) for n in z.namelist()),'Unsafe ZIP')
            z.extractall(root)
        print(json.dumps(verify(root,a.output),indent=2))


if __name__=='__main__':main()
