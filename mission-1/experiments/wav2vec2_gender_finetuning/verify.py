"""CPU-only complete-result validation and paired baseline comparison; never tunes."""
import argparse
from common import *


def main():
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    manifest,jobs=bundle()
    export=read(a.run/'export_manifest.json')
    for name,h in export['files'].items():
        path=(a.run/name).resolve()
        require(path.is_relative_to(a.run.resolve()),'Unsafe export path')
        require(sha(path)==h,f'Export file hash mismatch: {name}')
    actual={str(p.relative_to(a.run)) for p in a.run.rglob('*.json') if p.name!='export_manifest.json'}
    require(actual==set(export['files']),'Export extra/missing files')
    identity=read(a.run/'identity.json'); ih=digest(identity)
    require(identity['bundle_manifest_sha256']==sha(ROOT/'bundle_manifest.json'),'Run bundle mismatch')
    require(identity['split']==read(ROOT/'data/split_identity.json'),'Split identity mismatch')
    selection=read(a.run/'selection.json'); require(selection['identity']==ih and selection['status']=='locked','Selection mismatch')
    pilot=read(a.run/'benchmark.json')
    require(pilot['identity']==ih and pilot['status']=='complete','Pilot mismatch')
    require({c['layers'] for c in selection['candidates']}=={c['layers'] for c in pilot['candidates'] if c['eligible']},'Eligible candidate mismatch')
    for candidate in selection['candidates']:
        history_path=a.run/f"search_top{candidate['layers']}"/'training.json'
        require(sha(history_path)==candidate['training_sha256'],'Selection evidence hash')
        history=read(history_path)
        require(history['identity']==digest([ih,'search',candidate['layers']]) and history['epochs']==identity['config']['epochs'],'Incomplete search evidence')
        best=min(history['history'],key=lambda x:(-x['correct'],x['epoch']))
        require((best['epoch'],best['correct'],best['n'])==(candidate['epochs'],candidate['dev_correct'],candidate['dev_n']),'Dev selection mismatch')
    expected_control=next(c for c in selection['candidates'] if c['layers']==0)
    expected_fine=min((c for c in selection['candidates'] if c['layers']>0),key=lambda c:(-c['dev_correct'],c['layers'],c['epochs']))
    require(selection['control']==expected_control and selection['finetuned']==expected_fine,'Arm selection mismatch')
    val=[j for j in jobs if j['partition']=='internal_validation']; baselines=read(ROOT/'data/baselines.json')
    results={}; predictions={}
    events=[read(p) for p in (a.run/'events').glob('*.json')]
    require(events and not any(e['status']=='running' for e in events),'Unfinished process event: resolve before success report')
    require(any(e['stage']=='evaluate' and e['status']=='complete' for e in events),'No completed evaluation stage')
    require(all(e['identity']==ih for e in events),'Event identity mismatch')
    for arm in ('control','finetuned'):
        folder=a.run/f'final_{arm}'; ei=read(folder/'evaluation_identity.json'); summary=read(folder/'evaluation.json')
        sid=digest([ih,'refit',arm,digest(selection)])
        require(ei['run_identity']==ih and ei['refit_identity']==sid and ei['selection_sha256']==sha(a.run/'selection.json'),'Evaluation provenance')
        require(ei['identity']==digest([sid,ei['checkpoint_sha256']]),'Checkpoint identity linkage')
        # Weights can stay on Drive; if included in transfer, independently hash them too.
        if (folder/'last.pt').exists(): require(sha(folder/'last.pt')==ei['checkpoint_sha256'],'Checkpoint hash')
        training=read(folder/'training.json')
        require(training['status']=='complete' and training['identity']==sid and training['epochs']==selection[arm]['epochs'],'Training incomplete')
        audit=read(folder/'update_audit.json')
        require(training['trainable_groups_changed'] and audit['passed'],'Missing update audit')
        require(training['frozen_sha256']==audit['frozen_sha256'],'Frozen audit mismatch')
        require(len(audit['initial'])==selection[arm]['layers']+1 and audit['initial'].keys()==audit['after_first_step'].keys(),'Trainable group audit')
        require(all(v!=audit['after_first_step'][k] for k,v in audit['initial'].items()),'Unchanged trainable group')
        rr=[read(p) for p in sorted((folder/'validation_calls').glob('*.json'))]
        dd=validate(rr,val,ei['identity'])
        require(summary['status']=='complete' and summary['identity']==ei['identity'] and summary['completed']==summary['expected']==5597 and summary['pending']==summary['failed']==0,'Incomplete/failed export')
        score=metric(rr); require(summary['metrics']==score,'Metrics mismatch')
        comparison={}
        for name,base in baselines.items():
            require(set(base)==set(dd),'Baseline coverage')
            counts=dict(corrected=0,regressed=0,both_wrong=0,both_correct=0)
            for cid,r in dd.items():
                new=r['prediction']==r['gender']; old=base[cid]==r['gender']
                counts['both_correct' if new and old else 'corrected' if new else 'regressed' if old else 'both_wrong']+=1
            counts['net_correct_gain']=counts['corrected']-counts['regressed']; comparison[name]=counts
        results[arm]=dict(metrics=score,comparison=comparison,achieved_99=score['correct']>=5542,
                          improvement_over_frozen_lr_correct=score['correct']-5443)
        predictions[arm]=dd
    paired=dict(corrected=0,regressed=0,both_wrong=0,both_correct=0)
    for cid,r in predictions['finetuned'].items():
        new=r['prediction']==r['gender']; old=predictions['control'][cid]['prediction']==r['gender']
        paired['both_correct' if new and old else 'corrected' if new else 'regressed' if old else 'both_wrong']+=1
    require(not a.output.exists(),'Preserve previous local report; choose new output')
    a.output.mkdir(parents=True)
    report=dict(status='verified_complete',results=results,finetuned_vs_control=paired,
                historical_failures=[e for e in events if e['status']!='complete'],
                checkpoint_hash_scope='Recorded on Colab; rehashed locally only when last.pt supplied',
                warning='Already observed Internal Validation; not a new independent test. No tuning on these results.')
    save(a.output/'comparison.json',report)
    lines=['# Wav2Vec2 부분 fine-tuning · Internal Validation','', '| 모델 | 정답 | 오답 | Overall | Male | Female |','|---|---:|---:|---:|---:|---:|']
    for name,k,m,f in [('MFCC+SVM',5321,94.809160,95.297279),('Frozen Wav2Vec2+LR',5443,96.755725,97.682230),('Frozen ECAPA',5419,97.824427,95.935506)]:
        lines.append(f'| {name} | {k} | {5597-k} | {100*k/5597:.6f}% | {m:.6f}% | {f:.6f}% |')
    for arm,result in results.items():
        s=result['metrics']; g=s['gender_accuracy']
        lines.append(f"| {arm} | {s['correct']} | {s['wrong']} | {100*s['accuracy']:.6f}% | {100*g['M']:.6f}% | {100*g['F']:.6f}% |")
    for arm,result in results.items():
        s=result['metrics']
        lines.extend(['',f"{arm}: Frozen LR 대비 정답 {s['correct']-5443:+d}, 99% 달성 {s['correct']>=5542}. Confusion M/F: {s['confusion_matrix_M_F']}",''])
    lines+=['','| 새 arm | 기준 모델 | 교정 | 회귀 | 공통 오답 | 둘 다 정답 |','|---|---|---:|---:|---:|---:|']
    for arm,result in results.items():
        for name,c in result['comparison'].items():
            lines.append(f"| {arm} | {name} | {c['corrected']} | {c['regressed']} | {c['both_wrong']} | {c['both_correct']} |")
    lines+=['',f"Fine-tuning vs 새 head 대조군: {paired}",f"실패/중단 이력: {len(report['historical_failures'])}건 (내역 JSON 포함).",'',report['warning']]
    atomic(a.output/'REPORT.md',('\n'.join(lines)+'\n').encode()); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
