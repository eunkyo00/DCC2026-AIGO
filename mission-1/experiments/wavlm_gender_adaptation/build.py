"""Build checked call manifests, portable inner split, notebook and upload ZIP."""
import csv
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from common import *


def prepare():
    mission = ROOT.parents[1]
    for n, h in SOURCE_HASHES.items():
        require(sha(mission/n) == h, f'Source hash: {n}')
    split = rows(mission/'validation/split_assignments.csv')
    calls = {r['call_id']: r for r in rows(mission/'validation/manifests/calls.csv')}
    require(len(split) == len(calls) == len({r['call_id'] for r in split}) == 27985, 'Source coverage')
    caller, operator = defaultdict(list), defaultdict(list)
    mismatched = []
    for r in rows(mission/'eda/eda_outputs/segments.csv'):
        cid = r['file_ref']
        if r['split'] != 'Training' or cid not in calls:
            continue
        a, b = float(r['start_s']), float(r['end_s'])
        if r['speaker'] == '1':
            require(r['matched'] == 'True' and r['exceeds_wav'] == 'False' and 0 <= a < b, 'Invalid segment')
            caller[cid].append([a,b])
            if r['gender'] != calls[cid]['gender']:
                mismatched.append(cid)
        elif r['speaker'] == '0':
            operator[cid].append([a,b])
    jobs = []
    for r in sorted(split, key=lambda r:r['call_id']):
        cid = r['call_id']; c = calls[cid]
        require((r['gender'],r['partition']) == (c['gender'],c['partition']), 'Label mismatch')
        spans = sorted(caller[cid])
        require(len(spans) == int(c['n_segments']) and len(spans) > 0, 'Caller coverage')
        require(Path(c['wav_relative_path']).parts[0] == 'Training', 'Official Validation forbidden')
        jobs.append(dict(**r, intervals=spans, n_segments=len(spans), wav_relative_path=c['wav_relative_path']))
    require(sum(j['n_segments'] for j in jobs) == 442639, 'Segment count')
    require(Counter(j['gender'] for j in jobs if j['partition']=='internal_validation') == {'M':2620,'F':2977}, 'Validation counts')
    inner = inner_assign(jobs)
    d = ROOT/'data'; d.mkdir(exist_ok=True)
    save(d/'jobs.json', jobs)
    sio = io.StringIO(newline=''); w = csv.writer(sio, lineterminator='\n')
    w.writerow(['call_id','gender','partition'])
    for j in jobs:
        if j['partition']=='train': w.writerow([j['call_id'],j['gender'],inner[j['call_id']]])
    atomic(d/'inner_split.csv', sio.getvalue().encode())
    for p in ('inner_train','dev'):
        atomic(d/f'{p}_ids.txt', ''.join(x+'\n' for x in sorted(inner) if inner[x]==p).encode())
    train = [j for j in jobs if j['partition']=='train']
    # Metadata indicates risk, not actual speaker identity. No automatic relabel/drop.
    audit = []
    for j in train:
        ov = sum(max(0,min(b,d)-max(a,c)) for a,b in j['intervals'] for c,d in operator[j['call_id']])
        audit.append(dict(call_id=j['call_id'],gender=j['gender'],caller_seconds=samples(j)/16000,
                          overlap_pair_seconds=ov, label_mismatch=j['call_id'] in mismatched))
    ranked = sorted(audit,key=lambda r:(-r['overlap_pair_seconds']/r['caller_seconds'],r['call_id']))
    review = ranked[:24] + sorted(audit,key=lambda r:r['caller_seconds'])[:12]
    review += sorted(audit,key=lambda r:hashlib.sha256(r['call_id'].encode()).hexdigest())[:12]
    review = list({r['call_id']:r for r in review}.values())
    for r in review:
        r.update(review_status='pending', actual_speaker_matches_label='', notes='')
    with open(d/'train_audio_review.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(review[0]));w.writeheader();w.writerows(review)
    sec = sum(samples(j)/16000 for j in jobs)
    resource = dict(source_hashes=SOURCE_HASHES, inner_split_sha256=sha(d/'inner_split.csv'),
        inner_counts=dict(Counter(inner.values())), inner_gender_counts={p:dict(Counter(j['gender'] for j in train if inner[j['call_id']]==p)) for p in ('inner_train','dev')},
        rule='gender-stratified SHA256(wavlm-inner-v1|42|call_id), ascending digest then ID; first round(n/5) dev',
        seed=SEED, layers=LAYERS, window_seconds=WINDOW, caller_hours=sec/3600,
        pooled_float32_GiB=len(jobs)*4*2*1024*4/2**30,
        four_layer_frame_fp16_GiB_approx=sec*50*4*1024*2/2**30,
        all_25_states_frame_fp16_GiB_approx=sec*50*25*1024*2/2**30,
        lr_peak_RAM_budget_GiB=5, drive_free_required_GiB=8,
        metadata_label_mismatch_train=len(set(mismatched)&set(inner)),
        train_overlap_risk_calls=sum(r['overlap_pair_seconds']>0 for r in audit),
        audio_listening_status='pending; source WAV volume unavailable locally',
        notes='Frame figures exclude filesystem/metadata; pooled cache has no frame tensors. Audio overlap is annotation risk, not proof of a wrong speaker.')
    save(ROOT/'design_manifest.json',resource)


def package():
    from make_notebook import make
    names = ['common.py','audio_model.py','runner.py','verify.py','requirements.txt','README.md',
             'design_manifest.json','data/jobs.json','data/inner_split.csv','data/inner_train_ids.txt','data/dev_ids.txt','data/train_audio_review.csv']
    m=dict(model=MODEL,revision=REVISION,source_hashes=SOURCE_HASHES,baseline_comparison_sha256=sha(ROOT.parent/'gender_model_comparison/validation_results/call_comparison.csv'),files={n:sha(ROOT/n) for n in names})
    save(ROOT/'bundle_manifest.json',m)
    a=ROOT/'artifacts';a.mkdir(exist_ok=True)
    zpath=a/'wavlm_gender_adaptation_t4_v2_bundle.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for n in names+['bundle_manifest.json']:z.write(ROOT/n,n)
    make(sha(zpath))
    print(json.dumps(dict(bundle=str(zpath),sha256=sha(zpath),design=read(ROOT/'design_manifest.json')),indent=2))


if __name__=='__main__':
    prepare();package()
