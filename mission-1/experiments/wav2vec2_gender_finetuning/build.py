"""Build local data index and upload bundle without touching other experiments."""
import io
import zipfile
from collections import Counter, defaultdict
from common import *


def main():
    mission = ROOT.parents[1]
    for name,h in HASHES.items():
        require(sha(mission/name)==h, f'Source hash: {name}')
    split = rows(mission/'validation/split_assignments.csv')
    calls = rows(mission/'validation/manifests/calls.csv')
    require(len(split)==len(calls)==27985, 'Call count')
    ss = {r['call_id']: r for r in split}; cc = {r['call_id']:r for r in calls}
    require(len(ss)==len(cc)==27985 and ss.keys()==cc.keys(), 'Duplicate/coverage')
    require(Counter(r['partition'] for r in split)=={'train':22388,'internal_validation':5597}, 'Partitions')
    intervals=defaultdict(list)
    for r in rows(mission/'eda/eda_outputs/segments.csv'):
        if r['split']=='Training' and r['speaker']=='1' and r['file_ref'] in ss:
            require(r['matched']=='True' and r['exceeds_wav']=='False', 'Bad segment')
            a,b=float(r['start_s']),float(r['end_s'])
            require(math.isfinite(a) and math.isfinite(b) and 0<=a<b, 'Bounds')
            intervals[r['file_ref']].append([a,b])
    require(sum(map(len,intervals.values()))==442639 and intervals.keys()==ss.keys(), 'Segment coverage')
    jobs=[]
    for cid in sorted(ss):
        s,c=ss[cid],cc[cid]
        require(all(s[k]==c[k] for k in ('gender','partition')), 'Manifest mismatch')
        require(len(intervals[cid])==int(c['n_segments']), 'Segment count')
        rel=Path(c['wav_relative_path'])
        require(not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]=='Training', 'Unsafe WAV path')
        jobs.append(dict(s, wav_relative_path=str(rel), intervals=intervals[cid]))
    inner=inner_assign(jobs)
    # Discover a completed sibling assignment; require exact ID-to-partition equality.
    sources=[]; shared_csv=None
    for p in sorted((mission/'experiments/wavlm_gender_adaptation').rglob('*')):
        if p.is_file() and 'inner' in p.name and p.suffix in ('.csv','.json'):
            raw=read(p) if p.suffix=='.json' else rows(p)
            candidate = raw if isinstance(raw,dict) and set(raw)==set(inner) else None
            if isinstance(raw,list) and raw and 'call_id' in raw[0]:
                key=next((k for k in ('inner_partition','partition','inner_split') if k in raw[0]),None)
                if key:
                    candidate={r['call_id']:r[key] for r in raw}
                    require(len(candidate)==len(raw),'Duplicate sibling split IDs')
                    require(all(r.get('gender',ss[r['call_id']]['gender'])==ss[r['call_id']]['gender'] for r in raw),'Sibling split labels')
            if candidate is not None:
                require(candidate==inner, f'WavLM split differs: {p}')
                sources.append(dict(path=str(p.relative_to(mission)),sha256=sha(p)))
                if p.suffix=='.csv': shared_csv=p
    save(ROOT/'data/jobs.json',jobs)
    save(ROOT/'data/inner_split.json',inner)
    buf=io.StringIO(); w=csv.writer(buf,lineterminator='\n'); w.writerow(['call_id','gender','inner_partition'])
    for j in jobs:
        if j['call_id'] in inner:
            w.writerow([j['call_id'],j['gender'],inner[j['call_id']]])
    atomic(ROOT/'data/inner_split.csv',shared_csv.read_bytes() if shared_csv else buf.getvalue().encode())
    save(ROOT/'data/split_identity.json',dict(rule='wavlm-inner-v1|42|call_id SHA256 order; stratify F/M; dev=(n+2)//5',seed=42,
         source_hashes=HASHES, mapping_sha256=digest(inner), csv_sha256=sha(ROOT/'data/inner_split.csv'),
         sibling_files=sources,counts=dict(Counter(inner.values())),
         note='Sibling files discovered at build time; source code rule independently matched.'))
    baseline_paths={
      'wav2vec2':mission/'ssl/wav2vec2_frozen/results/full_l4/val_predictions.csv',
      'mfcc':mission/'baseline/mfcc_svm/results/val_predictions.csv',
      'ecapa':mission/'experiments/gender_model_comparison/validation_results/comparison.csv'}
    # ECAPA saved comparison name may vary; discover by header, never infer labels.
    if not baseline_paths['ecapa'].exists():
        candidates=[p for p in (mission/'experiments/gender_model_comparison/validation_results').glob('*.csv') if 'ecapa_prediction' in p.open().readline()]
        require(len(candidates)==1,'ECAPA comparison ambiguous'); baseline_paths['ecapa']=candidates[0]
    base={}; provenance={}
    val={j['call_id']:j for j in jobs if j['partition']=='internal_validation'}
    require(Counter(j['gender'] for j in val.values())=={'M':2620,'F':2977}, 'Val genders')
    for name,path in baseline_paths.items():
        rr=rows(path); dd={r['call_id']:r for r in rr}
        require(len(rr)==len(dd)==5597 and dd.keys()==val.keys(),'Baseline coverage')
        key='ecapa_prediction' if name=='ecapa' else 'prediction'
        require(all(r['gender']==val[c]['gender'] and r[key] in ('F','M') for c,r in dd.items()),'Baseline labels')
        base[name]={c:r[key] for c,r in dd.items()}; provenance[name]=dict(path=str(path.relative_to(mission)),sha256=sha(path))
    save(ROOT/'data/baselines.json',base); save(ROOT/'data/baseline_sources.json',provenance)
    names=['README.md','common.py','engine.py','run.py','verify.py','requirements.txt','config.template.json']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'data').glob('*'))]
    save(ROOT/'bundle_manifest.json',dict(model=MODEL,revision=REVISION,source_hashes=HASHES,files={n:sha(ROOT/n) for n in names}))
    with zipfile.ZipFile(ROOT/'wav2vec2_finetuning_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in names+['bundle_manifest.json']:
            z.write(ROOT/n,n)
    print('Bundle:',sha(ROOT/'wav2vec2_finetuning_bundle.zip'))
    print(read(ROOT/'data/split_identity.json'))

if __name__=='__main__': main()
