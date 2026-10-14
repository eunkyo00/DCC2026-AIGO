"""Build a standalone teammate distribution without changing the original Colab distribution."""
import csv
import io
import json
import shutil
import zipfile
from pathlib import Path
from common import ROOT, bundle, read, save, sha, atomic, SOURCE_HASHES


def replace_once(text, old, new):
    if text.count(old)!=1:raise RuntimeError('Upstream code changed; review the portable patch: '+old[:100])
    return text.replace(old,new,1)


def build():
    base,jobs=bundle()  # Verify original code/metadata before deriving a portable copy.
    dest=ROOT/'handoff';dest.mkdir(exist_ok=True)
    (dest/'data').mkdir(exist_ok=True)
    for n in ('common.py','audio_model.py','requirements.txt','design_manifest.json'):
        shutil.copyfile(ROOT/n,dest/n)
    for n in ('jobs.json','inner_split.csv','inner_train_ids.txt','dev_ids.txt','train_audio_review.csv'):
        shutil.copyfile(ROOT/'data'/n,dest/'data'/n)
    baseline=ROOT.parent/'gender_model_comparison/validation_results/call_comparison.csv'
    shutil.copyfile(baseline,dest/'data/baseline_predictions.csv')
    upstream={n:sha(ROOT/n) for n in ('common.py','audio_model.py','runner.py','verify.py','requirements.txt')}
    for p in (ROOT/'handoff_src').iterdir():
        if p.is_file():shutil.copyfile(p,dest/p.name)
    s=(ROOT/'runner.py').read_text()
    s=replace_once(s,'from common import *','from common import *\nfrom portable import validate_paths, require_dataset_check')
    s=replace_once(s,"    source=Path(config['source_output'])/'identity.json'\n    old=read(source)\n    require(old['data_root']==config['data_root'], 'Original identity data_root mismatch')", "    validate_paths(config)\n    require_dataset_check(config)")
    s=replace_once(s,"source_identity_sha256=sha(source)","source_identity_sha256=None, input_provenance='verified bundled metadata; fresh teammate execution'")
    s=replace_once(s,"shutil.disk_usage('/content')","shutil.disk_usage(ident['config']['hf_cache_dir'])")
    s=replace_once(s,"    config=read(args.config)","    config=read(args.config)\n    paths=validate_paths(config)\n    paths['hf_cache_dir'].mkdir(parents=True,exist_ok=True)\n    require(paths['gpu_lock_path'].parent.is_dir(), 'Create/select the shared host lock parent first')")
    s=replace_once(s,"os.environ['HF_HOME']='/content/gender_model_hf_cache'","os.environ['HF_HOME']=config['hf_cache_dir']")
    s=replace_once(s,"with lock('/content/.wavlm_gender_gpu.lock'),lock(out/'.run.lock'):","with lock(config['gpu_lock_path']),lock(out/'.run.lock'):")
    s=replace_once(s,"names=['identity.json'","names=['dataset_check.json','identity.json'")
    s=s.replace('free Drive','free output storage').replace('Drive metadata overhead','output-storage metadata overhead').replace('Drive reads','storage reads')
    atomic(dest/'runner.py',s.encode())
    s=(ROOT/'verify.py').read_text()
    s=replace_once(s,"ROOT.parent/'gender_model_comparison/validation_results/call_comparison.csv'","ROOT/'data/baseline_predictions.csv'")
    s=replace_once(s,"p.add_argument('--output',type=Path,default=ROOT/'outputs/local_verified')","p.add_argument('--output',type=Path,required=True)")
    s=replace_once(s,"required={'identity.json'","required={'dataset_check.json','identity.json'")
    s=replace_once(s,"ident=read(folder/'identity.json');ih=digest(ident)","ident=read(folder/'identity.json');ih=digest(ident)\n    dataset=read(folder/'dataset_check.json')\n    require(dataset['status']=='pass' and dataset['calls']==27985 and not dataset['failures'] and dataset['data_root']==ident['config']['data_root'] and dataset['bundle_sha256']==ident['bundle_sha256'],'Dataset provenance mismatch')")
    s=s.replace('실제 L4 시간','실제 GPU 시간')
    atomic(dest/'verify.py',s.encode())
    headers=ROOT.parents[1]/'eda/eda_outputs/audio_headers.csv'
    with headers.open(newline='') as f:by={r['file_ref']:r for r in csv.DictReader(f) if r['split']=='Training'}
    buf=io.StringIO(newline='');w=csv.writer(buf,lineterminator='\n')
    w.writerow(['call_id','wav_relative_path','size_bytes','frames'])
    total=0
    for j in jobs:
        r=by[j['call_id']];total+=int(r['size_bytes'])
        w.writerow([j['call_id'],j['wav_relative_path'],r['size_bytes'],r['frames']])
    atomic(dest/'data/wav_inventory.csv',buf.getvalue().encode())
    save(dest/'handoff_provenance.json',dict(parent_bundle_sha256=sha(ROOT/'bundle_manifest.json'),upstream_files=upstream,
        source_hashes=SOURCE_HASHES,audio_headers_sha256=sha(headers),required_wav_calls=len(jobs),required_wav_bytes=total,
        compatibility='Independent output identity; never resume original Colab output with portable code',
        patches=['Explicit data/output/HF/lock paths','No dependency on old Drive identity','CPU WAV inventory check','Self-contained baseline verification']))
    # Notebook lives outside code ZIP to avoid a circular ZIP hash dependency.
    names=['common.py','audio_model.py','runner.py','verify.py','portable.py','pack_audio.py','requirements.txt',
           'README.md','config.template.json','design_manifest.json','handoff_provenance.json',
           'data/jobs.json','data/inner_split.csv','data/inner_train_ids.txt','data/dev_ids.txt','data/train_audio_review.csv',
           'data/baseline_predictions.csv','data/wav_inventory.csv']
    save(dest/'bundle_manifest.json',dict(model=base['model'],revision=base['revision'],source_hashes=SOURCE_HASHES,
        baseline_comparison_sha256=sha(baseline),files={n:sha(dest/n) for n in names}))
    a=ROOT/'artifacts';a.mkdir(exist_ok=True)
    zip_path=a/'wavlm_teammate_bundle.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for n in names+['bundle_manifest.json']:z.write(dest/n,n)
    atomic(zip_path.with_name(zip_path.name+'.sha256'),(sha(zip_path)+'  '+zip_path.name+'\n').encode())
    from make_handoff_notebook import make
    make(sha(zip_path))
    print(json.dumps(dict(zip=str(zip_path),sha256=sha(zip_path),bundle_bytes=zip_path.stat().st_size,
        required_wav_calls=len(jobs),required_wav_GB=total/1e9),indent=2))


if __name__=='__main__':build()
