# config.py의 빈 경로를 채운 뒤 python3 run.py 한 번으로 실행합니다.
# 경로/환경 오류는 GPU 학습 전에 중단합니다. config.local.py가 있으면 우선 사용합니다.
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from common import ROOT, atomic, bundle, digest, gpu_profile, journal, lock, read, require, save, sha


def command(argv, env=None, capture=False):
    def parent_death_signal():
        import ctypes
        parent=os.getppid()
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=parent:os.kill(os.getpid(),signal.SIGTERM)
    p=subprocess.Popen(list(map(str,argv)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                       text=True,bufsize=1,env=env,start_new_session=True,
                       preexec_fn=parent_death_signal if platform.system()=='Linux' else None)
    output=[]
    try:
        for line in p.stdout:
            if capture:output.append(line)
            else:print(line,end='',flush=True)
        if p.wait():raise RuntimeError(f'Command failed ({p.returncode}): {argv}\n'+''.join(output))
        return ''.join(output)
    finally:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()


def load_settings():
    path=ROOT/'config.local.py'
    if not path.exists():path=ROOT/'config.py'
    spec=importlib.util.spec_from_file_location('user_settings',path)
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    result={}
    for key in ('INPUT_ZIP','DATA_ROOT','OUTPUT_DIR','HF_CACHE_DIR','GPU_LOCK_PATH'):
        value=getattr(c,key,None)
        require(isinstance(value,str) and value.strip(), f'Fill {key} in {path.name}')
        p=Path(value).expanduser()
        require(p.is_absolute(),f'{key} must be an absolute path')
        result[key]=str(p.resolve())
    require(Path(result['INPUT_ZIP']).is_file(),'INPUT_ZIP does not exist')
    require((Path(result['DATA_ROOT'])/'Training').is_dir(),'DATA_ROOT must directly contain Training/')
    require(Path(result['GPU_LOCK_PATH']).parent.is_dir(),'GPU_LOCK_PATH parent does not exist')
    return result


def prepare_metadata(archive):
    m=read(ROOT/'metadata_manifest.json')
    require(sha(archive)==m['archive_sha256'],'Wrong metadata ZIP; use wavlm_gender_inputs.zip supplied with this code')
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(m['files']),'Unexpected metadata ZIP members')
        for name,expected in m['files'].items():
            p=(ROOT/name).resolve()
            require(p.is_relative_to(ROOT.resolve()),'Unsafe metadata path')
            b=z.read(name)
            require(hashlib.sha256(b).hexdigest()==expected,f'Metadata hash mismatch: {name}')
            if p.exists():require(sha(p)==expected,f'Existing metadata changed: {name}')
            else:
                p.parent.mkdir(parents=True,exist_ok=True);atomic(p,b)
    bundle()


def probe(python):
    script="""import json,sys,torch,torchaudio
assert sys.version_info[:2]==(3,12), 'Python 3.12 required'
assert torch.cuda.is_available(), 'CUDA torch required'
assert torch.__version__==torchaudio.__version__, 'torch/torchaudio version or CUDA wheel mismatch'
print(json.dumps(dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,torchaudio=torchaudio.__version__,cuda=torch.version.cuda)))
"""
    return json.loads(command([python,'-c',script],capture=True).strip().splitlines()[-1])


def environment(out,hardware):
    envdir=out/'.venv';python=envdir/'bin/python';marker=envdir/'.wavlm_environment.json'
    expected=dict(base_python=str(Path(sys.executable).resolve()),torch=hardware['torch'],requirements_sha256=sha(ROOT/'requirements.txt'))
    if envdir.exists():require(marker.is_file() and read(marker)==expected,'Existing .venv is not this experiment environment')
    else:
        command([sys.executable,'-m','venv','--system-site-packages',envdir])
        source_sites=json.loads(command([sys.executable,'-c','import site,json; print(json.dumps(site.getsitepackages()))'],capture=True))
        target_sites=json.loads(command([python,'-c','import site,json; print(json.dumps(site.getsitepackages()))'],capture=True))
        local=next(Path(p) for p in target_sites if str(envdir) in p)
        (local/'existing_torch.pth').write_text('\n'.join(source_sites)+'\n')
        save(marker,expected)
    check="""import importlib.metadata as m,sys
for line in open(sys.argv[1]):
 if '==' in line:
  name,version=line.strip().split('=='); assert m.version(name)==version, name
"""
    try:command([python,'-c',check,ROOT/'requirements.txt'],capture=True)
    except RuntimeError:command([python,'-m','pip','install','--disable-pip-version-check','-r',ROOT/'requirements.txt'])
    command([python,'-c',check,ROOT/'requirements.txt'],capture=True)
    require(probe(python)==hardware,'GPU/torch changed in the virtual environment')
    return python


def stage_done(out,stage):
    events=journal(out/'events.jsonl')
    status=[r['status'] for r in events if r['stage']==stage]
    return bool(status) and status[-1]=='complete'


def pipeline(python,config_path,config,stop_after_benchmark=False):
    out=Path(config['output_dir'])
    env=dict(os.environ,HF_HOME=config['hf_cache_dir'],CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')
    command([python,'-u',ROOT/'portable.py','--config',config_path,'--check-data'],env=env)
    for stage in ('preflight','benchmark','extract','frozen-select','lock-frozen','final-fit','validation'):
        # runner identity checks remain authoritative; the launcher never changes selection after locking.
        if stage_done(out,stage) and stage not in ('preflight','validation'):
            print('Completed stage retained:',stage,flush=True)
        else:
            print('Starting:',stage,flush=True)
            command([python,'-u',ROOT/'runner.py',stage,'--config',config_path],env=env)
        if stage=='benchmark':
            print((out/'benchmark.json').read_text(),flush=True)
            if stop_after_benchmark:return
    command([sys.executable,ROOT/'verify.py',out/'wavlm_results.zip','--output',out/'verified'])
    print('Verified results:',out/'verified/REPORT.md',flush=True)
    print('Return ZIP:',out/'wavlm_results.zip',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark-only',action='store_true')
    args=parser.parse_args()
    settings=load_settings()
    require(platform.system()=='Linux','Run on a Linux CUDA GPU host')
    require(sys.version_info[:2]==(3,12),'Run with the Python 3.12 interpreter that has CUDA torch/torchaudio')
    prepare_metadata(settings['INPUT_ZIP'])
    hardware=probe(sys.executable);profile=gpu_profile(hardware['gpu'])
    config=dict(data_root=settings['DATA_ROOT'],output_dir=settings['OUTPUT_DIR'],hf_cache_dir=settings['HF_CACHE_DIR'],
                gpu_lock_path=settings['GPU_LOCK_PATH'],expected_gpu=hardware['gpu'],precision=profile['precision'],crop_seconds=12,accumulation=4)
    from portable import validate_paths
    validate_paths(config)
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    def stop(*_):raise KeyboardInterrupt('Terminated')
    signal.signal(signal.SIGTERM,stop)
    with lock(out/'.launcher.lock'):
        path=out/'runtime_config.json'
        if path.exists():require(read(path)==config,'Execution config changed; use a new output')
        else:save(path,config)
        python=environment(out,hardware)
        print('GPU:',hardware,'precision:',config['precision'],flush=True)
        pipeline(python,path,config,args.benchmark_only)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Stopped. Rerun the same command to resume.',file=sys.stderr);sys.exit(130)
