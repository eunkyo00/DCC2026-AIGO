# 아래 5개 경로만 실제 서버의 절대 경로로 입력하세요.
# 원본 WAV는 별도 준비: DATA_ROOT 바로 아래에 Training/ 폴더가 있어야 합니다.
# INPUT_JSON: 따로 전달받은 wavlm_gender_inputs.json (원본 음성 미포함, ZIP 불필요).
INPUT_JSON = ""
DATA_ROOT = ""
OUTPUT_DIR = ""
HF_CACHE_DIR = ""
GPU_LOCK_PATH = ""  # 같은 서버의 실험들이 공유할 잠금 파일 경로. 부모 폴더는 미리 준비하세요.

# Linux / Python 3.12 / 정상 CUDA torch+torchaudio 환경에서 실행하세요.
# L4는 BF16, T4는 FP32를 자동 선택합니다. GPU 두 종류만 지원합니다.
# 가상환경은 OUTPUT_DIR/.venv에 자동 생성하며 기존 torch는 재설치하지 않습니다.
# 실행: python3 run.py
# 기본 실행은 frozen 특징+LR입니다. 상위층 적응 학습은 자동으로 하지 않습니다.
# 결과: OUTPUT_DIR/wavlm_results.zip, OUTPUT_DIR/verified/REPORT.md
# 중단 후 같은 명령을 실행하면 완료 단계/통화는 확인 후 재사용합니다.
# 학습 checkpoint 사이의 미저장 step은 재실행 범위를 기록합니다.
# 실제 GPU 속도/메모리는 미측정입니다. benchmark 결과를 확인하세요.
# 실행 프로세스를 멈춰도 대여 GPU 요금은 서비스에서 별도로 정지해야 합니다.

# 이전 버전에서 실행했다면 새 OUTPUT_DIR을 지정하세요. 이전 결과는 보존합니다.

# 아래는 실행 코드입니다. 경로 외에는 수정하지 마세요.

# 공통 무결성 검사·저장
import contextlib
import csv
import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = 'microsoft/wavlm-large'
REVISION = 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'
SOURCE_HASHES = {
    'validation/split_assignments.csv': '04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06',
    'validation/manifests/calls.csv': 'dc57d0f7bea6e06017ac3e27444a0ff5bb299e401ea787400e84cbdf7d072e23',
    'eda/eda_outputs/segments.csv': '8bf8c83c0190232e4500b4067bc28d2611b09dd227a7ce4dc9c3741e6dcce584',
}
SEED = 42
LAYERS = [6, 12, 18, 24]
WINDOW = 12  # predeclared context/cost compromise, not selected on Validation


# 배포용 보조 파일 없이 사용하는 고정 환경·입력 해시입니다.
REQUIREMENTS = ['numpy==2.2.2',
 'scipy==1.15.3',
 'soundfile==0.13.1',
 'transformers==4.46.3',
 'tokenizers==0.20.3',
 'huggingface-hub==0.31.4',
 'safetensors==0.5.3',
 'scikit-learn==1.6.1',
 'joblib==1.4.2',
 'psutil==7.0.0']
INPUT_MANIFEST = {'model': 'microsoft/wavlm-large',
 'revision': 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c',
 'source_hashes': {'validation/split_assignments.csv': '04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06',
                   'validation/manifests/calls.csv': 'dc57d0f7bea6e06017ac3e27444a0ff5bb299e401ea787400e84cbdf7d072e23',
                   'eda/eda_outputs/segments.csv': '8bf8c83c0190232e4500b4067bc28d2611b09dd227a7ce4dc9c3741e6dcce584'},
 'baseline_comparison_sha256': 'e4c9f2addbdbc339f34f459cc562121a8df429e761a33b7b70a18a9a6dc99081',
 'files': {'data/jobs.json': '3e4f024ad7c0e5bec8fb2c5ae9af25c467dbc78129d8286faa79a54f237c4e89',
           'data/inner_split.csv': 'b6664bee9b72dd14a53743388274577d6c5b73e5ef5f51ab4d53da6fa3321e17',
           'data/inner_train_ids.txt': '1de6452fffaa1cbe75e0a0581ac6a989417a4acdd053b22e00e17a5a3ba724fc',
           'data/dev_ids.txt': '8b695c719a757ed275586846fb0a9ab12003732b5e967eb58dd21c4e271e4c80',
           'data/train_audio_review.csv': '2482439d77a9a2fd6ab81bf5e0cfeaba904da13d336a45e069f9614df11f1e53',
           'data/baseline_predictions.csv': 'e4c9f2addbdbc339f34f459cc562121a8df429e761a33b7b70a18a9a6dc99081',
           'data/wav_inventory.csv': 'ce7df76040db8075371d2160f753b8ea1da6646fb0269fd3987a6dbaac1e7bf1'},
 'input_json_sha256': 'd3d121bd843d1cfd26cb0bd3a03653d70097ba66f8f412b28f2268f435e1a13a'}
SETTINGS_KEYS = ('INPUT_JSON', 'DATA_ROOT', 'OUTPUT_DIR', 'HF_CACHE_DIR', 'GPU_LOCK_PATH')


def code_identity():
    # 입력 경로만 제외한 실제 코드의 해시입니다. 경로는 실행 config에 따로 기록합니다.
    # Python 버전별 AST 표현 차이가 없도록 원문 바이트를 정규화합니다.
    import ast
    source=Path(__file__).read_bytes()
    tree=ast.parse(source)
    lines=source.splitlines(keepends=True)
    offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    edits=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in SETTINGS_KEYS:
            v=node.value
            edits.append((offsets[v.lineno-1]+v.col_offset,offsets[v.end_lineno-1]+v.end_col_offset))
    require(len(edits)==len(SETTINGS_KEYS),'Settings declarations changed')
    for start,end in sorted(edits,reverse=True):source=source[:start]+b'""'+source[end:]
    return hashlib.sha256(source).hexdigest()


def runtime_manifest():
    return dict(INPUT_MANIFEST, code_sha256=code_identity(), requirements=REQUIREMENTS)


def bundle_identity():
    return digest(runtime_manifest())


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def gpu_profile(name, precision=None):
    """Explicit supported GPU/precision pairs; never emulate BF16 on T4."""
    families = {'T4': 't4', 'Tesla T4': 't4', 'NVIDIA T4': 't4',
                'L4': 'l4', 'NVIDIA L4': 'l4'}
    require(name in families, f'Expected T4 or L4, got {name!r}')
    family = families[name]
    precision = precision or ('fp32' if family == 't4' else 'bf16')
    require(precision in ('fp32', 'bf16'), 'Only verified FP32/BF16 paths are supported')
    require(family != 't4' or precision == 'fp32', 'T4 requires FP32 in this release; BF16 is unsupported')
    return dict(family=family, precision=precision, output_tag=f'{family}_{precision}_v2')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save(path, obj):
    atomic(path, (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())


def append(path, obj):
    with open(path, 'a') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def journal(path):
    """Fail closed on truncated records; never silently discard completed work."""
    p = Path(path)
    if not p.exists():
        return []
    b = p.read_bytes()
    require(not b or b.endswith(b'\n'), f'Truncated journal; preserve and inspect: {p}')
    return [json.loads(x) for x in b.splitlines()]


def inner_assign(jobs):
    """Version-independent gender-stratified SHA256 order, 20% dev per gender."""
    result = {}
    for g in ('F', 'M'):
        ids = [j['call_id'] for j in jobs if j['partition'] == 'train' and j['gender'] == g]
        ids.sort(key=lambda x: (hashlib.sha256(f'wavlm-inner-v1|{SEED}|{x}'.encode()).hexdigest(), x))
        n = (len(ids) + 2) // 5  # round(n*.2), no floating-point dependency
        result.update({x: 'dev' if i < n else 'inner_train' for i, x in enumerate(ids)})
    return result


def samples(j):
    return 2 * sum(round(b * 8000) - round(a * 8000) for a, b in j['intervals'])


def check_predictions(rr, jobs, identity, complete=True):
    jj = {j['call_id']: j for j in jobs}
    ids = [r['call_id'] for r in rr]
    require(len(ids) == len(set(ids)) and set(ids) <= set(jj), 'Duplicate/unexpected IDs')
    if complete:
        require(set(ids) == set(jj), 'Exact coverage failed; no partial score allowed')
    for r in rr:
        j = jj[r['call_id']]
        p = r['probabilities_F_M']
        require(len(p) == 2 and all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1 for x in p)
                and abs(sum(p)-1) < 1e-5, 'Invalid probability')
        require(r['prediction'] == ('M' if p[1] > p[0] else 'F'), 'Argmax mismatch')
        require(r['identity'] == identity and r['gender'] == j['gender'], 'Identity/label mismatch')
        require(r['samples'] == samples(j) and r['windows'] == math.ceil(samples(j)/(WINDOW*16000)),
                'Incomplete caller audio')
    return {r['call_id']: r for r in rr}


def metrics(rr):
    matrix = [[sum(r['gender'] == g and r['prediction'] == p for r in rr) for p in ('M', 'F')] for g in ('M', 'F')]
    n = len(rr)
    k = matrix[0][0] + matrix[1][1]
    return dict(calls=n, correct=k, wrong=n-k, accuracy=k/n, confusion_matrix_M_F=matrix,
                gender_accuracy={g: matrix[i][i]/sum(matrix[i]) for i, g in enumerate(('M', 'F'))})


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    owner = dict(pid=os.getpid(), host=socket.gethostname(), time=time.time(), token=os.urandom(16).hex())
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(owner, f)
        f.flush()
        os.fsync(f.fileno())
    try:
        yield
    finally:
        if path.exists() and read(path) == owner:
            path.unlink()


def bundle():
    m = runtime_manifest()
    for name, expected in m['files'].items():
        require(sha(ROOT/name) == expected, f'Bundle changed: {name}')
    jobs = read(ROOT/'data/jobs.json')
    require(len(jobs) == len({j['call_id'] for j in jobs}) == 27985, 'Jobs coverage')
    require(sum(j['partition'] == 'train' for j in jobs) == 22388, 'Train coverage')
    return m, jobs


# 입력 음성·경로 검사
import argparse
import os
import wave
from pathlib import Path

PATH_FIELDS = ('data_root', 'output_dir', 'hf_cache_dir', 'gpu_lock_path')


def validate_paths(config):
    paths = {}
    for key in PATH_FIELDS:
        value = config.get(key)
        require(isinstance(value, str) and value.strip(), f'Fill the blank path: {key}')
        p = Path(value).expanduser()
        require(p.is_absolute(), f'Use an absolute path for {key}')
        require(str(p.resolve()) == value, f'Use the resolved absolute path for {key}: {p.resolve()}')
        paths[key] = p
    data, out, cache, lock = (paths[k] for k in PATH_FIELDS)
    require(data.is_dir() and (data/'Training').is_dir(), 'data_root must directly contain Training/')
    protected = (data, ROOT.resolve())
    for p in (out, cache, lock):
        require(not any(p == q or p.is_relative_to(q) or q.is_relative_to(p) for q in protected),
                'Keep outputs, HF cache and GPU lock outside the original data and code folders')
    require(out != cache and not out.is_relative_to(cache) and not cache.is_relative_to(out),
            'Keep output_dir and hf_cache_dir separate')
    require(lock != out and lock != cache and not lock.is_relative_to(out) and not lock.is_relative_to(cache),
            'gpu_lock_path must be a shared host lock outside experiment output/cache')
    if (out/'identity.json').exists():
        old=read(out/'identity.json')
        require(old.get('bundle_sha256')==bundle_identity() and old.get('config')==config,
                'Existing output has a different execution identity; choose a new empty output')
    require(config.get('precision') in ('fp32','bf16'), 'Set precision: T4=fp32; L4=bf16 (or fp32)')
    require(isinstance(config.get('expected_gpu'), str) and config['expected_gpu'].strip(), 'Fill expected_gpu from nvidia-smi')
    return paths


def check_dataset(config):
    paths = validate_paths(config)
    manifest, jobs = bundle()
    inventory = rows(ROOT/'data/wav_inventory.csv')
    require(len(inventory) == len(jobs) and {r['call_id'] for r in inventory} == {j['call_id'] for j in jobs}, 'WAV inventory coverage')
    job_by = {j['call_id']: j for j in jobs}
    failures = []; total = 0
    for i, r in enumerate(inventory):
        rel = Path(r['wav_relative_path'])
        try:
            require(not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] == 'Training', 'Unsafe WAV path')
            require(str(rel) == job_by[r['call_id']]['wav_relative_path'], 'Inventory/job path mismatch')
            p = paths['data_root']/rel
            require(p.is_file() and p.stat().st_size == int(r['size_bytes']), 'WAV missing or byte size mismatch')
            with wave.open(str(p),'rb') as f:
                require((f.getframerate(),f.getnchannels(),f.getsampwidth(),f.getnframes()) ==
                        (8000,1,2,int(r['frames'])), 'WAV header mismatch')
            require(all(0 <= round(a*8000) < round(b*8000) <= int(r['frames'])
                        for a,b in job_by[r['call_id']]['intervals']), 'Caller crop exceeds WAV')
            total += p.stat().st_size
        except Exception as exc:
            failures.append(dict(call_id=r['call_id'],path=str(rel),error=str(exc)))
        if (i+1)%2000 == 0: print(f'WAV headers {i+1}/{len(inventory)}, failures={len(failures)}',flush=True)
    paths['output_dir'].mkdir(parents=True,exist_ok=True)
    result = dict(status='pass' if not failures else 'failed',data_root=config['data_root'],
                  bundle_sha256=bundle_identity(),calls=len(inventory),total_bytes=total,
                  failures=failures,verification='file names, sizes and 8kHz/mono/PCM16/frame-count headers; not full waveform content hashes')
    save(paths['output_dir']/'dataset_check.json',result)
    require(not failures, f'{len(failures)} WAV failures. See dataset_check.json; no GPU work started')
    print(result['status'], result['calls'], 'WAVs, bytes:', total,flush=True)
    return result


def require_dataset_check(config):
    result = read(Path(config['output_dir'])/'dataset_check.json')
    require(result['status']=='pass' and result['calls']==27985 and not result['failures'] and
            result['data_root']==config['data_root'] and result['bundle_sha256']==bundle_identity(),
            'Run run.py --worker check-data for this data path and code bundle first')


# 음성 처리·모델 (지연 로딩)
_audio_loaded = False


def load_audio():
    # 가상환경 준비가 끝난 작업 프로세스에서만 모델 라이브러리를 불러옵니다.
    global _audio_loaded, np, sf, torch, resample_poly, AutoFeatureExtractor, WavLMModel, load_wave, windows, crop, Encoder, Head, crop_stats
    if _audio_loaded:
        return
    import sys
    sys.modules['torchvision'] = None
    sys.modules['librosa'] = None
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoFeatureExtractor, WavLMModel


    def load_wave(j, root):
        # Adapted from gender_model_comparison/compare_models.py:load_wave.
        audio,sr=sf.read(Path(root)/j['wav_relative_path'],dtype='float32',always_2d=False)
        require(sr==8000 and audio.ndim==1 and np.isfinite(audio).all(),'Expected finite 8kHz mono WAV')
        parts=[]
        for start,end in j['intervals']:
            a,b=round(start*sr),round(end*sr)
            require(0<=a<b<=len(audio),'Invalid caller bounds')
            parts.append(resample_poly(audio[a:b],2,1,window=('kaiser',5.0)).astype(np.float32))
        result=np.concatenate(parts)
        require(len(result)==samples(j),'Caller sample coverage')
        return result


    def windows(x):
        # Balanced windows, no dropped tail; disjoint intervals can have boundary discontinuities.
        return np.array_split(x, math.ceil(len(x)/(WINDOW*16000)))


    def crop(x, seconds, seed):
        n=int(seconds*16000)
        start=int(np.random.default_rng(seed).integers(0,max(1,len(x)-n+1)))
        return x[start:start+n], start


    class Encoder(torch.nn.Module):
        def __init__(self, precision='bf16', mode='frozen'):
            super().__init__()
            self.precision=precision
            self.processor=AutoFeatureExtractor.from_pretrained(MODEL,revision=REVISION)
            self.backbone, self.loading=WavLMModel.from_pretrained(MODEL,revision=REVISION,output_loading_info=True)
            require(not self.loading['missing_keys'] and not self.loading['mismatched_keys'] and not self.loading['error_msgs'], 'Incomplete pretrained backbone load')
            self.backbone.config.apply_spec_augment=False
            self.backbone.config.layerdrop=0.0
            self.backbone.requires_grad_(False)
            if mode=='top2':
                for layer in self.backbone.encoder.layers[-2:]:layer.requires_grad_(True)
            self.backbone.eval()  # fixed dropout for controlled frozen/top2 comparison; gradients still enabled
            require(self.backbone.config.num_hidden_layers==24 and self.backbone.config.hidden_size==1024,'Unexpected backbone')
            require(self.processor.do_normalize and self.processor.sampling_rate==16000,'Normalization changed')
            self.cuda()

        def forward(self,x):
            require(len(x)>0,'Empty input')
            # Normalize real waveform first, then minimum conv-length padding. Batch size one: no temporal batching pad.
            inp=self.processor(x,sampling_rate=16000,return_tensors='pt',padding=False).input_values
            if inp.shape[1]<400: inp=torch.nn.functional.pad(inp,(0,400-inp.shape[1]))
            inp=inp.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.precision=='bf16'):
                hs=self.backbone(inp,output_hidden_states=True).hidden_states
            return torch.stack([torch.stack([hs[k][0].float().mean(0),hs[k][0].float().square().mean(0)]) for k in LAYERS])

        def pooled(self,x):
            moments=[];weights=[]
            with torch.no_grad():
                for part in windows(x):
                    moments.append(self(part));weights.append(len(part))
            w=torch.tensor(weights,device='cuda',dtype=torch.float32)
            stats=(torch.stack(moments)*w[:,None,None,None]).sum(0)/w.sum()
            return torch.stack([stats[:,0],(stats[:,1]-stats[:,0].square()).clamp_min(1e-7).sqrt()],1)

        def parameter_record(self):
            return dict(total=sum(p.numel() for p in self.parameters()),
                        trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                        trainable_names=[n for n,p in self.named_parameters() if p.requires_grad],
                        frozen_names=[n for n,p in self.named_parameters() if not p.requires_grad],
                        loading=self.loading, processor=self.processor.to_dict())


    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mix=torch.nn.Parameter(torch.zeros(4))
            self.norm=torch.nn.LayerNorm(2048)
            self.classifier=torch.nn.Linear(2048,2)

        def forward(self,stats):
            # stats [4,2,1024]: learned convex combination of layer mean/std.
            z=(stats*self.mix.softmax(0)[:,None,None]).sum(0).flatten()
            return self.classifier(self.norm(z))


    def crop_stats(m):
        return torch.stack([m[:,0],(m[:,1]-m[:,0].square()).clamp_min(1e-7).sqrt()],1)
    _audio_loaded = True


# 학습·평가 단계
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
    for line in REQUIREMENTS:
        if '==' in line:
            name,version=line.split('==')
            require(im.version(name)==version,f'Package mismatch: {name}')
    ident=dict(bundle_sha256=bundle_identity(), source_identity_sha256=None, input_provenance='verified bundled metadata; fresh teammate execution',
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
    load_audio()
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
    load_audio()
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
    load_audio()
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
    load_audio()
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
    load_audio()
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
    load_audio()
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
    load_audio()
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
    load_audio()
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


def worker_main():
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


# 결과 검증·기존 모델 비교
import argparse
import tempfile
import zipfile


def verify(folder, output):
    folder=Path(folder);output=Path(output)
    manifest,jobs=bundle()
    hh=read(folder/'export_manifest.json')
    for n,h in hh.items():
        require((folder/n).resolve().is_relative_to(folder.resolve()),'Unsafe export path')
        require(sha(folder/n)==h,'Export file hash mismatch: '+n)
    required={'dataset_check.json','identity.json','preflight.json','initial_parameters.json','benchmark.json','frozen_selection.json','selection.lock.json','final_models.json','validation_started.json','completion.json','events.jsonl'}
    require(required<=set(hh),'Missing provenance')
    ident=read(folder/'identity.json');ih=digest(ident)
    dataset=read(folder/'dataset_check.json')
    require(dataset['status']=='pass' and dataset['calls']==27985 and not dataset['failures'] and dataset['data_root']==ident['config']['data_root'] and dataset['bundle_sha256']==ident['bundle_sha256'],'Dataset provenance mismatch')
    require(ident['bundle_sha256']==bundle_identity(),'Unknown code bundle')
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
    basefile=ROOT/'data/baseline_predictions.csv'
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
    lines=['# WavLM Internal Validation results','',result['scope'],'',
           '| Model | Correct | Wrong | Overall | Male | Female |','|---|---:|---:|---:|---:|---:|']
    for name,r in summary.items():
        m=r['metrics'];lines.append(f"| {name} | {m['correct']} | {m['wrong']} | {m['accuracy']:.6%} | {m['gender_accuracy']['M']:.6%} | {m['gender_accuracy']['F']:.6%} |")
    for name,r in summary.items():
        m=r['metrics']
        lines+=['',f"{name} confusion (true/pred M,F): `{m['confusion_matrix_M_F']}`.",'']
        for b,o in r['versus_baselines'].items():lines.append(f"- {b}: corrected {o['corrected']}, regressed {o['regressed']}, both_wrong {o['both_wrong']}, both_correct {o['both_correct']}")
    lines+=['','99% requires at least 5542 correct calls. Do not tune on these results.',
            f'Historical failure records: {len(failures)}; unresolved failures: 0. See benchmark.json and execution records for measured runtime.']
    atomic(output/'REPORT.md',('\n'.join(lines)+'\n').encode());return result


def verify_main():
    p=argparse.ArgumentParser();p.add_argument('zip',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        with zipfile.ZipFile(a.zip) as z:
            require(len(z.namelist())==len(set(z.namelist())),'Duplicate ZIP paths')
            require(sum(i.file_size for i in z.infolist())<100*1024**2,'Unexpectedly large result ZIP')
            require(all((root/n).resolve().is_relative_to(root.resolve()) for n in z.namelist()),'Unsafe ZIP')
            z.extractall(root)
        print(json.dumps(verify(root,a.output),indent=2))


# 환경 준비·전체 실행
# 이 파일 맨 위의 빈 경로를 채운 뒤 python3 run.py 한 번으로 실행합니다.
# 경로/환경 오류는 GPU 학습 전에 중단합니다.
import argparse
import contextlib
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path


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
    result={}
    for key in SETTINGS_KEYS:
        value=globals().get(key)
        require(isinstance(value,str) and value.strip(), f'Fill {key} at the top of run.py')
        p=Path(value).expanduser()
        require(p.is_absolute(),f'{key} must be an absolute path')
        result[key]=str(p.resolve())
    require(Path(result['INPUT_JSON']).is_file(),'INPUT_JSON does not exist')
    require((Path(result['DATA_ROOT'])/'Training').is_dir(),'DATA_ROOT must directly contain Training/')
    require(Path(result['GPU_LOCK_PATH']).parent.is_dir(),'GPU_LOCK_PATH parent does not exist')
    return result


def prepare_metadata(path):
    # ZIP 대신 JSON 하나에서 입력을 복원합니다. 통화·split 정보는 동일합니다.
    m=runtime_manifest()
    require(sha(path)==m['input_json_sha256'],'Wrong input JSON; use wavlm_gender_inputs.json supplied with this code')
    package=read(path)
    require(package.get('format')=='wavlm-inputs-v1','Unsupported input format')
    files=package.get('files',{})
    expected={n:h for n,h in m['files'].items() if n.startswith('data/')}
    require(set(files)==set(expected),'Unexpected metadata members')
    for name,h in expected.items():
        p=(ROOT/name).resolve()
        require(p.is_relative_to((ROOT/'data').resolve()),'Unsafe metadata path')
        require(isinstance(files[name],str),'Metadata contents must be text')
        body=files[name].encode('utf-8')
        require(hashlib.sha256(body).hexdigest()==h,f'Metadata hash mismatch: {name}')
        if p.exists():require(sha(p)==h,f'Existing metadata changed: {name}')
        else:
            p.parent.mkdir(parents=True,exist_ok=True);atomic(p,body)
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
    expected=dict(base_python=str(Path(sys.executable).resolve()),torch=hardware['torch'],requirements_sha256=digest(REQUIREMENTS))
    if envdir.exists():require(marker.is_file() and read(marker)==expected,'Existing .venv is not this experiment environment')
    else:
        command([sys.executable,'-m','venv','--system-site-packages',envdir])
        source_sites=json.loads(command([sys.executable,'-c','import site,json; print(json.dumps(site.getsitepackages()))'],capture=True))
        target_sites=json.loads(command([python,'-c','import site,json; print(json.dumps(site.getsitepackages()))'],capture=True))
        local=next(Path(p) for p in target_sites if str(envdir) in p)
        (local/'existing_torch.pth').write_text('\n'.join(source_sites)+'\n')
        save(marker,expected)
    check="""import importlib.metadata as m,sys
for line in sys.argv[1:]:
 if '==' in line:
  name,version=line.strip().split('=='); assert m.version(name)==version, name
"""
    try:command([python,'-c',check,*REQUIREMENTS],capture=True)
    except RuntimeError:command([python,'-m','pip','install','--disable-pip-version-check',*REQUIREMENTS])
    command([python,'-c',check,*REQUIREMENTS],capture=True)
    require(probe(python)==hardware,'GPU/torch changed in the virtual environment')
    return python


def stage_done(out,stage):
    events=journal(out/'events.jsonl')
    status=[r['status'] for r in events if r['stage']==stage]
    return bool(status) and status[-1]=='complete'


def pipeline(python,config_path,config,stop_after_benchmark=False):
    out=Path(config['output_dir'])
    env=dict(os.environ,HF_HOME=config['hf_cache_dir'],CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')
    command([python,'-u',ROOT/'run.py','--worker','check-data','--config',config_path],env=env)
    for stage in ('preflight','benchmark','extract','frozen-select','lock-frozen','final-fit','validation'):
        # runner identity checks remain authoritative; the launcher never changes selection after locking.
        if stage_done(out,stage) and stage not in ('preflight','validation'):
            print('Completed stage retained:',stage,flush=True)
        else:
            print('Starting:',stage,flush=True)
            command([python,'-u',ROOT/'run.py','--worker',stage,'--config',config_path],env=env)
        if stage=='benchmark':
            print((out/'benchmark.json').read_text(),flush=True)
            if stop_after_benchmark:return
    command([python,ROOT/'run.py','--verify',out/'wavlm_results.zip','--output',out/'verified'])
    print('Verified results:',out/'verified/REPORT.md',flush=True)
    print('Return ZIP:',out/'wavlm_results.zip',flush=True)


def main():
    parser=argparse.ArgumentParser(description='run.py 맨 위의 경로 설정 후 실행. 기본: frozen 특징 + LR; 입력 ZIP 불필요.')
    parser.add_argument('--benchmark-only',action='store_true',help='벤치마크까지만 실행')
    args=parser.parse_args()
    settings=load_settings()
    require(platform.system()=='Linux','Run on a Linux CUDA GPU host')
    require(sys.version_info[:2]==(3,12),'Run with the Python 3.12 interpreter that has CUDA torch/torchaudio')
    prepare_metadata(settings['INPUT_JSON'])
    hardware=probe(sys.executable);profile=gpu_profile(hardware['gpu'])
    config=dict(data_root=settings['DATA_ROOT'],output_dir=settings['OUTPUT_DIR'],hf_cache_dir=settings['HF_CACHE_DIR'],
                gpu_lock_path=settings['GPU_LOCK_PATH'],expected_gpu=hardware['gpu'],precision=profile['precision'],crop_seconds=12,accumulation=4)
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
    try:
        if len(sys.argv)>1 and sys.argv[1]=='--worker':
            del sys.argv[1]
            if len(sys.argv)>1 and sys.argv[1]=='check-data':
                p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--config',required=True)
                check_dataset(read(p.parse_args().config))
            else:
                worker_main()
        elif len(sys.argv)>1 and sys.argv[1]=='--verify':
            del sys.argv[1]
            verify_main()
        else:
            main()
    except KeyboardInterrupt:
        print('Stopped. Rerun the same command to resume.',file=sys.stderr)
        sys.exit(130)
