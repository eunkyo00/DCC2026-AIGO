# Mission 1: WAV+JSON 29,200쌍 전체 재실험. Python 3.10~3.12 권장.
# Windows/Linux NVIDIA GPU에서 실행 권장. Mac/CPU는 --check-data 또는 느린 --device cpu 실행 가능.
# 이 파일 하나만 다운로드하면 됩니다. 다른 .py/requirements.txt 파일은 필요 없습니다.
# 이 파일에는 개인 경로/계정이 없습니다. 아래 SOURCES와 RUN_DIR만 각자 설정하세요.
#
# 설치 (새 가상환경 권장; 이 폴더에서 실행):
#   python -m venv .venv
#   Windows 활성화: .venv\Scripts\activate
#   Linux/Mac 활성화: source .venv/bin/activate
#   GPU용 PyTorch: https://pytorch.org/get-started/locally/ 에서 OS/GPU에 맞게 설치
#   예: Windows/Linux, RTX 50 시리즈/호환 드라이버:
#       python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
#   python -m pip install transformers==4.46.3 tokenizers==0.20.3 huggingface-hub==0.31.4 numpy==2.0.2 scipy==1.15.3 scikit-learn==1.6.1 soundfile==0.13.1 joblib==1.4.2 threadpoolctl==3.6.0
#   python run_full29200.py
#
# 파일 수정 없이 실행할 수도 있습니다 (경로에 공백이 있으면 따옴표 필수):
#   python run_full29200.py --source "D:/dataset/Training.zip" --run-dir "D:/results/full29200_01"
#   WAV/JSON ZIP이 따로 있으면 --source를 두 번 지정하세요.
#   원본 검사만: 위 명령 끝에 --check-data (GPU 불필요, 특징 추출/학습은 시작하지 않음).
#   메모리가 작으면: --wavlm-batch-seconds 8 --wav2vec2-batch-seconds 16
#   중단 후: 같은 명령/실행 폴더로 재실행. 다중 파일 버전의 저장 폴더와는 구분하세요.
#   재시작 시: GPU/코드/버전/정밀도/배치 설정을 유지하세요.
#   새 실험: 새 --run-dir 사용. 과거 27,985개 실험의 특징/모델은 가져오지 않습니다.
#
# 처리: 23,360 학습 + 5,840 내부 검증 → 성능/앙상블 비율 확정 → 제출 LR은 29,200 전체 재학습.
# 출력 accuracy는 재학습 전 검증 모델 기준. 공식 Validation은 사용하지 않습니다.
# 사전학습 WavLM/Wav2Vec2 본체는 고정하며, 특징 추출과 LR 학습을 처음부터 수행합니다.
# 누락/잘못된 라벨이 있으면 전체 실험을 중단합니다. 코드는 없는 음성 파일을 복구하지 않습니다.
#
# GPU 연산에 원본을 네트워크 디스크에서 반복 읽지 않도록 작업 폴더로 복사/압축 해제합니다.
# 작업 디스크는 원본 ZIP+압축 해제 데이터+모델/최종 ZIP 공간이 필요합니다.
# 작업 폴더를 로컬 SSD, 결과 폴더를 지속 저장 디스크로 지정할 수 있습니다.
# 64통화마다 두 모델 특징/체크섬 저장. 현재 미저장 블록만 재계산합니다.
# 첫 64개 이후 두 모델 실측 ETA 갱신. 복사/검사/모델 로딩/분류기 학습/ZIP 시간은 ETA에서 제외.
# 첫 실행은 공개 사전학습 가중치 다운로드용 인터넷 필요. 추론에는 인터넷이 필요 없습니다.
#
# 최종 결과: RUN_DIR/Mission1_full29200_result.zip (model.pt, 추론 코드, 재현 소스, 검증 기록).
# 중간 기록: features/, progress.json, events.jsonl, dataset_check.json, last_failure.json.
# 원본 WAV/JSON은 결과 ZIP에 들어가지 않습니다. 같은 RUN_DIR은 한 프로세스에서만 사용하세요.
# BF16 모델은 BF16 CUDA GPU 필요. CPU 추론도 필요하면 처음부터 PRECISION="fp32" 선택.
# 학습 완료 후 새 음성 예측 (이 파일과 model.pt만 필요):
#   python run_full29200.py --predict --audio-dir "WAV 경로" --label-dir "JSON 경로" --checkpoint "model.pt" --predictions "predictions.csv"
# 실제 전체 데이터/GPU 소요 시간·성능은 실행 후 결정됩니다. 합성 CPU 검증만 사전 수행했습니다.
import argparse
import csv
import functools
import hashlib
import io
import math
import platform
import shutil
import traceback
import unicodedata
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
import contextlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent

# [필수 설정 1] Training ZIP 또는 압축 해제된 Training 폴더. 둘 중 한 방식만 사용하세요.
# Windows 예시: SOURCES = [r"D:\dataset\Training.zip"]
# Linux 예시:   SOURCES = ["/data/emergency/Training"]
# 분리 ZIP:     SOURCES = [r"D:\dataset\TS_서울_구급.zip", r"D:\dataset\TL_서울_구급.zip"]
SOURCES = []

# [필수 설정 2] 실험 결과를 영구 보관할 폴더. 기본값은 코드 폴더/runs/full29200_01.
# Windows 예시: RUN_DIR = Path(r"D:\results\mission1_full29200_01")
RUN_DIR = HERE / "runs" / "full29200_01"

# [선택] None이면 RUN_DIR/.work, RUN_DIR/.hf_cache. 원본과 겹치지 않는 경로를 사용하세요.
# WORK_DIR = Path(r"E:\ssd_work\mission1_full29200_01")
WORK_DIR = None
HF_CACHE_DIR = None

# [선택] auto=CUDA 없으면 안내 후 중단. CPU는 매우 느리므로 명시적으로 "cpu" 선택.
DEVICE = "auto"          # "auto", "cuda", "cpu"
PRECISION = "auto"       # "auto", "bf16", "fp32" (지원 CUDA=BF16, 그 외=FP32)



# 외부 라이브러리 설치 목록. 결과 ZIP에 실제 PyTorch 버전과 함께 자동 기록합니다.
REQUIREMENTS = 'transformers==4.46.3\ntokenizers==0.20.3\nhuggingface-hub==0.31.4\nnumpy==2.0.2\nscipy==1.15.3\nscikit-learn==1.6.1\nsoundfile==0.13.1\njoblib==1.4.2\nthreadpoolctl==3.6.0\n'

def load_dependencies():
    # --help와 경로 설정 확인에는 GPU/외부 라이브러리 설치가 필요하지 않습니다.
    global np, sf, torch, resample_poly, expit
    global WavLMConfig, WavLMModel, Wav2Vec2Config, Wav2Vec2Model
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from scipy.special import expit
    from transformers import WavLMConfig, WavLMModel, Wav2Vec2Config, Wav2Vec2Model


def inference_only(function):
    # torch를 지연 로딩해 파일 하나로도 --help를 바로 확인할 수 있습니다.
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with torch.inference_mode():
            return function(*args, **kwargs)
    return wrapped


# 음성 전처리·특징 추출·체크포인트 추론
MODELS = {
    'wavlm': ('microsoft/wavlm-large', 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'),
    'wav2vec2': ('facebook/wav2vec2-base', '0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8'),
}
# Fixed padding buckets make features independent of other calls in the batch.
# Wav2Vec2-base uses group norm: do not pass attention_mask (HF model guidance).
# This is a new batching/padding protocol, not bit-identical to the old experiment.
BUCKETS = (8000, 16000, 32000, 48000, 64000, 96000, 128000, 192000, 240000)
PROTOCOL = 'full29200-fixed-buckets-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def intervals(doc):
    result = []
    for u in doc['utterances']:
        speaker = u.get('speaker', u.get('Speaker'))
        require(str(speaker) in ('0', '1'), 'Invalid speaker')
        if str(speaker) != '1':
            continue
        a, b = u['startAt'], u['endAt']
        require(all(type(v) in (int, float) and math.isfinite(v) for v in (a, b)), 'Invalid boundary')
        require(0 <= a < b, 'Invalid caller boundary')
        result.append((float(a), float(b)))
    require(bool(result), 'No caller segments')
    return result


def read_segments(path, bounds):
    audio, rate = sf.read(path, dtype='float32')
    require(rate == 8000 and audio.ndim == 1 and np.isfinite(audio).all(), 'Expected finite 8kHz mono WAV')
    segments = []
    for start, end in bounds:
        a, b = round(start * rate / 1000), round(end * rate / 1000)
        require(0 <= a < b <= len(audio), 'Caller boundary exceeds audio')
        segments.append(resample_poly(audio[a:b], 2, 1, window=('kaiser', 5.0)).astype(np.float32))
    require(bool(segments), 'No caller audio')
    return segments


def units(segments, kind):
    """Yield (normalized samples, segment index, aggregation weight)."""
    if kind == 'wavlm':
        audio = np.concatenate(segments)
        chunks = np.array_split(audio, math.ceil(len(audio) / 192000))
        for x in chunks:
            n = len(x)
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
            yield np.pad(x, (0, max(0, 400 - len(x)))), 0, n
    else:
        for segment_index, x in enumerate(segments):
            x = np.pad(x, (0, max(0, 400 - len(x))))
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
            for chunk in np.array_split(x, math.ceil(len(x) / 240000)):
                yield chunk, segment_index, 1


def configure_torch():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class Encoder:
    def __init__(self, kind, precision, cache_dir=None, entry=None, device='cuda'):
        self.kind, self.precision, self.device = kind, precision, torch.device(device)
        config_type, model_type = ((WavLMConfig, WavLMModel) if kind == 'wavlm'
                                   else (Wav2Vec2Config, Wav2Vec2Model))
        if entry is None:
            name, revision = MODELS[kind]
            self.model = model_type.from_pretrained(name, revision=revision, cache_dir=cache_dir)
        else:
            self.model = model_type(config_type.from_dict(entry['config']))
            self.model.load_state_dict(entry['weights'], strict=True)
        self.model.requires_grad_(False).eval().to(self.device)
        self.dim = self.model.config.hidden_size

    @inference_only
    def _forward(self, batch, size):
        try:
            return self._forward_once(batch, size)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise RuntimeError('GPU memory too small even for one window; choose a larger GPU') from None
        torch.cuda.empty_cache()
        middle = len(batch) // 2
        a, na = self._forward(batch[:middle], size)
        b, nb = self._forward(batch[middle:], size)
        return np.concatenate([a, b]), np.concatenate([na, nb])

    @inference_only
    def _forward_once(self, batch, size):
        # Retry smaller batches on OOM. Fixed padding is retained, so math does not change.
        lengths = torch.tensor([len(r[0]) for r in batch], device=self.device)
        x = np.zeros((len(batch), size), dtype=np.float32)
        for i, r in enumerate(batch):
            x[i, :len(r[0])] = r[0]
        values = torch.from_numpy(x).to(self.device)
        kwargs = {}
        if self.kind == 'wavlm':
            kwargs['attention_mask'] = (torch.arange(size, device=self.device)[None] < lengths[:, None]).long()
        amp = (torch.autocast('cuda', dtype=torch.bfloat16)
               if self.device.type == 'cuda' and self.precision == 'bf16' else contextlib.nullcontext())
        with amp:
            h = self.model(values, **kwargs).last_hidden_state
        frame_lengths = self.model._get_feat_extract_output_lengths(lengths).long()
        mask = torch.arange(h.shape[1], device=self.device)[None] < frame_lengths[:, None]
        mean = (h.float() * mask[:, :, None]).sum(1) / frame_lengths[:, None]
        result = mean.cpu().numpy()
        require(np.isfinite(result).all(), 'Nonfinite encoder feature')
        return result, frame_lengths.cpu().numpy()

    def encode(self, prepared, batch_seconds):
        buckets = {}
        counts = []
        for call, segments in enumerate(prepared):
            counts.append(1 if self.kind == 'wavlm' else len(segments))
            for x, segment, weight in units(segments, self.kind):
                size = next(n for n in BUCKETS if n >= len(x))
                buckets.setdefault(size, []).append((x, call, segment, weight))
        sums = [np.zeros((n, self.dim), np.float64) for n in counts]
        weights = [np.zeros(n, np.float64) for n in counts]
        for size, records in sorted(buckets.items()):
            batch_size = min(64, max(1, int(batch_seconds * 16000 / size)))
            for offset in range(0, len(records), batch_size):
                batch = records[offset:offset + batch_size]
                features, frames = self._forward(batch, size)
                for r, f, nf in zip(batch, features, frames):
                    _, call, segment, weight = r
                    weight = weight if self.kind == 'wavlm' else int(nf)
                    sums[call][segment] += f * weight
                    weights[call][segment] += weight
        return np.stack([(s / w[:, None]).mean(0) for s, w in zip(sums, weights)]).astype(np.float32)

    def entry(self):
        return {'config': self.model.config.to_dict(),
                'weights': {k: v.detach().cpu() for k, v in self.model.state_dict().items()}}


def probability(x, state):
    value = np.asarray(x, dtype=np.float32).copy()
    value -= state['mean'].numpy()
    value /= state['scale'].numpy()
    p = expit(value @ state['coef'].numpy().reshape(-1) + state['intercept'].item())
    return np.stack([1-p, p], axis=-1)


class GenderEnsemble:
    def __init__(self, checkpoint, device="auto"):
        configure_torch()
        state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
        require(state['format'] == PROTOCOL and state['classes'] == ['F', 'M'], 'Unsupported checkpoint')
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        require(device in ('cuda', 'cpu'), 'Choose cuda or cpu')
        require(device != 'cuda' or torch.cuda.is_available(), 'CUDA GPU not available')
        require(state['precision'] != 'bf16' or (device == 'cuda' and torch.cuda.is_bf16_supported()),
                'BF16 checkpoint requires BF16-capable CUDA GPU; use fp32 when training for CPU inference')
        self.state = state
        self.encoders = {k: Encoder(k, state['precision'], entry=state[k], device=device) for k in MODELS}

    def predict_segments(self, prepared):
        p = 0
        for kind, encoder in self.encoders.items():
            x = encoder.encode(prepared, self.state['batch_seconds'][kind])
            w = self.state['wavlm_weight'] if kind == 'wavlm' else 1-self.state['wavlm_weight']
            p = p + w * probability(x, self.state[kind]['lr'])
        require(np.isfinite(p).all(), 'Nonfinite predictions')
        return p

# 데이터 검사·학습·중간 저장·결과 패키징
EXPECTED = 29200
BLOCK = 64


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def save(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def read(path):
    return json.loads(Path(path).read_text())


def csv_bytes(rows, fields):
    s = io.StringIO()
    w = csv.DictWriter(s, fieldnames=fields)
    w.writeheader()
    w.writerows({k: row[k] for k in fields} for row in rows)
    return s.getvalue().encode()


def is_training(name):
    parts = Path(name).parts
    return not any(p.startswith('._') or p == '__MACOSX' or p.lower() == 'validation'
                   or p.upper().startswith(('VS_', 'VL_')) for p in parts)


def stage(sources, target, progress):
    """Copy/extract into work disk; keep network I/O outside GPU feature extraction."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    owner=target/'.stage_owner.json'
    require(not any(target.iterdir()) or (owner.exists() and read(owner).get('owner')=='mission1-full29200'),
            '임시 데이터 경로에 다른 파일이 있습니다. 비어 있는 work_dir를 지정하세요.')
    save(owner,{'owner':'mission1-full29200'})
    items = []
    for source in sources:
        source = Path(source)
        require(source.exists(), 'Selected source is missing')
        paths = sorted(source.rglob('*')) if source.is_dir() else [source]
        for p in paths:
            if p.is_file() and is_training(str(p)) and p.suffix.lower() in ('.zip', '.wav', '.json'):
                st = p.stat()
                items.append((p, st.st_size, st.st_mtime_ns))
    require(bool(items), 'No ZIP/WAV/JSON found; select Training source')
    identity = digest([(str(p), size, mtime) for p, size, mtime in items])
    marker = target / 'staging.json'
    if marker.exists() and read(marker)['identity'] == identity:
        progress('로컬 데이터 준비 완료 기록 재사용 (모델 특징 캐시는 별도 새 실험).')
        return target
    for sub in ('wav', 'json'):
        if (target / sub).exists():
            shutil.rmtree(target / sub)
        (target / sub).mkdir()
    started = last = time.monotonic()
    count = 0
    seen = set()

    def write_member(name, size, opener):
        nonlocal count, last
        suffix = Path(name).suffix.lower()
        if not is_training(name) or suffix not in ('.wav', '.json'):
            return
        stem = unicodedata.normalize('NFC', Path(name).stem)
        key = (suffix, stem)
        require(key not in seen, 'Duplicate source filename; select non-overlapping sources')
        seen.add(key)
        require(shutil.disk_usage(target).free > size + (2 << 30), 'Insufficient work disk space')
        dest = target / suffix[1:] / (stem + suffix)
        temp = dest.with_name(dest.name + '.tmp')
        with opener() as src, temp.open('wb') as dst:
            shutil.copyfileobj(src, dst, 4 << 20)
        require(temp.stat().st_size == size, 'Short source copy')
        temp.replace(dest)
        count += 1
        if time.monotonic() - last > 10:
            progress(f'로컬 준비: {count:,}파일 · 경과 {(time.monotonic()-started)/60:.1f}분')
            last = time.monotonic()

    def unpack(path, depth=0):
        require(depth <= 3, 'Too many nested ZIP layers')
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir() or not is_training(info.filename):
                    continue
                if Path(info.filename).suffix.lower() == '.zip':
                    nested = target / f'nested_{depth}.zip'
                    require(shutil.disk_usage(target).free > info.file_size + (2 << 30), 'Insufficient nested ZIP space')
                    with z.open(info) as src, nested.open('wb') as dst:
                        shutil.copyfileobj(src, dst, 4 << 20)
                    unpack(nested, depth + 1)
                    nested.unlink()
                else:
                    write_member(info.filename, info.file_size, lambda i=info: z.open(i))

    for p, size, _ in items:
        if p.suffix.lower() == '.zip':
            # A single sequential copy avoids repeated random ZIP reads over network filesystems.
            require(shutil.disk_usage(target).free > size + (2 << 30), 'Not enough work disk for ZIP; choose a larger work disk or an extracted folder')
            local = target / 'source.zip'
            progress('ZIP을 작업 디스크로 복사 중…')
            copied=0; copy_started=copy_last=time.monotonic()
            with p.open('rb') as src, local.open('wb') as dst:
                while chunk := src.read(8 << 20):
                    dst.write(chunk);copied+=len(chunk)
                    now=time.monotonic()
                    if now-copy_last>10:
                        eta=(now-copy_started)/copied*(size-copied)
                        progress(f'ZIP 로컬 복사 {copied/max(size,1):.1%} · 남은 복사 약 {eta/60:.1f}분')
                        copy_last=now
            require(local.stat().st_size==size,'Incomplete ZIP copy')
            unpack(local)
            local.unlink()
        else:
            write_member(p.name, size, lambda p=p: p.open('rb'))
    save(marker, {'identity': identity, 'files': count})
    progress(f'로컬 준비 완료: {count:,}파일 · {(time.monotonic()-started)/60:.1f}분')
    return target


def validate_dataset(root, out, progress, expected=EXPECTED):
    """Never silently turn the requested full run into a smaller experiment."""
    root, out = Path(root), Path(out)
    wavs = {p.stem: p for p in (root / 'wav').glob('*.wav')}
    labels = {p.stem: p for p in (root / 'json').glob('*.json')}
    report = {'expected': expected, 'wav_files': len(wavs), 'json_files': len(labels),
              'missing_wav': sorted(set(labels)-set(wavs)), 'missing_json': sorted(set(wavs)-set(labels)),
              'failures': []}
    jobs = []
    started = time.monotonic()

    def inspect(stem):
        p, label = wavs[stem], labels[stem]
        doc = json.loads(label.read_text(encoding='utf-8-sig'))
        require(doc.get('gender') in ('F', 'M'), 'Invalid gender')
        require(isinstance(doc.get('address'), str) and '서울' in doc['address'], 'Training must be Seoul only')
        bounds = intervals(doc)
        info = sf.info(p)
        require(info.samplerate == 8000 and info.channels == 1 and info.subtype == 'PCM_16', 'Expected PCM16/8kHz/mono')
        require(all(0 <= round(a*8) < round(b*8) <= info.frames for a, b in bounds), 'Boundary outside WAV')
        # Hash entire source; full decode/finite/short-read checked during feature extraction.
        audio_hash = sha(p)
        return dict(call_id=hashlib.sha256(stem.encode()).hexdigest(), gender=doc['gender'],
                    wav_path=str(p), wav_sha256=audio_hash, label_sha256=sha(label),
                    intervals_ms=bounds, audio_seconds=info.frames/8000,
                    caller_seconds=sum(round(b*8)-round(a*8) for a,b in bounds)/8000)

    common = sorted(set(labels) & set(wavs))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = pool.map(lambda stem: safe_inspect(inspect, stem), common)
        for i, (job, failure) in enumerate(futures):
            if failure:
                report['failures'].append(failure)
            else:
                jobs.append(job)
            if (i+1) % 500 == 0:
                progress(f'전수 검사 {i+1:,}/{len(common):,} · 오류 {len(report["failures"]):,} · 경과 {(time.monotonic()-started)/60:.1f}분')
    report['valid_calls'] = len(jobs)
    report['pass'] = (len(wavs) == len(labels) == len(jobs) == expected and not report['failures']
                      and not report['missing_wav'] and not report['missing_json'])
    report['verification'] = 'all file bytes SHA256, labels, headers, caller bounds; decode checked at extraction'
    save(out / 'dataset_check.json', report)
    require(report['pass'], f'29,200개 전수 검사 실패: WAV {len(wavs):,}, JSON {len(labels):,}, 정상 {len(jobs):,}. dataset_check.json 확인. GPU 추출은 시작하지 않습니다.')
    jobs.sort(key=lambda r:r['call_id'])
    return jobs


def safe_inspect(fn, stem):
    try:
        return fn(stem), None
    except Exception as e:
        return None, {'file_ref': hashlib.sha256(stem.encode()).hexdigest(), 'error': str(e)}


def split_jobs(jobs):
    from sklearn.model_selection import train_test_split
    y = np.array([r['gender'] for r in jobs])
    train, val = train_test_split(np.arange(len(jobs)), test_size=0.2, stratify=y, random_state=42)
    train = set(train.tolist())
    for i, job in enumerate(jobs):
        job['partition'] = 'train' if i in train else 'internal_validation'
    train_hashes={r['wav_sha256'] for r in jobs if r['partition']=='train'}
    val_hashes={r['wav_sha256'] for r in jobs if r['partition']=='internal_validation'}
    require(not train_hashes & val_hashes, 'Identical WAV bytes cross train/validation; resolve duplicate recordings before full experiment')
    require(sum(r['partition']=='train' for r in jobs) == round(len(jobs)*0.8), 'Split quota mismatch')
    return jobs


def new_identity(jobs, precision, batch_seconds, device="cuda"):
    import scipy, sklearn, torch, transformers
    return {'format': PROTOCOL, 'calls': len(jobs), 'seed': 42, 'precision': precision,
            'block_calls': BLOCK, 'batch_seconds': batch_seconds, 'device':device,
            'models': MODELS, 'scope': 'fresh features; no previous experiment cache',
            'source_sha256': digest([{k:v for k,v in r.items() if k!='wav_path'} for r in jobs]),
            'code': {Path(__file__).name:sha(__file__)},
            'versions': {k:v for k,v in [('python',platform.python_version()),('numpy',np.__version__),
                         ('scipy',scipy.__version__),('sklearn',sklearn.__version__),
                         ('torch',torch.__version__),('transformers',transformers.__version__)]}}


def block_path(out, start):
    return Path(out) / 'features' / f'block_{start:05d}.npz'


def load_block(path, ids, identity):
    side = path.with_suffix('.json')
    if not path.exists() or not side.exists():
        return None
    require(read(side)['sha256'] == sha(path), 'Checkpoint checksum mismatch; damaged block must be restored')
    with np.load(path, allow_pickle=False) as z:
        require(z['identity'].item() == digest(identity) and np.array_equal(z['ids'], ids), 'Checkpoint identity mismatch')
        arrays = {k:z[k].copy() for k in MODELS}
    for kind, dim in [('wavlm',1024),('wav2vec2',768)]:
        require(arrays[kind].shape == (len(ids),dim) and np.isfinite(arrays[kind]).all(), 'Invalid saved features')
    return arrays


def save_block(path, ids, identity, arrays):
    memory = io.BytesIO()
    np.savez_compressed(memory, ids=ids, identity=digest(identity), **arrays)
    atomic(path, memory.getvalue())
    save(path.with_suffix('.json'), {'sha256':sha(path), 'calls':len(ids)})


def extract(jobs, out, identity, progress, cache_dir=None):
    import gc, torch
    configure_torch()
    device=identity.get('device','cuda')
    require(device=='cpu' or torch.cuda.is_available(), 'CUDA GPU unavailable')
    kinds = list(MODELS)
    encoders = None
    started = time.monotonic()
    measured_seconds = measured_audio = 0.0
    model_times = {k:0.0 for k in kinds}
    measured_calls = 0
    pending_audio = sum(j['caller_seconds'] for j in jobs)
    all_arrays = {k:[] for k in kinds}
    resumed = 0
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0,len(jobs),BLOCK):
                block = jobs[start:start+BLOCK]
                ids = np.array([r['call_id'] for r in block])
                path = block_path(out,start)
                arrays = load_block(path,ids,identity)
                seconds = sum(j['caller_seconds'] for j in block)
                if arrays is None:
                    if encoders is None:
                        progress('사전학습 음성 모델 2개 로딩 중 (기존 실험 특징·분류기는 사용하지 않음)…')
                        encoders = {k:Encoder(k,identity['precision'],cache_dir=cache_dir,device=device) for k in kinds}
                    tick = time.monotonic()
                    prepared = list(pool.map(lambda j:read_segments(j['wav_path'],j['intervals_ms']),block))
                    arrays = {}
                    for kind in kinds:
                        progress(f'현재 블록 {start+1:,}~{start+len(block):,}: {kind} 처리 중…')
                        t = time.monotonic()
                        arrays[kind] = encoders[kind].encode(prepared,identity['batch_seconds'][kind])
                        model_times[kind] += time.monotonic()-t
                    save_block(path,ids,identity,arrays)  # durable checkpoint before progress says complete
                    measured_seconds += time.monotonic()-tick
                    measured_audio += seconds
                    measured_calls += len(block)
                else:
                    resumed += len(block)
                for kind in kinds:
                    all_arrays[kind].append(arrays[kind])
                done = start+len(block)
                pending_audio -= seconds
                eta = measured_seconds / measured_audio * pending_audio if measured_audio else None
                state = {'completed_calls':done,'total_calls':len(jobs),'resumed_calls':resumed,
                         'new_calls_this_session':measured_calls,'elapsed_session_s':time.monotonic()-started,
                         'remaining_feature_estimate_s':eta,'measured_pipeline_s':measured_seconds,
                         'model_seconds':model_times,'precision':identity['precision'],
                         'device':device, 'gpu':torch.cuda.get_device_name() if device=='cuda' else None, 'status':'extracting'}
                save(Path(out)/'progress.json',state)
                eta_text = (f'{eta/3600:.2f}시간' if eta is not None else '측정 중')
                confidence = '초기 추정' if measured_calls < 512 else '실측 기반 추정'
                progress(f'특징 저장 {done:,}/{len(jobs):,} ({done/len(jobs):.1%}) · 남은 특징 추출 {eta_text} ({confidence}) · 재사용 {resumed:,}개')
                if measured_calls and done==len(block):
                    estimate = measured_seconds/measured_audio*sum(j['caller_seconds'] for j in jobs)
                    progress(f'첫 블록 실측: 전체 특징 추출 약 {estimate/3600:.2f}시간. 원본 복사·검사·최종 ZIP 시간 제외; 이후 갱신합니다.')
    finally:
        encoders = None
        gc.collect()
        torch.cuda.empty_cache()
    return {k:np.concatenate(v) for k,v in all_arrays.items()}


def fit(x,y,c):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,random_state=42))
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x,y)
    require(list(model.classes_)==['F','M'],'Class order')
    return model


def score(y,p):
    require(p.shape==(len(y),2) and np.isfinite(p).all(),'Invalid probabilities')
    pred=np.array(['F','M'])[p.argmax(1)]
    return {'calls':len(y),'correct':int((pred==y).sum()),'accuracy':float((pred==y).mean()),
            'gender_accuracy':{g:float((pred[y==g]==g).mean()) for g in ('F','M')}}


def lr_state(model):
    import torch
    scaler, lr = model.steps[0][1], model.steps[1][1]
    return {k:torch.from_numpy(np.array(v,copy=True)) for k,v in
            [('mean',scaler.mean_),('scale',scaler.scale_),('coef',lr.coef_),('intercept',lr.intercept_)]}


def train(jobs, features, out, identity, progress):
    import joblib
    from sklearn.model_selection import StratifiedKFold
    out=Path(out)
    y=np.array([r['gender'] for r in jobs]); ids=np.array([r['call_id'] for r in jobs])
    tr=np.array([r['partition']=='train' for r in jobs]); va=~tr
    x={k:v[tr] for k,v in features.items()}; yt=y[tr]
    ps={k:np.zeros((len(yt),2)) for k in MODELS}
    for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(x['wavlm'],yt)):
        path=out/'oof'/f'fold_{fold}.npz'
        if path.exists() and path.with_suffix('.json').exists():
            require(read(path.with_suffix('.json'))['sha256']==sha(path),'OOF checksum mismatch')
            with np.load(path,allow_pickle=False) as z:
                require(np.array_equal(z['ids'],ids[tr][b]) and z['identity'].item()==digest(identity),'OOF identity mismatch')
                fold_p={k:z[k].copy() for k in MODELS}
        else:
            fold_p={k:fit(x[k][a],yt[a],0.1 if k=='wavlm' else 1.0).predict_proba(x[k][b]) for k in MODELS}
            buf=io.BytesIO(); np.savez_compressed(buf,ids=ids[tr][b],identity=digest(identity),**fold_p)
            atomic(path,buf.getvalue());save(path.with_suffix('.json'),{'sha256':sha(path)})
        for k in MODELS:
            ps[k][b]=fold_p[k]
        progress(f'Train OOF {fold+1}/3 저장 완료')
    candidates=[{'wavlm_weight':w,**score(yt,w*ps['wavlm']+(1-w)*ps['wav2vec2'])} for w in (1.,.75,.5,.25,0.)]
    # Only Training OOF chooses weight; internal Validation remains untouched until selection.
    selected=max(candidates,key=lambda r:(r['correct'],r['wavlm_weight']))['wavlm_weight']
    selection={'wavlm_weight':selected,'selection_scope':'3-fold OOF on Training partition only',
               'candidates':candidates,'identity':digest(identity)}
    save(out/'selection.json',selection)
    models={k:fit(x[k],yt,0.1 if k=='wavlm' else 1.0) for k in MODELS}
    temp=out/'evaluation_classifiers.joblib.tmp'; joblib.dump(models,temp);temp.replace(out/'evaluation_classifiers.joblib')
    p={k:m.predict_proba(features[k][va]) for k,m in models.items()}
    mixed=selected*p['wavlm']+(1-selected)*p['wav2vec2']
    metrics={'source_calls':len(jobs),'train_calls':int(tr.sum()),'internal_validation_calls':int(va.sum()),
             'ensemble':score(y[va],mixed),'individual':{k:score(y[va],v) for k,v in p.items()},
             'wavlm_weight':selected,'protocol':PROTOCOL,'final_refit_calls':len(jobs),
             'scope':'call-disjoint internal validation; not official Validation; not guaranteed person-disjoint',
             'changed_from_previous':'full 29200, new split, fixed bucket padding, auto BF16/FP32; old accuracy not directly comparable'}
    save(out/'metrics.json',metrics)
    rows=[dict(call_id=cid,gender=g,prediction=('F','M')[int(v.argmax())],p_f=float(v[0]),p_m=float(v[1]))
          for cid,g,v in zip(ids[va],y[va],mixed)]
    atomic(out/'validation_predictions.csv',csv_bytes(rows,['call_id','gender','prediction','p_f','p_m']))
    progress(f'내부 검증 완료: {int(va.sum()):,}통화 · 정확도 {metrics["ensemble"]["accuracy"]:.6%} · WavLM 가중치 {selected:.2f}')
    progress('검증 성능을 확정했습니다. 제출용 LR을 전체 29,200개로 재학습 중…')
    final_models={k:fit(features[k],y,0.1 if k=='wavlm' else 1.0) for k in MODELS}
    temp=out/'classifiers.joblib.tmp';joblib.dump(final_models,temp);temp.replace(out/'classifiers.joblib')
    return final_models,metrics


def export(jobs, features, models, metrics, identity, out, progress, work_dir=None, cache_dir=None):
    import gc, torch
    out=Path(out); local=Path(work_dir or out/'.work')/'export'
    local.mkdir(exist_ok=True,parents=True)
    state={'format':PROTOCOL,'classes':['F','M'],'precision':identity['precision'],
           'wavlm_weight':metrics['wavlm_weight'],'batch_seconds':identity['batch_seconds'],
           'source_calls':len(jobs),'train_calls':len(jobs),'evaluation_train_calls':metrics['train_calls'],'identity_sha256':digest(identity)}
    for kind in MODELS:
        progress(f'제출 체크포인트 구성: {kind}…')
        encoder=Encoder(kind,identity['precision'],cache_dir=cache_dir,device='cpu')
        state[kind]=encoder.entry();state[kind]['lr']=lr_state(models[kind])
        sample=features[kind][:128]
        require(np.allclose(probability(sample,state[kind]['lr']),models[kind].predict_proba(sample),atol=1e-6),'Exported LR differs')
        del encoder
        gc.collect()
    model=local/'model.pt';torch.save(state,model)
    # Load the exact exported bytes and check format; no pickle custom classes/network needed at inference.
    loaded=torch.load(model,map_location='cpu',weights_only=True)
    require(loaded['identity_sha256']==digest(identity),'Export roundtrip identity')
    del loaded,state
    gc.collect()
    shutil.copyfile(Path(__file__),local/'run_full29200.py')
    requirements=REQUIREMENTS
    atomic(local/'requirements.txt',(requirements+'\ntorch=='+torch.__version__.split('+')[0]+'\n').encode())
    report=(f'# Mission 1 full 29,200 fresh experiment\n\n'
            f'- Source: {len(jobs):,} calls; Train {metrics["train_calls"]:,}; Internal Validation {metrics["internal_validation_calls"]:,}.\n'
            f'- Internal Validation Accuracy: {metrics["ensemble"]["accuracy"]:.6%}.\n'
            f'- WavLM weight: {metrics["wavlm_weight"]}; precision: {identity["precision"]}.\n'
            f'- Final submission LR refit: all {len(jobs):,} calls AFTER metrics/weight were locked. The reported accuracy belongs to the evaluation model, not an independent test of the final refit model.\n'
            f'- Frozen pretrained encoders; LR heads are freshly fitted; no old experiment features.\n'
            f'- Protocol: {PROTOCOL}. Fixed bucket padding differs from previous experiment.\n'
            f'- Official Validation never used. Split is call-disjoint, not guaranteed person-disjoint.\n'
            f'- Verification: CPU tests, exported LR parity and checkpoint loading. Actual-device inference roundtrip: see export_verification.json.\n'
            f'- Run: python run_full29200.py --predict --audio-dir WAV --label-dir JSON --checkpoint model.pt --predictions predictions.csv\n')
    atomic(out/'REPORT.md',report.encode())
    # A real audio roundtrip is required before final completion is marked.
    verifier=GenderEnsemble(model,device=identity.get('device','cuda'))
    samples=jobs[:3]
    actual=verifier.predict_segments([read_segments(j['wav_path'],j['intervals_ms']) for j in samples])
    expected=sum((metrics['wavlm_weight'] if k=='wavlm' else 1-metrics['wavlm_weight'])*models[k].predict_proba(features[k][:3]) for k in MODELS)
    delta=float(np.max(np.abs(actual-expected)))
    tolerance=0.02 if identity['precision']=='bf16' else 0.002
    require(delta <= tolerance and np.array_equal(actual.argmax(1),expected.argmax(1)), 'Export inference roundtrip differs; no final ZIP released')
    save(out/'export_verification.json',{'calls':3,'max_probability_error':delta,'tolerance':tolerance,'predictions_match':True,'device':identity.get('device','cuda')})
    del verifier
    gc.collect();torch.cuda.empty_cache()
    names=('dataset_check.json','identity.json','split_assignments.csv','selection.json','metrics.json',
           'validation_predictions.csv','REPORT.md','export_verification.json','progress.json','classifiers.joblib','evaluation_classifiers.joblib','events.jsonl')
    save(out/'progress.json',{**read(out/'progress.json'),'status':'packaging','completed_calls':len(jobs)})
    if not (out/'events.jsonl').exists():
        atomic(out/'events.jsonl',b'')
    for name in names:
        shutil.copyfile(out/name,local/name)
    # Copy the exact runnable implementation for reproducibility, not just a stripped inference script.
    with zipfile.ZipFile(local/'training_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__),'run_full29200.py')
    archive=local.parent/'Mission1_full29200_result.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
        for name in ('model.pt','run_full29200.py','requirements.txt','training_source.zip',*names):
            z.write(local/name,name)
    progress('최종 ZIP을 결과 폴더에 저장 중…')
    target=out/archive.name
    temp=target.with_suffix('.zip.tmp');shutil.copyfile(archive,temp)
    require(sha(temp)==sha(archive),'Result ZIP copy checksum mismatch')
    temp.replace(target)
    save(out/'completion.json',{'status':'complete','source_calls':len(jobs),'zip_sha256':sha(target),
                                 'zip_bytes':target.stat().st_size,'metrics':metrics})
    save(out/'progress.json',{**read(out/'progress.json'),'status':'complete'})
    progress('완료: 결과 폴더의 Mission1_full29200_result.zip을 전달하면 됩니다.')
    return target


def resolve_config(config):
    # 작업/출력/캐시를 원본 안에 두지 않아 원본을 실수로 지우거나 재탐색하지 않습니다.
    config=dict(config)
    require(config.get('sources'), 'SOURCES 또는 --source에 원본 Training ZIP/폴더를 설정하세요.')
    sources=[Path(v).expanduser().resolve() for v in config['sources']]
    out=Path(config['output']).expanduser().resolve()
    work=Path(config.get('work_dir') or out/'.work').expanduser().resolve()
    cache=Path(config.get('hf_cache_dir') or out/'.hf_cache').expanduser().resolve()
    data=work/'data'
    for source in sources:
        require(source.exists(),'원본 경로가 없습니다: '+str(source))
        for dest in (out,work,cache):
            require(not (dest==source or dest.is_relative_to(source) or source.is_relative_to(dest)),
                    '원본과 출력/작업/캐시 경로가 겹칩니다. 별도 경로를 설정하세요.')
    require(not (out==work or out.is_relative_to(work)), 'output을 work_dir 안에 두면 안 됩니다.')
    require(not (cache==data or cache.is_relative_to(data) or data.is_relative_to(cache)), '캐시와 임시 데이터 경로가 겹칩니다.')
    require(not (out==data or out.is_relative_to(data)), '출력을 임시 데이터 안에 두면 안 됩니다.')
    config.update(sources=[str(s) for s in sources],output=str(out),work_dir=str(work),hf_cache_dir=str(cache))
    return config


def prepare(config,progress):
    out=Path(config['output']);out.mkdir(exist_ok=True,parents=True)
    root=stage(config['sources'],Path(config['work_dir'])/'data',progress)
    return split_jobs(validate_dataset(root,out,progress))


def run(config,progress=print):
    import torch
    config=resolve_config(config)
    out=Path(config['output']);out.mkdir(exist_ok=True,parents=True)
    device=config.get('device','auto')
    if device=='auto':
        require(torch.cuda.is_available(),'NVIDIA CUDA GPU가 없습니다. 설치를 확인하거나 느린 CPU 실행은 --device cpu로 명시하세요.')
        device='cuda'
    require(device in ('cuda','cpu'),'device: auto/cuda/cpu 중 선택')
    require(device!='cuda' or torch.cuda.is_available(),'CUDA를 사용할 수 없습니다. GPU용 PyTorch/드라이버를 확인하세요.')
    precision=config.get('precision','auto')
    if precision=='auto':precision='bf16' if device=='cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    require(precision in ('bf16','fp32'),'precision: auto/bf16/fp32 중 선택')
    require(precision!='bf16' or (device=='cuda' and torch.cuda.is_bf16_supported()),'이 장치는 BF16을 지원하지 않습니다. fp32를 선택하세요.')
    batches=({'wavlm':32.0,'wav2vec2':64.0} if precision=='bf16' else {'wavlm':12.0,'wav2vec2':32.0})
    batches.update(config.get('batch_seconds',{}))
    require(all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in batches.values()),'배치 초는 양수여야 합니다.')
    progress(f'장치: {torch.cuda.get_device_name() if device=="cuda" else "CPU (전체 실행 매우 느림)"} · 정밀도 {precision} · 전체 29,200통화 새 실험')
    try:
        jobs=prepare(config,progress)
        identity=new_identity(jobs,precision,batches,device)
        if (out/'identity.json').exists():
            require(read(out/'identity.json')==json.loads(json.dumps(identity)),'재시작 환경/데이터/코드가 달라졌습니다. 새 실행 폴더를 선택하세요.')
        else:
            save(out/'identity.json',identity)
        atomic(out/'split_assignments.csv',csv_bytes(jobs,['call_id','gender','partition']))
        if (out/'completion.json').exists():
            p=out/'Mission1_full29200_result.zip'
            require(sha(p)==read(out/'completion.json')['zip_sha256'],'Completed ZIP checksum mismatch')
            progress('이미 완료된 동일 실험입니다. 저장된 ZIP을 전달하세요.')
            return p
        save(out/'run_config.json',config)
        features=extract(jobs,out,identity,progress,cache_dir=config['hf_cache_dir'])
        models,metrics=train(jobs,features,out,identity,progress)
        return export(jobs,features,models,metrics,identity,out,progress,
                      work_dir=config['work_dir'],cache_dir=config['hf_cache_dir'])
    except BaseException as exc:
        save(out/'last_failure.json',{'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise

# 새 데이터 예측 CSV 생성
def predict_folder(audio_dir,label_dir,checkpoint,output,block_calls=32,device="auto"):
    audio_dir,label_dir,output=Path(audio_dir),Path(label_dir),Path(output)
    def index(root,suffix):
        result={}
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix.lower()==suffix and not p.name.startswith('._'):
                require(p.stem not in result,'Duplicate input stem')
                result[p.stem]=p
        return result
    wavs,labels=index(audio_dir,'.wav'),index(label_dir,'.json')
    require(bool(wavs) and set(wavs)<=set(labels),'Missing WAV or corresponding JSON')
    model=GenderEnsemble(checkpoint,device=device)
    names=sorted(wavs)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_name(output.name+'.tmp')
    try:
        with temp.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['audio file name','gender'])
            for offset in range(0,len(names),block_calls):
                batch=names[offset:offset+block_calls]
                prepared=[]
                for stem in batch:
                    doc=json.loads(labels[stem].read_text(encoding='utf-8-sig'))
                    # Only allowed segmentation fields reach the model. Never gender, text or address.
                    prepared.append(read_segments(wavs[stem],intervals(doc)))
                p=model.predict_segments(prepared)
                for stem,pred in zip(batch,p.argmax(1)):
                    writer.writerow([wavs[stem].name,('F','M')[int(pred)]])
                stream.flush()
                print(f'Predictions {min(offset+len(batch),len(names))}/{len(names)}',flush=True)
        temp.replace(output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return output



# 명령행 진입점
def arguments(argv=None):
    parser = argparse.ArgumentParser(description="Mission 1 full 29,200 fresh experiment; settings are documented in code comments.")
    parser.add_argument('--source',action='append',help='Training ZIP/folder; repeat for separate WAV/JSON archives')
    parser.add_argument('--run-dir',type=Path,default=RUN_DIR)
    parser.add_argument('--work-dir',type=Path,default=WORK_DIR)
    parser.add_argument('--hf-cache-dir',type=Path,default=HF_CACHE_DIR)
    parser.add_argument('--device',choices=('auto','cuda','cpu'),default=DEVICE)
    parser.add_argument('--precision',choices=('auto','bf16','fp32'),default=PRECISION)
    parser.add_argument('--wavlm-batch-seconds',type=float)
    parser.add_argument('--wav2vec2-batch-seconds',type=float)
    parser.add_argument('--check-data',action='store_true')
    parser.add_argument('--predict',action='store_true',help='Predict new WAVs using an exported model.pt')
    parser.add_argument('--audio-dir',type=Path)
    parser.add_argument('--label-dir',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--predictions',type=Path,default=Path('predictions.csv'))
    return parser.parse_args(argv)


@contextlib.contextmanager
def exclusive_run(out):
    # OS 파일 잠금은 프로세스 강제 종료 시 자동 해제됩니다. Windows/Linux/Mac 지원.
    out.mkdir(parents=True,exist_ok=True)
    handle=(out/'.run.lock').open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            handle.seek(0,os.SEEK_END)
            if handle.tell()==0:handle.write(b'0');handle.flush()
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except (OSError,BlockingIOError):
        handle.close()
        raise ValueError('이 결과 폴더에서 이미 다른 실행이 진행 중입니다.') from None
    try:
        yield
    finally:
        handle.close()


def main(argv=None):
    args=arguments(argv)
    config={'sources':[str(p) for p in (args.source or SOURCES)],'output':str(args.run_dir),
            'device':args.device,'precision':args.precision}
    for key in ('work_dir','hf_cache_dir'):
        value=getattr(args,key)
        if value is not None:config[key]=str(value)
    batches={}
    for kind in ('wavlm','wav2vec2'):
        value=getattr(args,kind+'_batch_seconds')
        if value is not None:batches[kind]=value
    if batches:config['batch_seconds']=batches
    if not args.predict and not config['sources']:
        print('SOURCES 주석 예시에 맞춰 경로를 입력하거나 --source를 지정하세요.',file=sys.stderr)
        return 2
    # 로그/진행 파일 저장; 노트북, Google Drive 계정, 외부 서비스 연결은 불필요합니다.
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    try:
        load_dependencies()
        if args.predict:
            if not (args.audio_dir and args.label_dir and args.checkpoint):
                raise ValueError('--predict에는 --audio-dir, --label-dir, --checkpoint가 필요합니다.')
            result=predict_folder(args.audio_dir,args.label_dir,args.checkpoint,args.predictions,device=args.device)
            print('예측 CSV: '+str(result),flush=True)
            return 0
        config=resolve_config(config)
        out=Path(config['output'])
        with exclusive_run(out):
            def progress(message):
                print(message,flush=True)
                with (out/'events.jsonl').open('a',encoding='utf-8') as log:
                    log.write(json.dumps({'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'message':message},ensure_ascii=False)+'\n')
            if args.check_data:
                jobs=prepare(config,progress)
                progress(f'원본 검사 통과: {len(jobs):,}통화. GPU 추출/학습을 시작하지 않았습니다.')
            else:
                result=run(config,progress)
                progress('결과 ZIP: '+str(result))
    except KeyboardInterrupt:
        print('중단되었습니다. 같은 명령으로 다시 실행하면 저장된 블록부터 이어집니다.',file=sys.stderr)
        return 130
    except Exception as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
