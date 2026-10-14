"""Frozen WavLM window features. Identical, label-independent processing for every split."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

import numpy as np

MODEL = 'microsoft/wavlm-large'
REVISION = 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'
INPUT_SHA = 'd3d121bd843d1cfd26cb0bd3a03653d70097ba66f8f412b28f2268f435e1a13a'
JOBS_SHA = '3e4f024ad7c0e5bec8fb2c5ae9af25c467dbc78129d8286faa79a54f237c4e89'
WINDOW = 12
DIM = 1024
BLOCK = 64


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def save(path, obj):
    atomic(path, (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def load_jobs(path):
    require(sha(path) == INPUT_SHA, 'Expected the existing wavlm_gender_inputs.json (verified version)')
    source = read(path)
    require(source['format'] == 'wavlm-inputs-v1', 'Input format')
    raw = source['files']['data/jobs.json']
    require(hashlib.sha256(raw.encode()).hexdigest() == JOBS_SHA, 'Job manifest checksum')
    jobs = sorted(json.loads(raw), key=lambda j: j['call_id'])
    require(len(jobs) == len({j['call_id'] for j in jobs}) == 27985, 'Expected 27985 unique calls')
    for partition, count in [('train', 22388), ('internal_validation', 5597)]:
        require(sum(j['partition'] == partition for j in jobs) == count, 'Partition coverage')
    for j in jobs:
        p = Path(j['wav_relative_path'])
        require(not p.is_absolute() and '..' not in p.parts and p.parts[0] == 'Training', 'Official Validation or unsafe audio path')
        require(j['gender'] in ('F', 'M') and j['intervals'], 'Labels/intervals')
    return jobs


def caller_samples(job):
    return sum((round(b * 8000) - round(a * 8000)) * 2 for a, b in job['intervals'])


def waveform(job, root):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(Path(root) / job['wav_relative_path'], dtype='float32', always_2d=False)
    require(sr == 8000 and x.ndim == 1 and np.isfinite(x).all(), 'Expected finite 8kHz mono audio')
    parts = []
    for a, b in job['intervals']:
        a, b = round(a * sr), round(b * sr)
        require(0 <= a < b <= len(x), 'Caller annotation outside audio')
        parts.append(resample_poly(x[a:b], 2, 1, window=('kaiser', 5.)).astype(np.float32))
    result = np.concatenate(parts)
    require(len(result) == caller_samples(job), 'Audio coverage mismatch')
    return result


def split_windows(x):
    require(len(x) > 0, 'Empty caller waveform')
    return np.array_split(x, math.ceil(len(x) / (WINDOW * 16000)))


class Encoder:
    def __init__(self):
        import sys
        # Avoid incompatible optional Colab image/audio packages; neither is used here.
        sys.modules['torchvision'] = None
        sys.modules['librosa'] = None
        import torch
        from transformers import AutoFeatureExtractor, WavLMModel
        require(torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(), 'Select Colab NVIDIA L4')
        self.torch = torch
        self.processor = AutoFeatureExtractor.from_pretrained(MODEL, revision=REVISION)
        self.model, loading = WavLMModel.from_pretrained(MODEL, revision=REVISION, output_loading_info=True)
        require(not loading['missing_keys'] and not loading['mismatched_keys'] and not loading['error_msgs'], 'Pretrained weight load incomplete')
        self.model.config.apply_spec_augment = False
        self.model.config.layerdrop = 0.
        self.model.requires_grad_(False).eval().cuda()
        require(self.processor.do_normalize and self.processor.sampling_rate == 16000, 'Processor changed')

    def encode(self, x):
        torch = self.torch
        feats, weights = [], []
        with torch.inference_mode():
            for part in split_windows(x):
                z = self.processor(part, sampling_rate=16000, return_tensors='pt', padding=False).input_values
                if z.shape[1] < 400:
                    z = torch.nn.functional.pad(z, (0, 400 - z.shape[1]))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    hidden = self.model(z.cuda()).last_hidden_state
                feats.append(hidden[0].float().mean(0).cpu().numpy())
                weights.append(len(part))
        return np.stack(feats).astype(np.float32), np.array(weights, dtype=np.int64)


def array_hash(x, weights, offsets):
    return hashlib.sha256(x.tobytes() + weights.tobytes() + offsets.tobytes()).hexdigest()


def write_block(path, jobs, values, identity):
    x = np.concatenate([a for a, _ in values]).astype(np.float32)
    weights = np.concatenate([b for _, b in values]).astype(np.int64)
    offsets = np.array([0] + list(np.cumsum([len(a) for a, _ in values])), dtype=np.int64)
    buf = io.BytesIO()
    np.savez_compressed(buf, ids=np.array([j['call_id'] for j in jobs]), x=x, weights=weights,
                        offsets=offsets, identity=identity, checksum=array_hash(x, weights, offsets))
    atomic(path, buf.getvalue())


def read_block(path, jobs, identity):
    with np.load(path, allow_pickle=False) as z:
        require(str(z['identity'].item()) == identity, 'Mixed feature identities')
        require(np.array_equal(z['ids'], [j['call_id'] for j in jobs]), 'Feature IDs/order changed')
        x, weights, offsets = z['x'].copy(), z['weights'].copy(), z['offsets'].copy()
        require(str(z['checksum'].item()) == array_hash(x, weights, offsets), 'Feature checksum mismatch')
    require(x.shape == (len(weights), DIM) and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid feature matrix')
    require(offsets.shape == (len(jobs) + 1,) and offsets[0] == 0 and offsets[-1] == len(x) and (np.diff(offsets) > 0).all(), 'Offsets/coverage')
    require((weights > 0).all() and (weights <= WINDOW * 16000).all(), 'Invalid real-window weights')
    result = []
    for i, j in enumerate(jobs):
        a, b = offsets[i:i+2]
        require(int(weights[a:b].sum()) == caller_samples(j), 'Caller samples missing')
        result.append((x[a:b], weights[a:b]))
    return result


def benchmark_jobs(jobs):
    # Equal coverage over the actual Train duration distribution; no label-specific processing.
    ordered = sorted((j for j in jobs if j['partition'] == 'train'), key=lambda j: (caller_samples(j), j['call_id']))
    return [ordered[i] for i in np.linspace(0, len(ordered)-1, 200, dtype=int)]


def extract(jobs, root, out, identity, partition, encoder=None):
    selected = [j for j in jobs if j['partition'] == partition]
    started = time.monotonic(); measured = []
    for start in range(0, len(selected), BLOCK):
        batch = selected[start:start+BLOCK]
        path = out / 'features' / partition / f'{start:05d}.npz'
        if path.exists():
            read_block(path, batch, identity)
            continue
        t = time.monotonic(); values = []
        for j in batch:
            bp = out / 'benchmark_features' / (j['call_id'] + '.npz')
            if bp.exists():
                values.extend(read_block(bp, [j], identity))
            else:
                if encoder is None:
                    encoder = Encoder()
                values.append(encoder.encode(waveform(j, root)))
        write_block(path, batch, values, identity)
        measured.append((time.monotonic()-t, len(batch)))
        sec = sum(a for a, _ in measured) / sum(b for _, b in measured)
        remaining = len(selected) - start - len(batch)
        print(f'{partition}: {start+len(batch)}/{len(selected)} | ETA {sec*remaining/3600:.2f} h', flush=True)
    save(out / ('extracted_' + partition + '.json'), dict(identity=identity, calls=len(selected), session_seconds=time.monotonic()-started))


def read_partition(jobs, out, identity, partition):
    selected = [j for j in jobs if j['partition'] == partition]; values = []
    for start in range(0, len(selected), BLOCK):
        values.extend(read_block(out / 'features' / partition / f'{start:05d}.npz', selected[start:start+BLOCK], identity))
    return selected, values
