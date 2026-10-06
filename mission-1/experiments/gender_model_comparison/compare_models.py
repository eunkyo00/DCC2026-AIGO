"""Frozen task-trained gender models: Drive-safe Train-only screening."""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

ROOT = Path(__file__).resolve().parent
MODELS = {
    'ecapa': ('JaesungHuh/voice-gender-classifier', 'db1222153bd60337e900be22add7af180452adc0'),
    'wavlm': ('tiantiaf/wavlm-large-age-sex', 'a4ad8039d8e298eb51d0bad71efb01a646ac56d5'),
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify_bundle():
    manifest = json.loads((ROOT / 'bundle_manifest.json').read_text())
    for filename, digest in manifest['files'].items():
        require(sha(ROOT / filename) == digest, f'Bundle hash mismatch: {filename}')
    return sha(ROOT / 'bundle_manifest.json')


def load_jobs():
    jobs = json.loads((ROOT / 'data/jobs.json').read_text())
    require(len(jobs) == len({j['call_id'] for j in jobs}) == 2000, 'Pilot IDs')
    require(all(j['partition'] == 'train' for j in jobs), 'Holdout prohibited')
    require(sum(j['smoke'] for j in jobs) == 6 and sum(j['benchmark'] for j in jobs) == 20, 'Fixtures')
    return jobs


def load_wave(job, data_root):
    path = data_root / job['wav_relative_path']
    require(path.is_file(), f'WAV missing: {path}')
    audio, sr = sf.read(path, dtype='float32', always_2d=False)
    require(sr == 8000 and audio.ndim == 1 and np.isfinite(audio).all(), 'Expected finite 8k mono WAV')
    parts = []
    for start, end in job['intervals']:
        a, b = round(start * sr), round(end * sr)
        require(0 <= a < b <= len(audio), f'Invalid crop: {job["call_id"]}')
        parts.append(resample_poly(audio[a:b], 2, 1, window=('kaiser', 5.0)).astype(np.float32))
    require(len(parts) == job['n_segments'], 'Segment coverage')
    return np.concatenate(parts)


def windows(signal):
    """Preserve all caller samples; balanced <=15s windows, single-input pad to 3s."""
    require(signal.ndim == 1 and len(signal) > 0, 'Empty waveform')
    n = math.ceil(len(signal) / 240000)
    for part in np.array_split(signal, n):
        count = len(part)
        yield np.pad(part, (0, max(0, 48000 - count))), count


def load_model(name, device):
    # Audio-only task: avoid optional torchvision/librosa imports and version conflicts.
    sys.modules['torchvision'] = None
    sys.modules['librosa'] = None
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sys.path.insert(0, str(ROOT / 'vendor'))
    if name == 'ecapa':
        from ecapa_model import ECAPA_gender
        cls = ECAPA_gender
    else:
        from wavlm_model import WavLMWrapper
        cls = WavLMWrapper
    repo, revision = MODELS[name]
    config_path = hf_hub_download(repo, 'config.json', revision=revision)
    config = json.loads(Path(config_path).read_text())
    if name == 'wavlm':
        require(config['pretrain_model'] == 'wavlm_large' and not config['apply_gradient_reversal'], 'Unsupported WavLM config')
    model = cls(**config)
    weight_path = hf_hub_download(repo, 'model.safetensors', revision=revision)
    model.load_state_dict(load_file(weight_path), strict=True)
    # eval once after strict loading: loralib merges its learned weights on eval.
    model.float().eval().requires_grad_(False).to(device)
    require(not model.training and all(not p.requires_grad for p in model.parameters()), 'Not frozen')
    require(all(p.dtype == torch.float32 for p in model.parameters()), 'Not FP32')
    return model, {'checkpoint': repo, 'revision': revision,
                   'weight_sha256': sha(weight_path), 'parameters': sum(p.numel() for p in model.parameters()),
                   'original_class_order': ['M', 'F'] if name == 'ecapa' else ['F', 'M']}


def predict(model, name, audio, device):
    import torch
    probabilities, lengths, padded = [], [], 0
    with torch.inference_mode():
        for part, length in windows(audio):
            x = torch.from_numpy(np.ascontiguousarray(part)).unsqueeze(0).to(device)
            out = model(x)
            logits = out if name == 'ecapa' else out[1]
            require(logits.shape == (1, 2) and torch.isfinite(logits).all().item(), 'Invalid logits')
            p = logits.softmax(-1)[0].detach().cpu().numpy()
            if name == 'ecapa':
                p = p[[1, 0]]  # Common output order: Female, Male.
            probabilities.append(p)
            lengths.append(length)
            padded += int(length < 48000)
    result = np.average(np.stack(probabilities), axis=0, weights=lengths)
    require(np.isfinite(result).all() and abs(result.sum() - 1) < 1e-5, 'Invalid probabilities')
    return result, len(lengths), padded


def recover(path):
    """Recover completed JSONL rows; drop only a truncated final line."""
    if not path.exists():
        return {}
    lines = path.read_bytes().splitlines(keepends=True)
    result, good = {}, []
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            require(i == len(lines) - 1, 'Corrupt interior cache line')
            path.write_bytes(b''.join(good))
            break
        require(row['call_id'] not in result, 'Duplicate cache call')
        result[row['call_id']] = row
        good.append(line if line.endswith(b'\n') else line + b'\n')
    else:
        if lines and not lines[-1].endswith(b'\n'):
            path.write_bytes(b''.join(good))
    return result


def metrics(rows):
    correct = sum(r['prediction'] == r['gender'] for r in rows)
    matrix = [[sum(r['gender'] == g and r['prediction'] == p for r in rows)
               for p in ('M', 'F')] for g in ('M', 'F')]
    return {'calls': len(rows), 'correct': correct, 'accuracy': correct / len(rows),
            'confusion_matrix_M_F': matrix,
            'gender_accuracy': {g: sum(r['gender'] == g and r['prediction'] == g for r in rows) /
                                sum(r['gender'] == g for r in rows) for g in ('M', 'F')}}


def evaluate(model, name, jobs, stage, output, root):
    import torch
    selected = [j for j in jobs if stage == 'pilot' or j[stage]]
    cache = output / f'{name}_{stage}.jsonl'
    completed = recover(cache)
    by_id = {j['call_id']: j for j in selected}
    require(set(completed) <= set(by_id), 'Unexpected cached IDs')
    for call, row in completed.items():
        require(row['gender'] == by_id[call]['gender'] and row['n_segments'] == by_id[call]['n_segments'], 'Cache mismatch')
        require(len(row['probabilities_F_M']) == 2 and np.isfinite(row['probabilities_F_M']).all(), 'Cache invalid')
    if len(completed) == len(selected) and (output / f'{name}_{stage}_summary.json').exists():
        print(f'RESUME {name} {stage}: complete ({len(completed)} calls)', flush=True)
        return
    torch.cuda.reset_peak_memory_stats()
    versions = [p._version for p in model.parameters()]
    with cache.open('a', buffering=1) as f:
        for j in selected:
            if j['call_id'] in completed:
                continue
            begin = time.perf_counter()
            try:
                audio = load_wave(j, root)
                read_seconds = time.perf_counter() - begin
                p, chunks, padded = predict(model, name, audio, 'cuda:0')
                torch.cuda.synchronize()
                row = {'call_id': j['call_id'], 'gender': j['gender'],
                       'prediction': ['F', 'M'][int(np.argmax(p))], 'probabilities_F_M': p.tolist(),
                       'n_segments': j['n_segments'], 'chunks': chunks, 'padded_chunks': padded,
                       'caller_samples_16k': len(audio), 'read_resample_seconds': read_seconds,
                       'elapsed_seconds': time.perf_counter() - begin}
                f.write(json.dumps(row, allow_nan=False) + '\n')
                completed[j['call_id']] = row
                if len(completed) % 10 == 0 or len(completed) == len(selected):
                    f.flush()
                    os.fsync(f.fileno())
                    print(f'{name} {stage}: {len(completed)}/{len(selected)} calls complete', flush=True)
            except Exception as exc:
                with (output / 'failure_attempts.jsonl').open('a') as errors:
                    errors.write(json.dumps({'model': name, 'stage': stage, 'call_id': j['call_id'],
                                              'error': repr(exc), 'utc': time.time()}) + '\n')
                raise
    require(versions == [p._version for p in model.parameters()], 'Encoder parameters changed')
    ordered = [completed[j['call_id']] for j in selected]
    require(len(ordered) == len(selected), 'Incomplete stage')
    elapsed = sum(r['elapsed_seconds'] for r in ordered)
    report = {'status': 'complete', 'model': name, 'stage': stage, **metrics(ordered),
              'prediction_shape': [len(ordered), 2], 'finite': True,
              'failed_calls': 0, 'successful_segments': sum(r['n_segments'] for r in ordered),
              'frozen_parameters_unchanged': True,
              'summed_call_seconds_excluding_model_load_and_cache_flush': elapsed,
              'seconds_per_call': elapsed / len(ordered),
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
              'cache_sha256': sha(cache), 'validation_evaluated': False,
              'projected_pilot_minutes_rough': elapsed / len(ordered) * 2000 / 60}
    save(output / f'{name}_{stage}_summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


def compare(jobs, output):
    report = {'scope': 'Train 2000-call preliminary screen, not final Validation',
              'validation_evaluated': False, 'models': {}, 'baseline_oof': {}}
    for name in MODELS:
        summary = json.loads((output / f'{name}_pilot_summary.json').read_text())
        cache = output / f'{name}_pilot.jsonl'
        require(summary['status'] == 'complete' and sha(cache) == summary['cache_sha256'], 'Pilot verification')
        rows_by_id = recover(cache)
        require(set(rows_by_id) == {j['call_id'] for j in jobs}, 'Pilot coverage')
        report['models'][name] = metrics([rows_by_id[j['call_id']] for j in jobs])
        base_wrong = [j['baseline_fusion_prediction'] != j['gender'] for j in jobs]
        new_wrong = [rows_by_id[j['call_id']]['prediction'] != j['gender'] for j in jobs]
        report['models'][name]['versus_fusion_oof'] = {
            'corrected': sum(a and not b for a, b in zip(base_wrong, new_wrong)),
            'regressed': sum(not a and b for a, b in zip(base_wrong, new_wrong)),
            'both_wrong': sum(a and b for a, b in zip(base_wrong, new_wrong))}
    for key in ('baseline_lr_prediction', 'baseline_fusion_prediction'):
        report['baseline_oof'][key] = metrics([{'gender': j['gender'], 'prediction': j[key]} for j in jobs])
    save(output / 'comparison.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['preflight', 'models', 'smoke', 'benchmark', 'pilot', 'compare'], required=True)
    args = parser.parse_args()
    bundle_sha = verify_bundle()
    config = json.loads(args.config.read_text())
    output, data_root = Path(config['output_dir']), Path(config['data_root'])
    output.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs()
    if args.stage == 'compare':
        require(json.loads((output / 'identity.json').read_text())['bundle_sha256'] == bundle_sha,
                'Comparison bundle differs from extraction')
        compare(jobs, output)
        return
    import torch
    require(torch.cuda.is_available(), 'CUDA GPU required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    require(importlib.metadata.version('transformers') == '4.46.3', 'Use bundled environment')
    identity = {'bundle_sha256': bundle_sha, 'models': MODELS, 'data_root': str(data_root),
                'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
                'versions': {p: importlib.metadata.version(p) for p in ['torch', 'torchaudio', 'numpy',
                             'scipy', 'transformers', 'loralib', 'huggingface-hub', 'safetensors']},
                'dtype': 'float32', 'autocast': False, 'tf32': False, 'batch_size': 1,
                'pipeline': 'caller_concat_balanced_3_to_15s_duration_weighted_probability_v1'}
    identity = json.loads(json.dumps(identity))
    identity_path = output / 'identity.json'
    if identity_path.exists():
        require(json.loads(identity_path.read_text()) == identity, 'Environment changed: choose a new output directory')
    else:
        require(not list(output.glob('*.jsonl')), 'Cache without identity')
        save(identity_path, identity)
    if args.stage == 'preflight':
        for j in jobs:
            require((data_root / j['wav_relative_path']).is_file(), f'Missing WAV: {j["wav_relative_path"]}')
        for j in jobs:
            if j['benchmark']:
                load_wave(j, data_root)
        save(output / 'preflight.json', {'status': 'pass', 'paths_checked': 2000,
                                         'wav_read_calls': 20, 'validation_evaluated': False})
        print('PREFLIGHT PASS: 2000 paths; 20 WAV reads; Train only', flush=True)
        return
    require((output / 'preflight.json').is_file(), 'Run preflight first')
    for name in MODELS:
        if args.stage in ('smoke', 'benchmark', 'pilot'):
            previous = {'smoke': 'models', 'benchmark': 'smoke', 'pilot': 'benchmark'}[args.stage]
            previous_file = output / (f'{name}_load.json' if previous == 'models' else f'{name}_{previous}_summary.json')
            require(previous_file.exists(), f'Run {previous} first for {name}')
        print(f'Loading {name} pinned FP32 model...', flush=True)
        started = time.perf_counter()
        model, info = load_model(name, 'cuda:0')
        if args.stage == 'models':
            p, _, _ = predict(model, name, np.zeros(48000, dtype=np.float32), 'cuda:0')
            save(output / f'{name}_load.json', {**info, 'status': 'pass', 'synthetic_shape': [1, 2],
                                               'finite': bool(np.isfinite(p).all()),
                                               'load_seconds': time.perf_counter() - started})
            print(f'{name}: STRICT MODEL LOAD + SYNTHETIC CUDA FORWARD PASS', flush=True)
        else:
            evaluate(model, name, jobs, args.stage, output, data_root)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
