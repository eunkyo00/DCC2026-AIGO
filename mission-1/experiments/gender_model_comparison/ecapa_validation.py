"""Pinned ECAPA Internal Validation. Run GPU inference only in the user's Colab."""
import argparse
import contextlib
import importlib.metadata
import os
import platform
import signal
import socket
import time
import uuid
import zipfile
from pathlib import Path

import compare_models as base
from validation_common import ROOT, bundle, digest, jsonl, read, require, samples, save, sha, validate


def append(path, row):
    import json
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


@contextlib.contextmanager
def lock(output):
    # Drive directory lock is also visible after runtime loss. Never auto-expire it.
    path = output / '.validation.lock'
    try:
        path.mkdir()
    except FileExistsError:
        raise RuntimeError(f'Active or stale lock: {path}. Stop the prior runtime first; see README.')
    owner = {'token': uuid.uuid4().hex, 'pid': os.getpid(), 'host': socket.gethostname(), 'utc': time.time()}
    try:
        save(path / 'owner.json', owner)
        yield
    finally:
        (path / 'owner.json').unlink(missing_ok=True)
        path.rmdir()


def environment(config, manifest):
    import torch
    import torchaudio
    require(torch.cuda.is_available(), 'CUDA required')
    require(torch.cuda.get_device_name(0) in ('NVIDIA L4', 'L4'),
            f'L4 GPU required, got {torch.cuda.get_device_name(0)}')
    source = Path(config['source_output'])
    old = read(source / 'identity.json')
    require(old['bundle_sha256'] == manifest['original_bundle_sha256'], 'Train source bundle mismatch')
    require(old['models']['ecapa'] == list(base.MODELS['ecapa']), 'Checkpoint/revision changed')
    require(old['data_root'] == config['data_root'], 'Use the original data_root')
    fixed = {'dtype': 'float32', 'autocast': False, 'tf32': False, 'batch_size': 1,
             'pipeline': 'caller_concat_balanced_3_to_15s_duration_weighted_probability_v1'}
    require(all(old[k] == v for k,v in fixed.items()), 'Train inference settings mismatch')
    require(old['python'] == platform.python_version(), 'Python differs from Train screen')
    require(old['gpu'] == torch.cuda.get_device_name(0), 'GPU differs from Train screen')
    versions = {p: importlib.metadata.version(p) for p in old['versions']}
    differences = [f'{p}: Train={expected!r}, current={versions[p]!r}'
                   for p, expected in old['versions'].items() if versions[p] != expected]
    require(not differences, 'Packages differ from Train screen; do not mix caches:\n' +
            '\n'.join(differences))
    versions['soundfile'] = importlib.metadata.version('soundfile')
    require(versions['soundfile'] == '0.13.1', 'Use the unchanged bundled soundfile version')
    versions['libsndfile'] = base.sf.__libsndfile_version__
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    info = read(source / 'ecapa_load.json')
    require(info['status'] == 'pass' and info['checkpoint'] == base.MODELS['ecapa'][0]
            and info['revision'] == base.MODELS['ecapa'][1], 'Train model load provenance')
    return {'bundle_sha256': sha(ROOT / 'validation_bundle_manifest.json'),
            'source_identity_sha256': sha(source / 'identity.json'),
            'source_model_load_sha256': sha(source / 'ecapa_load.json'),
            'model': {k: info[k] for k in ('checkpoint','revision','weight_sha256')},
            'data_root': config['data_root'], 'python': platform.python_version(),
            'gpu': torch.cuda.get_device_name(0), 'versions': versions,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'threads': 2, **fixed}


def initial_eta(config, jobs):
    source = Path(config['source_output'])
    screen = source.parent / (source.name + '_screen200')
    candidates = [(screen, 'ecapa_pilot'), (source, 'ecapa_benchmark')]
    for folder, stem in candidates:
        summary, cache = folder / (stem + '_summary.json'), folder / (stem + '.jsonl')
        if not summary.exists() or not cache.exists():
            continue
        s = read(summary)
        require(s['status'] == 'complete' and sha(cache) == s['cache_sha256'], 'Timing source hash mismatch')
        if folder == screen:
            require(read(screen / 'screen_identity.json')['source_identity'] == read(source / 'identity.json'),
                    'Screen timing identity mismatch')
        rows = jsonl(cache)
        require(rows and len(rows) == s['calls'] and len({r['call_id'] for r in rows}) == len(rows), 'Timing coverage')
        seconds = sum(r['elapsed_seconds'] for r in rows)
        audio = sum(r['caller_samples_16k'] for r in rows)/16000
        require(seconds > 0 and audio > 0, 'Invalid source timings')
        result = {'source': str(cache), 'source_sha256': sha(cache), 'measured_calls': len(rows),
                  'measured_seconds': seconds, 'measured_audio_seconds': audio,
                  'remaining_calls': len(jobs),
                  'by_call_minutes': seconds/len(rows)*len(jobs)/60,
                  'by_audio_minutes': seconds/audio*sum(samples(j) for j in jobs)/16000/60,
                  'note': 'Train measurements include WAV read/resample/inference, exclude model load and Drive flush; rough projections, not a confidence interval.'}
        print('INITIAL MEASURED ETA:', result, flush=True)
        return result
    print('No completed Train timing cache: ETA will appear after first Validation calls.', flush=True)
    return {'status': 'awaiting_validation_measurements'}


def export(output, jobs, identity):
    ih = digest(identity)
    rows = jsonl(output / 'predictions.jsonl')
    validate(rows, jobs, ih, complete=True)
    failures = jsonl(output / 'failure_attempts.jsonl')
    unresolved = sorted({r['call_id'] for r in failures if r.get('call_id')} - {r['call_id'] for r in rows})
    require(not unresolved, 'Unresolved failures')
    summary = {'status': 'complete', 'calls': len(rows), 'prediction_shape': [5597, 2],
               'identity_sha256': ih, 'cache_sha256': sha(output / 'predictions.jsonl'),
               'failed_calls': 0, 'historical_failure_attempts': len(failures),
               'caller_samples_16k': sum(r['caller_samples_16k'] for r in rows),
               'accuracy': 'computed only by local analysis'}
    save(output / 'completion.json', summary)
    names = ['identity.json','source_identity.json','source_ecapa_load.json','predictions.jsonl',
             'failure_attempts.jsonl','events.jsonl','completion.json','preflight.json','eta.json','progress.json',
             'lock_releases.jsonl']
    save(output / 'result_files.json', {n:sha(output/n) for n in names if (output/n).exists()})
    names.append('result_files.json')
    archive = output / 'ecapa_validation_results.zip'
    with zipfile.ZipFile(archive.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as z:
        for name in names:
            if (output / name).exists():
                z.write(output / name, name)
        z.write(ROOT / 'validation_bundle_manifest.json', 'validation_bundle_manifest.json')
    archive.with_suffix('.tmp').replace(archive)
    print('EXACT COVERAGE COMPLETE:', archive, flush=True)


def execute(config, stage):
    import torch
    manifest, jobs = bundle()
    output = Path(config['output_dir'])
    source = Path(config['source_output'])
    require(output.resolve() not in (source.resolve(), (source.parent/(source.name+'_screen200')).resolve()),
            'Validation output must be separate')
    output.mkdir(parents=True, exist_ok=True)
    with lock(output):
        identity = environment(config, manifest)
        ih = digest(identity)
        path = output / 'identity.json'
        if path.exists():
            require(read(path) == identity, 'Validation identity changed; refuse cache mixing')
        else:
            require(not list(output.glob('*.jsonl')), 'Results without identity')
            save(path, identity)
            (output / 'source_identity.json').write_bytes((source / 'identity.json').read_bytes())
            (output / 'source_ecapa_load.json').write_bytes((source / 'ecapa_load.json').read_bytes())
        for name in ('predictions.jsonl','failure_attempts.jsonl','events.jsonl'):
            jsonl(output / name, repair=True)
        rows = jsonl(output / 'predictions.jsonl')
        done = validate(rows, jobs, ih)
        pending = [j for j in jobs if j['call_id'] not in done]
        print(f'ENVIRONMENT PASS: {identity}; saved={len(done)}/5597', flush=True)
        if stage == 'preflight':
            began = time.perf_counter()
            probe = output / '.write_probe'
            append(probe, {'write': True})
            probe.unlink()
            ordered = sorted(jobs, key=samples)
            fixtures = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
            for j in fixtures:
                audio = base.load_wave(j, Path(config['data_root']))
                require(len(audio) == samples(j), 'Representative WAV sample coverage')
            save(output / 'preflight.json', {'status':'pass', 'identity_sha256':ih,
                 'representative_wav_ids':[j['call_id'] for j in fixtures],
                 'seconds': time.perf_counter()-began, 'all_other_paths_checked_on_use':True})
            save(output / 'eta.json', initial_eta(config, pending))
            return
        require(read(output / 'preflight.json')['identity_sha256'] == ih, 'Run preflight first')
        if not pending:
            export(output, jobs, identity)
            return
        (output / 'completion.json').unlink(missing_ok=True)
        (output / 'ecapa_validation_results.zip').unlink(missing_ok=True)
        append(output / 'events.jsonl', {'event':'run_start', 'utc':time.time(), 'remaining':len(pending), 'identity_sha256':ih})
        current = None
        try:
            start = time.perf_counter()
            model, info = base.load_model('ecapa', 'cuda:0')
            require(info['weight_sha256'] == identity['model']['weight_sha256'], 'Weight hash changed')
            print(f'Model loaded: {time.perf_counter()-start:.1f}s', flush=True)
            versions = [p._version for p in model.parameters()]
            loop_start = time.perf_counter()
            observed_samples = 0
            for index, j in enumerate(pending, 1):
                current = j['call_id']
                begin = time.perf_counter()
                # Only waveform metadata enters the model pipeline, never labels.
                wave_job = {k:j[k] for k in ('call_id','wav_relative_path','intervals','n_segments')}
                audio = base.load_wave(wave_job, Path(config['data_root']))
                read_seconds = time.perf_counter()-begin
                p, chunks, padded = base.predict(model, 'ecapa', audio, 'cuda:0')
                torch.cuda.synchronize()
                require(versions == [v._version for v in model.parameters()], 'Parameters changed')
                row = {'call_id':current, 'gender':j['gender'], 'identity_sha256':ih,
                       'prediction':['F','M'][int(p.argmax())], 'probabilities_F_M':p.tolist(),
                       'n_segments':j['n_segments'], 'caller_samples_16k':len(audio),
                       'chunks':chunks, 'padded_chunks':padded, 'read_resample_seconds':read_seconds,
                       'elapsed_seconds':time.perf_counter()-begin}
                validate([row], [j], ih)
                append(output / 'predictions.jsonl', row)
                done[current] = row
                observed_samples += len(audio)
                if index == 1 or index % 10 == 0 or index == len(pending):
                    elapsed = time.perf_counter()-loop_start
                    remaining = pending[index:]
                    progress = {'completed':len(done), 'total':5597, 'utc':time.time(),
                                'session_calls':index, 'session_wall_seconds_including_drive_flush':elapsed,
                                'remaining_minutes_by_call':elapsed/index*len(remaining)/60,
                                'remaining_minutes_by_audio':elapsed/observed_samples*sum(samples(x) for x in remaining)/60}
                    save(output / 'progress.json', progress)
                    print('PROGRESS:', progress, flush=True)
            append(output / 'events.jsonl', {'event':'inference_complete', 'utc':time.time(), 'identity_sha256':ih})
            export(output, jobs, identity)
        except BaseException as exc:
            append(output / 'failure_attempts.jsonl', {'call_id':current, 'error':repr(exc),
                   'utc':time.time(), 'identity_sha256':ih, 'kind':'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'})
            append(output / 'events.jsonl', {'event':'stopped', 'utc':time.time(), 'completed':len(done)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['preflight','run'], required=True)
    args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    execute(read(args.config), args.stage)


if __name__ == '__main__':
    main()
