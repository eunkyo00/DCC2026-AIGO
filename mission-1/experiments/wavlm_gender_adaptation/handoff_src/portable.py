"""Explicit teammate paths and CPU-only checks; no Colab or previous-run identity."""
import argparse
import os
import wave
from pathlib import Path
from common import ROOT, bundle, require, read, save, rows, sha

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
        require(old.get('bundle_sha256')==sha(ROOT/'bundle_manifest.json') and old.get('config')==config,
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
                  bundle_sha256=sha(ROOT/'bundle_manifest.json'),calls=len(inventory),total_bytes=total,
                  failures=failures,verification='file names, sizes and 8kHz/mono/PCM16/frame-count headers; not full waveform content hashes')
    save(paths['output_dir']/'dataset_check.json',result)
    require(not failures, f'{len(failures)} WAV failures. See dataset_check.json; no GPU work started')
    print(result['status'], result['calls'], 'WAVs, bytes:', total,flush=True)
    return result


def require_dataset_check(config):
    result = read(Path(config['output_dir'])/'dataset_check.json')
    require(result['status']=='pass' and result['calls']==27985 and not result['failures'] and
            result['data_root']==config['data_root'] and result['bundle_sha256']==sha(ROOT/'bundle_manifest.json'),
            'Run portable.py --check-data for this data path and code bundle first')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--check-data',action='store_true')
    a=p.parse_args();config=read(a.config);validate_paths(config)
    if a.check_data:check_dataset(config)
    else:print('Explicit paths validated')
