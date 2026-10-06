"""Resume a common 200-call screen without modifying the original signed bundle."""
import argparse
import gc
import contextlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import compare_models as base


def select_jobs(jobs):
    # Original pilot order is sorted call_id; selection never consults predictions.
    base.require(jobs == sorted(jobs, key=lambda j: j['call_id']), 'Unexpected pilot order')
    selected = jobs[:200]
    base.require(len(selected) == 200, 'Need 200 calls')
    base.require({g: sum(j['gender'] == g for j in selected) for g in ('M', 'F')} ==
                 {'M': 101, 'F': 99}, 'Unexpected screen composition')
    return selected


def read_source(path):
    """Read interrupted source logs without editing them."""
    if not path.exists():
        return {}
    lines = path.read_bytes().splitlines()
    rows = {}
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            base.require(i == len(lines) - 1, f'Corrupt interior line: {path}')
            break
        base.require(row['call_id'] not in rows, f'Duplicate source ID: {path}')
        rows[row['call_id']] = row
    return rows


def validate(rows, jobs):
    by_id = {j['call_id']: j for j in jobs}
    base.require(set(rows) <= set(by_id), 'Cache contains unexpected IDs')
    for call, row in rows.items():
        j = by_id[call]
        p = np.asarray(row['probabilities_F_M'])
        base.require(p.shape == (2,) and np.isfinite(p).all() and
                     (p >= 0).all() and (p <= 1).all() and abs(p.sum() - 1) < 1e-5,
                     f'Invalid probabilities: {call}')
        base.require(row['prediction'] == ['F', 'M'][int(np.argmax(p))], 'Prediction mismatch')
        base.require(row['gender'] == j['gender'] and row['n_segments'] == j['n_segments'],
                     f'Label/segment mismatch: {call}')
        expected = sum(round(b * 8000) - round(a * 8000) for a, b in j['intervals']) * 2
        base.require(row['caller_samples_16k'] == expected, f'Audio coverage mismatch: {call}')
        base.require(np.isfinite(row['elapsed_seconds']) and row['elapsed_seconds'] >= 0,
                     'Invalid timing')


def migrate(name, jobs, source, output):
    source_rows = read_source(source / f'{name}_pilot.jsonl')
    ids = {j['call_id'] for j in jobs}
    selected = {c: r for c, r in source_rows.items() if c in ids}
    validate(selected, jobs)
    cache = output / f'{name}_pilot.jsonl'
    existing = base.recover(cache)
    validate(existing, jobs)
    for c in set(existing) & set(selected):
        base.require(existing[c] == selected[c], 'Source and screen results conflict')
    with cache.open('a') as f:
        for j in jobs:
            c = j['call_id']
            if c in selected and c not in existing:
                f.write(json.dumps(selected[c], allow_nan=False) + '\n')
                existing[c] = selected[c]
        f.flush()
        os.fsync(f.fileno())
    return existing, len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    bundle = base.verify_bundle()
    config = json.loads(args.config.read_text())
    source, root = Path(config['output_dir']), Path(config['data_root'])
    identity = json.loads((source / 'identity.json').read_text())
    base.require(identity['bundle_sha256'] == bundle, 'Source bundle mismatch')
    base.require(identity['models'] == json.loads(json.dumps(base.MODELS)), 'Model revision mismatch')
    base.require(identity['data_root'] == str(root), 'Data root changed')
    import torch
    base.require(torch.cuda.is_available(), 'CUDA required')
    base.require(identity['gpu'] == torch.cuda.get_device_name(0), 'GPU changed')
    base.require(identity['python'] == platform.python_version(), 'Python changed')
    for key, version in identity['versions'].items():
        base.require(importlib.metadata.version(key) == version, f'Package changed: {key}')
    base.require(identity['dtype'] == 'float32' and identity['batch_size'] == 1 and
                 identity['autocast'] is False and identity['tf32'] is False and
                 identity['pipeline'] == 'caller_concat_balanced_3_to_15s_duration_weighted_probability_v1',
                 'Inference definition changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    jobs = select_jobs(base.load_jobs())
    output = source.parent / (source.name + '_screen200')
    output.mkdir(exist_ok=True)
    screen_identity = {'source_identity': identity, 'source_output': str(source),
                       'screen_code_sha256': base.sha(Path(__file__)),
                       'selection': 'first 200 original sorted call_ids; independent of predictions',
                       'call_ids': [j['call_id'] for j in jobs]}
    identity_path = output / 'screen_identity.json'
    if identity_path.exists():
        base.require(json.loads(identity_path.read_text()) == screen_identity, 'Screen identity changed')
    else:
        base.require(not list(output.glob('*.jsonl')), 'Screen cache without identity')
        base.save(identity_path, screen_identity)
    base.require(json.loads((source / 'preflight.json').read_text())['status'] == 'pass', 'Preflight missing')
    for name in base.MODELS:
        for stage in ('smoke', 'benchmark'):
            report = json.loads((source / f'{name}_{stage}_summary.json').read_text())
            base.require(report['status'] == 'complete' and report['failed_calls'] == 0,
                         f'{name} {stage} incomplete')
    print('SCREEN: same 200 calls, Male=101 Female=99; Validation untouched', flush=True)
    print('Output:', output, flush=True)
    started = time.perf_counter()
    provenance = {}
    for name in base.MODELS:
        rows, imported = migrate(name, jobs, source, output)
        missing = len(jobs) - len(rows)
        benchmark = json.loads((source / f'{name}_benchmark_summary.json').read_text())
        timing = list(rows.values())
        seconds_per_call = (sum(r['elapsed_seconds'] for r in timing) / len(timing)
                            if timing else benchmark['seconds_per_call'])
        print(f'{name}: saved={len(rows)}/200, remaining={missing}; estimated remaining inference '
              f'{missing * seconds_per_call / 60:.1f} min (Drive/model load may add time)', flush=True)
        provenance[name] = {'source_rows_reused': imported, 'new_calls_this_run': missing}
        if missing:
            print(f'Loading {name}...', flush=True)
            model, info = base.load_model(name, 'cuda:0')
            old_info = json.loads((source / f'{name}_load.json').read_text())
            base.require(info['weight_sha256'] == old_info['weight_sha256'], 'Model weights changed')
            base.evaluate(model, name, jobs, 'pilot', output, root)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        rows = base.recover(output / f'{name}_pilot.jsonl')
        validate(rows, jobs)
        base.require(len(rows) == 200, 'Incomplete screen')
        ordered = [rows[j['call_id']] for j in jobs]
        report = {'status': 'complete', 'scope': '200-call early screen', **base.metrics(ordered),
                  'cache_sha256': base.sha(output / f'{name}_pilot.jsonl'),
                  'prediction_shape': [200, 2], 'finite': True, 'failed_calls': 0,
                  'validation_evaluated': False,
                  'seconds_per_call': sum(r['elapsed_seconds'] for r in ordered) / 200}
        base.save(output / f'{name}_pilot_summary.json', report)
    # Original comparison checks exact coverage and cache hashes, then computes OOF overlap.
    with contextlib.redirect_stdout(io.StringIO()):
        base.compare(jobs, output)
    report = json.loads((output / 'comparison.json').read_text())
    report.update(scope='Train 200-call early screen; not a 99% claim or final Validation',
                  selection=screen_identity['selection'], gender_counts={'M': 101, 'F': 99},
                  provenance=provenance, elapsed_this_run_minutes=(time.perf_counter() - started) / 60)
    base.save(output / 'comparison.json', report)
    archive = output / 'screen200_results.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.iterdir()):
            if p.suffix in ('.json', '.jsonl'):
                z.write(p, p.name)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print('SCREEN 200 COMPLETE:', archive, flush=True)


if __name__ == '__main__':
    main()
