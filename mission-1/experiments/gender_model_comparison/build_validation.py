"""Build an isolated metadata/code bundle; never edit the Train bundle."""
import csv
import json
import math
import zipfile
from pathlib import Path
from validation_common import ROOT, sha, save, require, samples, metrics

MISSION = ROOT.parents[1]
SOURCES = {
    'validation/split_assignments.csv': '04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06',
    'validation/manifests/calls.csv': 'dc57d0f7bea6e06017ac3e27444a0ff5bb299e401ea787400e84cbdf7d072e23',
    'eda/eda_outputs/segments.csv': '8bf8c83c0190232e4500b4067bc28d2611b09dd227a7ce4dc9c3741e6dcce584',
}
BASELINES = {'mfcc': 'baseline/mfcc_svm/results/val_predictions.csv',
             'wav2vec2': 'ssl/wav2vec2_frozen/results/full_l4/val_predictions.csv'}


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def unique(data):
    out = {r['call_id']: r for r in data}
    require(len(out) == len(data), 'Duplicate source IDs')
    return out


def main():
    for path, expected in SOURCES.items():
        require(sha(MISSION / path) == expected, f'Source hash: {path}')
    original = json.loads((ROOT / 'bundle_manifest.json').read_text())
    for name, expected in original['files'].items():
        require(sha(ROOT / name) == expected, f'Original bundle changed: {name}')
    split = unique(rows(MISSION / 'validation/split_assignments.csv'))
    calls = unique(rows(MISSION / 'validation/manifests/calls.csv'))
    require(len(split) == 27985 and set(split) == set(calls), 'Source call coverage')
    selected = {c: r for c, r in split.items() if r['partition'] == 'internal_validation'}
    require(len(selected) == 5597, 'Validation size')
    intervals = {c: [] for c in selected}
    with (MISSION / 'eda/eda_outputs/segments.csv').open() as f:
        for r in csv.DictReader(f):
            c = r['file_ref']
            if c in selected and r['split'] == 'Training' and r['speaker'] == '1':
                require(r['matched'] == 'True' and r['exceeds_wav'] == 'False' and
                        r['gender'] == selected[c]['gender'], f'Segment mismatch: {c}')
                a, b = float(r['start_s']), float(r['end_s'])
                require(math.isfinite(a) and math.isfinite(b) and 0 <= a < b and
                        round(b*8000) > round(a*8000), 'Invalid interval')
                intervals[c].append([a, b])
    jobs = []
    for c, r in sorted(selected.items()):
        m = calls[c]
        require(m['gender'] == r['gender'] and m['partition'] == r['partition'], 'Manifest identity')
        require(len(intervals[c]) == int(m['n_segments']) > 0, 'All caller segments required')
        require(abs(sum(b-a for a,b in intervals[c])-float(m['total_duration'])) < 1e-6, 'Duration mismatch')
        path = Path(m['wav_relative_path'])
        require(not path.is_absolute() and '..' not in path.parts and path.parts[0] == 'Training', 'Unsafe WAV path')
        jobs.append({**r, 'wav_relative_path': path.as_posix(), 'n_segments': len(intervals[c]),
                     'intervals': sorted(intervals[c])})
    counts = {g: sum(j['gender'] == g for j in jobs) for g in ('M', 'F')}
    require(counts == {'M': 2620, 'F': 2977}, 'Class counts')
    baseline_info = {}
    for name, path in BASELINES.items():
        data = unique(rows(MISSION / path))
        require(set(data) == set(selected), f'{name} fixed Validation coverage')
        require(all(r['gender'] == selected[c]['gender'] and r['prediction'] in ('M','F') for c,r in data.items()), 'Baseline labels')
        score = metrics(list(data.values()))
        require(score['correct'] == {'mfcc': 5321, 'wav2vec2': 5443}[name], 'Baseline score mismatch')
        baseline_info[name] = {'path': path, 'sha256': sha(MISSION / path), **score}
    (ROOT / 'validation_data').mkdir(exist_ok=True)
    save(ROOT / 'validation_data/jobs.json', jobs)
    names = ['compare_models.py', 'validation_common.py', 'ecapa_validation.py',
             'analyze_ecapa_validation.py', 'model_sources.json', 'requirements.txt',
             'vendor/ecapa_model.py', 'vendor/ecapa_LICENSE', 'validation_data/jobs.json']
    manifest = {'scope': 'Fixed Internal Validation; ECAPA only; no training or tuning',
                'sources': SOURCES, 'original_bundle_sha256': sha(ROOT / 'bundle_manifest.json'),
                'calls': len(jobs), 'gender_counts': counts,
                'caller_segments': sum(j['n_segments'] for j in jobs),
                'caller_audio_seconds': sum(samples(j) for j in jobs)/16000,
                'windows': sum(math.ceil(samples(j)/240000) for j in jobs),
                'short_padded_calls': sum(samples(j)<48000 for j in jobs),
                'baselines': baseline_info, 'files': {n: sha(ROOT/n) for n in names}}
    save(ROOT / 'validation_bundle_manifest.json', manifest)
    archive = ROOT / 'artifacts/ecapa_validation_bundle.zip'
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in names + ['validation_bundle_manifest.json']:
            z.write(ROOT / name, name)
    from make_validation_notebook import make_notebook
    make_notebook(sha(archive))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','baselines')}, indent=2))
    print('ZIP:', archive, 'SHA256:', sha(archive))


if __name__ == '__main__':
    main()
