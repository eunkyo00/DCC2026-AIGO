"""Dependency-free integrity, fixed partitions, and result checks."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = 'facebook/wav2vec2-base'
REVISION = '0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8'
HASHES = {
 'validation/split_assignments.csv': '04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06',
 'validation/manifests/calls.csv': 'dc57d0f7bea6e06017ac3e27444a0ff5bb299e401ea787400e84cbdf7d072e23',
 'eda/eda_outputs/segments.csv': '8bf8c83c0190232e4500b4067bc28d2611b09dd227a7ce4dc9c3741e6dcce584'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def save(path, obj):
    atomic(path, (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode())


def rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def inner_assign(jobs):
    # Matches the concurrently prepared WavLM common.py, without importing/mutating it.
    result = {}
    for g in ('F', 'M'):
        ids = [j['call_id'] for j in jobs if j['partition'] == 'train' and j['gender'] == g]
        ids.sort(key=lambda x: (hashlib.sha256(f'wavlm-inner-v1|42|{x}'.encode()).hexdigest(), x))
        n = (len(ids)+2)//5
        result.update({x: 'dev' if i < n else 'inner_train' for i, x in enumerate(ids)})
    return result


def metric(rr):
    require(bool(rr), 'Empty evaluation')
    cm = [[sum(r['gender']==g and r['prediction']==p for r in rr) for p in ('M','F')] for g in ('M','F')]
    correct = cm[0][0]+cm[1][1]
    return dict(n=len(rr), correct=correct, wrong=len(rr)-correct, accuracy=correct/len(rr),
                confusion_matrix_M_F=cm, gender_accuracy={g: cm[i][i]/sum(cm[i]) for i,g in enumerate(('M','F'))})


def validate(rr, jobs, identity, complete=True):
    jj = {j['call_id']: j for j in jobs}
    ids = [r['call_id'] for r in rr]
    require(len(ids)==len(set(ids)) and set(ids)<=set(jj), 'Duplicate/unexpected IDs')
    if complete:
        require(set(ids)==set(jj), 'Incomplete exact coverage')
    for r in rr:
        j = jj[r['call_id']]
        p = r['probabilities_F_M']
        require(len(p)==2 and all(isinstance(x,(float,int)) and math.isfinite(x) and 0<=x<=1 for x in p)
                and abs(sum(p)-1)<1e-5, 'Invalid probabilities')
        require(r['prediction']==('M' if p[1]>p[0] else 'F'), 'Argmax mismatch')
        require(r['gender']==j['gender'] and r['identity']==identity, 'Label/identity mismatch')
        require(r['segments']==len(j['intervals']) and r['source_samples']==sum(round(b*8000)-round(a*8000) for a,b in j['intervals']), 'Incomplete caller coverage')
    return {r['call_id']: r for r in rr}


def bundle():
    m = read(ROOT/'bundle_manifest.json')
    for name, h in m['files'].items():
        require(sha(ROOT/name)==h, f'Bundle hash mismatch: {name}')
    jobs = read(ROOT/'data/jobs.json')
    require(len(jobs)==len({j['call_id'] for j in jobs})==27985, 'Jobs coverage')
    require(read(ROOT/'data/inner_split.json')==inner_assign(jobs), 'Inner partition mismatch')
    return m,jobs
