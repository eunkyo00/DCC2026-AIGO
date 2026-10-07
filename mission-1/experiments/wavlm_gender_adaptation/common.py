"""Stdlib integrity and durable storage; shared by CPU verifier and GPU runner."""
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
    m = read(ROOT/'bundle_manifest.json')
    for name, expected in m['files'].items():
        require(sha(ROOT/name) == expected, f'Bundle changed: {name}')
    jobs = read(ROOT/'data/jobs.json')
    require(len(jobs) == len({j['call_id'] for j in jobs}) == 27985, 'Jobs coverage')
    require(sum(j['partition'] == 'train' for j in jobs) == 22388, 'Train coverage')
    return m, jobs
