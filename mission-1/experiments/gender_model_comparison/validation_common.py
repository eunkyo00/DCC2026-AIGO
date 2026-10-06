"""CPU-only integrity checks shared by the ECAPA runner and local analysis."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def bundle():
    manifest = read(ROOT / 'validation_bundle_manifest.json')
    for name, expected in manifest['files'].items():
        require(sha(ROOT / name) == expected, f'Bundle hash mismatch: {name}')
    jobs = read(ROOT / 'validation_data/jobs.json')
    require(len(jobs) == len({j['call_id'] for j in jobs}) == 5597, 'Validation IDs')
    require(all(j['partition'] == 'internal_validation' for j in jobs), 'Validation partition')
    require({g: sum(j['gender'] == g for j in jobs) for g in ('M', 'F')} ==
            {'M': 2620, 'F': 2977}, 'Validation class counts')
    return manifest, jobs


def samples(job):
    return 2 * sum(round(b * 8000) - round(a * 8000) for a, b in job['intervals'])


def jsonl(path, repair=False):
    """Only an incomplete, non-newline-terminated final record is recoverable."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows, good = [], bytearray()
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            require(repair and i == len(lines)-1 and not line.endswith(b'\n'),
                    f'Corrupt JSONL: {path}, line {i+1}')
            # Preserve the interrupted bytes before repairing the append point.
            backup = path.with_name(path.name + '.interrupted-' + hashlib.sha256(data).hexdigest()[:12])
            backup.write_bytes(data)
            path.write_bytes(good)
            return rows
        rows.append(row)
        good += line if line.endswith(b'\n') else line + b'\n'
    if repair and data and not data.endswith(b'\n'):
        path.write_bytes(good)
    return rows


def validate(rows, jobs, identity_hash, complete=False):
    by_id = {j['call_id']: j for j in jobs}
    ids = [r['call_id'] for r in rows]
    require(len(ids) == len(set(ids)) and set(ids) <= set(by_id), 'Duplicate/unexpected IDs')
    if complete:
        require(set(ids) == set(by_id), 'Incomplete Validation coverage')
    for r in rows:
        j = by_id[r['call_id']]
        p = r['probabilities_F_M']
        require(isinstance(p, list) and len(p) == 2 and
                all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in p)
                and abs(sum(p)-1) < 1e-5, f'Invalid probabilities: {r["call_id"]}')
        require(r['prediction'] == ('M' if p[1] > p[0] else 'F'), 'Argmax mismatch')
        require(r['identity_sha256'] == identity_hash, 'Row identity mismatch')
        require(r['gender'] == j['gender'] and r['n_segments'] == j['n_segments'], 'Label/segments mismatch')
        n = samples(j)
        require(r['caller_samples_16k'] == n and r['chunks'] == math.ceil(n / 240000)
                and r['padded_chunks'] == int(n < 48000), 'Audio/window coverage mismatch')
        for key in ('elapsed_seconds', 'read_resample_seconds'):
            require(math.isfinite(r[key]) and r[key] >= 0, 'Invalid timing')
        require(r['read_resample_seconds'] <= r['elapsed_seconds'], 'Timing order')
    return {r['call_id']: r for r in rows}


def metrics(rows):
    matrix = [[sum(r['gender'] == g and r['prediction'] == p for r in rows)
               for p in ('M', 'F')] for g in ('M', 'F')]
    correct = matrix[0][0] + matrix[1][1]
    return {'calls': len(rows), 'correct': correct, 'wrong': len(rows)-correct,
            'accuracy': correct / len(rows), 'confusion_matrix_M_F': matrix,
            'gender_accuracy': {g: matrix[i][i] / sum(matrix[i]) for i, g in enumerate(('M', 'F'))}}
