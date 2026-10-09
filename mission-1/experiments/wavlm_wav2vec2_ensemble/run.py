"""Cache-only, Train-OOF-selected WavLM/Wav2Vec2 probability ensemble."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time
import warnings
import zipfile

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SPLIT_SHA = '04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06'
# Exact previously verified Wav2Vec2 export (not a user-supplied arbitrary NPZ).
EXPORT_SHA = '64cd8d608c5e7e18a0db86193f7b4b26fa08d689aa22e4dab6807b7360bd71c4'
WAVLM_ID = '49725c8a0b4daed276a2ba30dcde37ffab5e220fdf717b1e2be5e0e700ff9d15'
WEIGHTS = (0.75, 0.5)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


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
    atomic(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())


def npz(path, **arrays):
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    atomic(path, b.getvalue())


def probabilities(p, n):
    require(p.shape == (n, 2) and np.isfinite(p).all(), 'Invalid probability shape/finite')
    require(((p >= 0) & (p <= 1)).all() and np.allclose(p.sum(1), 1), 'Invalid probabilities')
    return p


def score(y, p):
    probabilities(p, len(y))
    pred = np.array(['F', 'M'])[p.argmax(1)]
    return dict(n=len(y), correct=int((pred == y).sum()), wrong=int((pred != y).sum()),
                accuracy=float((pred == y).mean()),
                gender_accuracy={g: float((pred[y == g] == g).mean()) for g in ('F', 'M')},
                confusion_F_M=[[int(((y == g) & (pred == h)).sum()) for h in ('F', 'M')] for g in ('F', 'M')])


def fit(x, y, c):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=42))
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y)
    require(list(model.classes_) == ['F', 'M'], 'Class order mismatch')
    return model


def choose(y, p1, p2, folds):
    """A predeclared conservative heuristic, not a statistical significance test."""
    baseline = score(y, p1)
    candidates = []
    for w in WEIGHTS:
        p = w * p1 + (1 - w) * p2
        s = score(y, p)
        deltas = [score(y[folds == k], p[folds == k])['accuracy'] -
                  score(y[folds == k], p1[folds == k])['accuracy'] for k in range(3)]
        eligible = min(deltas) >= 0 and s['accuracy'] - baseline['accuracy'] >= 0.001
        candidates.append(dict(wavlm_weight=w, metrics=s, fold_deltas=deltas, eligible=eligible))
    valid = [r for r in candidates if r['eligible']]
    best = max(valid, key=lambda r: (r['metrics']['correct'], r['wavlm_weight'])) if valid else None
    return dict(wavlm_weight=best['wavlm_weight'] if best else 1.0, baseline=baseline,
                candidates=candidates, rule='3 folds nonnegative gain AND total >=0.1 percentage point; ties prefer WavLM; otherwise baseline',
                scope='Train OOF selection, not an unbiased test; call split is not guaranteed speaker-disjoint')


def metadata(split):
    require(sha(split) == SPLIT_SHA, 'Wrong original split CSV')
    with Path(split).open(newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r['call_id'])
    require(len(rows) == len({r['call_id'] for r in rows}) == 27985, 'Split coverage')
    require(sum(r['partition'] == 'train' for r in rows) == 22388, 'Train coverage')
    require(sum(r['partition'] == 'internal_validation' for r in rows) == 5597, 'Validation coverage')
    require(all(r['gender'] in ('F', 'M') for r in rows), 'Invalid label')
    return rows


def sources(w1, w2):
    ident = read(w1 / 'identity.json')
    require(digest(ident) == WAVLM_ID, 'Expected the verified L4 WavLM run; select its output directory')
    require(read(w1 / 'completion.json')['status'] == 'complete', 'WavLM run incomplete')
    if w2.is_file():
        require(sha(w2) == EXPORT_SHA, 'Unknown Wav2Vec2 compact export; use original verified export or call_cache directory')
        return ident, dict(kind='export', sha256=EXPORT_SHA)
    summary = read(w2 / 'extraction_summary.json')
    wi = read(w2 / 'call_cache/identity.json')
    require(summary['status'] == 'complete' and summary['identity'] == wi, 'Wav2Vec2 extraction incomplete or identity changed')
    require(summary['completed_calls'] == 27985 and summary['failed_calls'] == 0, 'Wav2Vec2 incomplete coverage')
    require(wi['split_sha256'] == SPLIT_SHA and wi['checkpoint'] == 'facebook/wav2vec2-base', 'Wrong Wav2Vec2 split/model')
    require(wi['revision'] == '0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8', 'Wrong Wav2Vec2 revision')
    return ident, dict(kind='cache', identity=wi, summary_sha256=sha(w2 / 'extraction_summary.json'))


def load_one(row, kind, root, identity):
    cid = row['call_id']
    sub = 'pooled' if kind == 'wavlm' else 'call_cache'
    p = root / sub / cid[:2] / (cid + '.npz')
    require(p.exists(), f'Missing {kind} cache: {p}. Restore the old cache; this lightweight runner does not extract audio.')
    if kind == 'wavlm':
        require(read(p.with_suffix('.hash.json'))['sha256'] == sha(p), 'WavLM cache checksum mismatch')
    with np.load(p, allow_pickle=False) as z:
        require(str(z['call_id'].item()) == cid, 'Cache call ID mismatch')
        if kind == 'wavlm':
            require(str(z['identity'].item()) == digest(identity), 'WavLM cache identity mismatch')
            a = z['features']
            require(a.shape == (4, 2, 1024) and int(z['samples']) > 0, 'WavLM feature shape/coverage')
            a = a[-1, 0].copy()
        else:
            require(str(z['identity_sha256'].item()) == digest(identity), 'Wav2Vec2 cache identity mismatch')
            q = json.loads(str(z['quality'].item()))
            require(q['failed_segments'] == 0 and q['successful_segments'] > 0, 'Wav2Vec2 segment failure')
            a = z['embedding'].copy()
    require(a.shape == ((1024,) if kind == 'wavlm' else (768,)) and a.dtype == np.float32 and np.isfinite(a).all(), 'Invalid feature')
    return a


def matrix(rows, kind, root, identity, out, partition):
    """Drive reads in four threads; committed 512-call chunks allow interruption recovery."""
    blocks = []
    started = time.monotonic()
    for start in range(0, len(rows), 512):
        batch = rows[start:start + 512]
        ids = np.array([r['call_id'] for r in batch])
        p = out / 'packed' / f'{partition}_{kind}_{start:05d}.npz'
        if p.exists():
            require(read(p.with_suffix('.json'))['sha256'] == sha(p), 'Packed cache checksum mismatch')
            with np.load(p, allow_pickle=False) as z:
                require(np.array_equal(z['ids'], ids) and str(z['source'].item()) == digest(identity), 'Packed cache mismatch')
                x = z['x'].copy()
        else:
            with ThreadPoolExecutor(max_workers=4) as pool:
                x = np.stack(list(pool.map(lambda row: load_one(row, kind, root, identity), batch)))
            npz(p, ids=ids, x=x, source=digest(identity))
            save(p.with_suffix('.json'), dict(sha256=sha(p)))
        require(x.shape == (len(batch), 1024 if kind == 'wavlm' else 768) and np.isfinite(x).all(), 'Packed features invalid')
        blocks.append(x)
        done = start + len(batch)
        elapsed = time.monotonic() - started
        print(f'{partition} {kind}: {done}/{len(rows)}; read ETA {elapsed / done * (len(rows)-done) / 60:.1f} min', flush=True)
    return np.concatenate(blocks)


def export_matrix(rows, path):
    with np.load(path, allow_pickle=False) as z:
        ids = z['call_ids'].astype(str)
        require(len(ids) == len(set(ids)) == 27985, 'Export IDs')
        positions = {cid: i for i, cid in enumerate(ids)}
        ix = [positions[r['call_id']] for r in rows]
        require(np.array_equal(z['gender'][ix], [r['gender'] for r in rows]), 'Export labels')
        require(np.array_equal(z['partition'][ix], [r['partition'] for r in rows]), 'Export partitions')
        x = z['embeddings'][ix].copy()
    require(x.shape == (len(rows), 768) and x.dtype == np.float32 and np.isfinite(x).all(), 'Export features')
    return x


def oof(x1, x2, y, ids, out):
    p1, p2 = np.zeros((len(y), 2)), np.zeros((len(y), 2))
    folds = np.full(len(y), -1)
    for k, (tr, dev) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(x1, y)):
        require(not set(ids[tr]) & set(ids[dev]), 'Fold leakage')
        target = out / f'oof_fold_{k}.npz'
        if target.exists():
            require(read(target.with_suffix('.json'))['sha256'] == sha(target), 'OOF checksum mismatch')
            with np.load(target, allow_pickle=False) as z:
                require(np.array_equal(z['ids'], ids[dev]), 'OOF IDs changed')
                a, b = z['wavlm'].copy(), z['wav2vec2'].copy()
        else:
            start = time.monotonic()
            a = fit(x1[tr], y[tr], 0.1).predict_proba(x1[dev])
            b = fit(x2[tr], y[tr], 1.0).predict_proba(x2[dev])
            npz(target, ids=ids[dev], wavlm=a, wav2vec2=b)
            save(target.with_suffix('.json'), dict(sha256=sha(target)))
            print(f'OOF fold {k+1}/3: {(time.monotonic()-start):.1f}s', flush=True)
        p1[dev], p2[dev] = probabilities(a, len(dev)), probabilities(b, len(dev))
        folds[dev] = k
    require((folds >= 0).all(), 'Incomplete OOF')
    return p1, p2, folds


def result_zip(out):
    with zipfile.ZipFile(out / 'ensemble_results.zip.tmp', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('identity.json', 'selection.lock.json', 'models.json', 'validation_started.json', 'completion.json', 'REPORT.md', 'validation_predictions.csv'):
            z.write(out / name, name)
    os.replace(out / 'ensemble_results.zip.tmp', out / 'ensemble_results.zip')


def run(args):
    started = time.monotonic()
    w1, w2, out = Path(args.wavlm), Path(args.wav2vec2), Path(args.output)
    rows = metadata(args.split)
    i1, s2 = sources(w1, w2)
    ident = dict(code_sha256=sha(__file__), split_sha256=SPLIT_SHA, wavlm_identity=digest(i1), wav2vec2=s2,
                 versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                 candidates=list(WEIGHTS), folds=3, seed=42, wavlm_C=0.1, wav2vec2_C=1.0)
    out.mkdir(parents=True, exist_ok=True)
    require(out.resolve() != w1.resolve() and out.resolve() != w2.resolve(), 'Use a separate ensemble output')
    if (out / 'identity.json').exists():
        require(read(out / 'identity.json') == ident, 'Resume code/source/environment changed; use a new output directory')
    else:
        require(not any(out.iterdir()), 'Output must be empty for a new run')
        save(out / 'identity.json', ident)
    if (out / 'completion.json').exists():
        result_zip(out)
        print(json.dumps(read(out / 'completion.json'), indent=2)); return
    train = [r for r in rows if r['partition'] == 'train']
    y = np.array([r['gender'] for r in train]); ids = np.array([r['call_id'] for r in train])
    x1 = matrix(train, 'wavlm', w1, i1, out, 'train')
    x2 = export_matrix(train, w2) if s2['kind'] == 'export' else matrix(train, 'wav2vec2', w2, s2['identity'], out, 'train')
    selection_path = out / 'selection.lock.json'
    if not selection_path.exists():
        p1, p2, folds = oof(x1, x2, y, ids, out)
        selection = choose(y, p1, p2, folds)
        selection['identity'] = digest(ident)
        save(selection_path, selection)
    selection = read(selection_path)
    require(selection['identity'] == digest(ident), 'Selection identity mismatch')
    w = selection['wavlm_weight']
    print('Locked WavLM weight:', w, flush=True)
    model_path = out / 'models.joblib'
    if model_path.exists():
        require(read(out / 'models.json')['sha256'] == sha(model_path), 'Final model checksum changed')
        models = joblib.load(model_path)
    else:
        require(not (out / 'validation_started.json').exists(), 'Cannot refit after validation starts')
        models = dict(wavlm=fit(x1, y, 0.1), wav2vec2=fit(x2, y, 1.0) if w < 1 else None,
                      wavlm_weight=w, classes=['F', 'M'], selection_sha256=sha(selection_path))
        b = io.BytesIO(); joblib.dump(models, b); atomic(model_path, b.getvalue())
        save(out / 'models.json', dict(sha256=sha(model_path)))
    require(models['selection_sha256'] == sha(selection_path), 'Selection changed after refit')
    del x1, x2
    marker = dict(selection_sha256=sha(selection_path), model_sha256=sha(model_path))
    if (out / 'validation_started.json').exists():
        require(read(out / 'validation_started.json') == marker, 'Validation identity changed')
    else:
        save(out / 'validation_started.json', marker)
    # Validation features/labels never enter fit, folds, or weight selection.
    val = [r for r in rows if r['partition'] == 'internal_validation']
    yv = np.array([r['gender'] for r in val])
    v1 = matrix(val, 'wavlm', w1, i1, out, 'validation')
    a = probabilities(models['wavlm'].predict_proba(v1), len(val))
    if w < 1:
        v2 = export_matrix(val, w2) if s2['kind'] == 'export' else matrix(val, 'wav2vec2', w2, s2['identity'], out, 'validation')
        b = probabilities(models['wav2vec2'].predict_proba(v2), len(val))
    else:
        b = a
    p = w * a + (1-w) * b
    npz(out / 'validation_predictions.npz', ids=np.array([r['call_id'] for r in val]), gender=yv, probabilities_F_M=p, wavlm_probabilities_F_M=a)
    sio = io.StringIO(); writer = csv.writer(sio, lineterminator='\n')
    writer.writerow(['call_id', 'gender', 'prediction', 'probability_F', 'probability_M'])
    for row, prob in zip(val, p):
        writer.writerow([row['call_id'], row['gender'], 'FM'[int(prob.argmax())], *prob])
    atomic(out / 'validation_predictions.csv', sio.getvalue().encode())
    result = dict(status='complete', identity=digest(ident), wavlm_weight=w,
                  selected=score(yv, p), refit_wavlm_baseline=score(yv, a),
                  session_seconds=time.monotonic()-started,
                  scope='Previously observed Internal Validation; not an independent test. Official Validation unused. No speaker-disjoint guarantee.',
                  historical_wavlm_correct=5490, baseline_reproduced=score(yv, a)['correct'] == 5490)
    atomic(out / 'REPORT.md', (f"# WavLM + Wav2Vec2\n\nWavLM weight: {w}\n\nAccuracy: {result['selected']['accuracy']:.6%}\n\nCorrect: {result['selected']['correct']} / {len(val)}\n\n" + result['scope'] + '\n').encode())
    save(out / 'completion.json', result)
    result_zip(out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('wavlm', 'wav2vec2', 'split', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    # Runtime-local lock releases on Stop/crash; use only one Colab VM per output.
    import fcntl
    lock = open('/tmp/aigo-ensemble-' + hashlib.sha256(str(Path(args.output).resolve()).encode()).hexdigest() + '.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run(args)
