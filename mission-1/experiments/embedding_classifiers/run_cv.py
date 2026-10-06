"""Train-only comparison of saved Wav2Vec2/MFCC features; never evaluate holdout."""
import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
MISSION = BASE.parents[1]
EXPORT_SHA = '64cd8d608c5e7e18a0db86193f7b4b26fa08d689aa22e4dab6807b7360bd71c4'
SPLIT_SHA = '04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06'
FEATURES = [f'mfcc_{i:02d}_{temporal}_segment_{stat}'
            for i in range(1, 14) for temporal in ('temporal_mean', 'temporal_std')
            for stat in ('mean', 'std')] + ['n_segments', 'total_duration_s']
CANDIDATES = [(stage, c) for stage in ('lr', 'fusion', 'svm')
              for c in ((1.0, 0.1, 10.0) if stage != 'svm' else (1.0, 10.0))]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def write_csv(path, names, values):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(names)
        writer.writerows(values)
    temporary.replace(path)


def index_unique(records, name):
    indexed = {r['call_id']: r for r in records}
    require(len(indexed) == len(records), f'{name}: duplicate call_id')
    return indexed


def load_inputs(embedding_path, mfcc_path, split_path):
    require(sha(embedding_path) == EXPORT_SHA, 'Embedding hash mismatch')
    require(sha(split_path) == SPLIT_SHA, 'Fixed split hash mismatch')
    with np.load(embedding_path, allow_pickle=False) as z:
        x, ids, y, partition, segments = [z[k] for k in
            ('embeddings', 'call_ids', 'gender', 'partition', 'n_segments')]
    require(x.shape == (27985, 768) and x.dtype == np.float32, 'Embedding shape/dtype')
    require(np.isfinite(x).all(), 'Embedding NaN/inf')
    require(len(set(ids)) == 27985, 'Duplicate embedding call_id')
    split = index_unique(rows(split_path), 'split')
    mfcc = index_unique(rows(mfcc_path), 'MFCC')
    require(set(ids) == set(split) == set(mfcc), 'Call coverage mismatch')
    for call, gender, part, count in zip(ids, y, partition, segments):
        for record in (split[call], mfcc[call]):
            require((record['gender'], record['partition']) == (gender, part),
                    f'Label/partition mismatch: {call}')
        require(int(mfcc[call]['n_segments']) == count, f'Segment count: {call}')
        require(int(mfcc[call]['failed_segments']) == 0 and
                int(mfcc[call]['successful_segments']) == count and
                int(mfcc[call]['mfcc_success']) == 1, f'MFCC extraction failed: {call}')
    train = partition == 'train'
    require(train.sum() == 22388 and (partition == 'internal_validation').sum() == 5597,
            'Partition counts mismatch')
    require(segments.sum() == 442639, 'Segment total mismatch')
    # Only Train features/labels leave this function. Holdout is never scored.
    ids, y, x = ids[train], y[train], x[train]
    m = np.array([[float(mfcc[call][name]) for name in FEATURES] for call in ids])
    require(m.shape == (22388, 54) and np.isfinite(m).all(), 'MFCC shape/NaN/inf')
    return ids, y, x, m


def make_model(stage, c):
    classifier = (SVC(C=c, kernel='rbf', gamma='scale', class_weight=None,
                      probability=False, cache_size=512) if stage == 'svm' else
                  LogisticRegression(C=c, solver='lbfgs', max_iter=1000,
                                     class_weight=None, random_state=42, tol=1e-4))
    return make_pipeline(StandardScaler(), classifier)


def evaluate_fold(model, x, y, fit_indices, test_indices):
    require(not np.intersect1d(fit_indices, test_indices).size, 'Fold leakage')
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x[fit_indices], y[fit_indices])
    fit_seconds = time.perf_counter() - start
    scaler = model.named_steps['standardscaler']
    require(int(scaler.n_samples_seen_) == len(fit_indices), 'Scaler fit count')
    np.testing.assert_allclose(scaler.mean_, x[fit_indices].mean(axis=0, dtype=np.float64),
                               rtol=1e-10, atol=1e-12)
    start = time.perf_counter()
    pred = model.predict(x[test_indices])
    score = model.decision_function(x[test_indices])
    predict_seconds = time.perf_counter() - start
    require(model.steps[-1][1].classes_.tolist() == ['F', 'M'], 'Class order')
    require(np.isfinite(score).all(), 'Nonfinite decision score')
    truth = y[test_indices]
    report = {
        'fit_calls': len(fit_indices), 'test_calls': len(test_indices),
        'correct': int((pred == truth).sum()), 'accuracy': float((pred == truth).mean()),
        'gender_accuracy': {g: float((pred[truth == g] == g).mean()) for g in ('M', 'F')},
        'confusion_matrix_M_F': confusion_matrix(truth, pred, labels=['M', 'F']).tolist(),
        'fit_seconds': fit_seconds, 'predict_and_score_seconds': predict_seconds,
        'iterations': model.steps[-1][1].n_iter_.tolist(),
        'scaler_fit_calls': int(scaler.n_samples_seen_),
    }
    return pred, score, report


def summarize(output):
    candidates = []
    for stage, c in CANDIDATES:
        name = f'{stage}_C{c:g}'
        reports = []
        for fold in range(3):
            path = output / name / f'fold_{fold}.json'
            if path.exists():
                report = json.loads(path.read_text())
                require(sha(path.parent / f'fold_{fold}_oof.csv') == report['oof_sha256'],
                        f'Corrupt OOF artifact: {path}')
                reports.append(report)
        if len(reports) == 3:
            candidates.append({
                'candidate': name, 'mean_accuracy': float(np.mean([r['accuracy'] for r in reports])),
                'min_accuracy': min(r['accuracy'] for r in reports),
                'max_accuracy': max(r['accuracy'] for r in reports),
                'oof_errors': sum(r['test_calls'] - r['correct'] for r in reports),
                'fold_accuracy': [r['accuracy'] for r in reports],
                'fit_seconds': sum(r['fit_seconds'] for r in reports),
            })
    # Predeclared stable order breaks exact ties after mean and minimum fold accuracy.
    candidates.sort(key=lambda r: (-r['mean_accuracy'], -r['min_accuracy']))
    report = {'status': 'complete' if len(candidates) == 8 else 'partial',
              'completed_candidates': len(candidates), 'planned_candidates': 8,
              'validation_evaluated': False, 'candidates': candidates}
    write_json(output / 'cv_summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--mfcc', type=Path, default=MISSION / 'baseline/mfcc_svm/results/call_features.csv')
    parser.add_argument('--split', type=Path, default=MISSION / 'validation/split_assignments.csv')
    parser.add_argument('--output', type=Path, default=BASE / 'results/run_01')
    parser.add_argument('--stage', choices=['lr', 'fusion', 'svm', 'all'], default='lr')
    parser.add_argument('--max-new-fits', type=int, default=24)
    args = parser.parse_args()
    require(args.max_new_fits > 0, '--max-new-fits must be positive')
    require((np.__version__, scipy.__version__, sklearn.__version__) ==
            ('2.2.2', '1.18.1', '1.9.1'), 'Install pinned requirements first')
    started = time.perf_counter()
    ids, y, x, mfcc = load_inputs(args.embeddings, args.mfcc, args.split)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        'code_sha256': sha(Path(__file__)), 'embedding_sha256': EXPORT_SHA,
        'mfcc_sha256': sha(args.mfcc), 'split_sha256': SPLIT_SHA,
        'train_call_order_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'mfcc_features': FEATURES, 'train_calls': len(ids), 'seed': 42, 'n_splits': 3,
        'thread_limit': 2, 'svc_cache_mb': 512,
        'candidate_order': CANDIDATES,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
    }
    # JSON canonicalization converts tuples to lists for resume comparison.
    manifest = json.loads(json.dumps(manifest))
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        require(json.loads(manifest_path.read_text()) == manifest, 'Resume identity mismatch; choose new output')
    else:
        require(not list(output.iterdir()), 'Output lacks manifest but is not empty')
        write_json(manifest_path, manifest)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(x, y))
    assignment = np.full(len(ids), -1)
    for fold, (_, test) in enumerate(folds):
        assignment[test] = fold
    require((assignment >= 0).all(), 'Incomplete fold coverage')
    write_csv(output / 'fold_assignments.csv', ['call_id', 'gender', 'fold'], zip(ids, y, assignment))
    print(f'INPUT PASS: Train={len(ids)}, Wav2Vec2={x.shape}, MFCC={mfcc.shape}; '
          f'{time.perf_counter() - started:.1f}s; Validation not evaluated', flush=True)
    fit_count = 0
    with threadpool_limits(limits=2):
        for stage, c in CANDIDATES:
            if args.stage not in (stage, 'all'):
                continue
            features = np.concatenate((x, mfcc), axis=1) if stage == 'fusion' else x
            name = f'{stage}_C{c:g}'
            directory = output / name
            directory.mkdir(exist_ok=True)
            for fold, (fit, test) in enumerate(folds):
                path = directory / f'fold_{fold}.json'
                if path.exists():
                    print(f'RESUME {name} fold {fold + 1}/3', flush=True)
                    continue
                print(f'START {name} fold {fold + 1}/3, fit={len(fit)} test={len(test)}', flush=True)
                pred, score, report = evaluate_fold(make_model(stage, c), features, y, fit, test)
                oof = directory / f'fold_{fold}_oof.csv'
                write_csv(oof, ['call_id', 'gender', 'prediction', 'decision_score_M'],
                          zip(ids[test], y[test], pred, score))
                report.update(candidate=name, fold=fold, oof_sha256=sha(oof))
                write_json(path, report)
                print(f'PASS {name} fold {fold + 1}/3: accuracy={report["accuracy"]:.6%}, '
                      f'fit={report["fit_seconds"]:.2f}s, predict+score={report["predict_and_score_seconds"]:.2f}s',
                      flush=True)
                fit_count += 1
                if fit_count >= args.max_new_fits:
                    summarize(output)
                    return
    summarize(output)


if __name__ == '__main__':
    main()
