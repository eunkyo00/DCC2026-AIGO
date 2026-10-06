"""Audit all saved CV predictions and describe Train-only errors; no model fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from run_cv import (BASE, MISSION, CANDIDATES, SPLIT_SHA, index_unique, require,
                    rows, sha, write_csv, write_json)

BASELINE = 'lr_C1'
TUNED = 'lr_C0.1'
SELECTED = 'fusion_C0.1'


def overlap(old_wrong, new_wrong):
    return {'both_correct': int((~old_wrong & ~new_wrong).sum()),
            'corrected': int((old_wrong & ~new_wrong).sum()),
            'regressed': int((~old_wrong & new_wrong).sum()),
            'both_wrong': int((old_wrong & new_wrong).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=BASE / 'results/run_01')
    parser.add_argument('--output', type=Path, default=BASE / 'results/run_01/error_analysis')
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    split_path = MISSION / 'validation/split_assignments.csv'
    mfcc_path = MISSION / 'baseline/mfcc_svm/results/call_features.csv'
    f0_path = MISSION / 'baseline/f0_lr/results/call_features.csv'
    calls_path = MISSION / 'validation/manifests/calls.csv'
    require(sha(split_path) == manifest['split_sha256'] == SPLIT_SHA, 'Split changed')
    require(sha(mfcc_path) == manifest['mfcc_sha256'], 'MFCC changed')
    require(sha(BASE / 'run_cv.py') == manifest['code_sha256'], 'CV runner changed')
    split = index_unique(rows(split_path), 'split')
    assignment = rows(args.run / 'fold_assignments.csv')
    ids = np.array([r['call_id'] for r in assignment])
    y = np.array([r['gender'] for r in assignment])
    fold_ids = np.array([int(r['fold']) for r in assignment])
    require(len(set(ids)) == len(ids) == 22388, 'Fold coverage')
    require(set(ids) == {c for c, r in split.items() if r['partition'] == 'train'}, 'Train IDs')
    import hashlib
    require(hashlib.sha256('\n'.join(ids).encode()).hexdigest() ==
            manifest['train_call_order_sha256'], 'Train order changed')
    for call, gender in zip(ids, y):
        require(split[call]['gender'] == gender, 'Train label changed')
    for fold, (_, test) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(ids, y)):
        require(np.array_equal(np.flatnonzero(fold_ids == fold), test), 'Fold mismatch')
    names = [f'{stage}_C{c:g}' for stage, c in CANDIDATES]
    predictions, scores, wrong, candidate_stats = {}, {}, {}, {}
    for name in names:
        predictions[name] = np.full(len(ids), '', dtype='<U1')
        scores[name] = np.full(len(ids), np.nan)
        for fold in range(3):
            root = args.run / name
            csv_path = root / f'fold_{fold}_oof.csv'
            report = json.loads((root / f'fold_{fold}.json').read_text())
            require(sha(csv_path) == report['oof_sha256'], f'OOF hash: {name}/{fold}')
            table = index_unique(rows(csv_path), name)
            test = np.flatnonzero(fold_ids == fold)
            require(set(table) == set(ids[test]), f'OOF fold coverage: {name}/{fold}')
            for i in test:
                row = table[ids[i]]
                require(row['gender'] == y[i] and row['prediction'] in ('F', 'M'), 'OOF label')
                predictions[name][i] = row['prediction']
                scores[name][i] = float(row['decision_score_M'])
            require(int((predictions[name][test] == y[test]).sum()) == report['correct'], 'Metric mismatch')
            require(report['fit_calls'] == report['scaler_fit_calls'] == len(ids) - len(test), 'Fit count')
        require(np.isfinite(scores[name]).all(), 'OOF nonfinite score')
        wrong[name] = predictions[name] != y
        candidate_stats[name] = {
            'errors': int(wrong[name].sum()), 'pooled_oof_accuracy': float((~wrong[name]).mean()),
            'confusion_matrix_M_F': confusion_matrix(y, predictions[name], labels=['M', 'F']).tolist()}
    cv = json.loads((args.run / 'cv_summary.json').read_text())
    require(cv['status'] == 'complete' and cv['completed_candidates'] == 8, 'Incomplete CV')
    for record in cv['candidates']:
        require(candidate_stats[record['candidate']]['errors'] == record['oof_errors'], 'Summary mismatch')
    mfcc = index_unique(rows(mfcc_path), 'mfcc')
    f0 = index_unique(rows(f0_path), 'f0')
    calls = index_unique(rows(calls_path), 'calls')
    for name, table in [('MFCC', mfcc), ('F0', f0), ('calls', calls)]:
        require(set(ids) <= set(table), f'{name}: missing Train IDs')
        for call, gender in zip(ids, y):
            require((table[call]['gender'], table[call]['partition']) == (gender, 'train'), f'{name}: identity')
            require(int(table[call]['n_segments']) == int(mfcc[call]['n_segments']), f'{name}: segments')
    def values(table, key):
        return np.array([float(table[c][key]) for c in ids])
    duration = values(mfcc, 'total_duration_s')
    segments = values(mfcc, 'n_segments')
    silence = values(mfcc, 'mean_silence_ratio')
    reliable = values(f0, 'reliable_f0_segment_ratio')
    tone = values(f0, 'tone_dominated_segments')
    pitch = np.array([float(f0[c]['f0_median_hz_median'])
                      if f0[c]['f0_median_hz_median'] else np.nan for c in ids])
    union = values(calls, 'reporter_union_duration')
    require(np.allclose(duration, values(f0, 'total_duration_s'), atol=1e-6), 'F0 duration mismatch')
    require(np.allclose(duration, values(calls, 'total_duration'), atol=1e-6), 'Manifest duration mismatch')
    for a in (duration, segments, silence, reliable, tone, union):
        require(np.isfinite(a).all(), 'Nonfinite diagnostic')
    source = np.array([Path(calls[c]['wav_relative_path']).parent.name for c in ids])
    selected_wrong = wrong[SELECTED]
    def stats(mask):
        n = int(mask.sum())
        e = int((selected_wrong & mask).sum())
        return {'calls': n, 'errors': e, 'error_rate': e / n if n else None,
                'share_of_errors': e / int(selected_wrong.sum()),
                'by_gender': {g: {'calls': int((mask & (y == g)).sum()),
                                 'errors': int((mask & (y == g) & selected_wrong).sum())} for g in ('M', 'F')}}
    masks = {
        'all': np.ones(len(ids), dtype=bool), 'Male': y == 'M', 'Female': y == 'F',
        'duration_lt8': duration < 8, 'duration_8_20': (duration >= 8) & (duration < 20),
        'duration_20_40': (duration >= 20) & (duration < 40), 'duration_ge40': duration >= 40,
        'segments_le5': segments <= 5, 'segments_6_15': (segments > 5) & (segments <= 15),
        'segments_ge16': segments > 15,
        'mean_silence_lt0.1': silence < .1,
        'mean_silence_0.1_0.5': (silence >= .1) & (silence < .5),
        'mean_silence_ge0.5': silence >= .5,
        'any_high_silence': values(mfcc, 'high_silence_segments') > 0,
        'no_high_silence': values(mfcc, 'high_silence_segments') == 0,
        'any_very_short_segment': values(mfcc, 'short_segments') > 0,
        'no_very_short_segment': values(mfcc, 'short_segments') == 0,
        'reliable_f0_lt0.5': reliable < .5, 'reliable_f0_ge0.5': reliable >= .5,
        'any_tone_flag': tone > 0, 'no_tone_flag': tone == 0,
        'duration_ge20_low_silence': (duration >= 20) & (silence < .1),
    }
    groups = {name: stats(mask) for name, mask in masks.items()}
    all_wrong = np.logical_and.reduce([wrong[n] for n in names])
    cohort_masks = {
        'persistent': wrong[BASELINE] & selected_wrong,
        'corrected': wrong[BASELINE] & ~selected_wrong,
        'regressed': ~wrong[BASELINE] & selected_wrong,
        'control': ~wrong[BASELINE] & ~selected_wrong,
    }
    cohorts = {}
    for name, mask in cohort_masks.items():
        cohorts[name] = {**stats(mask), 'median_duration': float(np.median(duration[mask])),
                         'median_segments': float(np.median(segments[mask])),
                         'median_silence': float(np.median(silence[mask])),
                         'median_reliable_f0_ratio': float(np.median(reliable[mask]))}
    sources = {name: stats(source == name) for name in sorted(set(source))}
    f0_medians = {}
    for g in ('M', 'F'):
        for state, err in [('correct', False), ('wrong', True)]:
            group = pitch[(y == g) & (selected_wrong == err)]
            valid = group[np.isfinite(group)]
            f0_medians[f'{g}_{state}'] = {'median_hz': float(np.median(valid)) if len(valid) else None,
                                         'valid': len(valid), 'missing': int(len(group) - len(valid))}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        'scope': 'Train OOF only; descriptive post-selection analysis', 'train_calls': len(ids),
        'validated_folds': 24, 'validation_evaluated': False, 'audio_listened': False,
        'inputs': {p.name: sha(p) for p in (split_path, mfcc_path, f0_path, calls_path)},
        'analysis_code_sha256': sha(Path(__file__)),
        'candidate_stats': candidate_stats,
        'baseline_to_fusion': overlap(wrong[BASELINE], selected_wrong),
        'tuned_lr_to_fusion': overlap(wrong[TUNED], selected_wrong),
        'all_8_wrong': int(all_wrong.sum()),
        'fusion_errors_corrected_by_any_svm': int((selected_wrong & (~wrong['svm_C1'] | ~wrong['svm_C10'])).sum()),
        'groups': groups, 'cohorts': cohorts, 'source_folders': sources,
        'f0_medians_hz': f0_medians,
    }
    write_json(output / 'summary.json', summary)
    detailed = []
    for i, call in enumerate(ids):
        cohort = next(name for name, mask in cohort_masks.items() if mask[i])
        detailed.append([call, y[i], int(fold_ids[i]), cohort, predictions[BASELINE][i],
                         predictions[TUNED][i], predictions[SELECTED][i], float(scores[SELECTED][i]),
                         int(all_wrong[i]), float(duration[i]), int(segments[i]), float(silence[i]),
                         float(reliable[i]), int(tone[i]), source[i]])
    headers = ['call_id', 'gender', 'fold', 'cohort', 'baseline_prediction', 'tuned_lr_prediction',
               'fusion_prediction', 'fusion_decision_score_M', 'all_8_wrong', 'duration_s',
               'n_segments', 'mean_silence_ratio', 'reliable_f0_ratio', 'tone_segments', 'source_folder']
    write_csv(output / 'train_oof_analysis.csv', headers, detailed)
    write_csv(output / 'fusion_errors.csv', headers, [r for i, r in enumerate(detailed) if selected_wrong[i]])
    # 6 per gender per cohort, deterministic random selection; not prevalence sampling.
    rng = np.random.default_rng(42)
    review = []
    for cohort, mask in cohort_masks.items():
        for gender in ('M', 'F'):
            pool = np.flatnonzero(mask & (y == gender))
            chosen = rng.choice(pool, size=min(6, len(pool)), replace=False)
            for i in chosen:
                review.append(detailed[i] + [calls[ids[i]]['wav_relative_path'],
                                            '', '', '', '', '', ''])
    write_csv(output / 'listening_review.csv', headers + ['wav_relative_path', 'audible_caller',
              'multiple_speakers', 'noise_or_tone', 'crop_issue', 'label_uncertainty', 'notes'], review)
    require(sum(cohorts[k]['calls'] for k in cohorts) == 22388, 'Cohort total')
    require(len([r for i, r in enumerate(detailed) if selected_wrong[i]]) == 570, 'Error coverage')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'OOF ERROR ANALYSIS PASS; review sample={len(review)}; audio not listened; output={output}')


if __name__ == '__main__':
    main()
