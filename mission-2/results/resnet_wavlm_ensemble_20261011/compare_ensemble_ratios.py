"""CPU-only exploratory ratio comparison; no training or threshold tuning."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(ids, labels, probabilities):
    if ids.ndim != 1 or len(ids) != 111947 or len(set(ids.tolist())) != len(ids):
        raise ValueError('Expected 111947 unique utterance IDs')
    if labels.shape != ids.shape or probabilities.shape != ids.shape:
        raise ValueError('Array shapes differ')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError('Invalid labels/probabilities')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Probabilities outside [0,1]')


def score(y, p):
    prediction = p >= 0.5
    cm = [[int(((y == a) & (prediction == b)).sum()) for b in (0, 1)] for a in (0, 1)]
    return dict(accuracy=float((prediction == y).mean()),
                operator_recall=cm[0][0]/sum(cm[0]), caller_recall=cm[1][1]/sum(cm[1]),
                confusion_matrix=cm, correct=int((prediction == y).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive-dir', type=Path, required=True)
    parser.add_argument('--extended', action='store_true',
                        help='Compare WavLM 40/45/50/55/60 percent against 40 percent')
    args = parser.parse_args()
    d = args.drive_dir
    cache = d / 'mission02_resnet_wavlm4_comparison'
    source = d / 'mission02_ensemble_eval.zip'
    for path in (source, cache/'resnet18_probs.npz', cache/'resnet34_probs.npz'):
        if not path.is_file():
            raise FileNotFoundError(f'필요한 기존 파일이 없습니다: {path}. 기존 비교 결과를 보존해 주세요.')
    # Read only the two small members, not weights or feature shards.
    with zipfile.ZipFile(source) as z:
        spec_bytes = z.read('mission02_ensemble_colab/package.json')
        wavlm_bytes = z.read('mission02_ensemble_colab/aligned_wavlm.npz')
    spec = json.loads(spec_bytes)
    if digest(wavlm_bytes) != spec['sha256']['aligned_wavlm.npz']:
        raise ValueError('WavLM data hash mismatch; do not bypass')
    with np.load(io.BytesIO(wavlm_bytes), allow_pickle=False) as z:
        ids, y, w = [z[k].copy() for k in ('ids', 'labels', 'probabilities')]
    validate(ids, y, w)
    probabilities = []
    for name in ('resnet18', 'resnet34'):
        with np.load(cache/f'{name}_probs.npz', allow_pickle=False) as z:
            expected = digest(spec_bytes) + spec['sha256'][name+'.pt'] + 'float32_no_tf32_v1'
            if z['signature'].item() != expected:
                raise ValueError(f'{name}: checkpoint/package signature mismatch')
            if not np.array_equal(z['ids'], ids) or not np.array_equal(z['labels'], y):
                raise ValueError(f'{name}: ID order or labels mismatch')
            p = z['probabilities'].copy()
        validate(ids, y, p)
        probabilities.append(p)
    r18, r34 = probabilities
    def blend(weight):
        # Preserve the original comparison's arithmetic and baseline.
        return ((1-weight)/2)*r18 + ((1-weight)/2)*r34 + weight*w
    baseline_percent = 40 if args.extended else 25
    baseline = score(y, blend(baseline_percent/100))
    change_key = f'change_percentage_points_vs_{baseline_percent}'
    correct_key = f'additional_correct_vs_{baseline_percent}'
    weights = (.40, .45, .50, .55, .60) if args.extended else (.15, .20, .25, .30, .35, .40)
    results = []
    for weight in weights:
        row = dict(weights=dict(resnet18=(1-weight)/2, resnet34=(1-weight)/2, wavlm4=weight),
                   threshold=.5, **score(y, blend(weight)))
        row[change_key] = 100*(row['accuracy']-baseline['accuracy'])
        row[correct_key] = row['correct']-baseline['correct']
        results.append(row)
    results.sort(key=lambda row: row['accuracy'], reverse=True)
    report = dict(scope='official_validation_weight_selection_not_independent_test',
                  samples=len(y), threshold=.5, baseline_wavlm_percent=baseline_percent,
                  baseline=baseline, results=results,
                  provenance=spec['provenance'],
                  caveats=['Repeated Validation model selection; no unbiased test estimate.',
                           'WavLM probabilities rounded to 6 decimals; re-evaluate actual weights before submission.',
                           'Legacy ResNet checkpoint selection used tuned scores; this does not change that provenance.',
                           'No automatic replacement of models or submission settings.'])
    out = d/('mission02_ensemble_ratio_comparison_extended' if args.extended
             else 'mission02_ensemble_ratio_comparison')
    out.mkdir(exist_ok=True)
    target = out/'ratios.json'
    temporary = out/'ratios.json.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(target)
    print(f'검증 {len(y):,}개 / 임계값 0.5 / 재학습 없음')
    for row in results:
        print(f"WavLM {row['weights']['wavlm4']:.0%} | 정확도 {row['accuracy']:.3%} | "
              f"대원 {row['operator_recall']:.3%} | 신고자 {row['caller_recall']:.3%} | "
              f"현재 {baseline_percent}% 대비 {row[change_key]:+.3f}%p "
              f"({row[correct_key]:+,}개)")
    print(f'저장: {target}')
    print('Validation 비율 선택 결과입니다. 독립 Test 성능이 아니며 기존 모델은 변경하지 않았습니다.')


if __name__ == '__main__':
    main()
