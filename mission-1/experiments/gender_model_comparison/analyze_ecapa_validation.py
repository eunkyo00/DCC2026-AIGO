"""Validate the returned Colab ZIP, then compare only fixed Validation predictions."""
import argparse
import csv
import tempfile
import zipfile
from pathlib import Path
from validation_common import ROOT, bundle, digest, jsonl, metrics, read, require, save, sha, validate


def analyze(folder, output):
    manifest, jobs = bundle()
    require(sha(folder / 'validation_bundle_manifest.json') == sha(ROOT / 'validation_bundle_manifest.json'), 'Returned bundle differs')
    for name, expected in read(folder / 'result_files.json').items():
        require(Path(name).name == name and sha(folder/name) == expected, f'Result file hash: {name}')
    identity = read(folder / 'identity.json')
    require(identity['bundle_sha256'] == sha(ROOT / 'validation_bundle_manifest.json'), 'Identity bundle mismatch')
    source = read(folder / 'source_identity.json')
    load = read(folder / 'source_ecapa_load.json')
    require(sha(folder / 'source_identity.json') == identity['source_identity_sha256'], 'Source identity hash')
    require(sha(folder / 'source_ecapa_load.json') == identity['source_model_load_sha256'], 'Source model hash')
    require(source['bundle_sha256'] == manifest['original_bundle_sha256'], 'Train provenance')
    require(identity['model'] == {k:load[k] for k in ('checkpoint','revision','weight_sha256')}, 'Model provenance')
    require(identity['model']['checkpoint'] == 'JaesungHuh/voice-gender-classifier' and
            identity['model']['revision'] == 'db1222153bd60337e900be22add7af180452adc0', 'Pinned model mismatch')
    for key in ('gpu','python','data_root','dtype','autocast','tf32','batch_size','pipeline'):
        require(identity[key] == source[key], f'Train environment differs: {key}')
    require(identity['dtype'] == 'float32' and identity['autocast'] is False and
            identity['tf32'] is False and identity['batch_size'] == 1 and
            identity['pipeline'] == 'caller_concat_balanced_3_to_15s_duration_weighted_probability_v1', 'Inference definition')
    require(all(identity['versions'][k] == v for k,v in source['versions'].items()), 'Package provenance')
    ih = digest(identity)
    completion = read(folder / 'completion.json')
    require(completion['status'] == 'complete' and completion['identity_sha256'] == ih and
            completion['cache_sha256'] == sha(folder/'predictions.jsonl'), 'Completion/hash mismatch')
    require(read(folder/'preflight.json')['identity_sha256'] == ih, 'Preflight identity')
    data = validate(jsonl(folder / 'predictions.jsonl'), jobs, ih, complete=True)
    failures = jsonl(folder / 'failure_attempts.jsonl')
    require(all(r['identity_sha256'] == ih and (r['call_id'] is None or r['call_id'] in data) for r in failures), 'Unresolved/foreign failures')
    require(completion['historical_failure_attempts'] == len(failures) and completion['failed_calls'] == 0,
            'Failure accounting mismatch')
    require(completion['calls'] == 5597 and completion['prediction_shape'] == [5597,2], 'Completion shape')
    baseline_data, baseline_metrics, overlap = {}, {}, {}
    for name, info in manifest['baselines'].items():
        path = ROOT.parents[1] / info['path']
        require(sha(path) == info['sha256'], f'Baseline hash changed: {name}')
        with path.open(newline='') as f:
            values = list(csv.DictReader(f))
        by_id = {r['call_id']:r for r in values}
        require(len(values) == len(by_id) == 5597 and set(by_id) == set(data), 'Baseline exact coverage')
        require(all(r['gender'] == data[c]['gender'] and r['prediction'] in ('M','F') for c,r in by_id.items()), 'Baseline label mismatch')
        baseline_data[name] = by_id
        baseline_metrics[name] = metrics(values)
        groups = dict(corrected=[], regressed=[], both_wrong=[], both_correct=[])
        for c,r in data.items():
            a, b = by_id[c]['prediction'] == r['gender'], r['prediction'] == r['gender']
            key = ('both_correct' if b else 'regressed') if a else ('corrected' if b else 'both_wrong')
            groups[key].append(c)
        overlap[name] = {k:len(v) for k,v in groups.items()}
    score = metrics(list(data.values()))
    report = {'status':'independently_verified_locally', 'scope':'Previously observed fixed Internal Validation; not a new independent test',
              'split_sha256':manifest['sources']['validation/split_assignments.csv'],
              'identity_sha256':ih, 'cache_sha256':completion['cache_sha256'],
              'ecapa':score, 'baselines':baseline_metrics, 'versus_baselines':overlap,
              'reached_99_percent':score['correct'] >= 5542,
              'correct_needed_for_99':max(0,5542-score['correct']),
              'historical_failure_attempts':len(failures), 'unresolved_failures':0}
    output.mkdir(parents=True, exist_ok=True)
    save(output/'metrics.json', report)
    fields = ['call_id','gender','ecapa_prediction','probability_F','probability_M','mfcc_prediction','wav2vec2_prediction']
    with (output/'call_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c,r in sorted(data.items()):
            writer.writerow({'call_id':c,'gender':r['gender'],'ecapa_prediction':r['prediction'],
                             'probability_F':r['probabilities_F_M'][0], 'probability_M':r['probabilities_F_M'][1],
                             **{n+'_prediction':v[c]['prediction'] for n,v in baseline_data.items()}})
    lines = ['# ECAPA · Internal Validation 결과', '',
             '5,597 unique IDs exact coverage, 확률·argmax·샘플 수·identity·hash·실패 기록을 로컬 검증했다.', '',
             '| 모델 | 정답 | 오답 | Overall | Male | Female |', '|---|---:|---:|---:|---:|---:|']
    for name,m in {**baseline_metrics,'ECAPA':score}.items():
        lines.append(f'| {name} | {m["correct"]} | {m["wrong"]} | {100*m["accuracy"]:.6f}% | {100*m["gender_accuracy"]["M"]:.6f}% | {100*m["gender_accuracy"]["F"]:.6f}% |')
    lines += ['', f'ECAPA confusion matrix (행=정답, 열=예측, M/F): `{score["confusion_matrix_M_F"]}`.', '',
              '| ECAPA와 비교한 기준선 | 교정 | 회귀 | 공통 오답 | 둘 다 정답 |','|---|---:|---:|---:|---:|']
    for name,m in overlap.items():
        lines.append(f'| {name} | {m["corrected"]} | {m["regressed"]} | {m["both_wrong"]} | {m["both_correct"]} |')
    lines += ['', f'99% 기준: 최소 5,542 정답 / 최대 55 오답. 달성: {report["reached_99_percent"]}.',
              f'과거 실패/중단 기록 {len(failures)}건, 미해결 실패 0건.', '',
              '기존에 관찰한 Internal Validation이다. 새 독립 test 또는 공식 Validation 결과가 아니다.',
              '추가 학습·설정 선택 없이 고정 ECAPA 파이프라인을 평가했다. 이 결과에 맞춰 튜닝하지 않는다.',
              'ECAPA와 Wav2Vec2는 집계 방식도 다르므로 전체 파이프라인 비교로 해석한다.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='Returned ZIP or extracted result directory')
    parser.add_argument('--output', type=Path, default=ROOT/'validation_results')
    args = parser.parse_args()
    if args.results.is_dir():
        analyze(args.results, args.output)
    else:
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(args.results) as z:
            require(len(z.namelist()) == len(set(z.namelist())), 'Duplicate ZIP entries')
            require(all(Path(n).name == n and n not in ('.','..') for n in z.namelist()), 'Unsafe result ZIP')
            require(sum(i.file_size for i in z.infolist()) < 100_000_000, 'Unexpected result ZIP size')
            z.extractall(tmp)
            analyze(Path(tmp), args.output)


if __name__ == '__main__':
    main()
