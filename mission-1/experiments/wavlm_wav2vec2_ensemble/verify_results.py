"""Recount returned predictions and compare with the published WavLM baseline (stdlib only)."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT = ROOT/'results/2026-10-09'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(ok, message):
    if not ok: raise ValueError(message)


def verify():
    identity = read(RESULT/'identity.json')
    completion = read(RESULT/'completion.json')
    lock = read(RESULT/'selection.lock.json')
    marker = read(RESULT/'validation_started.json')
    ih = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(ih == completion['identity'] == lock['identity'], 'Run identity mismatch')
    require(sha(ROOT/'run.py') == identity['code_sha256'], 'Executed runner changed')
    require(sha(RESULT/'selection.lock.json') == marker['selection_sha256'], 'Selection changed')
    require(read(RESULT/'models.json')['sha256'] == marker['model_sha256'], 'Model identity changed')
    require(completion['status'] == 'complete' and completion['wavlm_weight'] == lock['wavlm_weight'] == .75, 'Unexpected selection/status')
    with (RESULT/'validation_predictions.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    old_path = ROOT.parent/'wavlm_gender_adaptation/results/2026-10-07/raw/validation_frozen_lr.jsonl'
    old = [json.loads(line) for line in old_path.read_text().splitlines()]
    baseline = {r['call_id']: r for r in old}
    require(len(rows) == len({r['call_id'] for r in rows}) == len(baseline) == len(old) == 5597, 'Missing/duplicate calls')
    require(set(baseline) == {r['call_id'] for r in rows}, 'Different evaluation calls')
    matrix = [[0, 0], [0, 0]]
    paired = dict(corrected=0, regressed=0, both_correct=0, both_wrong=0)
    for row in rows:
        p = [float(row['probability_F']), float(row['probability_M'])]
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in p) and abs(sum(p)-1) < 1e-8, 'Invalid probabilities')
        require(row['prediction'] == ('M' if p[1] > p[0] else 'F'), 'Argmax mismatch')
        prev = baseline[row['call_id']]
        require(prev['gender'] == row['gender'], 'Label changed')
        i, j = 'FM'.index(row['gender']), 'FM'.index(row['prediction'])
        matrix[i][j] += 1
        before, after = prev['prediction'] == prev['gender'], i == j
        paired['both_correct' if before and after else 'regressed' if before else 'corrected' if after else 'both_wrong'] += 1
    correct = matrix[0][0] + matrix[1][1]
    metric = completion['selected']
    require(metric['confusion_F_M'] == matrix and metric['correct'] == correct and metric['wrong'] == 5597-correct and metric['n'] == 5597, 'Metric counts differ')
    require(abs(metric['accuracy'] - correct/5597) < 1e-12, 'Accuracy differs')
    for i, g in enumerate('FM'):
        require(abs(metric['gender_accuracy'][g] - matrix[i][i]/sum(matrix[i])) < 1e-12, 'Gender accuracy differs')
    require(paired['corrected'] == 11 and paired['regressed'] == 4, 'Unexpected paired result')
    # This checks the returned OOF selection summary, not absent raw fold predictions.
    eligible = []
    for candidate in lock['candidates']:
        expected = min(candidate['fold_deltas']) >= 0 and candidate['metrics']['accuracy'] - lock['baseline']['accuracy'] >= .001
        require(candidate['eligible'] == expected, 'Selection rule mismatch')
        if expected: eligible.append(candidate)
    require(max(eligible, key=lambda r: (r['metrics']['correct'], r['wavlm_weight']))['wavlm_weight'] == lock['wavlm_weight'], 'Wrong candidate chosen')
    result = dict(status='verified_returned_predictions', calls=5597, metrics=metric, versus_wavlm=paired,
                  source_hashes={p.name: sha(p) for p in sorted(RESULT.iterdir()) if p.suffix in ('.json', '.csv') and p.name != 'verification.json'},
                  limitations=['Raw OOF probabilities and model weights are absent from the ZIP; their numerical contents were not independently recomputed.',
                               'Previously observed Internal Validation, not independent test; no speaker-disjoint guarantee.'])
    (RESULT/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], correct=correct, paired=paired), indent=2))


if __name__ == '__main__': verify()
