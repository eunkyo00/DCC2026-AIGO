"""Build small Train-only metadata/code bundle and a complete Colab notebook."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
MISSION = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    split_path = MISSION / 'validation/split_assignments.csv'
    assert sha(split_path) == '04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06'
    train = sorted([r for r in rows(split_path) if r['partition'] == 'train'], key=lambda r: r['call_id'])
    assert len(train) == 22388
    _, sample = train_test_split(train, test_size=2000, random_state=42, stratify=[r['gender'] for r in train])
    selected = {r['call_id']: r for r in sample}
    calls = {r['call_id']: r for r in rows(MISSION / 'validation/manifests/calls.csv')}
    intervals = {c: [] for c in selected}
    with (MISSION / 'eda/eda_outputs/segments.csv').open() as f:
        for row in csv.DictReader(f):
            c = row['file_ref']
            if c in selected and row['split'] == 'Training' and row['speaker'] == '1':
                assert row['matched'] == 'True' and row['exceeds_wav'] == 'False'
                intervals[c].append([float(row['start_s']), float(row['end_s'])])
    predictions = {}
    cv = MISSION / 'experiments/embedding_classifiers/results/run_01'
    for name in ('lr_C1', 'fusion_C0.1'):
        predictions[name] = {}
        for fold in range(3):
            p = cv / name / f'fold_{fold}_oof.csv'
            assert sha(p) == json.loads((p.parent / f'fold_{fold}.json').read_text())['oof_sha256']
            predictions[name].update({r['call_id']: r for r in rows(p)})
    # New Train-only smoke fixtures: avoid old fixtures containing holdout calls.
    smoke = set()
    for g in ('M', 'F'):
        group = [c for c in selected if selected[c]['gender'] == g]
        ordered = sorted(group, key=lambda c: (float(calls[c]['total_duration']), c))
        smoke.update([ordered[0], ordered[len(ordered)//2], ordered[-1]])
    benchmark = set(smoke)
    for c in sorted(selected, key=lambda c: hashlib.sha256(('benchmark42' + c).encode()).hexdigest()):
        if len(benchmark) == 20:
            break
        benchmark.add(c)
    jobs = []
    for c in sorted(selected):
        r = selected[c]
        assert calls[c]['partition'] == 'train' and calls[c]['gender'] == r['gender']
        assert len(intervals[c]) == int(calls[c]['n_segments'])
        assert all(predictions[n][c]['gender'] == r['gender'] for n in predictions)
        jobs.append({**r, 'wav_relative_path': calls[c]['wav_relative_path'],
                     'intervals': sorted(intervals[c]), 'n_segments': len(intervals[c]),
                     'smoke': c in smoke, 'benchmark': c in benchmark,
                     'baseline_lr_prediction': predictions['lr_C1'][c]['prediction'],
                     'baseline_fusion_prediction': predictions['fusion_C0.1'][c]['prediction']})
    (ROOT / 'data').mkdir(exist_ok=True)
    (ROOT / 'data/jobs.json').write_text(json.dumps(jobs, ensure_ascii=False))
    paths = [ROOT / 'compare_models.py', ROOT / 'requirements.txt', ROOT / 'model_sources.json',
             ROOT / 'data/jobs.json', *sorted((ROOT / 'vendor').iterdir())]
    paths = [p for p in paths if p.is_file() and p.suffix != '.pyc']
    manifest = {'scope': 'Train 2000-call screen only', 'split_sha256': sha(split_path),
                'sample_seed': 42, 'calls': 2000,
                'gender_counts': {g: sum(j['gender'] == g for j in jobs) for g in ('M', 'F')},
                'caller_segments': sum(j['n_segments'] for j in jobs),
                'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}
    (ROOT / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    archive = artifacts / 'gender_comparison_bundle.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in paths + [ROOT / 'bundle_manifest.json']:
            z.write(p, p.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    make_notebook(sha(archive))
    print(json.dumps({**{k: v for k, v in manifest.items() if k != 'files'},
                      'zip_bytes': archive.stat().st_size, 'zip_sha256': sha(archive)}, indent=2))


def make_notebook(bundle_sha):
    cells = []
    def md(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})
    def code(text):
        import ast
        ast.parse(text)
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(keepends=True),
                      'execution_count': None, 'outputs': []})
    md('''# Mission 1 · WavLM / ECAPA 전체 예비 비교

**이 notebook 하나로 환경 준비 → 두 모델 로딩 → smoke → benchmark → Train 2,000통화 비교까지 진행합니다.**

준비: Python 3 / L4 GPU. `gender_comparison_bundle.zip`을 **내 드라이브/DDC-Colab/**에 업로드하세요.
기존 원본 WAV 폴더 바로가기도 같은 Google 계정의 내 드라이브에 있어야 합니다.
embedding NPZ나 원본 WAV를 새로 업로드할 필요는 없습니다.

위에서 아래로 실행하세요. 오류 셀에서 멈추고 아래 셀을 실행하지 마세요.
전체 27,985통화 추출·추가 학습·Validation 평가는 포함하지 않습니다.

새 입력 방식: 모든 caller crop을 기존 방식으로 16kHz 변환 → 시간순 연결 → 균등한 최대 15초 창.
3초보다 짧은 통화는 단일 입력을 3초까지 0으로 채우고 기록합니다. 모든 원본 caller 샘플을 유지합니다.
창별 Female/Male 확률을 실제 음성 길이로 가중 평균합니다. 두 모델에 동일한 입력을 사용합니다.
기존 segment 동일 가중 embedding 평균과 다른 **파이프라인 비교**입니다.
''')
    md('## 1. Drive 연결 · ZIP 연결\n예상 30초–2분, 로그인 시간 별도.\n')
    code('''from google.colab import drive
from pathlib import Path
import json, hashlib, zipfile, subprocess, sys, os

drive.mount('/content/drive')
BUNDLE = Path('/content/drive/MyDrive/DDC-Colab/gender_comparison_bundle.zip')
assert BUNDLE.is_file(), f'ZIP을 이 위치에 업로드하세요: {BUNDLE}'
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == "''' + bundle_sha + '''", 'ZIP 버전이 notebook과 다릅니다.'
WORK = Path('/content/gender_model_comparison')
WORK.mkdir(exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    for name in z.namelist():
        assert (WORK / name).resolve().is_relative_to(WORK.resolve()), 'Unsafe ZIP path'
    z.extractall(WORK)
print('BUNDLE PASS:', WORK, flush=True)
''')
    md('## 2. 독립 실행 환경 설치\n예상 2–5분. 모델은 아직 다운로드하지 않습니다. 새 subprocess를 사용하므로 NumPy 설치 후 notebook 재시작이 필요 없습니다.\n')
    code('''def command(argv, env=None):
    print('Running:', ' '.join(map(str, argv)), flush=True)
    p = subprocess.Popen(list(map(str, argv)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    for line in p.stdout:
        print(line, end='', flush=True)
    if p.wait() != 0:
        raise RuntimeError(f'Command failed with return code {p.returncode}; 아래 셀을 실행하지 마세요.')

ENV = Path('/content/gender_compare_env')
if not (ENV / 'bin/python').exists():
    command([sys.executable, '-m', 'venv', '--system-site-packages', ENV])
PY = ENV / 'bin/python'
# An interrupted setup can leave an existing venv isolated from Colab's torch.
venv_cfg = ENV / 'pyvenv.cfg'
cfg_text = venv_cfg.read_text()
if 'include-system-site-packages = true' not in cfg_text.lower():
    import re
    cfg_text, count = re.subn(r'(?im)^include-system-site-packages\s*=\s*.*$',
                              'include-system-site-packages = true', cfg_text)
    if count != 1:
        raise RuntimeError('Cannot repair venv system site packages setting')
    venv_cfg.write_text(cfg_text)
command([PY, '-c', 'import torch; print("torch:", torch.__version__)'])
# Repair a partially created/Colab venv that lacks pip, without deleting it.
if subprocess.run([PY, '-m', 'pip', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
    command([sys.executable, '-m', 'pip', '--python', PY, 'install', '--disable-pip-version-check', 'pip'])
command([PY, '-m', 'pip', 'install', '--disable-pip-version-check', '-r', WORK / 'requirements.txt'])
torch_version = subprocess.check_output([sys.executable, '-c', 'import torch; print(torch.__version__)'], text=True).strip()
if '+cu128' not in torch_version:
    raise RuntimeError(f'Expected Colab CUDA 12.8 torch, got {torch_version}')
command([PY, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-deps',
         '--force-reinstall', 'torchaudio==' + torch_version,
         '--index-url', 'https://download.pytorch.org/whl/cu128'])
command([PY, '-c', 'import sys; sys.modules["torchvision"]=None; sys.modules["librosa"]=None; import torch, torchaudio, transformers, numpy, scipy; from transformers import WavLMModel; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0)); print("torch:",torch.__version__,"torchaudio:",torchaudio.__version__,"transformers:",transformers.__version__); print("ENVIRONMENT PASS")'])
''')
    md('## 3. 원본 WAV 경로 확인 · 결과 저장 위치\n예상 수 초. 경로를 자동 확인하며, 못 찾으면 `DATA_ROOT_OVERRIDE`에 실제 경로를 입력합니다.\n')
    code('''DATA_ROOT_OVERRIDE = ''  # 자동 확인 실패 시에만 실제 Training 폴더의 부모 경로 입력
OUTPUT = Path('/content/drive/MyDrive/DDC-Colab/gender_comparison_v1')
options = []
old = Path('/content/drive/MyDrive/DDC-Colab/wav2vec2_frozen_cuda_fp32/config.json')
if DATA_ROOT_OVERRIDE:
    options.append(Path(DATA_ROOT_OVERRIDE))
else:
    if old.is_file():
        cfg = json.loads(old.read_text())
        value = cfg.get('data_root') or cfg.get('DATA_ROOT')
        if value: options.append(Path(value))
    for outer in Path('/content/drive/MyDrive').iterdir():
        if outer.is_dir() and '2026' in outer.name and 'DCC' in outer.name:
            options.append(outer)
            options.extend(p for p in outer.iterdir() if p.is_dir())
jobs = json.loads((WORK / 'data/jobs.json').read_text())
found = {}
for p in options:
    if (p / 'Training').is_dir() and (p / jobs[0]['wav_relative_path']).is_file():
        found[str(p.resolve())] = p
assert len(found) == 1, f'실제 원본 경로를 정확히 하나 확인해야 합니다. 후보: {list(found)}. DATA_ROOT_OVERRIDE를 지정하세요.'
DATA_ROOT = next(iter(found.values()))
OUTPUT.mkdir(parents=True, exist_ok=True)
CONFIG = WORK / 'config.json'
CONFIG.write_text(json.dumps({'data_root': str(DATA_ROOT), 'output_dir': str(OUTPUT)}, ensure_ascii=False))
RUN_ENV = os.environ.copy()
RUN_ENV['HF_HOME'] = '/content/gender_model_hf_cache'
RUN_ENV['TOKENIZERS_PARALLELISM'] = 'false'
def run(stage):
    command([PY, '-u', WORK / 'compare_models.py', '--config', CONFIG, '--stage', stage], env=RUN_ENV)
print('DATA_ROOT:', DATA_ROOT)
print('OUTPUT:', OUTPUT)
print('Train calls:', len(jobs), 'No Validation inference')
''')
    md('## 4. Preflight\n예상 1–5분, Drive 상태에 따라 증가. 2,000개 경로와 benchmark 20통화의 실제 WAV·crop·resampling을 확인합니다.\n')
    code("run('preflight')\n")
    md('## 5. 모델 다운로드 · CUDA 로딩 확인\n첫 실행 예상 3–10분, 네트워크에 따라 증가. 두 모델을 차례로 로딩합니다. 실제 음성 대신 3초 합성 입력으로 shape·finite·클래스 순서를 확인합니다.\n')
    code("run('models')\n")
    md('## 6. 두 모델 6-call smoke\n예상 1–5분, 모델 재로딩 포함. **여기부터 실제 WAV 추론**입니다. 새 Train 전용 6통화를 두 모델이 동일하게 사용합니다.\n')
    code("run('smoke')\n")
    md('## 7. 두 모델 20-call benchmark\n예상 1–5분, 실제 시간은 출력으로 확인. 정확도 우열을 판단하는 표본이 아닙니다. projection은 로딩·캐시 flush 등을 제외한 거친 참고값입니다.\n')
    code("run('benchmark')\n")
    code('''for name in ('ecapa', 'wavlm'):
    s = json.loads((OUTPUT / f'{name}_benchmark_summary.json').read_text())
    print(name, 'sec/call=', round(s['seconds_per_call'], 3),
          '2000-call rough minutes=', round(s['projected_pilot_minutes_rough'], 1))
print('Drive 대기·파일 저장·세션 상태에 따라 더 오래 걸릴 수 있습니다.')
''')
    md('## 8. 같은 Train 2,000통화 예비 비교\n**소요시간은 바로 위 두 모델의 실측 예상치 합계를 참고하세요.** 사전에 고정 시간을 보장하지 않습니다.\nDrive에 통화별 예측을 저장하며 중단 시 동일 환경에서 이 셀을 다시 실행하면 이어갑니다.\n두 모델 전체 완료 전에는 부분 집계로 승자를 선택하지 마세요.\n')
    code("run('pilot')\n")
    md('## 9. 최종 비교표 · 로컬 전달 ZIP\n예상 수 초. 이 결과는 예비 Train 표본 결과이며 최종 Validation 정확도가 아닙니다.\n')
    code("run('compare')\n")
    code('''archive = OUTPUT / 'gender_comparison_results.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUTPUT.iterdir()):
        if p.is_file() and p.suffix in ('.json', '.jsonl'):
            z.write(p, p.name)
    z.write(WORK / 'bundle_manifest.json', 'bundle_manifest.json')
print('RESULT ZIP:', archive)
print('비교 결과와 ZIP을 전달하세요. 다음 실험은 결과 검토 후 결정합니다.')
''')
    md('''## 세션 종료 후 복구

같은 notebook의 1–3번을 다시 실행한 뒤 8번을 실행합니다. Drive에 완료된 preflight·smoke·benchmark가 있어야 합니다.
다른 GPU·패키지·코드 버전이면 cache 혼합 방지를 위해 중단합니다. 임의로 identity 검사를 지우지 마세요.
실패 통화는 `failure_attempts.jsonl`에 기록하고 즉시 중단합니다. 문제 해결 후 해당 통화를 재시도합니다.
같은 OUTPUT으로 notebook 두 개를 동시에 실행하지 마세요.
''')
    for i, cell in enumerate(cells):
        cell['id'] = f'cell-{i:02}'
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
          'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                       'colab': {'provenance': []}, 'language_info': {'name': 'python'}}}
    (ROOT / 'Gender_Model_Comparison.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
