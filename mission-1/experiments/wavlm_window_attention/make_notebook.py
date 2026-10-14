"""Readable single-upload Colab notebook; no embedded base64 or large metadata payload."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def make():
    cells = []
    def md(text): cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))
    def code(text, folded=False):
        cells.append(dict(cell_type='code', metadata={'cellView': 'form'} if folded else {}, execution_count=None, outputs=[], source=text.splitlines(True)))
    md('''# WavLM 구간 가중치 실험 · NVIDIA L4

**Frozen WavLM → 구간 특징 1회 저장 → LR / 평균 head / 가중치 head 비교 → 설정 잠금 → 결과 ZIP.**

본체 fine-tuning은 하지 않습니다. 작은 분류기만 학습합니다. 기존 통화 평균 cache에는 구간 정보가 없어서 **음성 특징 재추출은 필요**합니다.
학습에 쓰는 원본은 서울 Training 데이터이며, 공식 Validation은 사용하지 않습니다.

### 사용법
1. 이 노트북 하나를 Colab에 업로드하고 런타임을 **L4**로 선택합니다.
2. 아래 셀을 순서대로 실행합니다. 경로는 기존 Drive 결과에서 자동으로 읽습니다.
3. **200통화 benchmark 결과의 예상 시간을 확인한 뒤** 전체 실험 셀을 실행합니다.
4. 마지막 셀에서 결과 ZIP을 다운로드합니다. 실행 후 노트북도 저장해 로그를 보존하세요.

긴 압축 문자열이 없으며 폴더 전체 자동 탐색도 하지 않습니다. 코드 셀은 기본으로 접혀 있습니다.

### 시간·공간
- 첫 준비: 보통 수분. 모델 최초 다운로드 시간은 별도입니다.
- 전체 음성 특징 추출: **시간 단위 작업**입니다. 실제 200통화의 읽기·추론·저장 시간을 측정해 Train/평가 예상 시간을 표시합니다.
- head 학습: 작은 모델이지만 3-fold 내부 early stopping과 재학습을 포함하므로 분~시간 단위입니다. 첫 fold가 끝나면 남은 fold 시간을 추정합니다.
- 여유 Drive 공간 4GiB 이상. L4 실측 전 고정 완료시간이나 99% 달성은 보장하지 않습니다.
- 완료한 64통화 블록, head의 epoch, 완료 fold부터 재개합니다. 중단된 블록/epoch의 미저장 작업은 다시 수행합니다.
- 하나의 출력 폴더에는 한 런타임만 연결하세요.
''')
    code('''#@title 1. Drive 연결 및 경로 확인
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import json, os, sys, subprocess
ROOT = Path('/content/drive/MyDrive/DDC-Colab')
INPUT_JSON = str(ROOT/'wavlm_gender_inputs.json')
DATA_ROOT = ''  # 기존 identity에서 자동으로 읽음. 필요하면 Training/의 상위 폴더 입력
OUTPUT = str(ROOT/'wavlm_window_attention_l4_v1')
HF_CACHE = '/content/wavlm_window_hf_cache'
previous = ROOT/'wavlm_gender_adaptation_l4_bf16_v2/identity.json'
if not DATA_ROOT and previous.exists():
    DATA_ROOT = json.loads(previous.read_text())['config']['data_root']
if not DATA_ROOT:
    DATA_ROOT = input('Training/ 폴더를 포함한 데이터 루트 경로: ').strip()
assert Path(INPUT_JSON).is_file(), 'DDC-Colab/wavlm_gender_inputs.json을 확인하세요.'
assert (Path(DATA_ROOT)/'Training').is_dir(), 'DATA_ROOT 바로 아래에 Training/이 있어야 합니다.'
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
assert 'L4' in gpu, f'L4 런타임을 선택하세요: {gpu}'
print('GPU:', gpu, '\\n데이터:', DATA_ROOT, '\\n새 결과:', OUTPUT)
WORK = Path('/content/wavlm_window_attention_v1')
WORK.mkdir(exist_ok=True)
CONFIG = WORK/'config.json'
CONFIG.write_text(json.dumps(dict(input_json=INPUT_JSON, data_root=DATA_ROOT, output=OUTPUT, hf_cache=HF_CACHE), ensure_ascii=False, indent=2))
''')
    md('''## 2. 실행 코드 준비
아래 세 셀은 읽을 수 있는 Python 코드를 런타임에 저장합니다. 원본 음성과 이전 결과는 수정하지 않습니다.
''')
    for file, title in [('features.py', '음성 구간 특징 코드 저장'), ('heads.py', '분류기·과적합 점검 코드 저장'), ('run.py', '실험 실행 코드 저장')]:
        source = (ROOT/file).read_text()
        assert "'''" not in source
        code(f"#@title {title}\n(WORK/{file!r}).write_text(r'''{source}''', encoding='utf-8')\nprint({file!r} + ' 준비 완료')\n", folded=True)
    md('''## 3. 별도 환경 준비
현재 Colab의 CUDA PyTorch를 재사용합니다. 새 venv에 CPU 처리·Transformers 패키지만 설치합니다.
''')
    requirements = (ROOT/'requirements.txt').read_text()
    code(f'''#@title 실행 환경 준비
assert (3, 10) <= sys.version_info[:2] < (3, 14), 'Python 3.10–3.13 런타임이 필요합니다.'
ENV = Path('/content/wavlm_window_env_v1')
PY = ENV/'bin/python'
(WORK/'requirements.txt').write_text({requirements!r})
if not PY.exists():
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip', str(ENV)], check=True)
subprocess.run([sys.executable, '-m', 'pip', '--python', str(PY), 'install', '-q', '--ignore-installed', '-r', str(WORK/'requirements.txt')], check=True)
subprocess.run([str(PY), '-c', "import torch,numpy,scipy; assert torch.cuda.is_available(); print('PyTorch',torch.__version__, 'GPU',torch.cuda.get_device_name(0), 'NumPy',numpy.__version__)"] , check=True)
print('환경 준비 완료')
''', folded=True)
    code('''#@title 실행 함수 준비 — 진행 출력과 로그 저장, Stop 시 자식 종료
import signal, time
def run(stage):
    logs = Path(OUTPUT)/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HOME=HF_CACHE, CUBLAS_WORKSPACE_CONFIG=':4096:8',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    def kill_when_parent_dies():
        import ctypes
        parent = os.getppid()
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != parent: os.kill(os.getpid(), signal.SIGTERM)
    p = subprocess.Popen([str(PY), '-u', str(WORK/'run.py'), stage, '--config', str(CONFIG)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env=env, start_new_session=True, preexec_fn=kill_when_parent_dies)
    try:
        with (logs/(stage+'.log')).open('a') as log:
            for line in p.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
        if p.wait(): raise RuntimeError(f'{stage} 실패. 출력과 Drive logs 폴더를 확인하고 같은 셀로 재개하세요.')
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
''', folded=True)
    md('''## 4. 200통화 속도 측정
모델 로드/워밍업은 별도입니다. 음성 길이 분포를 대표하는 Train 200통화를 측정하고 특징을 저장합니다.
아래 출력의 `total_extraction_hours`는 전체 Train+Internal Validation 추출의 추정 시간입니다.
Drive 속도 변화로 초과할 수 있고, 작은 분류기 학습 시간은 별도입니다.
''')
    code("run('benchmark')\n")
    md('''## 5. 전체 실험
Train 특징 → 내부 early stopping을 포함한 3-fold 비교 → 후보 선택 잠금 → 전체 Train 재학습 → 최종 평가.

- **LR:** 기존 방식의 길이 가중 평균 + LR(C=0.1).
- **Mean:** 같은 평균 + 작은 분류기.
- **Attention:** 같은 분류기 + 학습한 구간 비중. dropout·weight decay·길이 가중치에서 과도하게 벗어나지 않도록 하는 규제를 사용.
- 후보는 LR보다 모든 fold에서 하락하지 않고 합계 +0.1%p 이상일 때 채택합니다.
- Attention은 Mean보다도 좋아야 채택합니다. 조건 미달이면 LR 유지.
- 후보 선택용 OOF 점수 자체는 독립 test 성능이 아닙니다. 화자 ID가 없어 동일 사람의 통화가 서로 다른 fold에 들어갈 가능성은 남아 있습니다.

이 셀은 여러 시간 실행될 수 있습니다. Stop 후 같은 셀로 재개하면 완료 단계는 보존합니다.
오류가 나면 다음 단계로 넘어가지 않습니다. 평가 결과를 본 뒤 후보/규칙을 바꾸지 마세요.
''')
    code('''out = Path(OUTPUT)
if not (out/'selection.lock.json').exists():
    run('extract-train')
    run('develop')
if not (out/'final_models.json').exists():
    run('final-fit')
run('evaluate')
''')
    md('''## 6. 결과 다운로드
`model.pt`는 작은 분류기 및 정규화 설정입니다. frozen WavLM 가중치는 별도로 필요합니다.
이번 산출물은 실험용이며, 대회 제출용 새 음성 `inference.py`와 최종 CSV 포장은 별도입니다.
기존 앙상블 98.213329%는 과거 참고값으로만 표시하며 현재 후보 선택에 사용하지 않습니다.
''')
    code('''from google.colab import files
out = Path(OUTPUT)
result = json.loads((out/'completion.json').read_text())
assert result['status'] == 'complete'
print((out/'REPORT.md').read_text())
print('선택:', result['selected'], '| 99% 달성:', result['reached_99'])
print('모델:', out/'model.pt')
files.download(str(out/'window_attention_results.zip'))
''')
    nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                language_info=dict(name='python'), accelerator='GPU', colab=dict(name='WavLM_Window_Attention_L4.ipynb')),
              nbformat=4, nbformat_minor=4)
    (ROOT/'WavLM_Window_Attention_L4.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__': make()
