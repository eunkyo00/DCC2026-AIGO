"""Create a titled Colab notebook bound to the exact upload ZIP."""
from common import *


def main():
    cells=[]
    def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s): cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
    md('''# Mission 1 · Wav2Vec2 상위 층 fine-tuning
새 head 대조군 → 상위 2/4층 후보 → Train 내부 선택 → Train 전체 재학습 → 고정 Internal Validation.
99%는 목표이며 보장하지 않습니다. 기존 결과·환경을 보존합니다.

**준비:** L4 런타임, 원본 WAV가 있는 Drive, `wav2vec2_finetuning_bundle.zip`을 `MyDrive/DDC-Colab/`에 업로드.
WavLM 및 다른 CUDA 작업을 먼저 종료하세요. 셀은 순서대로 실행합니다. GPU 학습은 이 Colab에서만 실행합니다.
''')
    md('## 1. Drive 연결과 새 코드 설치\nZIP hash를 검사하며 전용 경로에만 설치합니다.')
    code('''from google.colab import drive
from pathlib import Path
import hashlib, json, os, signal, subprocess, sys, time, zipfile

drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/DDC-Colab')
WORK=Path('/content/wav2vec2_finetuning')
BUNDLE=BASE/'wav2vec2_finetuning_bundle.zip'
EXPECTED_SHA='''+repr(sha(ROOT/'wav2vec2_finetuning_bundle.zip'))+'''
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest()==EXPECTED_SHA, 'Notebook/ZIP mismatch'
WORK.mkdir(exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    assert len(z.namelist())==len(set(z.namelist()))
    assert all((WORK/n).resolve().is_relative_to(WORK.resolve()) for n in z.namelist())
    z.extractall(WORK)
''')
    md('## 2. 기존 원본 경로 확인 · 별도 config\n기존 config가 있으면 실제 `data_root`를 읽고, 없으면 기존 Drive identity를 읽습니다. 추정 경로를 사용하지 않습니다.')
    code('''OLD=Path('/content/gender_model_comparison/config.json')
SOURCE=BASE/'gender_comparison_v1/identity.json'
previous=json.loads(OLD.read_text()) if OLD.exists() else json.loads(SOURCE.read_text())
DATA_ROOT=previous['data_root']
if SOURCE.exists():
    assert DATA_ROOT==json.loads(SOURCE.read_text())['data_root'], 'Existing identity path mismatch'
assert Path(DATA_ROOT).is_dir(), 'Original WAV root inaccessible'
OUTPUT=BASE/'wav2vec2_finetuning_v1'
CONFIG=WORK/'config.json'
cfg=json.loads((WORK/'config.template.json').read_text())
cfg.update(data_root=DATA_ROOT,output=str(OUTPUT))
if CONFIG.exists():
    assert json.loads(CONFIG.read_text())==cfg, 'Preserve existing config'
else:
    CONFIG.write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
print('Original WAV:',DATA_ROOT,'\\nNew output:',OUTPUT)
print(json.loads((WORK/'data/split_identity.json').read_text()))
''')
    md('''## 3. 현재 torch/CUDA 확인 · 새 venv
기존 정상 torch/torchaudio는 재설치하지 않습니다. `--system-site-packages` 전용 venv에서
필요한 non-torch 패키지만 설치합니다. torch 또는 torchaudio 자체가 비정상이면 자동 교체하지 않고 중단합니다.
과거 `2.11.0+cu128`은 참고값이며 현재 import·CUDA 상태를 먼저 확인합니다.
''')
    code('''def command(argv, env=None):
    def parent_death():
        import ctypes
        parent=os.getppid()
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=parent: os.kill(os.getpid(),signal.SIGTERM)
    proc=subprocess.Popen(list(map(str,argv)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,bufsize=1,env=env,start_new_session=True,preexec_fn=parent_death)
    try:
        for line in proc.stdout: print(line,end='',flush=True)
        if proc.wait(): raise RuntimeError(f'Worker failed: {proc.returncode}')
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL); proc.wait()

PY=Path('/content/wav2vec2_finetuning_env/bin/python')
# Reapply even if a failed ensurepip attempt left bin/python behind. No deletion.
command([sys.executable,'-m','venv','--without-pip','--system-site-packages',PY.parents[1]])
command([PY,'-c','import sys; assert sys.prefix != sys.base_prefix; print("venv:", sys.prefix)'])
# pip is inherited from Colab; ensurepip is intentionally not invoked.
command([PY,'-m','pip','--version'])
probe="""import torch, importlib.util
print('torch',torch.__version__,'CUDA',torch.version.cuda)
assert torch.cuda.is_available()
assert 'L4' in torch.cuda.get_device_name(0)
x=torch.ones(2,device='cuda'); print('GPU check:',x.sum().item())
if importlib.util.find_spec('torchaudio'):
 import torchaudio
 print('torchaudio',torchaudio.__version__)
 assert torchaudio.__version__.split('+')[0]==torch.__version__.split('+')[0]
 if '+' in torchaudio.__version__ and '+' in torch.__version__:
  assert torchaudio.__version__.split('+')[1]==torch.__version__.split('+')[1]
"""
command([PY,'-c',probe])
# Resolve in the new venv; protect the installed torch version via constraints.
version=subprocess.check_output([PY,'-c','import importlib.metadata; print(importlib.metadata.version("torch"))'],text=True).strip()
constraints=WORK/'torch_constraints.txt'; constraints.write_text('torch=='+version+'\\n')
command([PY,'-m','pip','install','--constraint',constraints,'-r',WORK/'requirements.txt'])
command([PY,'-m','pip','check'])
command([PY,'-c',probe+'\\nfrom transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor; import scipy, soundfile; print("Imports PASS")'])
ENV=os.environ.copy(); ENV['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; ENV['PYTHONUNBUFFERED']='1'
def run(stage): command([PY,WORK/'run.py','--config',CONFIG,stage],env=ENV)
''')
    md('''## 4. L4 benchmark · 100 optimizer steps/후보와 dev 평가 표본
상위 0·2·4층을 독립 초기화해 측정합니다. Train 길이 분위수와 양 극단, 양 성별을 포함합니다.
전체 caller 구간, 통화 4개 gradient accumulation을 적용하므로 후보마다 400통화입니다.
메모리 85% 미만·평균 step 120초 이하라는 사전 자원 기준을 통과해야 다음 탐색에 포함됩니다.
데이터 읽기/전처리, forward/backward, checkpoint, 평가 ETA를 확인하세요. 추정 시간은 보장이 아닙니다.
''')
    code("run('benchmark')\nprint((OUTPUT/'benchmark.json').read_text())\n")
    md('''## 5. Train 내부 설정 선택용 학습
적격 후보만 inner-train으로 최대 4 epochs 학습하고 매 epoch **전체 inner-dev**를 평가합니다.
최고 dev 정답 수, 동률이면 작은 층 수·이른 epoch를 선택합니다. Validation은 아직 읽지 않습니다.
중단 시 동일 셀을 재실행하면 마지막 원자적 optimizer checkpoint에서 재개합니다.
''')
    code("run('search')\n")
    md('## 6. 설정 잠금\n대조군과 fine-tuning 후보의 dev 선택 결과·epoch를 저장합니다. 잠금 이후 search는 금지됩니다.')
    code("if not (OUTPUT/'selection.json').exists(): run('lock')\nprint((OUTPUT/'selection.json').read_text())\n")
    md('''## 7. 최종 Train 22,388통화 재학습
두 arm 모두 원래 pretrained weights와 동일 seed의 새 head에서 시작합니다.
선택한 정수 epoch만큼 전체 Train으로 학습합니다. dev checkpoint 자체를 최종 평가하지 않습니다.
고정 LR이므로 refit에서 step 수 증가가 LR schedule을 바꾸지 않습니다.
''')
    code("run('refit')\n")
    md('''## 8. 고정 Internal Validation 5,597통화
두 arm의 모든 caller 발화를 사용합니다. 통화별 확률을 지속 저장하고 재개 시 검증합니다.
5,597 exact coverage와 실패 0을 확인하기 전에는 성공 점수를 내지 않습니다.
이미 관찰한 Internal Validation이며 새 독립 test가 아닙니다. 결과를 보고 재튜닝하지 마세요.
''')
    code("run('evaluate')\n")
    md('''## 9. 로컬 검증용 결과 내보내기
weights·원본 WAV·캐시는 Drive에 유지합니다. identity, 선택 근거, benchmark, 업데이트 감사,
checkpoint hash, 통화별 결과와 실패 이력을 ZIP으로 내려받습니다.
''')
    code('''EXPORT=OUTPUT/'wav2vec2_finetuning_results.zip'
export_hashes={}
with zipfile.ZipFile(EXPORT,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUTPUT.rglob('*.json')):
        # Include full provenance and failures, but omit dev per-call predictions (scores retained).
        if any(part.startswith('dev_epoch') or part.startswith('eval_sample-') for part in path.parts): continue
        if path.name=='export_manifest.json': continue
        name=str(path.relative_to(OUTPUT))
        export_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        z.write(path,name)
    z.writestr('export_manifest.json',json.dumps(dict(files=export_hashes),indent=2))
print(EXPORT)
from google.colab import files
files.download(str(EXPORT))
''')
    md('''## 중단·장애 복구
Stop은 worker process group에 TERM 후 최대 10초 뒤 KILL을 보냅니다. 정상 예외는 실패/중단 event를 기록합니다.
강제 런타임 종료는 `running` event와 Drive `ACTIVE_SESSION.lock`을 남길 수 있습니다.
다른 세션/프로세스가 종료됐는지 확인한 뒤 아래 복구 셀을 실행합니다. 자동 잠금 삭제는 하지 않습니다.
복구는 모델 step을 변경하지 않습니다. 마지막 checkpoint 이후 미커밋 step만 다시 계산합니다.
''')
    code('''# Normally leave False. Set True ONLY after confirming all other Colab sessions stopped.
RECOVER_CONFIRMED_DEAD_SESSION=False
if RECOVER_CONFIRMED_DEAD_SESSION:
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not active, 'GPU still occupied'
    for event in (OUTPUT/'events').glob('*.json'):
        obj=json.loads(event.read_text())
        if obj['status']=='running':
            obj.update(status='interrupted',recovery_note='Operator confirmed dead session',recovered_at=time.time())
            tmp=event.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2)); os.replace(tmp,event)
    (OUTPUT/'ACTIVE_SESSION.lock').unlink(missing_ok=True)
    print('Recovery logged. Rerun the interrupted stage.')
''')
    notebook=dict(cells=cells,metadata=dict(colab=dict(name='Wav2Vec2_Gender_Finetuning.ipynb'),kernelspec=dict(display_name='Python 3',language='python',name='python3'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
    save(ROOT/'Wav2Vec2_Gender_Finetuning.ipynb',notebook)

if __name__=='__main__': main()
