import json
from common import ROOT, atomic


def make(bundle_hash):
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None))
    md('''# WavLM 성별 분류 · 팀원 인계용

코드 ZIP: `wavlm_teammate_bundle.zip`. 원본 WAV 27,985개는 **별도로 준비**해야 합니다.
Linux + NVIDIA L4 또는 T4 + 정상 CUDA PyTorch 환경에서 실행합니다. Jupyter/Runpod/Colab 모두 실제 경로를 직접 입력합니다.
Colab이면 Drive 연결은 먼저 팀원 본인 계정으로 수행하세요. 이 노트북은 Drive 경로/계정을 가정하지 않습니다.

기본 범위: frozen WavLM 다층 특징 → Train 내부 dev 선택 → 전체 Train LR → Internal Validation.
99%는 목표입니다. 약 20시간/2만원은 과거 예산용 가정이며 실측·보장값이 아닙니다. benchmark부터 확인합니다.
기존 실험 output에서 재개하지 말고 팀원 실행용 빈 출력 폴더를 선택하세요.''')
    md('## 0. 현재 환경 확인 (읽기만)\n출력된 Python 실행파일 경로와 GPU 이름을 다음 셀에 입력하세요. GPU 없는 로컬에서도 파일 준비는 가능합니다.')
    code('''import sys, platform, subprocess
from pathlib import Path
print('Python executable:', Path(sys.executable).resolve())
print('Python:', sys.version)
print('OS:', platform.platform())
r = subprocess.run(['nvidia-smi'], text=True, capture_output=True)
print(r.stdout or r.stderr)
''')
    md('## 1. 팀원이 채울 항목\n모든 경로는 빈칸입니다. 절대 경로를 직접 입력하세요. DATA_ROOT는 `Training/` 바로 위 폴더, OUTPUT_DIR는 결과가 영구 보존되는 새 폴더입니다. GPU_LOCK_PATH는 같은 서버의 다른 실험도 공유할 잠금 **파일** 경로입니다. 학습 설정 12초/accumulation 4는 사전 설계값입니다.')
    code('''from pathlib import Path
import hashlib, json, os, signal, shutil, time, zipfile

BUNDLE_PATH = ""
WORK_DIR = ""
BASE_PYTHON = ""
VENV_DIR = ""
DATA_ROOT = ""
OUTPUT_DIR = ""
HF_CACHE_DIR = ""
GPU_LOCK_PATH = ""
EXPECTED_GPU = ""
PRECISION = ""  # T4: fp32 / L4: bf16 (또는 fp32). fp16은 이 배포본에서 지원하지 않습니다.

RUN_ADAPTATION = False
CROP_SECONDS = 12
ACCUMULATION = 4
''')
    md('## 2. ZIP hash 확인·코드 풀기\n원본 WAV ZIP은 이 셀에서 풀지 않습니다. DATA_ROOT 아래의 Training/ 구조를 보존해 미리 준비하세요. 필요한 파일 목록은 코드 ZIP의 data/wav_inventory.csv에 있습니다.')
    code('''EXPECTED_BUNDLE_SHA256 = '''+repr(bundle_hash)+'''
path_values = dict(BUNDLE_PATH=BUNDLE_PATH, WORK_DIR=WORK_DIR, BASE_PYTHON=BASE_PYTHON,
                   VENV_DIR=VENV_DIR, DATA_ROOT=DATA_ROOT, OUTPUT_DIR=OUTPUT_DIR,
                   HF_CACHE_DIR=HF_CACHE_DIR, GPU_LOCK_PATH=GPU_LOCK_PATH)
for key, value in path_values.items():
    assert isinstance(value,str) and value.strip(), f'경로 빈칸을 채워주세요: {key}'
    assert Path(value).is_absolute(), f'절대 경로를 입력하세요: {key}'
    assert str(Path(value).resolve()) == value, f'실제 절대 경로를 사용하세요: {key} -> {Path(value).resolve()}'
assert EXPECTED_GPU.strip() and PRECISION in ('fp32','bf16')
BUNDLE, WORK, ENV = Path(BUNDLE_PATH), Path(WORK_DIR), Path(VENV_DIR)
assert BUNDLE.is_file() and Path(BASE_PYTHON).is_file()
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == EXPECTED_BUNDLE_SHA256, '노트북/ZIP 버전 불일치'
ENV_MARKER = dict(base_python=BASE_PYTHON,bundle_sha256=EXPECTED_BUNDLE_SHA256)
assert not ENV.exists() or ((ENV/'.wavlm_handoff_env.json').is_file() and json.loads((ENV/'.wavlm_handoff_env.json').read_text())==ENV_MARKER), '새 venv 또는 이 배포본에서 만든 전용 venv만 사용하세요'
WORK.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    assert len(z.namelist()) == len(set(z.namelist()))
    assert all((WORK/n).resolve().is_relative_to(WORK.resolve()) for n in z.namelist())
    manifest_bytes = z.read('bundle_manifest.json')
    if any(WORK.iterdir()):
        assert (WORK/'bundle_manifest.json').is_file() and (WORK/'bundle_manifest.json').read_bytes()==manifest_bytes, '다른 코드 폴더에 덮어쓰지 마세요'
    z.extractall(WORK)
CONFIG = WORK/'config.json'
config = dict(data_root=DATA_ROOT, output_dir=OUTPUT_DIR, hf_cache_dir=HF_CACHE_DIR,
              gpu_lock_path=GPU_LOCK_PATH, expected_gpu=EXPECTED_GPU, precision=PRECISION,
              crop_seconds=CROP_SECONDS, accumulation=ACCUMULATION)
if CONFIG.exists(): assert json.loads(CONFIG.read_text())==config, '기존 config를 바꾸어 재개할 수 없습니다'
CONFIG.write_text(json.dumps(config,ensure_ascii=False,indent=2))
print('Code:', WORK, '\\nConfig:', config)
''')
    md('## 3. 기존 torch 확인·별도 venv·패키지 설치\n정상 torch/torchaudio는 재설치하지 않습니다. CUDA wheel 불일치·지원하지 않는 GPU이면 여기서 중단합니다. 새로운 환경의 torch/CUDA는 실제 값을 기록합니다. Stop은 학습 자식 프로세스까지 종료하지만 GPU 대여 요금은 제공업체에서 별도로 정지해야 합니다.')
    code('''def command(argv, env=None):
    def parent_death_signal():
        import ctypes
        parent = os.getppid()
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != parent: os.kill(os.getpid(), signal.SIGTERM)
    proc = subprocess.Popen(list(map(str,argv)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env, start_new_session=True, preexec_fn=parent_death_signal)
    try:
        for line in proc.stdout: print(line,end='',flush=True)
        if proc.wait(): raise RuntimeError(f'실패 ({proc.returncode}); 다음 단계로 넘어가지 마세요')
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL); proc.wait()

probe = "import sys,platform,torch,torchaudio; print(sys.version,torch.__version__,torchaudio.__version__,torch.version.cuda); assert platform.system()=='Linux'; assert sys.version_info[:2]==(3,12), 'Python 3.12 environment required'; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); assert torch.__version__==torchaudio.__version__, 'torch/torchaudio CUDA wheel mismatch'"
command([BASE_PYTHON,'-c',probe])
command([BASE_PYTHON, WORK/'portable.py','--config',CONFIG])
command([BASE_PYTHON,'-c',"import sys,torch; sys.path.insert(0,sys.argv[1]); from common import gpu_profile; print(gpu_profile(torch.cuda.get_device_name(0),sys.argv[2])); assert torch.cuda.get_device_name(0)==sys.argv[3]",WORK,PRECISION,EXPECTED_GPU])
PY = ENV/'bin/python'
if not PY.exists():
    command([BASE_PYTHON,'-m','venv','--system-site-packages',ENV])
    (ENV/'.wavlm_handoff_env.json').write_text(json.dumps(ENV_MARKER))
    source_sites=json.loads(subprocess.check_output([BASE_PYTHON,'-c','import site,json; print(json.dumps(site.getsitepackages()))'],text=True))
    target_sites=json.loads(subprocess.check_output([PY,'-c','import site,json; print(json.dumps(site.getsitepackages()))'],text=True))
    local_site=next(Path(p) for p in target_sites if str(ENV) in p)
    (local_site/'existing_torch_environment.pth').write_text('\\n'.join(source_sites)+'\\n')
command([PY,'-m','pip','install','-r',WORK/'requirements.txt'])
command([PY,'-c',probe])
RUN_ENV=dict(os.environ,HF_HOME=HF_CACHE_DIR,CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')
def run(stage): command([PY,'-u',WORK/'runner.py',stage,'--config',CONFIG],env=RUN_ENV)
''')
    md('## 4. 데이터 경로·27,985 WAV 전수 점검\n파일 이름·용량·헤더·crop 경계 원본 manifest를 확인합니다. 실패가 있으면 목록을 저장하고 중단합니다. 공식 Validation은 사용하지 않습니다.')
    code("command([PY,'-u',WORK/'portable.py','--config',CONFIG,'--check-data'])")
    md('## 5. Preflight·실측 benchmark\n6개 Train 표본 수치 검사 후 frozen/top2 각각 100 optimizer steps(+2 warmup), 전체 caller 60통화 평가를 측정합니다. frozen-only 선택이어도 benchmark는 top2를 짧게 측정한 뒤 가중치를 버립니다. 실제 시간/VRAM/RAM/저장 공간을 확인하고 다음 단계로 진행하세요.')
    code("run('preflight')\nrun('benchmark')\nprint((Path(OUTPUT_DIR)/'benchmark.json').read_text())")
    md('## 6. Train 특징 추출·분류기 선택\n22,388 Train만 사용. inner-train 17,910/dev 4,478의 고정 ID 목록. 완료 통화는 검증 후 재사용합니다.')
    code("run('extract')\nrun('frozen-select')\nprint((Path(OUTPUT_DIR)/'frozen_selection.json').read_text())")
    md('## 7. 선택 사항: Train 청취 검토·상위층 적응\n기본 RUN_ADAPTATION=False이면 건너뜁니다. 실행하려면 audit 결과의 caller/crop을 듣고 audio_review/review.csv를 채운 뒤 다음 셀을 실행하세요. 각 4 epochs가 추가되며 Frozen만의 시간/비용 예상에 포함되지 않습니다.')
    code("if RUN_ADAPTATION:\n    run('audit')\n    print('청취 후 작성할 CSV:',Path(OUTPUT_DIR)/'audio_review/review.csv')")
    code("if RUN_ADAPTATION:\n    run('develop-frozen')\n    run('develop-top2')")
    md('## 8. 설정 잠금·최종 Train 재학습\nInternal Validation을 보기 전에 고정합니다. 최종 LR은 Train 22,388개로 fit. 채택된 top2는 초기 모델부터 선택 epoch만큼 재학습합니다. 잠금 이후 dev 설정 변경은 금지합니다.')
    code("if not (Path(OUTPUT_DIR)/'selection.lock.json').exists():\n    run('lock-adapted' if RUN_ADAPTATION else 'lock-frozen')\nprint((Path(OUTPUT_DIR)/'selection.lock.json').read_text())\nrun('final-fit')")
    md('## 9. Internal Validation 전체 평가·반환 ZIP\n이미 관찰한 5,597통화입니다. F/M argmax, threshold tuning 금지. 전체 완료 후에만 결과 ZIP이 생성됩니다.')
    code("run('validation')\nprint('반환 파일:',Path(OUTPUT_DIR)/'wavlm_results.zip')\nprint((Path(OUTPUT_DIR)/'completion.json').read_text())")
    md('## 10. 팀원 또는 로컬 CPU에서 결과 검증\n아래 경로도 직접 입력합니다. 같은 코드 ZIP을 푼 verify.py를 써야 합니다. 기존 Colab용 verifier와 혼용하지 마세요. torch 없이 표준 Python만으로 검증됩니다.')
    code('''RETURNED_ZIP = ""
VERIFIED_OUTPUT_DIR = ""
assert RETURNED_ZIP.strip() and VERIFIED_OUTPUT_DIR.strip(), '두 절대 경로를 입력하세요'
assert Path(RETURNED_ZIP).is_absolute() and Path(VERIFIED_OUTPUT_DIR).is_absolute()
command([BASE_PYTHON,WORK/'verify.py',RETURNED_ZIP,'--output',VERIFIED_OUTPUT_DIR])
''')
    md('## 중단/재개와 남은 lock\n보통 Stop은 자식까지 종료하고 lock을 정리합니다. 강제 종료 후 lock이 남았다면 소유 프로세스/다른 런타임 종료를 확인하고 token으로 해제합니다. checkpoint는 100 optimizer steps마다 저장하며 미저장 step 재실행 범위를 출력합니다. VM/Pod 종료 전 ZIP과 최종 checkpoint를 보존하세요.')
    code('''def inspect_locks():
    for p in (Path(GPU_LOCK_PATH),Path(OUTPUT_DIR)/'.run.lock'):
        if p.exists(): print(p,p.read_text())
    print(subprocess.check_output(['nvidia-smi'],text=True))
def release_stale_lock(path, token, previous_process_stopped=False):
    assert previous_process_stopped, '이전 실행이 실제 종료됐는지 확인하세요'
    p=Path(path)
    assert p in (Path(GPU_LOCK_PATH),Path(OUTPUT_DIR)/'.run.lock')
    assert json.loads(p.read_text())['token']==token
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    p.unlink()
inspect_locks()
''')
    for i,c in enumerate(cells):c['id']=f'handoff-{i:02d}'
    nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),nbformat=4,nbformat_minor=5)
    atomic(ROOT/'handoff/WavLM_Teammate.ipynb',(json.dumps(nb,ensure_ascii=False,indent=2)+'\n').encode())
