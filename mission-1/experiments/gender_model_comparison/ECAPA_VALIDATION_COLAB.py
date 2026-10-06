# %% [markdown]
# # Mission 1 · 고정 ECAPA Internal Validation 5,597통화
#
# Train 200통화 비교에서 선택한 ECAPA 설정을 그대로 평가합니다. 추가 학습이나 설정 탐색은 없습니다.
# Male 2,620 / Female 2,977, 기존에 관찰한 fixed Internal Validation입니다.
#
# `ecapa_validation_bundle.zip`을 **내 드라이브/DDC-Colab/**에 올린 후 위에서 아래로 실행하세요.
# Colab 런타임 유형을 **Python 3 · L4 GPU**로 선택하세요. 기존 환경·Drive 원본·모델 캐시를 사용합니다. 기존 Train 실행을 먼저 중지하세요.
# 오류가 나면 아래 셀로 넘어가지 말고 출력을 전달하세요. 추론 완료 후 ZIP을 로컬로 전달하면 검증·비교합니다.

# %% [markdown]
# ## 1. Drive 연결 · 별도 Validation 코드 설치
# 기존 Train 코드·config·결과를 덮어쓰지 않습니다.

# %%
from google.colab import drive
from pathlib import Path
import hashlib, json, os, signal, subprocess, sys, time, zipfile

drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/DDC-Colab')
BUNDLE = BASE / 'ecapa_validation_bundle.zip'
WORK = Path('/content/ecapa_gender_validation')
EXPECTED_SHA256 = 'dfa8b82630038807a32cdbe4b9d80cdc48dc8bbb3e65a36db636a8c978fef25b'
assert BUNDLE.is_file(), f'업로드할 위치: {BUNDLE}'
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == EXPECTED_SHA256, 'Notebook과 ZIP 버전 불일치'
WORK.mkdir(exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    assert len(z.namelist()) == len(set(z.namelist()))
    assert all((WORK/n).resolve().is_relative_to(WORK.resolve()) for n in z.namelist())
    z.extractall(WORK)
SOURCE = BASE / 'gender_comparison_v1'
SOURCE_IDENTITY = json.loads((SOURCE/'identity.json').read_text())
OUTPUT = BASE / 'ecapa_gender_validation_v1'
old_config = Path('/content/gender_model_comparison/config.json')
previous = json.loads(old_config.read_text()) if old_config.exists() else SOURCE_IDENTITY
DATA_ROOT = previous['data_root']
assert DATA_ROOT == SOURCE_IDENTITY['data_root'], '기존 config와 실행 identity의 원본 경로 불일치'
CONFIG = WORK / 'validation_config.json'
CONFIG.write_text(json.dumps(dict(data_root=DATA_ROOT, source_output=str(SOURCE), output_dir=str(OUTPUT)), ensure_ascii=False))
print('원본:', DATA_ROOT, '\n결과:', OUTPUT)


# %% [markdown]
# ## 2. 기존 환경 확인 · 필요한 경우에만 복구
# 정상 환경에는 설치하지 않습니다. 새 런타임은 기존 identity 버전을 기준으로 복구하며, 다른 GPU/Python/패키지를 캐시에 혼합하지 않습니다.

# %%
def command(argv, env=None):
    # A new process group lets notebook Stop terminate the worker and descendants.
    def parent_death_signal():
        import ctypes
        parent = os.getppid()
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != parent:
            os.kill(os.getpid(), signal.SIGTERM)
    proc = subprocess.Popen(list(map(str, argv)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env, start_new_session=True,
                            preexec_fn=parent_death_signal)
    try:
        for line in proc.stdout:
            print(line, end='', flush=True)
        if proc.wait():
            raise RuntimeError(f'실행 실패 ({proc.returncode}). 출력 오류를 전달하세요.')
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

ENV = Path('/content/gender_compare_env')
PY = ENV / 'bin/python'
if not PY.exists():
    command([sys.executable, '-m', 'venv', '--system-site-packages', ENV])
cfg = ENV / 'pyvenv.cfg'
if 'include-system-site-packages = true' not in cfg.read_text().lower():
    import re
    value, n = re.subn(r'(?im)^include-system-site-packages\s*=\s*.*$', 'include-system-site-packages = true', cfg.read_text())
    assert n == 1
    cfg.write_text(value)
command([PY, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA required"; name=torch.cuda.get_device_name(0); print("GPU:", name); assert name in ("NVIDIA L4", "L4"), "L4 GPU required"'])
# Restore only missing/mismatched non-torch packages to the frozen Train environment.
expected = dict(line.strip().split('==') for line in (WORK/'requirements.txt').read_text().splitlines() if line.strip())
expected.update(SOURCE_IDENTITY['versions'])
repairs = []
for package, version in expected.items():
    if package in ('torch', 'torchaudio'):
        continue
    check = subprocess.run([PY, '-c', f'import importlib.metadata; print(importlib.metadata.version({package!r}))'], capture_output=True, text=True)
    current = check.stdout.strip() if check.returncode == 0 else 'missing'
    if current != version:
        print(f'Restore {package}: {current} -> {version}', flush=True)
        repairs.append(f'{package}=={version}')
if repairs:
    command([sys.executable, '-m', 'pip', '--python', PY, 'install', '--no-deps', *repairs])
for package, version in expected.items():
    if package in ('torch', 'torchaudio'):
        continue
    current = subprocess.check_output([PY, '-c', f'import importlib.metadata; print(importlib.metadata.version({package!r}))'], text=True).strip()
    assert current == version, f'Package restoration failed: {package}: {current} != {version}'
torch_version = subprocess.check_output([PY, '-c', 'import torch; print(torch.__version__)'], text=True).strip()
assert torch_version == SOURCE_IDENTITY['versions']['torch'], f'torch 변경: {torch_version}'
check = subprocess.run([PY, '-c', 'import torchaudio; print(torchaudio.__version__)'], capture_output=True, text=True)
if check.returncode:
    print(check.stderr)
    assert torch_version == '2.11.0+cu128', '자동 torchaudio 복구 대상 환경 아님'
    command([sys.executable, '-m', 'pip', '--python', PY, 'install', '--no-deps', '--force-reinstall',
             'torchaudio==2.11.0+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128'])
RUN_ENV = os.environ.copy()
RUN_ENV['HF_HOME'] = '/content/gender_model_hf_cache'
RUN_ENV['TOKENIZERS_PARALLELISM'] = 'false'
# Detect surviving old inference before starting another GPU job.
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        args = (p/'cmdline').read_bytes().decode().split('\0')
    except (OSError, UnicodeDecodeError):
        continue
    assert not any(Path(a).name in ('compare_models.py','screen_200.py','ecapa_validation.py') for a in args if a), f'기존 추론 프로세스 {p.name} 먼저 중지'
def run(stage):
    command([PY, '-u', WORK/'ecapa_validation.py', '--config', CONFIG, '--stage', stage], env=RUN_ENV)
print('환경 준비 완료. 다음 preflight에서 전체 identity를 확인합니다.')


# %% [markdown]
# ## 3. Preflight · 기존 실측 기반 시간 예상
#
# 대표 WAV 3개(길이 최소·중앙·최대), GPU/패키지/고정 설정, Drive 쓰기를 확인합니다.
# 기존 ECAPA screen200 또는 benchmark 기록의 hash와 시간을 읽어 통화 수·오디오 길이 기준 예상을 표시합니다.
# 기존 기록에는 Drive flush와 모델 로딩이 빠져 있으므로 참고값입니다. 이 단계에는 GPU 추론이 없습니다.

# %%
run('preflight')


# %% [markdown]
# ## 4. 5,597통화 추론 · 동일 셀로 resume
#
# 위 실측 예상치를 확인하고 실행합니다. 첫 통화와 이후 매 10통화마다 **Drive 저장 포함** 실측 잔여 시간을 갱신합니다.
# 완료된 통화는 재추론하지 않습니다. 설정·환경·가중치가 다르면 재개를 거부합니다.
# Stop은 자식 프로세스 그룹도 종료합니다. 런타임 강제 소실은 lock이 남을 수 있습니다(아래 복구 안내).
# 정확도는 여기서 설정 선택에 사용하지 않고 완료 후 로컬에서 계산합니다.

# %%
run('run')


# %% [markdown]
# ## 5. 완료 결과 ZIP 다운로드 · 로컬 전달
# 5,597개 exact coverage 검사를 통과해야 ZIP이 생성됩니다.

# %%
from google.colab import files
result = OUTPUT / 'ecapa_validation_results.zip'
assert result.is_file(), '전체 검증 완료 ZIP이 없습니다.'
print('결과 ZIP:', result)
print('SHA256:', hashlib.sha256(result.read_bytes()).hexdigest())
files.download(str(result))
old_zip = BASE / 'gender_comparison_v1_screen200/screen200_results.zip'
if old_zip.exists():
    print('Train 200 원본도 로컬 독립 검증에 전달 가능:', old_zip)


# %% [markdown]
# ## 중단·복구 안내
#
# 정상 Stop 후에는 4번을 다시 실행합니다. 새 런타임에서는 1–3번 후 4번을 실행합니다.
# 완료 통화는 `predictions.jsonl`, 실패/중단은 `failure_attempts.jsonl`, 실행 이력은 `events.jsonl`에 남습니다.
# SIGKILL/런타임 소실은 실패 로그를 남길 수 없지만, exact coverage로 미완료 통화를 재개합니다.
#
# `.validation.lock` 오류: 먼저 이전 notebook/런타임의 추론이 종료되었는지 확인하세요.
# `OUTPUT/.validation.lock/owner.json`의 host·pid·token을 확인하고, **이전 실행이 끝난 경우에만** 아래 함수를 직접 호출합니다.
# 다른 런타임이 살아 있는지 자동 판정할 수 없으므로 시간 경과만으로 잠금을 해제하지 않습니다.
# Drive lock은 동기화 지연이 있는 여러 VM에서 분산 트랜잭션을 보장하지 않습니다. 같은 OUTPUT은 한 런타임에서만 실행하세요.

# %%
def release_stale_lock(expected_token, previous_runtime_stopped=False):
    assert previous_runtime_stopped, '이전 런타임 종료 확인이 필요합니다.'
    lock = OUTPUT / '.validation.lock'
    owner = json.loads((lock/'owner.json').read_text())
    assert owner['token'] == expected_token, 'Lock 소유자가 바뀌었습니다.'
    import socket
    if owner['host'] == socket.gethostname():
        assert not (Path('/proc')/str(owner['pid'])).exists(), '이전 프로세스가 아직 존재합니다.'
    with (OUTPUT/'lock_releases.jsonl').open('a') as f:
        f.write(json.dumps(dict(owner=owner, released_utc=time.time())) + '\n')
    (lock/'owner.json').unlink()
    lock.rmdir()
# owner를 확인한 뒤에만 수동 호출:
# print((OUTPUT/'.validation.lock/owner.json').read_text())
# release_stale_lock('확인한 token', previous_runtime_stopped=True)
