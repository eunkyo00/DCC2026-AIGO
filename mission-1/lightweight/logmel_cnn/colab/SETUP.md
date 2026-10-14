# 새 Google 계정 Colab 준비

이 절차는 경량 CNN 실험만 준비한다. 기존 Frozen Wav2Vec2 노트북이나 cache는 열거나
변경하지 않는다. 새 계정에서 본인 Drive에 복제한 데이터 폴더를 사용한다.

## 1. Colab과 데이터 경로

새 계정으로 [Colab](https://colab.research.google.com/)에 접속해 새 노트북을 만들고,
상단 `런타임 → 런타임 유형 변경`에서 하드웨어 가속기를 `GPU`로 고른다.
선택지가 있으면 T4 GPU로 시작하고, 런타임 버전은 기본 최신을 유지한다.
GPU를 바꾸면 런타임이 재시작될 수 있으므로 그 뒤 Drive를 연결한다.
아래 셀로 Drive를 마운트한다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

`Drive already mounted`는 정상이다. 아래 경로 검사에서 `False`가 나오면
강제 재마운트부터 하지 말고 Drive에서 실제 복제 위치를 찾는다.

```python
from pathlib import Path
mydrive = Path('/content/drive/MyDrive')
print('MyDrive:', mydrive.exists())
print('이름에 대학부가 있는 1단계 폴더:', [str(p) for p in mydrive.iterdir() if '대학부' in p.name])
print('대상 폴더의 바로 아래:', [p.name for p in (mydrive / '대학부 데이터').iterdir()] if (mydrive / '대학부 데이터').is_dir() else '폴더 없음')
```

`대학부 데이터` 아래에 다른 `대학부 데이터`가 한 번 더 있으면 더 안쪽을
`DATA_ROOT`로 지정한다. 드라이브에 복사 진행 중이라면 완료 후 다시 확인한다.
현재 새 계정에서 `MyDrive` 바로 아래에는 `Colab Notebooks`와 `VoCat`만 확인됐다.
다음 셀은 폴더만 최대 4단계 탐색한다. WAV 파일 수만 개를 나열하지 않는다.

```python
from collections import deque
from pathlib import Path
queue = deque([(Path('/content/drive/MyDrive'), 0)])
found = []
while queue:
    folder, depth = queue.popleft()
    if (folder / 'Training/1.원천데이터/TS_서울_구급').is_dir():
        found.append(folder)
        continue
    if depth < 4:
        queue.extend((child, depth + 1) for child in folder.iterdir() if child.is_dir())
print('DATA_ROOT candidates:', found)
assert len(found) == 1, 'Drive 계정과 복제 위치를 확인하세요'
DATA_ROOT = found[0]
```

`found`가 비어 있으면 현재 마운트한 Google 계정의 `MyDrive`에 복제가 보이지 않는다.
Drive 웹 화면에서 복제 완료 상태와 로그인 계정을 확인한다.

Drive에서 `Training/1.원천데이터/TS_서울_구급`이 바로 아래에 있는 폴더를
`DATA_ROOT`로 지정한다. 현재 사용자가 확인한 위치는 `MyDrive/대학부 데이터`다.

```python
from pathlib import Path
print(DATA_ROOT.exists())
print((DATA_ROOT / 'Training/1.원천데이터/TS_서울_구급').exists())
```

런타임 확인 셀:

```python
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
!free -h
!df -h /content
```

둘 다 `True`여야 한다. 공유 드라이브의 바로가기만 만든 경우와 실제 파일을
내 드라이브에 복제한 경우는 다를 수 있으므로 WAV 하나를 직접 열 수 있는지 확인한다.

## 2. 실험 코드 올리기

로컬에서 `python mission-1/lightweight/logmel_cnn/colab/build_bundle.py`로 만든
`artifacts/logmel_colab_bundle.zip`을 Colab 파일 탭에 업로드한다. 이 ZIP은
CNN 코드와 고정 split, 기존 manifest/segment 인덱스, MFCC 예측만 포함한다.
WAV는 포함하지 않는다. ZIP을 업로드한 뒤 다음 셀을 실행한다.

```python
import json, zipfile
from pathlib import Path
bundle = Path('/content/logmel_colab_bundle.zip')
assert bundle.is_file(), 'Colab 파일 탭에 ZIP을 업로드하세요'
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
    z.extractall('/content/logmel_exp')
print('extracted', bundle.stat().st_size, 'bytes')
```

런타임이 재시작되면 Colab에 업로드한 ZIP도 사라질 수 있다. 여러 세션에 걸쳐
작업하려면 ZIP을 본인 Drive의 실험 전용 폴더에도 보관하고 `bundle` 경로를 바꾼다.

## 3. 읽기 전용 사전 점검

```python
import subprocess, sys
root = Path('/content/logmel_exp')
script = root / 'mission-1/lightweight/logmel_cnn/colab/preflight.py'
subprocess.run([sys.executable, str(script), '--data-root', str(DATA_ROOT)], check=True)
```

출력에서 `calls=27985`, `segments=442639`, Train `22388`, Internal Validation
`5597`, `missing_count=0`, `cuda_available=true`를 확인한다. 세 파일을 먼저
확인하는 빠른 점검이며, 전체 27,985개 WAV의 경로를 확인하려면 같은 명령에
`--check-all-paths`를 추가한다. 이 명령은 모델이나 특징을 계산하지 않는다.

Colab의 Torch가 이미 설치됐는지 확인한 다음 필요한 의존성만 설치한다.

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
%pip install -q 'soundfile>=0.12' 'pytest>=7'
```

패키지 설치가 Torch/CUDA 환경을 바꾸지 않았는지 확인하고, 테스트를 실행한다.

```python
subprocess.run([sys.executable, '-m', 'pytest', '-q',
                str(root / 'mission-1/lightweight/logmel_cnn/test_logmel_cnn.py')], check=True)
```

## 4. 저장 공간과 실행 순서

전수 audit 결과는 92,916,809 valid frame, 610,649 chunk다. raw valid frame만
14.87 GB이며, 현재 chunk padding 포함 저장은 약 25.01 GB이다. 실험 전용
Drive 폴더와 `/content` 임시 저장 공간의 여유를 확인한다. Drive에 동일 shard를
백업한다면 저장 공간은 더 필요하다. Drive에 작은 파일 수만 개를 만들지 않는다.

이 셀까지는 Drive 경로, 고정 cohort, GPU, 코드 배포를 확인하는 준비 단계다.
전수 추출 전에 실제 WAV 접근과 Torch 모델 테스트 결과를 확인한다. 데이터 경로가
확인되면 `DATA_ROOT`와 사전 점검 출력을 공유해 추출 경로를 확정한다.
