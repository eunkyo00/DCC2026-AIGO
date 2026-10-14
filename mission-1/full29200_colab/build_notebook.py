"""Build a compact notebook and uploadable runtime ZIP. No encoded megacells."""
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).parent
OUT=ROOT.parents[1]/'output'/'mission1_full29200_colab'
OUT.mkdir(parents=True,exist_ok=True)


def cell(kind,text):
    result={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':
        result.update(execution_count=None,outputs=[])
        result['metadata']={'cellView':'form','jupyter':{'source_hidden':True},'collapsed':True}
    return result


cells=[cell('markdown', '''# Mission 1 — 29,200통화 처음부터 재실험
**런타임 → 런타임 유형 변경 → GPU**를 선택하고 위에서 아래로 실행하세요.

1. 이 노트북과 함께 받은 **`mission1_runtime.zip`**을 1번 셀에서 업로드합니다.
2. Drive의 원본 Training ZIP/폴더를 버튼으로 선택합니다. 누락 없이 WAV+JSON **29,200쌍**이 필요합니다.
3. 새 실행 이름으로 시작합니다. 중단 후에는 같은 이름을 선택해 재개합니다.
4. 완료 후 결과 ZIP 다운로드 버튼을 누릅니다. Drive에도 자동 보관됩니다.

**기존 27,985개 특징/모델은 사용하지 않습니다.** 공개 사전학습 WavLM-Large/Wav2Vec2-base를 고정하고 특징을 새로 추출합니다.
23,360/5,840통화로 검증 결과를 확정한 뒤, 제출용 LR은 전체 29,200통화로 다시 학습합니다.
표시되는 정확도는 재학습 전 검증 모델의 성능입니다. 공식 Validation은 사용하지 않습니다.

길이별 고정 패딩·GPU 배치·L4/A100 BF16을 사용한 새 처리 방식이라 이전 결과와 동일한 수치를 보장하지 않습니다.
**GPU·Drive에 따라 여러 시간이 걸릴 수 있습니다.** 첫 64개부터 두 모델의 실제 처리 속도로 ETA를 갱신합니다.
데이터 복사·검사·모델 다운로드·최종 ZIP 저장 시간은 특징 추출 ETA에 포함되지 않습니다.
'''),cell('code','''#@title 1. 실행 파일 업로드 (mission1_runtime.zip)
from google.colab import files
from pathlib import Path
import io, zipfile, sys
uploaded = files.upload()
assert len(uploaded) == 1, 'mission1_runtime.zip 하나만 업로드하세요.'
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as z:
    assert all('/' not in n and n not in ('.', '..') for n in z.namelist())
    z.extractall('/content/aigo_runtime')
sys.path.insert(0, '/content/aigo_runtime')
del uploaded
print('실행 파일 준비 완료')
'''),cell('code','''#@title 2. 환경 준비 (설치 로그는 숨김)
import subprocess, os
os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS='1', TOKENIZERS_PARALLELISM='false')
with open('/content/aigo_runtime/install.log', 'w') as log:
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
        '/content/aigo_runtime/requirements.txt'], stdout=log, stderr=log)
assert r.returncode == 0, '설치 실패: 왼쪽 파일 탭에서 install.log 확인'
from importlib.metadata import version
assert 'numpy' not in sys.modules or sys.modules['numpy'].__version__ == version('numpy'), '런타임 → 세션 다시 시작 후 1번부터 실행하세요 (설치된 NumPy 적용).'
print('환경 준비 완료. 다음 셀로 진행하세요.')
'''),cell('code','''#@title 3. Google Drive 연결
from google.colab import drive, output
output.enable_custom_widget_manager()
drive.mount('/content/drive')
print('Drive 연결 완료')
'''),cell('code','''#@title 4. 데이터 선택 · 새 실험 / 재시작 설정
import colab_ui as ui
ui.setup()
'''),cell('markdown','''### 실행 전 확인
- ZIP 하나에 WAV/JSON이 모두 있으면 하나만, 각각의 ZIP이면 둘 다 추가하세요.
- 전체 데이터 폴더를 선택해도 `Validation`, `VS_`, `VL_` 데이터는 제외합니다.
- 하나라도 누락·라벨 오류·서울 외 데이터가 있으면 **GPU 추출 전에 중단**하고 진단 파일을 저장합니다.
- 결과 위치: Drive → **AIGO_Mission1_full29200 → 실행 이름**.
- 64통화씩 두 모델의 특징을 저장합니다. 갑자기 끊기면 진행 중인 블록만 다시 계산합니다.
- 같은 실행 폴더를 두 코랩 런타임에서 동시에 실행하지 마세요.
- 첫 실험은 새 이름을 사용하세요. 같은 실험의 재시작만 해당 이름의 체크포인트를 사용합니다.
'''),cell('code','''#@title 5. 전체 실험 실행 · 자동 저장 · 예상 시간 표시
ui.execute()
'''),cell('code','''#@title 6. 최종 결과 / 중간 기록 다운로드
ui.downloads()
'''),cell('markdown','''### 결과 ZIP
`model.pt`, `inference.py`, `model.py`, `core.py`, `requirements.txt`, `training_source.zip`,
학습/검증 분할표, 전체 데이터 검사 결과, OOF 선택 기록, 평가용/제출용 LR, 검증 예측 CSV, 성능 보고서가 포함됩니다.
원본 WAV/JSON과 특징 캐시는 ZIP에 넣지 않습니다. 특징 캐시는 Drive 실행 폴더에 보관됩니다.

학습은 CUDA가 필요합니다. BF16 체크포인트는 L4/A100 등 BF16 지원 GPU에서 추론하세요.
다운로드가 브라우저에서 막히면 Drive 실행 폴더의 `Mission1_full29200_result.zip`을 직접 다운로드하세요.
코랩 세션이 종료되면 1~4번 셀을 다시 실행하고, 기존 실행 이름을 선택한 뒤 5번 셀을 실행하세요.
''')]
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Mission1_full29200.ipynb'},
          'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
for i,c in enumerate(cells):
    c['id']=f'm1-full-{i:02d}'
(OUT/'Mission1_full29200.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
with zipfile.ZipFile(OUT/'mission1_runtime.zip','w',zipfile.ZIP_DEFLATED) as z:
    z.write(OUT/'Mission1_full29200.ipynb','train.ipynb')
    for p in sorted(ROOT.iterdir()):
        if p.name in ('core.py','pipeline.py','inference.py','colab_ui.py','requirements.txt','README.md'):
            z.write(p,p.name)
with zipfile.ZipFile(OUT/'Mission1_full29200_Colab_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ('Mission1_full29200.ipynb','mission1_runtime.zip'):
        z.write(OUT/name,name)
print(OUT)
