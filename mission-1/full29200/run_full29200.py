# Mission 1: WAV+JSON 29,200쌍 전체 재실험. Python 3.10~3.12 권장.
# Windows/Linux NVIDIA GPU에서 실행 권장. Mac/CPU는 --check-data 또는 느린 --device cpu 실행 가능.
# 이 파일에는 개인 경로/계정이 없습니다. 아래 SOURCES와 RUN_DIR만 각자 설정하세요.
#
# 설치 (새 가상환경 권장; 이 폴더에서 실행):
#   python -m venv .venv
#   Windows 활성화: .venv\Scripts\activate
#   Linux/Mac 활성화: source .venv/bin/activate
#   GPU용 PyTorch: https://pytorch.org/get-started/locally/ 에서 OS/GPU에 맞게 설치
#   예: Windows/Linux, RTX 50 시리즈/호환 드라이버:
#       python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
#   python -m pip install -r requirements.txt
#   python run_full29200.py
#
# 파일 수정 없이 실행할 수도 있습니다 (경로에 공백이 있으면 따옴표 필수):
#   python run_full29200.py --source "D:/dataset/Training.zip" --run-dir "D:/results/full29200_01"
#   WAV/JSON ZIP이 따로 있으면 --source를 두 번 지정하세요.
#   원본 검사만: 위 명령 끝에 --check-data (GPU 불필요, 특징 추출/학습은 시작하지 않음).
#   메모리가 작으면: --wavlm-batch-seconds 8 --wav2vec2-batch-seconds 16
#   중단 후: 같은 명령/실행 폴더로 재실행. GPU/코드/버전/정밀도/배치 설정을 유지하세요.
#   새 실험: 새 --run-dir 사용. 과거 27,985개 실험의 특징/모델은 가져오지 않습니다.
#
# 처리: 23,360 학습 + 5,840 내부 검증 → 성능/앙상블 비율 확정 → 제출 LR은 29,200 전체 재학습.
# 출력 accuracy는 재학습 전 검증 모델 기준. 공식 Validation은 사용하지 않습니다.
# 사전학습 WavLM/Wav2Vec2 본체는 고정하며, 특징 추출과 LR 학습을 처음부터 수행합니다.
# 누락/잘못된 라벨이 있으면 전체 실험을 중단합니다. 코드는 없는 음성 파일을 복구하지 않습니다.
#
# GPU 연산에 원본을 네트워크 디스크에서 반복 읽지 않도록 작업 폴더로 복사/압축 해제합니다.
# 작업 디스크는 원본 ZIP+압축 해제 데이터+모델/최종 ZIP 공간이 필요합니다.
# 작업 폴더를 로컬 SSD, 결과 폴더를 지속 저장 디스크로 지정할 수 있습니다.
# 64통화마다 두 모델 특징/체크섬 저장. 현재 미저장 블록만 재계산합니다.
# 첫 64개 이후 두 모델 실측 ETA 갱신. 복사/검사/모델 로딩/분류기 학습/ZIP 시간은 ETA에서 제외.
# 첫 실행은 공개 사전학습 가중치 다운로드용 인터넷 필요. 추론에는 인터넷이 필요 없습니다.
#
# 최종 결과: RUN_DIR/Mission1_full29200_result.zip (model.pt, 추론 코드, 재현 소스, 검증 기록).
# 중간 기록: features/, progress.json, events.jsonl, dataset_check.json, last_failure.json.
# 원본 WAV/JSON은 결과 ZIP에 들어가지 않습니다. 같은 RUN_DIR은 한 프로세스에서만 사용하세요.
# BF16 모델은 BF16 CUDA GPU 필요. CPU 추론도 필요하면 처음부터 PRECISION="fp32" 선택.
# 실제 전체 데이터/GPU 소요 시간·성능은 실행 후 결정됩니다. 합성 CPU 검증만 사전 수행했습니다.
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent

# [필수 설정 1] Training ZIP 또는 압축 해제된 Training 폴더. 둘 중 한 방식만 사용하세요.
# Windows 예시: SOURCES = [r"D:\dataset\Training.zip"]
# Linux 예시:   SOURCES = ["/data/emergency/Training"]
# 분리 ZIP:     SOURCES = [r"D:\dataset\TS_서울_구급.zip", r"D:\dataset\TL_서울_구급.zip"]
SOURCES = []

# [필수 설정 2] 실험 결과를 영구 보관할 폴더. 기본값은 코드 폴더/runs/full29200_01.
# Windows 예시: RUN_DIR = Path(r"D:\results\mission1_full29200_01")
RUN_DIR = HERE / "runs" / "full29200_01"

# [선택] None이면 RUN_DIR/.work, RUN_DIR/.hf_cache. 원본과 겹치지 않는 경로를 사용하세요.
# WORK_DIR = Path(r"E:\ssd_work\mission1_full29200_01")
WORK_DIR = None
HF_CACHE_DIR = None

# [선택] auto=CUDA 없으면 안내 후 중단. CPU는 매우 느리므로 명시적으로 "cpu" 선택.
DEVICE = "auto"          # "auto", "cuda", "cpu"
PRECISION = "auto"       # "auto", "bf16", "fp32" (지원 CUDA=BF16, 그 외=FP32)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description="Mission 1 full 29,200 fresh experiment; settings are documented in code comments.")
    parser.add_argument('--source',action='append',help='Training ZIP/folder; repeat for separate WAV/JSON archives')
    parser.add_argument('--run-dir',type=Path,default=RUN_DIR)
    parser.add_argument('--work-dir',type=Path,default=WORK_DIR)
    parser.add_argument('--hf-cache-dir',type=Path,default=HF_CACHE_DIR)
    parser.add_argument('--device',choices=('auto','cuda','cpu'),default=DEVICE)
    parser.add_argument('--precision',choices=('auto','bf16','fp32'),default=PRECISION)
    parser.add_argument('--wavlm-batch-seconds',type=float)
    parser.add_argument('--wav2vec2-batch-seconds',type=float)
    parser.add_argument('--check-data',action='store_true')
    return parser.parse_args(argv)


@contextlib.contextmanager
def exclusive_run(out):
    # OS 파일 잠금은 프로세스 강제 종료 시 자동 해제됩니다. Windows/Linux/Mac 지원.
    out.mkdir(parents=True,exist_ok=True)
    handle=(out/'.run.lock').open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            handle.seek(0,os.SEEK_END)
            if handle.tell()==0:handle.write(b'0');handle.flush()
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except (OSError,BlockingIOError):
        handle.close()
        raise ValueError('이 결과 폴더에서 이미 다른 실행이 진행 중입니다.') from None
    try:
        yield
    finally:
        handle.close()


def main(argv=None):
    args=arguments(argv)
    config={'sources':[str(p) for p in (args.source or SOURCES)],'output':str(args.run_dir),
            'device':args.device,'precision':args.precision}
    for key in ('work_dir','hf_cache_dir'):
        value=getattr(args,key)
        if value is not None:config[key]=str(value)
    batches={}
    for kind in ('wavlm','wav2vec2'):
        value=getattr(args,kind+'_batch_seconds')
        if value is not None:batches[kind]=value
    if batches:config['batch_seconds']=batches
    if not config['sources']:
        print('SOURCES 주석 예시에 맞춰 경로를 입력하거나 --source를 지정하세요.',file=sys.stderr)
        return 2
    # 로그/진행 파일 저장; 노트북, Google Drive 계정, 외부 서비스 연결은 불필요합니다.
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    try:
        from pipeline import resolve_config, prepare, run
        config=resolve_config(config)
        out=Path(config['output'])
        with exclusive_run(out):
            def progress(message):
                print(message,flush=True)
                with (out/'events.jsonl').open('a',encoding='utf-8') as log:
                    log.write(json.dumps({'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'message':message},ensure_ascii=False)+'\n')
            if args.check_data:
                jobs=prepare(config,progress)
                progress(f'원본 검사 통과: {len(jobs):,}통화. GPU 추출/학습을 시작하지 않았습니다.')
            else:
                result=run(config,progress)
                progress('결과 ZIP: '+str(result))
    except KeyboardInterrupt:
        print('중단되었습니다. 같은 명령으로 다시 실행하면 저장된 블록부터 이어집니다.',file=sys.stderr)
        return 130
    except Exception as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
