# 아래 5개 경로만 실제 서버의 절대 경로로 입력하세요.
# 원본 WAV는 별도 준비: DATA_ROOT 바로 아래에 Training/ 폴더가 있어야 합니다.
# INPUT_ZIP: 따로 전달받은 wavlm_gender_inputs.zip (원본 음성 미포함).
INPUT_ZIP = ""
DATA_ROOT = ""
OUTPUT_DIR = ""
HF_CACHE_DIR = ""
GPU_LOCK_PATH = ""  # 같은 서버의 실험들이 공유할 잠금 파일 경로. 부모 폴더는 미리 준비하세요.

# Linux / Python 3.12 / 정상 CUDA torch+torchaudio 환경에서 실행하세요.
# L4는 BF16, T4는 FP32를 자동 선택합니다. GPU 두 종류만 지원합니다.
# 가상환경은 OUTPUT_DIR/.venv에 자동 생성하며 기존 torch는 재설치하지 않습니다.
# 실행: python3 run.py
# 기본 실행은 frozen 특징+LR입니다. 상위층 적응 학습은 자동으로 하지 않습니다.
# 결과: OUTPUT_DIR/wavlm_results.zip, OUTPUT_DIR/verified/REPORT.md
# 중단 후 같은 명령을 실행하면 완료 단계/통화는 확인 후 재사용합니다.
# 학습 checkpoint 사이의 미저장 step은 재실행 범위를 기록합니다.
# 실제 GPU 속도/메모리는 미측정입니다. benchmark 결과를 확인하세요.
# 실행 프로세스를 멈춰도 대여 GPU 요금은 서비스에서 별도로 정지해야 합니다.
