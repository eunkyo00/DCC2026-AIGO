# Mission 2 WavLM 4층 로컬 GPU 학습 전달용

NVIDIA GPU가 있는 Windows/Linux PC에서 실행합니다. Colab/Google Drive 마운트는 필요 없습니다. **새 실험이며 94% 달성 결과가 아닙니다.** 기존 ResNet 결과를 대체하지 않습니다.

## 전달할 파일 2개

| 파일 | 받는 방법 | 용도 |
|---|---|---|
| [mission02_wavlm4_local.zip](mission02_wavlm4_local.zip) | 이 폴더에서 다운로드 | 모델·학습·추론 코드, LOCAL_GPU.md, 테스트 |
| `mission02_wavlm4_data.zip` | 데이터 접근 권한이 있는 팀원에게 비공개로 별도 전달 | Training에서 분리한 학습 100,000발화와 내부 검증 4,000발화 |

**데이터 ZIP은 이 저장소에 업로드하지 않았습니다.** 약 1.9GiB이며, 실제 음성 파형과 발화 ID를 포함합니다. 일반 GitHub 파일 제한(100MiB)을 넘고 저장소의 기존 데이터 재배포 제외 원칙에 따라 비공개 전달 대상으로 유지합니다. 공개 Drive 링크, Git LFS 또는 분할 업로드로 우회하지 않습니다. 코드 ZIP만 받으면 학습을 실행할 수 없습니다.

## 실행 부탁할 때 전달할 내용

> NVIDIA GPU가 있는 PC에서 Mission 2 WavLM 학습 부탁드립니다.
> 코드 ZIP을 별도 폴더에 풀고 같은 폴더에 비공개로 전달받은 데이터 ZIP을 넣어 주세요.
> 코드 ZIP 안의 `LOCAL_GPU.md`에 따라 Python 가상환경과 GPU에 맞는 CUDA PyTorch를 설치해 주세요.
> 해당 폴더 터미널에서 아래 명령을 순서대로 실행하면 됩니다.

```bash
python -m pip install -r requirements.txt
python run_local.py doctor
python run_local.py unpack --zip mission02_wavlm4_data.zip
python run_local.py benchmark
python run_local.py train
```

`doctor`가 실제 GPU 연산을 확인합니다. GPU 드라이버 또는 CUDA PyTorch가 맞지 않으면 먼저 설치를 수정해야 합니다. PyTorch 설치 명령은 [공식 설치 안내](https://pytorch.org/get-started/locally/)에서 해당 GPU/OS에 맞춰 선택합니다. 코드 ZIP은 파일들이 최상위에 있으므로 별도 작업 폴더를 만든 뒤 풀어 주세요.

### 기본 학습 및 재개

- batch size 16, 최대 6 epoch, 내부 검증 4회 연속 미개선 시 조기 종료.
- WavLM 앞 4개 Transformer 층까지만 실행. 초기에는 집계층·분류기, 이후에는 3·4층도 미세조정.
- 최종 임계값 **0.5 고정**. 내부 검증은 Training 통화에서 분리하며 공식 Validation은 학습/모델 선택에 사용하지 않음.
- 로컬에서는 30분 시간 제한 없이 실행. 절전/잠자기 설정을 확인할 것.
- 저장 위치 `outputs/wavlm4`. 데이터 압축 해제 공간 외에 체크포인트용 최소 3GiB 및 초기 모델 다운로드 여유 공간 필요.
- 중단 후 `python run_local.py train`으로 마지막 저장 지점부터 재개. 마지막 저장 이후 최대 100 step은 다시 계산할 수 있음.
- GPU 메모리가 부족하면 **새 실행 폴더에서** 아래처럼 시작. 기존 실행의 batch size를 바꿔 이어갈 수는 없음.

```bash
python run_local.py train --batch-size 8 --output outputs/wavlm4_bs8
```

Colab 체크포인트를 이어 쓰려면 ZIP 안의 `LOCAL_GPU.md`의 3-B 절을 따릅니다. Drive 용량 초과가 발생했으므로 정상 파일인지 먼저 검사해야 합니다. `.tmp`를 재개 파일로 사용하지 말고 원본과 이전 사본을 보존하세요.

## 학습 완료 후 돌려받을 자료

1. **사용한 output 폴더 전체** (기본 `outputs/wavlm4`): best_model.pt, last_checkpoint.pt, history.json 등.
2. **실제로 사용한 코드**: 변경했다면 변경본 포함.
3. 마지막 학습 로그와 `doctor` 출력: GPU, Python, PyTorch, CUDA 정보.
4. 환경 패키지 목록: `python -m pip freeze > requirements-lock.txt`.

코드와 결과는 최종 검증을 마칠 때까지 삭제하지 않습니다. 가중치와 결과 폴더도 비공개로 전달하며 이 GitHub 폴더에 무작정 업로드하지 않습니다.

## 공식 평가와 최종 제출은 별도 단계

현재 데이터 ZIP에는 공식 Validation이 없습니다. 출력되는 INTERNAL DEV 정확도는 최종 공식 성능이 아닙니다. 공식 전체 평가에는 담당자가 별도로 준비한 `mission02_wavlm4_official.zip`이 필요하며, 준비/실행 방법은 ZIP 안의 `LOCAL_GPU.md` 4절에 있습니다.

최종적으로 실제 학습 가중치와 코드로 아래 명령의 실행 및 CSV를 검증해야 합니다.

```bash
python inference.py --audio_dir 입력음성폴더 --label_dir 입력JSON폴더 --ckpt_path outputs/wavlm4/best_model.pt --output outputs/mission2.csv
```

CSV 열은 `audio file name,startAt,endAt,speaker`입니다. 음성 구간 추출에 startAt/endAt을 사용하며 text·발화 순서 등을 분류 입력에 넣지 않습니다. 추론은 로컬 가중치/config를 사용하도록 구성했지만, 실제 전달받은 가중치의 오프라인 실행 및 새 환경 재현 확인을 마쳐야 제출 완료라고 할 수 있습니다.

## 검증 범위

로컬 소규모 테스트에서 4층 출력, 학습/재개, 오프라인 CSV 추론을 확인했습니다. 다른 GPU PC에서 전체 학습하거나 공식 Accuracy 94%를 검증한 상태는 아닙니다. 포함한 ZIP은 사용자가 전달한 Desktop 코드 ZIP 원본입니다.
