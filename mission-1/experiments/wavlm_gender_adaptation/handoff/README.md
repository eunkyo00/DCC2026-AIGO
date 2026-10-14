# 팀원 인계 · WavLM 성별 분류

이 문서와 `WavLM_Teammate.ipynb`, `wavlm_teammate_bundle.zip`을 전달하세요.
**코드 ZIP에는 음성이 없습니다.** 원본 Training WAV는 별도로 접근 가능하게 해야 합니다.
기존 T4/L4 Colab ZIP과 이 팀원용 ZIP을 섞지 마세요. 이번 배포본은 새 output identity를 사용합니다.
기존 Colab 코드·노트북·결과를 변경하거나 이어받지 않습니다. 기존 ECAPA/Colab `identity.json`도 필요 없습니다.

## 팀원에게 그대로 보낼 요청

> WavLM-Large frozen 다층 특징+LR 실험을 실행해 주세요. 첨부된 팀원용 노트북과 코드 ZIP을 함께 사용하고,
> 첫 설정 셀의 빈 경로를 실제 서버 기준으로 채워 주세요. Training 원본 27,985개 WAV가 별도로 필요합니다.
> GPU는 현재 L4/T4를 지원하고, T4는 FP32/L4는 BF16 기본입니다.
> 먼저 데이터 확인·preflight·benchmark까지만 실행하고 benchmark.json을 공유해 주세요.
> 실제 예상 시간과 비용 확인 후 frozen 학습·Internal Validation 평가를 진행합니다.
> RUN_ADAPTATION=False가 기본이며, 상위층 학습은 별도 추가 실험입니다.
> 최종 wavlm_results.zip과 verify.py가 만든 REPORT.md/metrics.json을 반환하고,
> 최종 모델/checkpoint는 영구 저장소에 보존해 주세요. Validation 점수에 맞춰 설정을 바꾸지 마세요.

## 보내는 파일과 별도 준비물

| 항목 | 전달/포함 여부 |
|---|---|
| `WavLM_Teammate.ipynb` | 함께 전달; Jupyter에서 열거나 Colab에 업로드 |
| `wavlm_teammate_bundle.zip` + `.sha256` | 함께 전달; notebook이 SHA256 확인 후 코드 경로에 풂 |
| 원본 WAV 27,985개 | **별도 전달 또는 팀원이 기존 데이터 이용**. 약 32.3 GB |
| `data/jobs.json` | ZIP 포함. WAV 상대경로, caller 구간, label, split |
| inner split/ID 목록/원본 metadata hash | ZIP 포함; 새 split 생성 금지 |
| baseline 예측(MFCC/Wav2Vec2/ECAPA) | ZIP 포함; 결과 분석에서만 사용, 모델 선택에 사용 금지 |
| 기존 EDA/Validation CSV 전체·원본 JSON | 실행에 추가로 필요 없음. 필요한 정보는 jobs/inventory에 포함 |
| 사전학습 model weights | ZIP 미포함. 실행 시 pinned 공개 checkpoint를 HF에서 다운로드 |
| 기존 Drive identity·예전 모델·cache | 필요 없음; 새 실행 기록 생성 |

WAV의 정확한 필수 목록과 파일 크기는 `data/wav_inventory.csv`에 있습니다.
`data_root` 밑에 manifest의 `Training/.../*.wav`가 그대로 있어야 합니다.
예를 들어 실제 경로를 X라고 할 때 `X/Training/...` 구조면 DATA_ROOT에는 X를 지정합니다.
폴더명/파일명을 임의로 바꾸지 말고, 공식 Validation 3,640개는 사용하지 않습니다.
Training 원본의 추가 파일 2개가 같이 있어도 runner는 지정된 정상 27,985개만 읽습니다.

원본 음성은 작성자 로컬에서 접근할 수 없어 이번 전달 ZIP에 넣지 못했습니다.
팀원이 데이터가 없다면 사용자가 접근 가능한 Drive/원본 저장소에서 파일을 별도로 전달해야 합니다.
이미 팀원 서버에 같은 Training 구조가 있으면 재업로드하지 않습니다.
새 WAV ZIP이 필요한 경우 `pack_audio.py`로 **발신자 환경에서** 필요한 27,985개만 묶을 수 있습니다.
모든 경로는 아래처럼 비워 두었습니다. 빈 값인 채로 실행하면 거부됩니다.

```bash
DATA_ROOT=""
AUDIO_ZIP=""
CODE_DIR=""
: "${DATA_ROOT:?원본 Training 상위 절대 경로를 입력하세요}"
: "${AUDIO_ZIP:?생성할 ZIP 절대 경로를 입력하세요}"
: "${CODE_DIR:?코드 ZIP을 푼 절대 경로를 입력하세요}"
python3 "$CODE_DIR/pack_audio.py" --data-root "$DATA_ROOT" --output "$AUDIO_ZIP"
```

이 ZIP은 CPU 압축 시간을 줄이기 위해 무압축 ZIP64이며 약 32.3 GB입니다.
생성 시 원본 외 추가 33GB 이상, 수신 후 압축파일과 풀린 파일을 함께 보관하면 약 65GB가 필요합니다.
코드·weights·checkpoint 여유까지 포함해 데이터 저장소 80GB 정도를 확보하거나 압축파일 보관 위치를 분리하세요.
추가 `.sha256`도 함께 전달해 수신자가 WAV ZIP 무결성을 확인합니다.
기존 원본 압축파일을 전달할 수도 있지만, 풀린 뒤 아래 전수 점검은 반드시 수행합니다.

## 반드시 지정할 항목 — 실제 경로는 모두 빈칸

| notebook 변수 | 입력할 값 |
|---|---|
| `BUNDLE_PATH` | 서버에 업로드한 코드 ZIP의 절대 경로: ` ` |
| `WORK_DIR` | 코드 ZIP을 풀 새 폴더: ` ` |
| `BASE_PYTHON` | 정상 CUDA torch/torchaudio가 설치된 Python 실행파일: ` ` |
| `VENV_DIR` | 새 전용 Python 환경 폴더: ` ` |
| `DATA_ROOT` | `Training/` 바로 위 원본 데이터 폴더: ` ` |
| `OUTPUT_DIR` | 결과·cache·checkpoint가 영구 보존될 새 폴더: ` ` |
| `HF_CACHE_DIR` | model 다운로드 cache 폴더: ` ` |
| `GPU_LOCK_PATH` | 서버에서 실험끼리 공유할 잠금 파일의 절대 경로: ` ` |
| `EXPECTED_GPU` | 실제 출력된 GPU 이름: ` ` |
| `PRECISION` | 실제 GPU에 맞춰 입력: ` ` |

값을 채우지 않으면 데이터 읽기/학습 전에 중단합니다. `config.template.json`도 경로가 전부 빈칸입니다.
노트북이 실행용 `config.json`을 생성합니다. template나 bundle 내부 파일을 수정하면 hash 검증이 실패하므로
**경로는 notebook 첫 설정 셀 또는 template를 복사한 별도 config.json에만 작성**하세요.
GPU lock의 부모 폴더는 미리 존재해야 합니다. 같은 GPU의 다른 실험도 같은 lock 파일 경로를 써야 합니다.
Data/code/output/HF cache를 분리합니다. 원본 데이터나 코드 폴더 안에 결과·HF cache를 만들지 않습니다.
재개 시 경로·GPU·정밀도·패키지·모델 identity가 같아야 합니다. 바꾸려면 새 output으로 처음부터 시작합니다.

## 환경 확인

- Linux / Python **3.12** / 정상 CUDA torch + 같은 버전·CUDA 빌드의 torchaudio.
- 현재 허용 GPU: `NVIDIA L4`/`L4`, `Tesla T4`/`NVIDIA T4`/`T4`.
  A40/4090/A5000 등 다른 GPU는 현재 검사에서 거부하므로 해당 GPU를 대여하기 전에 코드 지원을 별도로 조정해야 합니다.
- L4: `bf16` 또는 `fp32`; T4: **`fp32`만**. FP16은 이 배포본에 구현하지 않았습니다.
- 정상 torch는 재설치하지 않습니다. notebook은 기존 torch를 읽는 별도 venv에 비 torch requirements만 설치합니다.
  torch가 없거나 torchaudio CUDA wheel이 다르면 환경 제공자가 맞춘 뒤 재실행합니다.
- 이전 Colab의 torch `2.11.0+cu128`은 참고 이력입니다. 팀원 환경에 강제 설치하지 않습니다.
  실제 버전을 새 identity에 기록하며 GPU 실행 호환성은 preflight/benchmark로 확인합니다.
- 외부 네트워크: PyPI 패키지 설치 및 공개 Hugging Face 모델 다운로드 가능해야 합니다.
- RAM: 실행 전 가용 5 GiB 이상 검사. 시스템 RAM 16GB 이상을 권장 계획값으로 잡되 실제 RSS는 측정합니다.
- 원본 WAV 별도 약 32.3GB, pooled cache 약 0.85GiB; output 가용 8GiB/HF cache 가용 3GiB 이상 검사.
- 중복 GPU 작업 금지. GPU 프로세스·host lock·output lock을 점검합니다.
- Runpod 등 대여 서버라면 OUTPUT_DIR를 종료 정책에 맞는 영구 저장소로 지정합니다. notebook Stop은 대여를 종료하지 않습니다.

## 실행 순서

1. 원본 WAV 접근/업로드·압축 해제. 코드 ZIP/노트북 업로드.
2. 노트북 0번에서 실제 Python/GPU를 확인하고 1번 빈칸 작성.
3. ZIP 검증 → 별도 venv 및 패키지 확인.
4. CPU로 **27,985 WAV 전수** 파일명·용량·8kHz/mono/PCM16/frames 확인. `dataset_check.json` 저장.
   이는 원본 metadata에 기록된 헤더/용량 검증이며, 원본 waveform 전체 hash 일치 주장과는 다릅니다.
5. Preflight 및 frozen/top2 각 **100 optimizer steps + 60통화 평가 benchmark**. 모델은 benchmark 후 폐기.
   `RUN_ADAPTATION=False`여도 짧은 top2 benchmark를 측정한다는 점에 주의하세요.
6. `benchmark.json`의 실제 ETA/메모리/저장량 확인 후 Train 특징 추출·LR 후보 비교.
7. 기본은 frozen-only. 필요한 경우에만 Train 청취 검토 후 frozen neural control/top2 적응 비교.
8. 설정 잠금 → 전체 Train 22,388개 재학습 → Internal Validation 5,597개 평가.
9. `wavlm_results.zip` 반환. 같은 팀원용 ZIP의 `verify.py`로 CPU 검증·비교.

CLI 사용자는 config.template.json을 config.json으로 복사하고 빈칸을 채운 뒤 아래와 같이 실행합니다.
전용 환경 설치는 노트북 3번과 동일하게 해야 합니다.

```bash
PYTHON_BIN=""
CODE_DIR=""
CONFIG_PATH=""
: "${PYTHON_BIN:?전용 Python 절대 경로}"
: "${CODE_DIR:?코드 폴더 절대 경로}"
: "${CONFIG_PATH:?작성한 config.json 절대 경로}"
"$PYTHON_BIN" "$CODE_DIR/portable.py" --config "$CONFIG_PATH" --check-data
"$PYTHON_BIN" "$CODE_DIR/runner.py" preflight --config "$CONFIG_PATH"
"$PYTHON_BIN" "$CODE_DIR/runner.py" benchmark --config "$CONFIG_PATH"
# 실측 확인 후 실행
"$PYTHON_BIN" "$CODE_DIR/runner.py" extract --config "$CONFIG_PATH"
"$PYTHON_BIN" "$CODE_DIR/runner.py" frozen-select --config "$CONFIG_PATH"
"$PYTHON_BIN" "$CODE_DIR/runner.py" lock-frozen --config "$CONFIG_PATH"
"$PYTHON_BIN" "$CODE_DIR/runner.py" final-fit --config "$CONFIG_PATH"
"$PYTHON_BIN" "$CODE_DIR/runner.py" validation --config "$CONFIG_PATH"
```

CLI도 `dataset_check.json` 없이는 preflight를 시작하지 않습니다. 원본 `runner.py` 대신 이 코드 ZIP의 runner를 사용하세요.

## 변경하지 않을 실험 조건

- microsoft/wavlm-large revision `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`; 외부 gender checkpoint와 다름.
- 기존 split Train 22,388/Internal Validation 5,597 유지. 공식 Validation 사용 금지.
- inner-train/dev **17,910/4,478**, seed42의 성별별 SHA256 정렬 분할. 같은 call의 segment 분리 금지.
- `inner_split.csv` SHA256 `b6664bee9b72dd14a53743388274577d6c5b73e5ef5f51ab4d53da6fa3321e17`.
- F0: layer24 mean / F1: layer6·12·18·24 mean/std concat, 각각 LR C=.1/1 (4개 후보).
- 시간순 모든 speaker=1 구간, round(t*8000), 구간별 polyphase 8→16k resample, caller 전체 연결.
- 평가: 전체 caller를 최대12초 균등 창, 모델 고유 normalization, 실제 sample 수 가중 moment pooling; tail 생략 없음.
- 신경망 학습 crop 기본12초, 최대4epochs. 이는 dev와 비용 기준의 별도 실험이며 frozen-only 필수 단계가 아님.
- BF16/FP32 모두 scaler 비활성 상태 저장; gradient checkpointing 기본 미사용, accumulation4/microbatch1.
- F/M argmax(동률 F); Internal Validation checkpoint 선택/early stopping/threshold tuning 금지.
- 기존에 관찰한 Internal Validation이며 독립 test가 아님. 99% 달성 보장 없음.

## 중단·재개·반환

통화별 결과/실패/identity를 저장하고 전체 5,597 unique ID가 모여야 완료합니다.
학습 checkpoint는 model trainable delta/head·optimizer·scheduler·AMP scaler·RNG·epoch/step·sampler 상태 포함.
100 optimizer steps/epoch 끝 기준으로 저장하며, 미저장 step은 재개 시 재실행 범위를 로그에 명시합니다.
일반 notebook Stop은 process group을 종료합니다. 강제 종료 뒤 남은 lock은 실제 이전 실행 종료 확인 후 token으로 해제합니다.
손상된 journal/미완료 cache는 조용히 무시하지 않고 거부합니다.

반환할 파일:
- **`OUTPUT_DIR/wavlm_results.zip`**: identity·환경·실측·설정·5,597 통화 결과·실패 이력.
- **`benchmark.json`**: 전체 실행 전에 먼저 공유.
- verifier의 **`REPORT.md`, `metrics.json`, `call_comparison.csv`**.
- 재현용으로 **`final_lr.joblib`**, 채택된 적응 모델의 **`final_top2/latest.pt`**와 원본 output 기록을 보존.
  weights는 결과 ZIP에 자동 포함되지 않으므로 필요한 경우 별도 전달. 저장소 Git에는 넣지 않습니다.

반환 ZIP의 로컬 분석도 사용자 경로는 빈칸으로 둡니다.

```bash
CODE_DIR=""
RESULT_ZIP=""
VERIFIED_OUTPUT=""
: "${CODE_DIR:?팀원용 코드 ZIP을 푼 경로}"
: "${RESULT_ZIP:?반환 ZIP 절대 경로}"
: "${VERIFIED_OUTPUT:?분석 결과를 쓸 절대 경로}"
python3 "$CODE_DIR/verify.py" "$RESULT_ZIP" --output "$VERIFIED_OUTPUT"
```

검증은 source/bundle/identity/hash·exact ID coverage·F/M 확률/argmax·NaN/inf·실패·전체 caller samples를 확인합니다.
Overall/Male/Female accuracy, M/F confusion, MFCC/Wav2Vec2/ECAPA 대비 corrected/regressed/both_wrong를 계산합니다.
실제 GPU 시간/성능은 아직 미측정입니다. 약20시간/2만원은 2초/통화 등의 가정에 기반한 예산 예시이며 실행 계약이나 보장값이 아닙니다.
