> **과거 인계 기록:** 이 문서는 ECAPA 평가 전에 작성됐다. 현재는 평가·로컬 검증 완료이며,
> 최신 결과는 [README](README.md)와 [Validation REPORT](validation_results/REPORT.md)를 참고한다.
> 아래 실행 요청은 보관용이고 새로운 작업 지시가 아니다.

# 새 창 시작 프롬프트

아래 내용을 새 Codex 창에 붙여넣어 주세요.

---

Mission 1의 성별 분류 실험을 이어서 진행해줘. **사전학습된 성별 분류용 ECAPA의 설정을 고정하고, 기존 Internal Validation 5,597통화를 평가하는 실행 코드를 준비해줘.**

## 작업 위치와 역할

- 저장소: `/Users/joeunkyo/Documents/ChatGPT/AIGO`, GitHub `eunkyo00/DDC2026-AIGO`.
- 브랜치는 `mission1/eunkyo` 유지. 다른 미추적 작업을 지우거나 함께 커밋하지 마.
- GPU 추론은 내가 Colab에서 실행하고, Codex는 코드·결과 검증·로컬 분석·문서 정리를 담당해.
- 전체 실행 코드와 제목이 있는 Colab notebook을 준비해줘. 긴 실행은 실측 기반 예상 시간을 표시해줘.
- 기존 Colab 환경과 Drive를 최대한 재사용해. 불필요한 재설치·재추론을 피하고 출력·오류를 확인하면서 진행해.

## 이미 결정한 다음 단계

Train 200통화 비교를 완료했고 **ECAPA만 선택**했어. 추가 2,000통화 실험과 전체 27,985통화 추론을 생략하고, ECAPA를 현재 설정 그대로 Internal Validation 5,597통화에서 평가할 거야.
추가 LR/Scaler 학습, WavLM 재실행, fine-tuning, threshold tuning, ensemble, 새로운 전처리 탐색을 이 작업에 넣지 마.

먼저 다음 파일을 읽고 기존 코드를 활용해:
- `mission-1/experiments/gender_model_comparison/SCREEN_200_REPORT.md`
- `mission-1/experiments/gender_model_comparison/compare_models.py`
- `mission-1/experiments/gender_model_comparison/screen_200.py`
- `mission-1/experiments/gender_model_comparison/vendor/ecapa_model.py`
- 같은 폴더의 `model_sources.json`, `bundle_manifest.json`, `requirements.txt`, notebook
- `mission-1/validation/split_assignments.csv`
- `mission-1/validation/manifests/calls.csv`
- `mission-1/eda/eda_outputs/segments.csv`
- `mission-1/README.md`, `mission-1/EXPERIMENT_PLAN_99.md`

## 현재 결과

동일 Train 200통화(Male 101 / Female 99), 사용자 제공 Colab 출력 기준:
- 기존 Wav2Vec2 LR OOF: 195/200 = 97.5%, confusion M/F `[[97,4],[1,98]]`.
- 기존 Wav2Vec2+MFCC LR OOF: 195/200 = 97.5%, 동일 confusion.
- Gender ECAPA: 198/200 = 99%, confusion `[[100,1],[1,98]]`.
  Fusion 오답 교정 4, 정답 회귀 1, 공통 오답 1.
- Gender WavLM: 190/200 = 95%, confusion `[[91,10],[0,99]]`.
  Fusion 오답 교정 1, 정답 회귀 6, 공통 오답 4.

200개 결과는 최종 Validation 결과가 아니며 99% 달성을 보장하지 않아.
결과 ZIP 원본은 아직 로컬에서 독립 검증하지 않았으므로, 접근 가능하면 identity/해시/실행 환경부터 확인해.
그 때문에 기존 200개를 다시 추론하지는 마.

## 고정 ECAPA 정의

- checkpoint `JaesungHuh/voice-gender-classifier`
- revision `db1222153bd60337e900be22add7af180452adc0`
- 기존 vendored 모델과 strict checkpoint loading 유지.
- FP32, eval, Frozen, batch size 1. autocast/TF32/FP16/BF16/padded batching 금지.
- 실제 8kHz mono WAV에서 caller 모든 segments를 시간순으로 처리.
- crop 인덱스 `round(start*8000)` / `round(end*8000)`.
- 기존 `resample_poly(up=2, down=1, window=('kaiser',5.0))`, float32.
- 모든 caller crop을 연결하고 `np.array_split`로 최대 15초 균등 창으로 분할.
- 전체 입력이 3초 미만이면 단일 입력을 3초까지 zero-pad하며 기록.
- 창별 softmax 확률을 실제 샘플 수로 가중 평균. 원본 M/F 순서를 F/M으로 변환하고 argmax.
- 음성 길이 절삭·일부 segment 생략 금지. 모델 고유 전처리는 기존 vendor 코드 그대로.

## 데이터와 분할

- 정상 Training 원본 27,985 calls, Train 22,388 / Internal Validation 5,597.
- split SHA256 `04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06`.
- manifest SHA256 `dc57d0f7bea6e06017ac3e27444a0ff5bb299e401ea787400e84cbdf7d072e23`.
- segments SHA256 `8bf8c83c0190232e4500b4067bc28d2611b09dd227a7ce4dc9c3741e6dcce584`.
- Internal Validation Male 2,620 / Female 2,977. 공식 Validation 원본 3,640과 혼동하지 마.
- metadata에서 기존 fixed split에 속한 5,597개의 정확한 call_id와 모든 caller 구간을 구성하고 검증해.
- 기존 `load_jobs()`는 Train 2,000개 전용이므로 그대로 Validation에 사용하면 안 돼.
- 기존 bundle/Train cache identity를 덮어쓰거나 hash 검사를 끄지 마. 별도 Validation runner·manifest·output을 만들어.
- 정답은 평가용으로만 사용하고 모델 입력이나 설정 선택에 쓰지 마.

## 현재 Colab/Drive

- Python `/content/gender_compare_env/bin/python`
- 코드 `/content/gender_model_comparison/`
- 기존 config `/content/gender_model_comparison/config.json`의 실제 data_root를 확인해서 재사용.
- 모델 캐시 `HF_HOME=/content/gender_model_hf_cache`
- 기존 Train 결과 `/content/drive/MyDrive/DDC-Colab/gender_comparison_v1/`
- 200개 결과 `/content/drive/MyDrive/DDC-Colab/gender_comparison_v1_screen200/`
- ZIP `screen200_results.zip`, 요약 `comparison.json`, 실행 정보 `screen_identity.json`.
- Validation 출력은 예를 들어 `/content/drive/MyDrive/DDC-Colab/ecapa_gender_validation_v1/`처럼 별도 폴더를 사용.
- 이전 환경: torch `2.11.0+cu128`, torchaudio는 CUDA 12.8 배포본으로 복구했음.
- 일반 PyPI에서 torchaudio 버전 숫자만 맞추면 CUDA 13 wheel이 설치되어 `libcudart.so.13` 오류가 났음.
  필요 시 공식 cu128 index에서 `torchaudio==2.11.0+cu128 --no-deps`로 설치하며 정상 환경을 매번 재설치하지 마.
- venv의 `include-system-site-packages = true`가 필요했고 pip이 없는 환경은 host pip의 `--python`으로 복구했음.
- 현재 환경은 실제 identity를 읽어서 확인하고, 환경이 바뀌면 기존 캐시와 무조건 합치지 마.

## 실행과 검증 요구

1. 로컬에서 설정·코드·5,597통화 metadata를 확인하고 Validation용 bundle/notebook을 준비해.
2. Colab에서 현재 GPU/환경, 데이터 경로, Drive 저장과 대표 WAV 접근만 필요한 만큼 확인해.
3. 시간 추정은 기존 ECAPA의 실측 시간·오디오 길이와 실제 Drive 상황을 활용해. 긴 full 실행을 Codex가 직접 하지 마.
4. Validation 추론은 완료 call별 확률·예측·call_id·identity·실패 기록을 Drive에 지속 저장하고 resume 가능하게 해.
   중지 시 자식 프로세스도 종료되도록 하고 중복 실행을 막아. 실패를 제외한 채 성공으로 보고하지 마.
5. 완료 후 5,597 unique IDs exact coverage, 두 클래스 확률 shape, NaN/inf, 합=1, argmax 일치, 누락/실패를 검증해.
6. 로컬에서 Overall/Male/Female Accuracy, confusion matrix, 정답/오답 수를 구해.
7. 같은 fixed Validation의 기존 baseline 예측과 call_id로 연결해 MFCC·Wav2Vec2 대비 corrected/regressed/both_wrong를 분석해.
   Train fusion OOF를 Validation 결과처럼 비교하지 마.
8. 기존 baseline: Majority 53.189209%, F0+LR 91.209577%, MFCC+SVM 95.068787%, Wav2Vec2+LR 97.248526%.
   99% 이상은 최소 5,542 정답 / 최대 55 오답. 새 ECAPA 점수는 실제 검증 완료 후 표에 반영해.
9. 원본 WAV·모델 weight·대용량 cache는 GitHub에서 제외하고 재현 코드·metrics·REPORT·README를 정리해.

이번 설정은 평가 전에 고정한다. 이 Validation은 기존 baseline에서 이미 관찰했으므로 새 독립 test라고 주장하지 마.
결과가 낮더라도 Validation 점수에 맞춰 반복 튜닝하지 말고, 후속 개선은 Train 실험으로 분리해.

**지금은 기존 파일 확인부터 시작해서 내가 Colab에 올릴 실행 코드 전체를 준비해줘.**
