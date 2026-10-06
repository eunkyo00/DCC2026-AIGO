# 기존 embedding 기반 분류기 비교

[실험 이력](../../EXPERIMENT_LOG.md)에 기록한 A1~A3의 실행 코드다.
현재 후속 후보는 [99% 목표 계획](../../EXPERIMENT_PLAN_99.md)에 정리했다.
**CPU만 사용하며 Colab GPU·WAV 재추출·파일 재업로드가 필요하지 않다.**
기존 baseline 결과는 보존한다. 이 코드는 Train 내부 CV만 실행하고
Internal Validation 5,597개에 대한 예측·평가는 수행하지 않는다.

현재 상태: **A1~A3 총 24회 CV와 정량 오답 분석 완료**, 청취·holdout 평가는 미실시.
→ [전체 비교·오답 분석](ERROR_ANALYSIS.md) · [A1 단계 기록](A1_REPORT.md)

## 입력과 검증

- `wav2vec2_l4_full_embeddings.npz`: 검증된 전체 L4 export, SHA256 고정.
- `baseline/mfcc_svm/results/call_features.csv`: 기존 MFCC 특징 54개.
- `validation/split_assignments.csv`: 기존 fixed split, SHA256 고정.

`call_id` 기준으로 결합해 중복·누락·정답·partition·segment 수·실패 수를 확인한다.
Train 22,388개만 학습 함수로 전달한다. MFCC가 유한하지 않으면 임의 보간하지 않고 중단한다.
각 fold의 Scaler fit 통화 수와 평균을 직접 확인하며, 수렴 경고는 실패로 처리한다.

## 실행 순서

저장소 루트에서 실행한다. 환경 설치는 최초 한 번, 보통 수 분 이내이나 네트워크에 따라 달라진다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r mission-1/experiments/embedding_classifiers/requirements.txt
```

아래 embedding 경로는 자신의 파일 위치로 바꾼다. 기본 output은
`mission-1/experiments/embedding_classifiers/results/run_01/`이다.

### A1 · LR 기준 비교

`C=1 → 0.1 → 10` 순서로 각각 동일한 3-fold를 실행한다. 총 9회 학습.
분 단위를 예상하되 첫 fold의 측정값으로 갱신한다.

```bash
.venv/bin/python -u mission-1/experiments/embedding_classifiers/run_cv.py \
  --embeddings "$HOME/Downloads/wav2vec2_l4_full_embeddings.npz" --stage lr
```

첫 fold만 확인하려면 `--max-new-fits 1`을 추가한다.

### A2 · MFCC 결합

A1 결과 확인 후 실행한다. 822차원, `C=1 → 0.1 → 10`, 총 9회 학습.

```bash
.venv/bin/python -u mission-1/experiments/embedding_classifiers/run_cv.py \
  --embeddings "$HOME/Downloads/wav2vec2_l4_full_embeddings.npz" --stage fusion
```

### A3 · RBF SVM

먼저 첫 fold의 시간·자원 사용을 확인한다. LR보다 오래 걸릴 수 있으며 실행 전에는
전체 소요시간을 확정하지 않는다. `C=1,10`, `gamma="scale"`, 총 6회 학습.

```bash
.venv/bin/python -u mission-1/experiments/embedding_classifiers/run_cv.py \
  --embeddings "$HOME/Downloads/wav2vec2_l4_full_embeddings.npz" --stage svm --max-new-fits 1
```

이후 같은 명령에서 `--max-new-fits 1`을 빼면 남은 fold를 실행한다.
CPU 수치연산 thread는 2개, SVM kernel cache는 512MiB로 제한했다.
한 output에 여러 프로세스를 동시에 실행하지 않는다.

## 정상 출력과 복구

```text
INPUT PASS: Train=22388, Wav2Vec2=(22388, 768), MFCC=(22388, 54); ...
START lr_C1 fold 1/3, fit=14925 test=7463
PASS lr_C1 fold 1/3: accuracy=..., fit=...s, predict+score=...s
```

fold 완료 시 OOF 예측과 지표를 저장한다. 동일 명령 재실행 시 완료된 fold는
`RESUME`으로 건너뛰고, 중단된 fold는 처음부터 다시 학습한다.
입력·코드·패키지 버전이 달라지면 resume을 거부한다. 다른 실험은 새 `--output`을 사용한다.

| 산출물 | 내용 |
|---|---|
| `manifest.json` | 입력·코드 hash, 패키지 버전, 54개 feature 목록 |
| `fold_assignments.csv` | Train 전체 통화의 고정 fold |
| `*/fold_*.json` | fold별 성별 정확도·confusion matrix·시간·수렴 반복 수 |
| `*/fold_*_oof.csv` | 해당 fold의 OOF 예측·M 방향 decision score |
| `cv_summary.json` | 3-fold를 모두 끝낸 후보만 비교; 8개 완료 전에는 `partial` |

OOF CSV에는 feature·원본 음성이 없다. 실행 산출물은 우선 gitignore로 보호하고,
검토를 마친 요약·필요한 예측만 나중에 별도로 정리한다. 최종 후보 선택과 holdout
평가는 이 스크립트의 자동 동작에 포함하지 않는다.

```bash
.venv/bin/python -m unittest discover -s mission-1/experiments/embedding_classifiers -p 'test_*.py'
```

## Train OOF 오답 분석

24회 CV 완료 후 다음 명령으로 hash·fold·정답을 재검증하고 정량 분석한다.
기존 F0 특징 CSV와 manifest도 로컬에 있어야 한다. 학습이나 WAV 접근은 하지 않는다.
예상 시간: 수 초~1분.

```bash
.venv/bin/python mission-1/experiments/embedding_classifiers/analyze_oof.py
```

`results/run_01/error_analysis/`에 집계 JSON, Train 전체 분석 CSV, 결합 모델 오답
570개 CSV, 청취용 48통화 검토표를 저장한다. JSON의 지표는 정량 관찰이며 청취 원인
분류와 구분한다. 공개 요약은 `ERROR_ANALYSIS_SUMMARY.json`에 보관한다.
