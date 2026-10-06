# Small-stem ResNet18 — Validation 결과 보고서

> 과거 임계값 튜닝 실험의 원본 기록이다. 현재 규정은 0.5 고정이므로 0.535 결과는 제출 기준 성능이 아니다. 현재 코드는 학습 선택·평가·추론 모두 0.5를 적용한다. 원본 JSON 로그는 수정하지 않았다.

## 결과 출처와 검증 범위

사용자가 제공한 `mission02_experiment_smallstem-20260919T042641Z-1-001.zip`의 `history.json`, `best_model.pt`를 대조했다. ZIP 무결성을 확인하고 체크포인트의 모델 가중치를 현재 `model.py`에 strict loading했으며 `[1,1,64,24]` 입력 forward 연산을 확인했다. **원본 Validation 음성으로 독립 추론을 재실행한 것은 아니다.**

| 파일 | SHA-256 | GitHub 포함 여부 |
|---|---|---|
| `history.json` | `cdbfbacbb463603d475dc57acfd89f7e7906ebbd5d1196bf08525f51edfea3e5` | 포함 |
| `best_model.pt` | `5a5dfa14227bb7c0961ff13e8f845c6d58bb66124b5875d316cf1e73ce9e84f5` | 미포함 |

## 학습·평가 조건

| 항목 | 값 |
|---|---:|
| 서울 지역 Training | 889,358발화 |
| 공식 Validation | 111,947발화 |
| 모델 | `resnet18_smallstem` |
| 모델 파라미터 | 11,169,858 |
| 특징 | `first_1p5`, 앞 1.5초, 64-band log-Mel `[64,24]` |
| 최고 epoch | 16 |
| Batch / 초기 학습률 | 512 / `2e-4` |
| Dropout / label smoothing / Mixup alpha | 0.25 / 0.03 / 0.10 |

학습에 정답 `speaker`는 사용하지만 모델 입력 특징에는 넣지 않는다. Validation은 모델 가중치의 역전파에는 사용하지 않으나 최고 epoch와 threshold 선택에 사용했다.

## 공식 Validation 결과

| 지표 | 값 |
|---|---:|
| Validation loss | 0.2737853731 |
| Accuracy (threshold 0.5) | 90.139977% |
| Accuracy (선택 threshold 0.535) | **90.248957%** |
| 정답 / 전체 발화 | 101,031 / 111,947 |

| 실제 / 예측 | 119대원 (`0`) | 신고자 (`1`) | 실제 역할 내 정답률 |
|---|---:|---:|---:|
| 119대원 (`0`) | 46,396 | 7,449 | 86.17% |
| 신고자 (`1`) | 3,467 | 54,635 | 94.03% |

혼동행렬은 `[[TN, FP], [FN, TP]]`이다. 발화 단위 전체 Accuracy가 90.25%여도 119대원 발화의 오류 비율은 더 높다. 역할별 차이의 원인은 이 보고서에서 확정하지 않는다. 전체 16 epoch 기록은 [history.json](history.json), 기계 판독 가능한 요약은 [metrics.json](metrics.json)에 있다.

## 재현성·한계

`python verify_results.py`는 공개 `history.json`과 `metrics.json`의 해시·최고 epoch·혼동행렬·Accuracy의 내부 일치를 검사한다. 별도 보관된 가중치를 받으면 `--checkpoint`로 해당 파일도 대조할 수 있다. 검증 스크립트가 성공해도 원본 음원을 다시 읽어 모델 성능을 측정한 것은 아니다.

같은 Validation에서 최고 epoch와 threshold를 선택했으므로 이 수치를 비공개 Test 성능으로 해석하면 안 된다. 저장소에는 가중치·음원·대용량 특징이 없어 공개 파일만으로 성능을 독립 재평가할 수 없다. `valid_elapsed_seconds`는 [history.json](history.json)의 **사전 계산 특징 로더 기준 평가 구간 시간**이며 WAV 읽기·전처리까지 포함한 제출 추론 시간으로 볼 수 없다.
