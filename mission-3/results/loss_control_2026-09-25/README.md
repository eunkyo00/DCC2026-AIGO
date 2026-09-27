# 손실함수 통제 실험: BCE vs Weighted BCE

사용자가 Colab에서 실행한 `m3_loss_control_fixed_v2_results.tar.gz`를 2026-09-25에 정리했다.
개별 예측 배열에서 Macro F1을 재계산하여 저장 지표와 일치함을 확인했다.
원문, 통화 ID, split manifest, 개별 예측 배열 및 모델 가중치는 Git에 포함하지 않는다.

## 실험 조건과 변인 통제

- 내부 학습 23,408건 / 내부 검증 5,792건. 공식 Validation 평가는 아니다.
- KLUE-RoBERTa Base + 증상별 attention head, 발화 사이 SEP + First-512.
- seed 42, 4 epoch, batch 8, LR 2e-5, warmup 1,000, weight decay 0.01.
- FP32, gradient clipping 1.0, 마지막 epoch 평가. 모든 임계값은 0.5 고정.
- B0: 일반 BCE, 양성 가중치 모두 1.
- L1: 내부 학습 데이터로만 `clip(sqrt(음성 수 / 양성 수), 1, 3)` 계산.
- 두 실행의 초기 파라미터 해시·분할 해시가 같고 학습 설정은 출력 경로만 다름을 확인했다.
- 다섯 결과의 검증 ID 순서·정답·target 순서가 같고 예측이 유한함을 확인했다.
- TF-IDF는 문자 2~5 gram + OvR Logistic Regression, 원문 전체로 한 번 학습해 재사용.
- 앙상블은 RoBERTa 확률 0.5 + TF-IDF 확률 0.5. 비율·임계값 탐색 없음.

## 전체 결과

| 조건 | Macro F1 | 오심 F1 |
|---|---:|---:|
| TF-IDF 단독 | 0.4706 | 0.0474 |
| B0 일반 BCE RoBERTa | 0.6107 | 0.1840 |
| B0 + TF-IDF | 0.5841 | 0.0964 |
| L1 Weighted BCE RoBERTa | **0.6400** | **0.3586** |
| L1 + TF-IDF | 0.6164 | 0.2063 |

L1 단독은 B0보다 Macro F1이 0.02936 높다. 목표 0.65까지 차이는 0.00998이다.
현재 50:50 확률 평균은 두 조건 모두 RoBERTa 단독보다 낮다. 이는 모든 결합 방법이
효과 없다는 증거는 아니며, 이번 결합 조건의 관측 결과다.

## 증상별 변화 (RoBERTa 단독)

| 증상 | B0 F1 | L1 F1 | B0 recall | L1 recall |
|---|---:|---:|---:|---:|
| 고열 | 0.6622 | 0.6553 | 0.5757 | 0.6297 |
| 구토 | 0.5886 | 0.5998 | 0.4894 | 0.5789 |
| 두통 | 0.4843 | 0.5256 | 0.3855 | 0.5301 |
| 복통 | 0.7891 | 0.7889 | 0.7282 | 0.7629 |
| 어지러움 | 0.6520 | 0.6717 | 0.5856 | 0.6638 |
| 열상 | 0.8974 | 0.8813 | 0.8961 | 0.8947 |
| 오심 | 0.1840 | 0.3586 | 0.1123 | 0.3608 |
| 전신쇠약 | 0.5593 | 0.5848 | 0.5085 | 0.6215 |
| 호흡곤란 | 0.6790 | 0.6942 | 0.6331 | 0.7287 |

오심 양성 668건 중 TP는 75→241, FN은 593→427로 개선됐다. 반면 FP는 72→435로
증가해 precision은 0.5102→0.3565로 낮아졌다. 가중 손실이 놓치는 사례를 줄였지만
오탐도 늘렸다. 고열·복통·열상의 F1은 소폭 하락했다.

## 학습 로그와 이전 실패 처리

| Epoch | B0 Macro F1 | L1 Macro F1 |
|---|---:|---:|
| 1 | 0.5684 | 0.6318 |
| 2 | 0.5846 | 0.6384 |
| 3 | 0.5876 | 0.6323 |
| 4 | 0.6107 | 0.6400 |

첫 기록 손실은 B0 0.7032, L1 0.8249였다. 이전의 F1=0/비정상 손실 현상과 달리
이번 실행은 B0가 약 0.61을 회복했다. 서로 다른 가중치를 쓰므로 B0/L1 loss의 크기를
직접 비교해 우열을 판단하지 않는다. 각 학습은 약 2시간 26분 걸렸다.

이전 코드의 non-persistent `loss_pos_weight` buffer가 `from_pretrained` 로딩 후
설정과 달라지는 문제를 작은 CPU 모델에서 재현했다(기대 1, 실제 0).
v2는 config 숫자로 가중치를 보관하고 실제 장치에서 매 forward 생성한다.
독립 BCE 식 비교, 양성 포함 입력 검사, 저장/재로딩 검사, 비정상 loss/gradient 중단을
추가했다. 실패한 이전 실행은 유효한 손실함수 비교 결과에 포함하지 않는다.

## 해석과 다음 실험

이번 동일 seed 비교에서는 Weighted BCE가 도움이 됐다. 다만 하나의 내부 분할·seed
결과이며 통계적 유의성이나 공식 Validation 일반화 성능을 확인한 것은 아니다.
가중 손실은 학습 목표를 바꾸지만 예측 threshold는 계속 0.5다.
다음에는 L1을 기준으로 손실·head·분할·seed를 고정하고 문맥 처리 한 요인만 바꾸어
비교할 수 있다. 여러 seed 재현성과 최종 독립 평가가 추가로 필요하다.

## 파일과 재실행

- [수정본 Colab 노트북](../../models/roberta/M3_BCE_vs_WeightedBCE_fixed05_fixed_v2.ipynb)
- `summary.csv`: 5개 조건의 전체 지표
- `label_comparison.csv`: B0/L1 증상별 precision/recall/F1 및 TP/FP/FN
- `weight_rule.json`, `training_pos_weights.csv`: 학습에서 계산한 가중치
- 각 실행의 `metrics.json`, `per_label.csv`, `run_config.json`, `training_log.json`: 제공된 기록

새 Colab GPU 노트북에서 기존과 같은 Training ZIP, `split_manifest.csv`,
`m3_fixed_results.tar.gz`를 준비하고 위에서부터 실행한다. 새 출력 폴더는
`/content/m3_loss_control_fixed_v2`, 결과 백업은 `m3_loss_control_fixed_v2_results.tar.gz`다.
작은 결과 백업에는 모델 가중치가 없으므로 새 데이터 추론·학습 재개에는 별도 모델 백업이 필요하다.
