# 2-3 실험: KLUE-RoBERTa Large와 W1 가중치

사용자가 Colab에서 실행한 `m3_2_3_large_seed42_results.tar`를 분석했다. 아래 점수는 **내부 검증 5,792건의 9개 증상 Macro F1**이며 예측 임계값은 **0.5로 고정**했다. 공식 Validation 결과가 아니다. 사전에 정한 평가 대상은 **4 epoch 최종 모델**이다.

## 실험 조건과 비교 범위

- seed 42, 동일 내부 train 23,408건 / dev-valid 5,792건. 분할 SHA-256은 `69817218fd5677067294a6d65ec44a3d0b4d7d5d4690f71f6fa4eaea1cfcf64e`.
- `klue/roberta-large`, 발화 경계 SEP + First-512, 증상별 label-attention head, Weighted BCE. 두통·오심·전신쇠약의 양성 가중치를 W0 대비 1.25배로 높인 W1 규칙을 사용했다.
- 4 epoch, 학습률 `2e-5`, micro batch 2 × gradient accumulation 4로 유효 batch 8, FP32, gradient checkpointing. 마지막 epoch를 선택했다. TF-IDF는 포함하지 않았다.
- 이전 2-1 W1 Base와 **검증 파일 순서·타깃 순서·정답·분할 해시·양성 가중치**가 일치한다. 새 예측 배열에서 Macro F1을 재계산해 저장된 값과 일치함을 확인했다.
- 이전 Base는 micro batch 8 × accumulation 1, 이번 Large는 2 × 4다. 따라서 아래 비교는 인코더 크기만 바꾼 엄밀한 통제 실험이 아니라 **같은 유효 batch의 역사적 비교**다. 동일 실행 조건의 Base 대조군은 아직 실행하지 않았다.

## 전체 성능

| 조건, seed 42 | 고정 임계값 0.5 Macro F1 | 목표 0.66과 차이 |
|---|---:|---:|
| 2-1 W1 RoBERTa Base | 0.640102 | −0.019898 |
| **2-3 W1 RoBERTa Large, 최종 4 epoch** | **0.637518** | **−0.022482** |

Large의 최종 점수는 이전 W1 Base보다 **0.002584 낮다**. seed 42 한 번만 실행했으므로 이 차이를 일반화 성능의 차이로 확정할 수 없다. 목표 0.66에는 도달하지 못했다.

| epoch | Large 내부 검증 Macro F1 | 검증 loss |
|---:|---:|---:|
| 1 | 0.637491 | 0.462672 |
| 2 | 0.641799 | 0.453611 |
| 3 | 0.634391 | 0.462231 |
| 4 | 0.637518 | 0.486141 |

2 epoch 점수가 4 epoch보다 높았지만, 이번 실험의 사전 지정 선택은 **최종 epoch**다. 검증 결과를 보고 2 epoch를 선택한다면 이는 새로운 모델 선택 규칙이 되므로, 그 점수를 독립적인 확인 결과로 취급할 수 없다. 후속 실험에서 선택 규칙을 미리 고정하고 새 seed로 검증해야 한다.

## 증상별 결과

| 증상 | 이전 W1 Base F1 | 이번 Large F1 | 변화 |
|---|---:|---:|---:|
| 고열 | 0.6673 | 0.6569 | −0.0104 |
| 구토 | 0.6081 | 0.6035 | −0.0046 |
| 두통 | 0.5140 | 0.5065 | −0.0075 |
| 복통 | 0.7889 | 0.7848 | −0.0042 |
| 어지러움 | 0.6719 | 0.6687 | −0.0032 |
| 열상 | 0.8830 | 0.8760 | −0.0070 |
| 오심 | 0.3754 | 0.3727 | −0.0027 |
| 전신쇠약 | 0.5677 | 0.5816 | +0.0139 |
| 호흡곤란 | 0.6845 | 0.6869 | +0.0024 |

전신쇠약은 개선됐지만 두통·오심을 포함한 7개 증상의 F1이 낮아졌다. 오심은 실제 양성 668건 중 Large가 295건을 맞혔고 373건을 놓쳤다. 오탐은 620건으로, 이전 W1 Base의 TP 296·FP 613·FN 372보다 나빠졌다.

## 다음 판단

이번 단일 결과만으로 Large를 채택할 근거는 없다. 인코더 크기의 효과를 분리하려면 같은 micro batch·gradient accumulation·저장 조건의 Base 대조군을 먼저 실행한다. 2 epoch 모델 선택을 검토할 때는 선택 규칙을 사전에 정하고 다른 seed 또는 별도 검증에서 확인한다. 같은 내부 검증셋에서 반복해서 고른 최고 점수를 최종 성능으로 보고하지 않는다.

## 보관 파일

- [실행 Colab 노트북](../../models/roberta/M3_2-3_RoBERTaLarge_W1_fixed05.ipynb)
- `seed_42/comparison.csv`, `seed_42/label_comparison.csv`: 전체·증상별 이전 W1 Base 비교
- `seed_42/w1_encoder/metrics.json`, `per_label.csv`, `run_config.json`, `training_log.json`
- `training_pos_weights.csv`, `weight_rule.json`, `environment.json`, `model_revision.json`, `seed_42/initial_parameter_sha256.txt`

원본 TAR의 `dev_predictions.npz`와 `split_manifest.csv`에는 개별 통화 파일명이 들어갈 수 있어 Git에 올리지 않았다. TAR에는 모델 가중치·체크포인트가 없어, Git 자료만으로 새 데이터 추론이나 학습 재개는 할 수 없다.
