# 2-7 실험: 앞·뒤 발화를 읽는 RoBERTa Base

사용자가 Colab에서 실행한 `m3_2_7_base_seed42_results.tar`를 분석했다. **내부 검증 5,792건·9개 증상 Macro F1**, 임계값 **0.5 고정**이다. 공식 Validation 결과가 아니다.

## 조건

- `klue/roberta-base`, seed 42, 동일 내부 분할, 발화 경계 보존, 증상별 attention head, W1 가중 BCE, 4 epoch.
- 앞 510개와 뒤 510개 본문 토큰을 각각 최대 512토큰 창으로 인코딩하고 창별 logit을 평균했다. 짧은 통화는 한 창만 사용한다.
- 긴 통화의 뒤쪽 증상 단서를 읽는 것이 가설이다. 기존 W1 Base는 첫 512토큰만 사용했다.
- 이번 실행은 micro batch 2 × 누적 4, gradient checkpointing을 사용했다. 기존 W1 Base는 batch 8 × 누적 1이므로 문맥 창만 바꾼 엄밀한 단일 요인 비교는 아니다.
- 동일 검증 파일 순서·정답·분할 해시와 W1 양성 가중치를 확인했다. 저장된 예측 배열에서 최종 F1을 재계산해 보고값과 일치함을 확인했다.

## 결과

| 조건 | Macro F1 |
|---|---:|
| 기존 W1 Base, First-512, seed 42 | **0.640102** |
| 2-7 앞·뒤 두 창, 최종 4 epoch | 0.637320 |
| 참고: 2-5 전체 앙상블 | **0.650944** |

2-7은 기존 W1 Base보다 **0.002782 낮다**. 단일 RoBERTa 결과와 전체 앙상블 결과는 구성 모델 수가 다르므로 동일 조건 비교가 아니다. 목표 0.66에 도달하지 못했다.

| epoch | 2-7 내부 검증 Macro F1 |
|---:|---:|
| 1 | 0.630643 |
| 2 | 0.635136 |
| 3 | 0.635063 |
| 4 | **0.637320** |

사전에 정한 최종 4 epoch를 평가했다. 학습은 완료됐고 1→4 epoch 동안 점수가 올랐으나 기준 모델에는 못 미쳤다.

| 증상 | 기존 W1 Base F1 | 2-7 F1 |
|---|---:|---:|
| 두통 | 0.5140 | 0.5129 |
| 오심 | 0.3754 | 0.3771 |
| 전신쇠약 | 0.5677 | 0.5669 |

오심은 소폭 올랐지만 전체 개선으로 이어지지 않았다. 이 설정에서 두 창 방식은 채택할 근거가 부족하다. 같은 내부 검증셋을 반복 사용한 탐색 결과이므로 일반화 성능을 단정할 수 없다. 현재 기준점은 2-5 앙상블 0.650944로 유지한다.

## 보관 파일

- [실행 Colab 노트북](../../models/roberta/M3_2-7_DualWindow_W1_Base_fixed05.ipynb)
- `seed_42/comparison.csv`, `label_comparison.csv`: 이전 W1과의 전체·증상별 비교
- `seed_42/w1_encoder/metrics.json`, `per_label.csv`, `run_config.json`, `training_log.json`
- `training_pos_weights.csv`, `weight_rule.json`, `environment.json`, `model_revision.json`, `seed_42/initial_parameter_sha256.txt`

개별 통화 파일명이 포함될 수 있는 예측 배열과 split manifest는 Git에 올리지 않았다. 제공된 결과 TAR에도 모델 가중치·체크포인트는 포함되어 있지 않다.
