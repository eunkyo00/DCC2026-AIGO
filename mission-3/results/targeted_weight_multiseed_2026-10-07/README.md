# 2-1 실험: 낮은 recall 증상 가중치 반복 검증

사용자가 Colab에서 실행해 제공한 `m3_2_1_results.tar`를 분석했다. seed 42의 결과는
앞선 `m3_targeted_weight_results.tar.gz`에서 재사용했고, seed 123·2026은 새로 학습했다.
아래 점수는 **내부 검증 5,792건의 9개 증상 Macro F1**이다. 공식 Validation 점수가
아니며, 예측 임계값은 모든 조건에서 **0.5로 고정**했다.

## 통제 조건

- 동일한 내부 train 23,408건 / dev-valid 5,792건, 분할 SHA-256
  `69817218fd5677067294a6d65ec44a3d0b4d7d5d4690f71f6fa4eaea1cfcf64e`.
- `klue/roberta-base`, 발화 경계 SEP + First-512, 증상별 label-attention head,
  Weighted BCE, 4 epoch, batch 8, 학습률 `2e-5`, 마지막 epoch 모델.
- W0 양성 가중치는 `clip(sqrt(음성/양성), 1, 3)`. W1은 두통·오심·전신쇠약의
  양성 가중치만 1.25배(최대 4)로 높였다. 실제 가중치는 `training_pos_weights.csv` 참고.
- 각 seed 안에서 W0/W1의 초기 파라미터 해시·분할 해시·학습 설정(출력 경로와
  조건별 가중치 제외)을 맞췄다. seed 123·2026의 예측 배열을 다시 계산해 저장
  Macro F1 및 검증 행 순서·정답과 일치함을 확인했다.
- 이번 반복 검증은 **RoBERTa 단독** 비교다. TF-IDF 앙상블은 포함하지 않는다.

## 전체 결과

| seed | W0 Macro F1 | W1 Macro F1 | W1 − W0 |
|---:|---:|---:|---:|
| 42 (기존 결과 재사용) | 0.638448 | 0.640102 | +0.001654 |
| 123 | 0.638675 | 0.639002 | +0.000327 |
| 2026 | 0.635014 | 0.640935 | +0.005921 |
| **3 seed 평균** | **0.637379** | **0.640013** | **+0.002634** |
| seed 간 표본 표준편차 | 0.002051 | 0.000970 | 0.002923 |

W1이 세 seed에서 모두 W0보다 높았지만 평균 개선 폭은 **0.0026**으로 작다.
W1 평균은 0.6400으로 목표 0.65에 미치지 못한다. 3개 seed만으로 일반화 성능의
유의미한 향상이나 공식 Validation 개선을 확정할 수 없다.

## 낮은 recall 증상

아래 값은 seed별 precision·recall·F1의 산술평균이다. 개별 값은
`seed_per_label.csv`에 있다.

| 증상 | W0 precision → W1 | W0 recall → W1 | W0 F1 → W1 |
|---|---:|---:|---:|
| 두통 | 0.5119 → 0.4843 | 0.5307 → 0.5468 | 0.5211 → 0.5136 |
| 오심 | 0.3424 → 0.3212 | 0.3802 → 0.4571 | 0.3602 → 0.3772 |
| 전신쇠약 | 0.5605 → 0.5347 | 0.5929 → 0.6230 | 0.5760 → 0.5753 |

오심은 세 seed 모두 recall과 F1이 올랐지만 precision은 떨어졌다. 두통은 recall은
상승했으나 F1은 세 seed 모두 하락했다. 전신쇠약은 recall이 상승했고 F1의 방향은
seed마다 달랐다. 따라서 W1은 오심 누락을 줄이는 데는 유망하지만 두통·전신쇠약까지
일괄 보강하는 규칙을 확정하기에는 근거가 부족하다.

## 해석과 다음 단계

이번 결과는 사전 지정한 두 조건의 반복 검증이며, 검증셋에서 배율이나 threshold를
추가 탐색하지 않았다. 최종 모델로 W1을 바로 확정하지 말고, 오심을 놓치는 비용과
오탐 증가의 비용을 평가 목표에 맞춰 비교해야 한다. 후속 실험을 한다면 사전에
조건을 고정하고 별도 검증·공식 평가를 구분한다. 최종 제출 모델을 정한 뒤에는
전체 Training 재학습과 동일 전처리의 추론 코드 검증이 별도로 필요하다.

## 보관 파일

- [재실행 Colab 노트북](../../models/roberta/M3_2-1_TargetedWeight_RuntimeOnly_fixed05.ipynb)
- `seed_comparison.csv`: seed별 Macro F1과 paired 차이
- `seed_per_label.csv`: seed·조건·증상별 precision, recall, F1, support
- `seed_*/<조건>/metrics.json`, `per_label.csv`, `run_config.json`, `training_log.json`
- `training_pos_weights.csv`, `weight_rule.json`, `seed_*/initial_parameter_sha256.txt`

원본 TAR의 `dev_predictions.npz`와 `split_manifest.csv`는 개별 통화 ID가 포함될 수
있어 Git에 올리지 않았다. 모델 가중치·체크포인트도 이 결과 TAR에 없다. 따라서
Git에 보관한 자료만으로 새 데이터 추론이나 학습 재개는 할 수 없다.
