# Weighted BCE 문맥 선택 실험: First-512 vs Head-Tail

2026-10-06에 Colab에서 실행한 `m3_context_control_results.tar.gz`를 정리한 기록이다.
이 실험은 현재 최고 설정인 KLUE-RoBERTa Base + 증상별 label-attention head + Weighted BCE를
유지한 채, 512토큰을 초과하는 통화에서 어떤 내용을 보존할지만 비교했다.

원문, 통화 ID, split manifest, 개별 예측 배열, 모델 가중치는 Git에 포함하지 않는다.
저장된 지표·설정·학습 로그는 결과 백업에서 가져왔고, `summary.csv`와 `metrics.json`의
고정 threshold 0.5 Macro F1을 기준으로 해석했다.

## 통제 조건

- 내부 학습 23,408건, 내부 검증 5,792건. 공식 Validation 결과가 아니다.
- `klue/roberta-base`, 증상별 label-attention head, 발화 사이 SEP 경계 보존.
- Weighted BCE: `clip(sqrt(negative / positive), 1, 3)`. 내부 학습 데이터로만 계산.
- seed 42, 4 epoch, batch 8, LR 2e-5, warmup 1,000, weight decay 0.01, FP32,
  gradient clipping 1.0, 마지막 epoch 선택.
- 모든 증상의 예측 threshold는 0.5로 고정.
- C0/C1의 초기 파라미터 SHA256은 `8a65c4…b3a35d`로 같고, 분할도 같다.
- 고정 TF-IDF 예측은 직전 성공 실험의 동일 검증 예측을 재사용했다.

## 입력 차이

| 조건 | 510개 이하 내용 토큰 | 510개 초과 내용 토큰 |
|---|---|---|
| C0 First-512 | 전체 | 앞 510개 |
| C1 Head-Tail | 전체 | 앞 255개 + 뒤 255개 |

긴 통화가 아니면 두 입력은 동일하다. 검증 통화 중 512토큰 초과는 313건(5.4%)이며,
학습 통화에서는 1,079건(4.6%)이다.

## 전체 결과

| 조건 | 고정 0.5 Macro F1 | 오심 F1 |
|---|---:|---:|
| 이전 Weighted BCE First-512 | **0.6400** | 0.3586 |
| C0 First-512 재학습 | **0.6387** | 0.3671 |
| C1 Head-Tail | 0.6382 | **0.3687** |
| C0 + TF-IDF 0.5/0.5 | 0.6180 | 0.2114 |
| C1 + TF-IDF 0.5/0.5 | 0.6179 | 0.2171 |

이번 직접 비교에서 Head-Tail은 C0보다 **0.00046 낮다**. 이전 First-512 기록과 C0의
차이 0.00134는 런타임·학습의 작은 변동 범위로 보이며, 이번 입력 방식의 판단에는 같은
실행에서 다시 학습한 C0와 C1의 비교를 사용한다.

## 긴 통화 결과

| 검증 통화 길이 | C0 First-512 | C1 Head-Tail | 차이 C1-C0 |
|---|---:|---:|---:|
| 512토큰 이하, 5,479건 | 0.6372 | 0.6381 | +0.0009 |
| 512토큰 초과, 313건 | **0.6520** | 0.6276 | **-0.0244** |

Head-Tail은 길이가 긴 통화에서 성능을 낮췄다. 통화 후반을 포함하는 이점보다 앞쪽 255개와
중간 문맥을 버리는 손실이 컸을 가능성이 있다. 긴 통화 표본은 313건이므로 이 하위 분석은
참고 지표이며, 추가 seed에서 재현하기 전까지 강한 일반화 결론으로 쓰지 않는다.

## 증상별 비교

| 증상 | C0 F1 | C1 F1 | 변화 |
|---|---:|---:|---:|
| 고열 | 0.6633 | 0.6586 | -0.0047 |
| 구토 | 0.6031 | 0.6033 | +0.0002 |
| 두통 | 0.5081 | 0.5110 | +0.0029 |
| 복통 | 0.7876 | 0.7855 | -0.0021 |
| 어지러움 | 0.6720 | 0.6696 | -0.0024 |
| 열상 | 0.8768 | 0.8790 | +0.0023 |
| 오심 | 0.3671 | 0.3687 | +0.0015 |
| 전신쇠약 | 0.5761 | 0.5802 | +0.0042 |
| 호흡곤란 | 0.6939 | 0.6878 | -0.0061 |

Head-Tail은 오심·두통·전신쇠약에서 미세한 상승이 있었지만, 전체 Macro F1을 높일 정도는
아니었다. 현재 기준 입력은 First-512로 유지한다.

## TF-IDF 결합

고정 TF-IDF와 0.5/0.5 확률 평균은 C0와 C1 모두 단독 RoBERTa보다 낮았다.
따라서 이번 고정 비율 결합은 채택하지 않는다. TF-IDF를 다시 결합하려면 학습 데이터의
out-of-fold 예측으로 증상별 결합기를 학습하는 별도 실험이 필요하다.

## 다음 단계

문맥 토큰 선택 방식은 이번 Head-Tail에서 개선되지 않았다. 다음 실험은 First-512를
기준으로 유지하고, 손실 가중치 공식만 변경하는 비교가 적절하다. 예를 들어 현재
`clip(sqrt(negative / positive), 1, 3)`과 최대 2·4 또는 사전 정의한 다른 공식을 비교한다.
입력·모델·head·분할·seed·threshold는 고정해야 한다.

## 파일 안내

- [실행 노트북](../../models/roberta/M3_WeightedBCE_First512_vs_HeadTail_fixed05.ipynb)
- `summary.csv`: 전체 결과와 오심 지표
- `length_group_comparison.csv`: 긴/짧은 통화별 Macro F1
- `label_comparison.csv`: C0/C1 증상별 precision, recall, F1, TP/FP/FN
- `training_pos_weights.csv`, `weight_rule.json`: Weighted BCE 가중치
- 각 조건의 `metrics.json`, `per_label.csv`, `run_config.json`, `training_log.json`

원본 결과 백업에는 개별 예측 배열이 포함되지만, Git에는 통화 ID가 포함될 수 있어 넣지
않았다. 모델 가중치도 Git에 없으므로 새 데이터 추론이나 학습 재개에는 별도 모델 백업이 필요하다.
