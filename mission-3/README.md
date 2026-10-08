# Mission 3 — 통화 텍스트의 9개 증상 다중 라벨 분류

통화의 `utterances[].text`를 원래 순서대로 공백으로 연결해 0~9개 증상을 예측한다.
클래스 순서는 **고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란**이다.
타깃 외 증상만 제거하고, 해당 통화 자체는 버리지 않는다. WAV, speaker, recordId,
audioPath, 주소, 정답 라벨은 입력 텍스트에 넣지 않는다.

`mission-1`의 EDA → 고정 검증 분할 → baseline/사전학습 모델 → 결과 보고서 구성을 따른다.
Mission 1의 음성 매칭 cohort와 다르게 Mission 3는 JSON 29,200건을 사용한다.

## 폴더

| 경로 | 역할 |
|---|---|
| `data/raw/` | 로컬 Training/Validation JSON ZIP |
| `data/processed/` | 텍스트·9개 정답 열을 가진 JSONL |
| `data/splits/` | E0 원본 및 검증된 split manifest |
| `eda/` | 데이터 점검 코드와 기존 조사 보고서 |
| `validation/` | Training 내부 분할 생성/복원 |
| `baseline/tfidf/` | 문자 TF-IDF + One-vs-Rest Logistic Regression |
| `models/roberta/` | E0 First-512, E1 Base Head-Tail, E3 Large Head-Tail |
| `ensemble/` | 확률 혼합 + 증상별 threshold 탐색 |
| `results/` | 기존 실험의 집계 결과와 출처 |
| `artifacts/` | 로컬 모델·확률·체크포인트 (Git 제외) |
| `tests/` | 텍스트 입력 경계와 데이터 처리 테스트 |

원본 JSON, 텍스트, 개별 통화 ID/예측, 모델 가중치는 GitHub에 올리지 않는다.
빈 데이터 폴더와 사용 안내만 버전 관리한다.

## 지금까지의 결과

후속 실험 기록:

- [고정 임계값 0.5 재실험](results/FIXED_THRESHOLD_0_5_2026-09-20.md): E0+E3 0.6174.
- [분류 head 비교](results/head_comparison_2026-09-22/README.md): CLS 0.6102 / Mean 0.6116 / 증상별 Attention 0.6144.
- [새 Colab head 비교 노트북](models/roberta/M3_fixed05_head_experiments.ipynb): 오류 분석부터 결과 백업까지.

아래 표는 이전 threshold 조정 실험 기록이며 위 후속 실험과 구분한다.

모두 사용자가 Colab에서 수행하고 화면/백업으로 제공한 **내부 검증 결과**다.
이번 코드 정리 작업에서 모델을 다시 학습한 수치가 아니다.

| 실험 | 설정 | 기본 F1 (0.5) | threshold 조정 F1 |
|---|---|---:|---:|
| E0 | KLUE-RoBERTa Base First-512 | 0.6063 | 0.6426 |
| E1 | KLUE-RoBERTa Base Head-Tail | 0.6092 | 0.6448 |
| TF-IDF | 문자 2~5 gram + LR | — | 0.6028 |
| E2 | E1 0.55 + TF-IDF 0.45 | — | 0.6494 |
| E3 | KLUE-RoBERTa Large Head-Tail | 0.6194 | 0.6479 |
| E3 + TF-IDF | 0.50 / 0.50 | — | 0.6519 |
| E3 + E2 | 0.35 / 0.65 | — | **0.6562** |

최종 비율은 Large 0.35 / Base 0.3575 / TF-IDF 0.2925다.
threshold와 비율을 같은 내부 검증 세트에서 선택했으므로 독립 평가 성능이 아니며,
공식 Validation 성능과 구분해야 한다. 0.9~0.98 도달을 보장하지 않는다.

## 실행

저장소 루트에서 실행한다. Colab 절차는 [COLAB.md](COLAB.md)를 따른다.

```bash
pip install -r mission-3/requirements.txt
python mission-3/data/prepare.py --train-zip mission-3/data/raw/mission3_train_json.zip --validation-zip mission-3/data/raw/mission3_validation_json.zip
python mission-3/validation/create_split.py --manifest mission-3/data/splits/e0_original/split_manifest.csv
python mission-3/eda/run_eda.py --out-dir mission-3/artifacts/audit
python mission-3/baseline/tfidf/run_tfidf.py --out-dir mission-3/artifacts/tfidf
python mission-3/models/roberta/run_roberta.py --experiment e1 --out-dir mission-3/artifacts/e1
python mission-3/models/roberta/run_roberta.py --experiment e3 --out-dir mission-3/artifacts/e3
python mission-3/ensemble/run_ensemble.py --a mission-3/artifacts/e1/dev_predictions.npz --b mission-3/artifacts/tfidf/dev_predictions.npz --out-dir mission-3/artifacts/e2
python mission-3/ensemble/run_ensemble.py --a mission-3/artifacts/e3/dev_predictions.npz --b mission-3/artifacts/e2/dev_predictions.npz --out-dir mission-3/artifacts/e3_e2
python -m unittest discover -s mission-3/tests -v
```

멘토링 지침처럼 threshold를 0.5로 고정하는 재실험에서는 각 단독 모델 명령에
`--fixed-threshold`를 추가한다. 이때 `metrics.json`의 `fixed_threshold_macro_f1`만 비교한다.
앙상블도 동일 원칙을 적용하며, `ensemble/run_ensemble.py`에 `--fixed-threshold --weight-a 0.5`를 함께 전달하면 가중치와 threshold 모두 탐색하지 않는다.

기존 결과를 덮어쓰지 않는다. 재실험은 새로운 `--out-dir`를 지정한다.
이 코드는 대화의 실험 절차를 재구성한 것으로 당시 노트북 자체는 아니다.
당시 전체 라이브러리 버전과 초기 가중치 seed가 불명확해 수치의 완전 일치를 보장하지 않는다.
새 실행에서는 환경 정보, 분할 순서, 예측 ID와 설정을 저장한다.

## 다음 단계

먼저 오분류(특히 오심)의 텍스트와 정답 대응을 살핀다. 새 모델도 동일한 manifest를 사용하고,
후보 선택 후 설정을 고정해 별도 평가한다. 최종 제출에는 전체 Training 재학습,
동일한 텍스트 전처리를 쓰는 추론과 주최 측 CSV 형식 확인이 별도로 필요하다.
미검증 제출용 추론 코드는 포함하지 않았다.

## 손실함수 통제 실험 (2026-09-25)

임계값 0.5 고정, 같은 초기 가중치·분할·학습 설정에서 일반 BCE와 Weighted BCE를 비교했다.
RoBERTa 단독 Macro F1은 **0.6107 → 0.6400**, 오심 recall은 **0.1123 → 0.3608**로
개선됐다. Weighted BCE + TF-IDF 50:50은 0.6164로 단독보다 낮았다.
단일 seed 내부 검증 결과이며 공식 Validation 성능과 구분한다.

- [결과·증상별 분석·초기화 버그 수정 기록](results/loss_control_2026-09-25/README.md)
- [재실행 Colab 노트북](models/roberta/M3_BCE_vs_WeightedBCE_fixed05_fixed_v2.ipynb)

## 문맥 토큰 선택 통제 실험 (2026-10-06)

Weighted BCE 설정에서 First-512와 Head-Tail을 비교했다. Head-Tail의 Macro F1은 0.6382로
First-512 재학습 0.6387보다 0.00046 낮았고, 512토큰 초과 통화 313건에서는 0.6520에서
0.6276으로 하락했다. 현재 기준 입력은 First-512로 유지한다.

- [결과 보고서·증상별 분석](results/context_control_2026-10-06/README.md)
- [재실행 Colab 노트북](models/roberta/M3_WeightedBCE_First512_vs_HeadTail_fixed05.ipynb)

## 낮은 recall 증상 가중치 통제 실험 (2026-10-07)

현재 Weighted BCE 대비 두통·오심·전신쇠약 양성 가중치만 1.25배 올려 비교했다.
Macro F1은 0.63845에서 0.64010으로 소폭 상승했지만, 두통과 전신쇠약 F1은 하락했다.
오심 recall은 0.3728에서 0.4431로 높아졌고 오탐도 증가했다. 이전 최고 기록
0.64002와는 사실상 동률에 가까워 여러 seed 재현 확인이 필요하다.

- [결과 보고서·증상별 지표](results/targeted_weight_2026-10-07/README.md)
- [재실행 Colab 노트북](models/roberta/M3_WeightedBCE_TargetedLowRecallWeights_fixed05.ipynb)

## 2-1 실험: 낮은 recall 증상 가중치 반복 검증 (2026-10-07)

동일한 W0/W1 조건을 seed 42·123·2026에서 비교했다. seed 42는 기존 결과를
재사용했다. W1이 세 seed에서 모두 높았고 평균 Macro F1은 **0.63738 → 0.64001**
(평균 paired 차이 **+0.00263**)이었다. 오심 평균 recall은 0.3802 → 0.4571로
높아졌지만 precision은 하락했다. 목표 0.65에는 도달하지 못했으며 공식 Validation
결과도 아니다.

- [반복 검증 보고서·seed별 지표](results/targeted_weight_multiseed_2026-10-07/README.md)
- [2-1 Colab 노트북](models/roberta/M3_2-1_TargetedWeight_RuntimeOnly_fixed05.ipynb)

## 2-2 실험: 오심 단독 가중치, seed 42 (2026-10-07)

2-1에서 오심 F1은 세 seed 모두 올랐지만 두통 F1은 모두 하락했다. 다음 조건은
두통·전신쇠약 가중치를 W0로 되돌리고 오심 가중치만 1.25배로 높였다. seed 42에서
내부 검증 Macro F1은 W0 0.638448 → 0.638592로 **+0.000144**였다. 오심 recall은
0.3728 → 0.4521, 오심 F1은 0.3617 → 0.3763으로 올랐지만 precision은 하락했다.
같은 seed의 W1 0.640102보다 전체 점수는 낮았다. 임계값은 0.5로 고정했다.
**seed 123·2026은 아직 실행하지 않았고 공식 Validation 결과도 아니다.**

- [seed 42 결과 보고서·증상별 지표](results/nausea_only_seed42_2026-10-07/README.md)
- [2-2 단일 seed 실행 Colab 노트북](models/roberta/M3_2-2_NauseaOnly_OneSeed_fixed05.ipynb)

## 2-3 실험: KLUE-RoBERTa Large + W1 가중치 (2026-10-08)

seed 42의 내부 검증에서 고정 임계값 0.5 Macro F1은 **0.637518**이었다. 이전
2-1 W1 Base의 0.640102보다 0.002584 낮고 목표 0.66에도 미치지 못했다.
2 epoch에서는 0.641799였으나 사전에 정한 최종 평가 대상은 4 epoch 모델이다.
Large는 micro batch 2 × 누적 4, 이전 Base는 batch 8 × 누적 1로 학습했으므로
인코더 크기의 효과만 분리한 비교는 아니다. 공식 Validation 점수도 아니다.

- [2-3 결과 보고서·증상별 지표](results/roberta_large_w1_seed42_2026-10-08/README.md)
- [2-3 실행 Colab 노트북](models/roberta/M3_2-3_RoBERTaLarge_W1_fixed05.ipynb)

## 2-4 실험: Base 다중 seed와 Large 고정 비율 앙상블 (2026-10-08)

W1 Base seed 42·123·2026을 동일 비율로 결합한 A는 Macro F1 **0.645129**,
A와 W1 Large seed 42를 50:50으로 결합한 B는 **0.647423**이었다. 임계값은
0.5로 고정했으며 결합 비율도 실행 전에 정했다. B는 Base seed 42보다 약
0.73%p 높지만 목표 0.66에는 미치지 못했다. 기존 내부 검증 예측을 결합한 탐색
결과이며 공식 Validation 결과가 아니다. TF-IDF는 포함하지 않았다.

- [2-4 결과 보고서·증상별 지표](results/fixed_ensemble_2026-10-08/README.md)
- [2-4 실행 Colab 노트북](models/roberta/M3_2-4_FixedSeedEnsemble_fixed05.ipynb)

## 2-5 실험: 가중 TF-IDF와 기존 앙상블 결합 (2026-10-08)

2-4 B 앙상블 80%와 가중 TF-IDF 20%의 고정 결합은 내부 검증 Macro F1
**0.650944**였다. 기존 B의 0.647423보다 약 0.35%p 높고 일반 TF-IDF 결합
0.648975보다도 높았다. 임계값은 0.5로 고정했다. 목표 0.66에는 미달했으며
오심 F1은 0.3815 → 0.3763으로 하락했다. 공식 Validation 결과가 아니다.

- [2-5 결과 보고서·증상별 지표](results/weighted_tfidf_blend_2026-10-08/README.md)
- [2-5 실행 Colab 노트북](models/roberta/M3_2-5_WeightedTFIDF_Blend_fixed05.ipynb)
