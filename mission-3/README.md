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
