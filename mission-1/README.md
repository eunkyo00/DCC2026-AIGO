# Mission 1 · 신고자 음성 성별 분류

**최종 제출 모델은 Frozen WavLM-Large + Wav2Vec2-base 특징에 각각 학습한 Logistic Regression(LR)의 확률을 50:50으로 결합한다.** Training 29,200통화 전체로 제출용 LR을 재학습했으며, 공식 Validation 3,640통화에서 **98.241758% (3,576정답 / 64오답)**를 기록했다. 사전학습 음성 모델 본체는 동결했고 fine-tuning하지 않았다.

[최종 제출 코드·실행 노트북](final_submission/README.md) · [학습·선택 기록](final_submission/run_records/) · [공식 Validation 예측 CSV](final_submission/outputs/mission1.csv) · [실험별 안내](EXPERIMENT_INDEX.md)

## 진행 과정

### 1. 데이터와 평가 기준 구성

119 통화의 WAV와 JSON을 파일명으로 매칭하고, JSON의 `speaker=1`인 신고자 구간을 `startAt`/`endAt`(ms)으로 잘랐다. 상담원(`speaker=0`) 구간은 분류 대상에서 제외했다. 성별은 학습 정답으로만 사용하며, 추론에서는 텍스트·성별·주소를 모델 입력으로 사용하지 않는다.

초기 실험은 당시 파이프라인에서 사용한 **27,985통화**를 Train 22,388 / Internal Validation 5,597로 분리했다. 같은 통화의 발화가 양쪽에 섞이지 않도록 통화 단위로 고정했고, 동일 분할에서 모델과 오답을 비교했다. 반복되는 약 329Hz 추정값을 합성 신호와 스펙트럼으로 점검해 F0 품질 기준에 반영했다. [EDA](eda/REPORT.md)와 [분할 기록](validation/REPORT.md)은 당시 조건의 기록이다.

### 2. 음향 특징에서 사전학습 음성 표현으로 확장

F0·음향 특징 + LR, MFCC + RBF SVM, Frozen Wav2Vec2 + LR을 차례로 비교했다. 고정 Gender ECAPA와 WavLM-Large도 평가했다. 단독 정확도뿐 아니라 이전 모델만 맞힌 통화를 확인해 특징·모델의 보완 가능성을 살폈다.

| 초기 동일 Internal Validation 5,597통화 | 정답 | 오답 | 정확도 |
|---|---:|---:|---:|
| Female 고정 baseline | 2,977 | 2,620 | 53.189209% |
| F0 + Acoustic → LR | 5,105 | 492 | 91.209577% |
| MFCC → RBF SVM | 5,321 | 276 | 95.068787% |
| Frozen Wav2Vec2 → LR | 5,443 | 154 | 97.248526% |
| Frozen Gender ECAPA | 5,419 | 178 | 96.819725% |
| Frozen WavLM-Large → LR | 5,490 | 107 | 98.088262% |
| WavLM 75% + Wav2Vec2 25% | 5,497 | 100 | 98.213329% |

저장된 Wav2Vec2 특징의 LR/SVM·MFCC 결합을 Train 3-fold에서 비교하고, OOF 오답을 길이·무음·발화 수·F0 지표로 분석했다. 오류 원인을 잡음·화자 혼입·라벨 오류로 확정하지는 않았다. WavLM은 Wav2Vec2 대비 82건을 교정하고 35건의 신규 오류가 생겼다.

### 3. 초기 75:25 앙상블 검증

기존 특징 캐시를 재사용해 Train 3-fold OOF에서 75:25와 50:50을 비교했다. 당시 규칙은 모든 fold에서 WavLM 단독 대비 하락이 없고 전체 정확도 +0.1%p 이상 개선하는 것이었다. 75:25를 채택하고 Internal Validation에서 11건 교정·4건 신규 오류, 순오답 7건 감소를 확인했다. [상세 결과](experiments/wavlm_wav2vec2_ensemble/results/2026-10-09/README.md)

### 4. 전체 29,200통화로 최종 재실험

최종 제출에서는 WAV/JSON **29,200쌍 모두**를 다시 검사했고 정상 29,200·오류 0건을 확인했다. 이전 특징·분할·LR을 가져오지 않고 두 모델의 특징 추출과 LR 학습을 새로 실행했다.

1. 성별 층화·seed 42로 Train **23,360** / Internal Validation **5,840**통화를 분리했다.
2. 신고자 구간을 8→16kHz로 변환했다. WavLM은 연결된 신고자 음성을 최대 12초 창으로 나누고 마지막 층 특징을 실제 샘플 수로 가중 평균했다. Wav2Vec2는 발화별 최대 15초 창 특징을 집계한 뒤 발화 간 평균을 사용했다. 고정 길이 버킷 패딩을 적용했다.
3. 두 backbone은 동결하고 표준화 + LR만 학습했다. C는 WavLM 0.1, Wav2Vec2 1.0으로 고정했으며, OOF fold마다 해당 학습 부분에서만 표준화와 LR을 fit했다.
4. Train 3-fold OOF에서 WavLM 비중 **1.0 / 0.75 / 0.5 / 0.25 / 0.0**을 비교해 **정답 수 최대**인 50:50을 선택했다. 50:50은 22,937정답, 75:25는 22,933정답이었다. 이는 초기 실험의 채택 규칙과 다르다.
5. 비중을 고정한 뒤 내부 검증을 평가했다. 이후 제출용 LR은 **Training 29,200통화 전체**로 다시 학습했다.
6. 두 backbone 가중치·설정·LR을 `model.pt`에 담아 내보내고 실제 CUDA 음성 3건의 재추론을 확인했다. 최종 추론 명령으로 공식 Validation 예측 CSV를 생성해 채점했다.

[데이터 검사](final_submission/run_records/dataset_check.json) · [OOF 선택](final_submission/run_records/selection.json) · [내부 검증](final_submission/run_records/metrics.json) · [모델 내보내기 검증](final_submission/run_records/export_verification.json)

## 최종 결과와 평가 범위

| 평가 | 모델 | 정답 / 전체 | 정확도 |
|---|---|---:|---:|
| 새 Internal Validation | WavLM + LR | 5,724 / 5,840 | 98.013699% |
| 새 Internal Validation | Wav2Vec2 + LR | 5,661 / 5,840 | 96.934932% |
| 새 Internal Validation | 50:50 앙상블 | 5,730 / 5,840 | 98.116438% |
| **공식 Validation** | **전체 Training 재학습 후 50:50 앙상블** | **3,576 / 3,640** | **98.241758%** |

공식 Validation의 성별별 정답률은 Female **1,929/1,959 (98.468606%)**, Male **1,647/1,681 (97.977394%)**이며, 혼동행렬은 행=실제 F/M, 열=예측 F/M 기준 `[[1929, 30], [34, 1647]]`이다. 공식 평가 수치는 [실행된 노트북](final_submission/Mission1_full29200_train.ipynb)과 [제출 자료 설명](final_submission/README.md)에 기록되어 있다.

새 내부 검증 수치는 23,360통화로 학습한 평가 모델의 결과다. 29,200통화 전체 재학습 모델의 별도 독립 test 점수로 해석하지 않는다. 공식 Validation은 학습·비중 선택에 사용하지 않고 학습 후 채점에만 사용했다. 비공개 Test 결과는 없다.

초기 98.213329%와 최종 내부 98.116438%는 데이터 수·분할·패딩 방식·선택 규칙이 달라 직접적인 성능 증감으로 비교하지 않는다. 통화 단위 분할은 동일 화자의 다른 통화가 양쪽에 있을 가능성까지 배제하지 않는다. 99% 목표는 최종 공식 Validation에서도 달성하지 못했다.

## 실행과 제출 파일

실제 최종 실행본은 **`final_submission/`**이다. 다른 실험 폴더의 추론 코드를 섞지 않는다.

```bash
cd mission-1/final_submission
python -m pip install -r requirements.txt
python inference.py --audio_dir WAV_FOLDER --label_dir JSON_FOLDER --ckpt_path model.pt --output outputs/mission1.csv

# Training 원본으로 새 결과 폴더에서 학습
python -u run_full29200.py --source TRAINING_FOLDER --run-dir NEW_RESULT_FOLDER
```

GPU용 PyTorch와 드라이버는 실행 환경에 맞게 준비한다. 최종 실행 환경은 Windows 11, RTX 5060 Laptop GPU 8GB, Python 3.11.9, PyTorch 2.8.0+cu128, BF16이며, 세부 버전은 [requirements-lock.txt](final_submission/requirements-lock.txt)에 있다. 실행된 노트북 전체 기록은 약 163.7분, 공식 Validation 추론 셀은 약 17분이다. 서로 다른 환경의 속도를 보장하는 값은 아니다.

| 제출 구성 | 파일 |
|---|---|
| 추론 진입점 | [inference.py](final_submission/inference.py) |
| 모델·전처리·학습·추론 구현 | [run_full29200.py](final_submission/run_full29200.py) |
| 출력과 학습 로그가 있는 노트북 | [Mission1_full29200_train.ipynb](final_submission/Mission1_full29200_train.ipynb) |
| 패키지 버전 | [requirements.txt](final_submission/requirements.txt), [requirements-lock.txt](final_submission/requirements-lock.txt) |
| 실행 근거 | [run_records/](final_submission/run_records/) |
| 공식 Validation 예측 | [outputs/mission1.csv](final_submission/outputs/mission1.csv) |
| 학습 완료 가중치 | `model.pt` — 별도 최종 제출 ZIP에 포함 |

`model.pt`는 약 1.64GB로 GitHub에 포함하지 않는다. SHA-256은 `abbfae10784130564362c0e2d6153f9458d154d9261abdf300c30407d29a17a7`이다. 원본 음성·라벨과 특징 캐시도 저장소에서 제외한다.

과거 완료 실험, 미실행 후보, 대체 실행 코드는 [실험별 안내](EXPERIMENT_INDEX.md)에서 구분한다. [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), [EXPERIMENT_PLAN_99.md](EXPERIMENT_PLAN_99.md), [PPT_SUPPORT.md](PPT_SUPPORT.md)는 이전 실험 조건의 이력이며 최종 29,200통화 실험 설정은 위 설명과 `final_submission/`을 따른다.
