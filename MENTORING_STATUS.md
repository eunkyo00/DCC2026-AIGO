# Mission 1 · 2 · 3 멘토링 진행 보고

> **확인일: 2026-09-19** · GitHub [DDC2026-AIGO](https://github.com/eunkyo00/DDC2026-AIGO) 직접 조회 및 fetch 후 코드·결과·문서 대조.
> 원격 `main` / `mission/sumin`: `5b47c83` · 작업 개인 브랜치 `mission1/eunkyo`: `84919c5`.
> Mission 3 최신 변경은 원격에서 확인했다. 아래 GitHub 출처는 확인한 커밋으로 고정했다.

## 한눈에 보는 현재 상황

| Mission | 목표 | 현재 최고 기록 | 현재 단계 | 우선 다음 단계 |
|---|---|---|---|---|
| **1** | 신고자 음성의 남/여 분류 | **Accuracy 97.2485%** · 내부 검증 5,597통화 | Frozen Wav2Vec2 평가·기본 오류 비교 완료 | 개별 오답 분석, 정답 화자 정보 없는 통합 추론 검증 |
| **2** | 발화별 신고자/119대원 분류 | **Accuracy 90.2490%** · 공식 Validation 111,947발화 | Small-stem ResNet18 16 epoch 학습 기록·추론 코드 확보 | 역할·길이별 오답 분석, 전체 발화 활용 실험 |
| **3** | 대화 텍스트에서 9개 증상 다중 라벨 예측 | **Macro F1 0.6562** · 내부 검증 5,792통화 | RoBERTa·TF-IDF·앙상블 실험 기록 확보 | 오심 등 취약 증상 분석, 설정 고정 후 별도 평가 |

**세 점수는 평가 단위·분할·지표가 달라 직접 비교하지 않는다.** 이 문서는 저장된 기록을 대조한 결과이며 모델을 새로 학습하거나 원본 데이터로 재평가한 결과가 아니다. 목표·입력 제한·공식 지표는 [문제 분석][rules]에 근거한다.

## Mission 1 — 신고자 성별 분류

### 목표 → 현재 방법

- **목표:** 음성에서 신고자의 성별을 예측한다. 공식 지표는 Accuracy.
- **평가 설계:** 유효 Training 27,985통화를 Train 22,388 / 내부 검증 5,597로 고정 분할(seed 42, 성별 층화). 같은 통화의 발화는 같은 분할에 배정하며 scaler는 Train에만 fit한다.
- **현재 입력:** 정답 `speaker=1` 구간의 특징을 통화 단위로 집계한다. 공식 Validation 3,640통화 성능과는 구분한다.

| 방법 | 모델·주요 설정 | Accuracy | 결과 출처 |
|---|---|---:|---|
| 다수 클래스 기준선 | Female 고정 | 53.1892% | [F0 metrics][m1f0] |
| F0 + 음향 특징 | 발화별 F0·RMS 등 → 통화별 평균/중앙값 등 19차원 → Logistic Regression, max_iter 1,000 | 91.2096% | [F0 metrics][m1f0] |
| MFCC + SVM | 8kHz, 13 MFCC·26 mel, 50–3,800Hz; 통화별 54차원 → RBF SVM, C=1, gamma=scale | 95.0688% | [MFCC metrics][m1mfcc] |
| **Frozen Wav2Vec2 + LR** | facebook/wav2vec2-base, FP32·encoder 고정, 8→16kHz; 시간 평균→발화 동일 가중 평균 768차원; LR C=1 | **97.2485%** | [Wav2Vec2 metrics][m1ssl] |

### 성능에서 확인한 것

- MFCC 대비 **+2.1797%p**, 오답 **276 → 154통화**. Wav2Vec2 남성 정답률 96.7557%, 여성 97.6822%.
- Wav2Vec2만 정답 182건, MFCC만 정답 60건, 둘 다 오답 94건. 상호 보완 가능성은 있으나 앙상블 개선은 아직 검증하지 않았다. [오류 비교][m1ssl]

### 진행 상태 → 문제점 → 다음 단계

| 상태 | 확인된 작업 |
|---|---|
| **완료** | EDA·고정 split, 위 기준선 평가, 성별별 지표·혼동행렬·모델 간 오류 겹침 비교 |
| **진행 중인 별도 작업** | 로컬 미추적 `lightweight/logmel_cnn/`: 8kHz·40 mel·5,954 파라미터 CNN과 추출/학습 코드 존재. GitHub 게시 및 전체 학습 성능은 확인되지 않음 |
| **아직 하지 않은 것** | 개별 오답 원인 분석, 새 WAV/JSON → 제출 CSV 통합 추론; HuBERT·fine-tuning·앙상블 등의 완료 결과 없음 |

- **검증 한계:** 사람 식별자가 없어 동일 신고자의 분할 간 재등장을 검증할 수 없다. 라벨 구간 겹침만으로 실제 동시 발성을 단정할 수도 없다. [Mission 1 문서][m1readme]
- **제출과의 간극:** 현재 결과는 정답 화자 구간을 사용하지만 규정상 추론 시 화자 정답은 제공 입력이 아니다. 따라서 97.2485%를 그대로 최종 파이프라인 성능으로 볼 수 없다. [규정][rules] · [구간 선택 코드][m1code]
- **다음 단계:** 문서에 기록된 개별 오답 청취·분류를 우선 수행. 추가 제안으로, 화자 예측 등 정답 `speaker`에 의존하지 않는 방식을 정하고 성별 분류까지 연결한 성능을 별도로 측정한다.

## Mission 2 — 신고자 / 119대원 발화 분류

### 목표 → 현재 방법

- **목표:** 발화 하나의 음향만으로 `0=119대원 / 1=신고자`를 예측. 텍스트·발화 순서는 모델 입력에서 제외한다.
- **입력:** 8kHz 음성 → 앞 1.5초 crop/pad → 64-band log-Mel `[1,64,24]`. Training 889,358발화 / 공식 Validation 111,947발화.
- **모델:** ImageNet 초기화 Small-stem ResNet18, 입력 convolution `3×3/stride 1`, 초기 max-pool 제거. Batch 512, LR `2e-4`, dropout 0.25, label smoothing 0.03, Mixup α=0.10, SpecAugment·AdamW·cosine scheduler. 최고 epoch 16. [코드·설정][m2readme]

### 성능

| 비교 | Validation Accuracy | 근거 |
|---|---:|---|
| 신고자 고정 기준선 | 51.9013% | 역할별 수 58,102 / 111,947로 계산 · [집계][m2readme] |
| Small-stem, threshold 0.5 | 90.1400% | [metrics][m2metrics] |
| **Small-stem, 선택 threshold 0.535** | **90.2490%** | [metrics][m2metrics] · [16 epoch 로그][m2history] |

- Validation loss **0.273785**. 혼동행렬 `[[46396,7449],[3467,54635]]`(행=실제, 열=예측, 순서 0/1). 119대원 정답률 **86.17%**, 신고자 **94.03%**로 역할별 차이가 크다. [보고서][m2report]
- 기본 ResNet18·Hybrid 점수는 기존 README에 공유 요약으로만 남아 있고 해당 원본 로그·checkpoint가 없어, 위 검증 가능한 결과표에는 포함하지 않았다.
- 기록의 근거는 제공 checkpoint와 로그의 대조다. **원본 Validation 음성 재추론은 수행되지 않았고**, 가중치도 GitHub에 없다. 같은 Validation으로 epoch·threshold를 선택해 독립 Test 성능을 뜻하지 않는다. [출처 범위][m2report]

### 진행 상태 → 문제점 → 다음 단계

| 상태 | 확인된 작업 |
|---|---|
| **완료** | EDA·특징 사전 계산, Small-stem 학습 기록 확보, 단일 명령 CSV 추론 코드 구현 |
| **후속 구현 단계** | `first_plus_whole` 전처리·추론 분기와 길이별 오류 분석 코드 존재. 해당 개선의 완료 성능은 확인되지 않음 |
| **아직 하지 않은 것** | 공개 기록 기준 후속 특징 실험의 성능 비교, 원본 음성 기반 독립 재평가·제출 완료 확인 |

- **문제점:** 앞 1.5초 이후 음향이 잘린다. 119대원 오분류가 많지만 원인은 아직 확정하지 않았다.
- **다음 단계:** 짧은/긴 발화·무음·겹침별 오류를 분석하고 `first_1p5`와 전체 발화 활용 방식을 비교한다. 추가 제안으로, epoch·threshold를 고정한 평가와 WAV→CSV 전체 추론 시간 측정을 함께 남긴다. [기존 후속 방향][m2readme]

## Mission 3 — 9개 증상 다중 라벨 분류

### 목표 → 현재 방법

- **목표:** 대화 텍스트만으로 고열·구토·두통·복통·어지러움·열상·오심·전신쇠약·호흡곤란을 예측. 공식 지표는 **9개 클래스 Macro F1**.
- **전처리·분할:** `utterances[].text`를 순서대로 연결하고 타깃 밖 라벨만 제거한다. JSON 29,200통화 → Train 23,408 / 내부 검증 5,792. 원본 E0 split manifest를 기준으로 복원·확인했다. [구현 검사][m3checks]
- **주요 설정:** TF-IDF는 문자 2–5 gram + One-vs-Rest LR. RoBERTa는 512토큰 입력, E0 처음 510토큰 / E1·E3 앞255+뒤255에 특수 토큰 추가. Base는 batch 8·LR `2e-5`, Large는 batch 4·gradient accumulation 2·LR `1e-5`; 4 epoch, BCE, FP16. **현재 코드는 과거 실험을 재구성한 것으로 당시 seed·버전의 완전 일치는 미확인**이다. [모델 설정][m3model]

### 성능 — 내부 검증 Macro F1

| 실험 | threshold 0.5 | 증상별 threshold 조정 | 저장된 기록의 출처 수준 |
|---|---:|---:|---|
| TF-IDF + LR | — | 0.6028 | Colab 화면 전사·반올림 |
| E0 · RoBERTa Base First-512 | 0.6063 | 0.6426 | 조정값은 로컬 export 확인 기록; 기본값은 README |
| E1 · Base Head-Tail | 0.6092 | 0.6448 | Colab 화면 전사·반올림 |
| E2 · E1 0.55 + TF-IDF 0.45 | — | 0.6494 | 로컬 export 확인 기록 |
| E3 · Large Head-Tail | 0.6194 | 0.6479 | Colab 화면 전사·반올림 |
| E3 0.50 + TF-IDF 0.50 | — | 0.6519 | Colab 화면 전사·반올림 |
| **E3 0.35 + E2 0.65** | — | **0.6562** | Colab 화면 전사·반올림; 최종 개별 예측 미제공 |

출처: [metrics][m3metrics] · [결과/출처 보고서][m3report] · [README][m3readme]. `—`는 해당 값 미기록. **0.6562는 Accuracy 65.62%가 아니다.** 원본 화면·전체 학습 로그·가중치로 이번에 독립 재현한 점수도 아니다.

### 진행 상태 → 문제점 → 다음 단계

| 상태 | 확인된 작업 |
|---|---|
| **완료** | E0/E1/TF-IDF/E2/E3/앙상블 결과 기록, 전처리·split·평가·앙상블 코드 정리, 구현 검사 7개 통과 기록 |
| **후속 정리 단계** | 과거 실험 재현용 코드 확보. 원본 notebook·seed·버전 추가 대조가 남아 있음 |
| **아직 하지 않은 것** | E4 weighted BCE 결과, 공식 Validation 확정 성능, 전체 Training 최종 재학습·제출용 통합 추론 |

- **취약 클래스:** E3 조정 F1은 오심 **0.3841**, 두통 **0.5252**, 열상 **0.8966**. 증상별 편차가 크다. [클래스별 결과][m3report]
- **개선 효과·한계:** Head-Tail과 Large 단독 개선 폭은 작다. 증상별 threshold와 앙상블 비율을 동일 내부 검증에서 반복 선택해 점수가 낙관적일 수 있다. 최종 앙상블의 개별 예측 파일도 확보해야 한다.
- **다음 단계:** 기록된 계획대로 오심 등의 오답 텍스트·정답 대응을 점검하고 고정 manifest로 비교한다. E4는 미실시 후보로 두고, 후보 설정을 고정한 뒤 별도 평가 → 최종 재학습·추론 구현으로 진행한다. [남은 작업][m3report]

## 멘토링 질문

1. **Mission 1의 실전 검증:** 현재 97.2485%는 정답 신고자 구간을 쓴 결과입니다. 정답 화자 없이 추론해야 할 때 Mission 2 예측을 연결하는 방식부터 검증할까요? 화자 분류 오류가 성별 성능에 미치는 영향을 어떻게 분리 측정하면 좋을까요?
2. **Mission 1 개선의 우선순위:** MFCC만 맞힌 60건과 두 모델 공통 오답 94건이 있습니다. 오답 유형을 확인한 뒤 집계 방식·앙상블·fine-tuning 중 무엇부터 비교하고, 어느 정도 개선을 다음 단계 진행 기준으로 삼으면 좋을까요?
3. **Mission 2 입력 개선:** 119대원 정답률 86.17%와 신고자 94.03%의 차이를 줄이려면 길이별 오류 분석 후 전체 발화 활용과 학습 손실 조정 중 무엇을 먼저 시험할까요? 입력 길이 효과와 모델 효과는 어떻게 분리하면 좋을까요?
4. **Mission 3 취약 증상:** 오심 F1 0.3841의 원인을 부정 표현·정답 불일치·다른 증상과의 혼동으로 나눠 점검한 뒤, weighted BCE와 텍스트 표현 개선 중 어떤 증거를 기준으로 선택하면 좋을까요?
5. **검증·다음 단계 결정:** Mission 2의 epoch/threshold와 Mission 3의 threshold/앙상블 비율이 같은 검증 세트에서 선택됐습니다. 추가 내부 분할 또는 교차검증으로 선택 편향을 어느 정도 확인한 뒤 설정을 고정하고 최종 평가·제출 준비로 넘어가면 좋을까요?

[rules]: https://github.com/eunkyo00/DDC2026-AIGO/blob/5b47c836eded319609db3efc85ef09b999693305/0.%EC%A4%91%EC%9A%94/%EB%AC%B8%EC%A0%9C_%EB%B6%84%EC%84%9D.md
[m1readme]: mission-1/README.md
[m1f0]: mission-1/baseline/f0_lr/results/metrics.json
[m1mfcc]: mission-1/baseline/mfcc_svm/results/metrics.json
[m1ssl]: mission-1/ssl/wav2vec2_frozen/results/full_l4/metrics.json
[m1code]: mission-1/ssl/wav2vec2_frozen/run_wav2vec2_frozen.py
[m2readme]: mission-2/README.md
[m2metrics]: mission-2/smallstem_resnet18/results/metrics.json
[m2history]: mission-2/smallstem_resnet18/results/history.json
[m2report]: mission-2/smallstem_resnet18/results/REPORT.md
[m3checks]: https://github.com/eunkyo00/DDC2026-AIGO/blob/5b47c836eded319609db3efc85ef09b999693305/mission-3/results/IMPLEMENTATION_CHECKS.md
[m3model]: https://github.com/eunkyo00/DDC2026-AIGO/blob/5b47c836eded319609db3efc85ef09b999693305/mission-3/models/roberta/README.md
[m3metrics]: https://github.com/eunkyo00/DDC2026-AIGO/blob/5b47c836eded319609db3efc85ef09b999693305/mission-3/results/metrics.json
[m3report]: https://github.com/eunkyo00/DDC2026-AIGO/blob/5b47c836eded319609db3efc85ef09b999693305/mission-3/results/REPORT.md
[m3readme]: https://github.com/eunkyo00/DDC2026-AIGO/blob/5b47c836eded319609db3efc85ef09b999693305/mission-3/README.md
