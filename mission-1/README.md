# Mission 1 · 신고자 음성 성별 분류

신고자(caller)의 음성과 발화 구간으로 통화별 Male / Female을 예측한다.
**현재 최고 Internal Validation Accuracy는 Frozen Wav2Vec2 + LR의 97.248526%다.**
고정 ECAPA 평가까지 완료했으며 99% 목표는 아직 달성하지 못했다. 기록 기준: 2026-09-22.

[전체 실험 이력](EXPERIMENT_LOG.md) · [후속 실험 계획](EXPERIMENT_PLAN_99.md) ·
[Wav2Vec2 결과](ssl/wav2vec2_frozen/results/REPORT.md) ·
[ECAPA 결과](experiments/gender_model_comparison/validation_results/REPORT.md)

## 동일 Internal Validation 결과

모든 모델은 Training 원본에서 분리한 **같은 5,597통화(Male 2,620 / Female 2,977)**로 평가했다.
성별별 Accuracy는 해당 실제 성별 통화 중 정답 비율이다.

| 모델 | 정답 | 오답 | Overall | Male | Female |
|---|---:|---:|---:|---:|---:|
| Majority · Female 고정 | 2,977 | 2,620 | 53.189209% | 0.000000% | 100.000000% |
| F0 + Acoustic → LR | 5,105 | 492 | 91.209577% | 89.961832% | 92.307692% |
| MFCC → RBF SVM | 5,321 | 276 | 95.068787% | 94.809160% | 95.297279% |
| **Frozen Wav2Vec2 → LR** | **5,443** | **154** | **97.248526%** | 96.755725% | **97.682230%** |
| Frozen Gender ECAPA | 5,419 | 178 | 96.819725% | **97.824427%** | 95.935506% |

ECAPA는 Wav2Vec2의 오답 83개를 교정했지만 정답 107개가 회귀해, 정답이 24개 적었다.
Male은 개선되고 Female은 하락했다. 현재 최고 모델은 Wav2Vec2로 유지한다.
99%에는 최소 **5,542 정답 / 최대 55 오답**이 필요하다.

| 실측 작업 | 시간 | 측정 범위 |
|---|---:|---|
| Wav2Vec2 L4 embedding 추출·검사 | 11.19시간 | 정상 Training 원본 27,985통화; 읽기·저장·검사 포함 |
| Wav2Vec2 특징 → LR 학습 | 약 3초 | 저장된 특징으로 로컬 학습 |
| ECAPA L4 추론·Drive 저장 | 2시간 8분 41초 | Internal Validation 5,597통화; 모델 로딩·최종 ZIP 생성 제외 |

처리 통화 수와 작업 범위가 달라 위 시간을 모델 간 속도 배율로 비교하지 않는다.

## 지금까지 한 시도

| 단계 | 상태 | 확인한 내용 |
|---|---|---|
| EDA·고정 분할 | 완료 | 정상 통화·caller 구간 검증, Train/Validation 통화 단위 분리 |
| Majority → F0 → MFCC → Wav2Vec2 | 완료 | 같은 Validation에서 baseline 비교 |
| 저장된 Wav2Vec2 특징의 LR/SVM·MFCC 결합 | 완료 | 8개 설정 × 3-fold; 최고 Train CV 평균 97.453991% |
| Train OOF 오류 분석 | 정량 분석 완료 | 길이·segment 수·무음·F0 연관 분석, 청취용 48통화 표본 준비 |
| ECAPA / WavLM Train 200통화 비교 | 예비 비교 완료 | ECAPA 99%, WavLM 95%; 사용자 전달 출력 기준, 원본 ZIP 독립 검증은 미완료 |
| 고정 ECAPA Validation | 평가·로컬 검증 완료 | 96.819725%, 5,597개 exact coverage, 기록된 실패 0건 |
| 최신 모델 조사·추가 학습 설계 | 제안·인계 단계 | MERaLiON-GR, WavLM 학습, Wav2Vec2 부분 fine-tuning 후보 정리 |

**Train CV·Train 200통화·Internal Validation은 서로 다른 평가다.**
Train 200통화의 99%나 MFCC 결합 CV 결과를 위 Validation 성능표에 섞지 않는다.
기존 Wav2Vec2 본체는 완전히 고정했고 LR/SVM만 학습했다. 부분 fine-tuning은 완료 결과가 없다.
추가 Train 2,000통화 비교와 전체 원본에 대한 성별 모델 추론은 생략했다.
자세한 후보별 수치와 선택 이유는 [실험 이력](EXPERIMENT_LOG.md)에 정리했다.

## 데이터와 해석 범위

- 정상 Training 원본 27,985통화, caller 442,639구간 → Train 22,388 / Internal Validation 5,597.
- 실제 8kHz mono WAV, JSON `speaker=1` 구간 사용. `speaker=0`은 상담원이다.
- seed 42의 고정 [split_assignments.csv](validation/split_assignments.csv)를 모든 실험에서 사용한다.
- 공식 Validation 3,640통화는 별도 데이터이며 위 성능표에 포함하지 않는다.
- 사람 식별자가 없어 통화 단위 분리가 사람 단위 분리를 보장하지 않는다.
- 이미 여러 baseline에서 관찰한 Internal Validation이므로 새 독립 test라고 부르지 않는다.
- 잡음·다중 화자·라벨 오류 원인은 아직 청취로 확정하지 않았다.
- ECAPA와 Wav2Vec2는 통화 집계 방식도 다르다. 전체 파이프라인 비교이며 차이 전부를 backbone 효과로 해석하지 않는다.

원본 매칭 실패와 음향 품질은 [EDA 보고서](eda/REPORT.md), 분할 검증은
[Validation 보고서](validation/REPORT.md)에서 확인한다.

## 파일 안내

| 찾는 내용 | 문서·폴더 |
|---|---|
| 완료된 시도와 결과·판단 | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) |
| 아직 실행하지 않은 후보·평가 규칙 | [EXPERIMENT_PLAN_99.md](EXPERIMENT_PLAN_99.md) |
| F0 baseline | [baseline/f0_lr/](baseline/f0_lr/README.md) |
| MFCC baseline | [baseline/mfcc_svm/](baseline/mfcc_svm/README.md) |
| Frozen Wav2Vec2 | [ssl/wav2vec2_frozen/](ssl/wav2vec2_frozen/README.md) |
| Wav2Vec2 최종 L4 지표·예측 | [results/full_l4/](ssl/wav2vec2_frozen/results/full_l4/metrics.json) |
| Train 내부 분류기·특징 결합 비교 | [experiments/embedding_classifiers/](experiments/embedding_classifiers/README.md) |
| ECAPA/WavLM 예비 비교와 ECAPA 최종 검증 | [experiments/gender_model_comparison/](experiments/gender_model_comparison/README.md) |
| ECAPA Colab 재현·resume 안내 | [ECAPA_VALIDATION_README.md](experiments/gender_model_comparison/ECAPA_VALIDATION_README.md) |

과거 인계 문서와 초기 smoke/benchmark 결과는 당시 단계의 기록으로 보존한다.
Wav2Vec2 `results/` 바로 아래의 초기 Mac 측정과 `results/full_l4/` 최종 결과를 구분한다.
원본 WAV·가중치·embedding cache·업로드 ZIP은 GitHub에 포함하지 않는다.
별도 경량 모델 및 다른 창의 작업은 이 문서의 검증 완료 결과에 포함하지 않았다.
