# Mission 1 · 실험 이력과 판단

2026-09-22 기준. [현재 성능표](README.md) · [다음 실험 계획](EXPERIMENT_PLAN_99.md).
검증된 결과, 사용자 전달 예비 결과, 아직 실행하지 않은 제안을 구분한다.

## 1. 데이터 확인과 평가 고정

EDA에서 파일 매칭·라벨·WAV 헤더·시간 경계를 확인했다. 정상 Training 27,985통화의
caller 442,639구간을 통화 단위로 Train 22,388 / Internal Validation 5,597에 배정했다.
성별을 유지한 seed 42 분할이며 공식 Validation 3,640개와 구분한다.

| 검증 원본 | SHA256 |
|---|---|
| `validation/split_assignments.csv` | `04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06` |
| `validation/manifests/calls.csv` | `dc57d0f7bea6e06017ac3e27444a0ff5bb299e401ea787400e84cbdf7d072e23` |
| `eda/eda_outputs/segments.csv` | `8bf8c83c0190232e4500b4067bc28d2611b09dd227a7ce4dc9c3741e6dcce584` |

원본 metadata 일부는 Git 제외 대상이다. [EDA](eda/REPORT.md) · [분할 검증](validation/REPORT.md).

## 2. baseline을 단계적으로 비교

| 방법 | 변경 내용 | Internal Validation | 판단 |
|---|---|---:|---|
| Majority | Female 고정 | 53.189209% | 최소 비교 기준 |
| F0 + Acoustic + LR | 높낮이·음량·유성음·길이 특징 | 91.209577% | 기본 음향 정보로 구분 가능 |
| MFCC + RBF SVM | MFCC 통화 특징 54차원, C=1 | 95.068787% | F0보다 개선 |
| Frozen Wav2Vec2 + LR | 768차원 사전학습 특징, Train-only Scaler/LR | 97.248526% | 현재 최고 |

Wav2Vec2는 `facebook/wav2vec2-base`의 본체를 FP32·eval·Frozen으로 고정했다.
마지막 hidden state의 시간 평균 → segment 동일 가중 통화 평균을 사용했다.
본체 학습 가능 파라미터는 0개이며 LR만 학습했다. **부분 fine-tuning 결과가 아니다.**

MFCC 대비 Wav2Vec2는 182개 오답 교정, 60개 정답 회귀, 공통 오답 94개였다.
[Wav2Vec2 최종 보고서](ssl/wav2vec2_frozen/results/REPORT.md).

## 3. 기존 특징으로 LR·SVM·MFCC 결합 비교

원본 음성이나 GPU를 다시 사용하지 않고 같은 Train 22,388통화의 3-fold로 8개 설정을 비교했다.
각 fold의 Scaler는 해당 fold의 학습 통화에만 fit했다. 총 24회 학습을 완료했다.

| 후보 | Train CV 평균 Accuracy | Train OOF 오답 |
|---|---:|---:|
| **Wav2Vec2 + MFCC + LR, C=0.1** | **97.453991%** | **570** |
| Wav2Vec2 + LR, C=0.1 | 97.297651% | 605 |
| Wav2Vec2 + MFCC + LR, C=1 | 97.217253% | 623 |
| Wav2Vec2 + LR, C=1 | 97.096657% | 650 |
| Wav2Vec2 + SVM, C=1 | 96.663390% | 747 |
| Wav2Vec2 + LR, C=10 | 96.654461% | 749 |
| Wav2Vec2 + SVM, C=10 | 96.654454% | 749 |
| Wav2Vec2 + MFCC + LR, C=10 | 96.565125% | 769 |

CV 평균은 fold별 정확도의 단순 평균이며 pooled OOF 정확도와 미세하게 다를 수 있다.
최고 결합 후보는 LR C=1 대비 174개 교정·94개 회귀로 오답이 80개 줄었다.
하지만 99%에는 미달했고, 이 후보들의 새 Internal Validation 평가는 하지 않았다.
[재현 코드·안내](experiments/embedding_classifiers/README.md).

## 4. Train OOF 정량 오류 분석

- 8개 후보가 모두 틀린 통화는 298개였다.
- 최고 결합 후보의 570오답은 Male 290 / Female 280이었다.
- 오류는 짧은 발화나 높은 무음 비율에만 집중되지 않았다.
- 오답 집단에서 추정 F0가 성별의 전형적 방향과 다른 경향을 관찰했다. 원인을 확정한 것은 아니다.
- 정답 대조군을 포함한 48통화 청취표를 준비했지만 실제 청취·라벨 재판정은 하지 않았다.

[전체 분석](experiments/embedding_classifiers/ERROR_ANALYSIS.md).

## 5. 외부 성별 모델 ECAPA / WavLM 예비 비교

같은 Train 200통화(Male 101 / Female 99), 같은 caller 음성으로 비교했다.
초기 2,000통화 계획은 전체 완료하지 않고 200통화 공통 표본 비교로 정리했다.
아래 수치는 사용자 제공 Colab 출력 기준이며 **Train 200 원본 ZIP의 로컬 독립 검증은 미완료**다.

| 모델 | 정답 / 200 | Accuracy | Fusion OOF 대비 교정 | 회귀 | 공통 오답 |
|---|---:|---:|---:|---:|---:|
| Wav2Vec2 + LR OOF | 195 | 97.5% | — | — | — |
| Wav2Vec2 + MFCC + LR OOF | 195 | 97.5% | — | — | — |
| Gender ECAPA | 198 | 99.0% | 4 | 1 | 1 |
| Gender WavLM | 190 | 95.0% | 1 | 6 | 4 |

ECAPA는 `JaesungHuh/voice-gender-classifier`, WavLM은 `tiantiaf/wavlm-large-age-sex`를 사용했다.
둘 다 추가 학습 없이 사용한 외부 성별 classifier이며, 우리 Train으로 backbone을 학습한 실험이 아니다.

**결정:** ECAPA의 설정을 고정하고 기존 Internal Validation을 평가한다.
추가 2,000통화 비교와 정상 원본 전체 성별 추론은 생략한다.
[예비 비교 보고서](experiments/gender_model_comparison/SCREEN_200_REPORT.md).

## 6. 고정 ECAPA Internal Validation 평가

5,597통화·caller 88,290구간을 모두 사용했다. 8kHz crop과 기존 polyphase resampling,
caller 시간순 연결, 최대 15초 균등 창, 실제 샘플 수 가중 확률 평균을 고정했다.
FP32·Frozen·batch 1이며 추가 학습·threshold 조정·전처리 탐색을 하지 않았다.

| 항목 | 검증 결과 |
|---|---|
| Overall | **96.819725%**, 5,419 정답 / 178 오답 |
| Male | 97.824427%, 2,563 / 2,620 |
| Female | 95.935506%, 2,856 / 2,977 |
| Confusion matrix | 행=실제, 열=예측, M/F: `[[2563,57],[121,2856]]` |
| Wav2Vec2 대비 | 교정 83 / 회귀 107 / 공통 오답 71 |
| MFCC 대비 | 교정 204 / 회귀 106 / 공통 오답 72 |
| 실행 | NVIDIA L4, 추론·Drive 저장 7,720.671초 |
| 무결성 | 5,597 unique IDs exact coverage, 확률·argmax·샘플 수·hash·identity 검증 통과 |
| 실패 기록 | 과거 실패 0건 / 미해결 실패 0건 |

반환 ZIP SHA256은 `2a26d6d23a15d182c9e7fbf7085a0eb6a4bf409f0bb2e6b7b7aad67bc835271e`다.
실행 이후 변경한 패키지 오류 표시 문구만 이전 상태로 복원해 당시 bundle의 모든 파일 hash를 검증했다.
검사를 비활성화하거나 결과 identity를 수정하지 않았다.

**판단:** Train 200통화의 99%는 전체 Validation에서 유지되지 않았다.
Wav2Vec2보다 0.428801%p 낮고 오답이 24개 많아, 기존 최고 baseline을 유지한다.
ECAPA는 Male 성능은 높았지만 Female 성능이 낮았다. Validation에 맞춰 반복 튜닝하지 않는다.

[검증 보고서](experiments/gender_model_comparison/validation_results/REPORT.md) ·
[기계 판독 지표](experiments/gender_model_comparison/validation_results/metrics.json) ·
[검증 출처](experiments/gender_model_comparison/validation_results/verification_provenance.json).

## 7. 실행 중 해결한 문제

| 문제 | 해결·검증 |
|---|---|
| torchaudio CUDA wheel 불일치 | 기존 torch를 유지하고 공식 cu128의 동일 버전 torchaudio 사용 |
| venv의 패키지 버전 차이 | Train identity와 비교해 불일치 패키지만 기존 버전으로 복구 |
| preflight 오류에 차이 항목이 안 보임 | 패키지별 기존/현재 버전이 보이도록 진단 메시지 보완 |
| 긴 Drive 추론 중 중단 위험 | 통화별 flush/fsync, identity 검사, 미완성 마지막 행 복구, 실패 기록 |
| 중복 실행·Stop 후 프로세스 잔류 | output lock, subprocess group 종료, 수동 stale lock 복구 안내 |
| 불확실한 실행시간 | 기존 기록 기반 예상 후 실제 wall time으로 갱신; 최종 실측 시간 별도 기록 |

13개 CPU 테스트로 입력 샘플 보존, 손상·중복 거부, 실패 후 resume, 합성 결과 export/분석을 확인했다.
실제 GPU 추론은 사용자가 Colab에서 실행했고, 반환 ZIP은 로컬에서 독립 검증했다.

## 8. 아직 완료하지 않은 것

MERaLiON-GR 조사, WavLM 자체 Train 학습, Wav2Vec2 부분 fine-tuning은 후속 후보다.
WavLM·Wav2Vec2 작업을 다른 창에서 시작할 인계 프롬프트는 작성했으나,
이 기록에 반영할 학습 완료·검증 결과는 없다. 구체적 후보와 평가 규칙은
[후속 계획](EXPERIMENT_PLAN_99.md)에 모았다.

표본 청취, 새로운 집계 실험, augmentation 비교, ensemble·threshold tuning,
새 WAV/JSON에서 예측 CSV까지 이어지는 통합 추론 진입점 역시 완료 결과에 포함하지 않는다.
