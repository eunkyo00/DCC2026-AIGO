# Train 200통화 비교 결과와 다음 결정

후속 상태: 고정 ECAPA Validation은 **96.819725%**로 평가·로컬 검증을 완료했다.
[최종 결과](validation_results/REPORT.md). 아래 내용은 Train 200통화 예비 비교 기록이다.

기록: 2026-09-22. 사용자 제공 Colab `comparison.json` 출력 기준이다.
`screen200_results.zip` 원본을 로컬로 받아 call_id·캐시 hash를 독립 재검증하는 작업은 아직 남아 있다.

## 비교 범위

Train 2,000통화 후보 중 원래 call_id 정렬 순서의 첫 200통화를 고정했다(Male 101 / Female 99).
정오답을 보고 선택하지 않았다. 두 새 모델은 같은 caller 음성과 집계 방식을 사용했다.
기존 LR/fusion 값은 동일 통화의 Train OOF 예측이다. 아래 수치는 Internal Validation 결과가 아니다.

| 모델 | 정답 / 200 | Overall | Male | Female | Fusion 오답 교정 | Fusion 정답 회귀 | 공통 오답 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wav2Vec2 + LR OOF | 195 | 97.5% | 96.039604% | 98.989899% | — | — | — |
| Wav2Vec2 + MFCC + LR OOF | 195 | 97.5% | 96.039604% | 98.989899% | — | — | — |
| Gender ECAPA | 198 | 99.0% | 99.009901% | 98.989899% | 4 | 1 | 1 |
| Gender WavLM | 190 | 95.0% | 90.099010% | 100.0% | 1 | 6 | 4 |

혼동행렬은 행=실제, 열=예측, 순서=M/F이다.
기존 LR/fusion 각각 `[[97, 4], [1, 98]]`, ECAPA `[[100, 1], [1, 98]]`, WavLM `[[91, 10], [0, 99]]`.

## 판단

ECAPA는 기존 fusion보다 정답이 3개 많아 후속 평가 후보로 선택했다.
200통화에서는 한 오류가 0.5%p이므로 통계적으로 우월하거나 전체에서 99%라고 결론내리지 않는다.
WavLM 확대 실행은 보류한다. 새로운 학습 없이 사용하는 성별 분류용 사전학습 모델이므로
Train 전체를 추론한다고 모델이 학습되거나 성능이 높아지는 것은 아니다.

사용자와 합의한 다음 단계는 추가 2,000통화 비교를 생략하고 **ECAPA 설정을 잠근 뒤
기존 Internal Validation 5,597통화를 한 번 평가**하는 것이다. 전체 27,985통화 추론은 이번 범위에 없다.

## 고정할 ECAPA 설정

- `JaesungHuh/voice-gender-classifier`, revision `db1222153bd60337e900be22add7af180452adc0`.
- 기존 `vendor/ecapa_model.py`, checkpoint strict loading, FP32 / eval / Frozen / batch size 1.
- autocast·TF32·FP16·BF16·padded batching 금지. 기존 200통화 실험과 동일한 전처리 유지.
- caller 모든 구간을 8kHz에서 `round(start*8000)`~`round(end*8000)`으로 crop.
- `scipy.signal.resample_poly(up=2, down=1, window=('kaiser',5.0))`, float32.
- caller crop을 시간순 연결하고 `np.array_split`로 최대 15초의 균등 창으로 분할.
- 3초 미만 단일 입력만 3초까지 0-padding, 실제 음성 샘플 수를 가중치로 사용.
- 창별 softmax, 원본 M/F를 공통 F/M 순서로 변환, 실제 샘플 수 가중 확률 평균, argmax.
- 추가 LR·Scaler 학습, threshold tuning, ensemble, fine-tuning 없음.

ECAPA는 기존 Wav2Vec2의 segment 동일 가중 embedding 평균과 집계 방식이 다르다.
이번 비교는 전체 파이프라인 비교이며 차이 전부를 backbone 효과라고 해석하지 않는다.

## 당시 결정한 후속 평가 및 저장

기존 split SHA256 `04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06`를 유지한다.
Internal Validation은 Training 원본에서 분리한 5,597통화(Male 2,620 / Female 2,977)이며
공식 Validation 원본 3,640통화와 다르다. 전체 Validation caller 구간을 사용한다.

Colab에서 통화별 F/M 확률, 예측, call_id, 완료/실패 로그, 실행 identity를 Drive에 지속 저장한다.
기존 Train screen cache와 구분되는 새 폴더를 사용하고 동일 실행 조건에서 resume한다.
로컬에서는 5,597 고유 ID의 정확한 coverage, 실패 수, 유한 확률, normalization·예측 일치,
Overall/Male/Female Accuracy, confusion matrix, MFCC·Wav2Vec2와의 오류 겹침을 검증한다.
99% 이상 기준은 정답 최소 5,542개 / 오답 최대 55개이다.

기존 baseline 표에 ECAPA를 추가하는 것은 해당 검증 후에만 수행한다.
Internal Validation은 이전 baseline에서 이미 관찰했으므로 새 독립 test set이라고 부르지 않는다.
이 결과를 보고 반복 튜닝하지 않고 후속 개선은 Train에서 개발한다.

[다른 창에 전달할 프롬프트](ECAPA_VALIDATION_HANDOFF.md)
