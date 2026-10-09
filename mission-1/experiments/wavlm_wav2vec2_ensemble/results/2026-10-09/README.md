# 앙상블 실험 결과 · 2026-10-09 공유

**WavLM 75% + Wav2Vec2 25% → 98.213329%, 정답 5,497 / 오답 100.**
날짜는 결과 공유 기준이다. 반환 ZIP에는 실행 시작·종료 시각이 기록되지 않았다.

## WavLM 단독과 비교

| 항목 | WavLM 단독 | 앙상블 |
|---|---:|---:|
| 전체 정확도 | 98.088262% | **98.213329%** |
| 정답 / 5,597 | 5,490 | **5,497** |
| 오답 | 107 | **100** |
| Male 정확도 | 97.748092% | **97.900763%** |
| Female 정확도 | 98.387639% | **98.488411%** |

교정 11건, 회귀 4건, 모두 정답 5,486건, 모두 오답 96건.
전체 정확도 +0.125067%p, 오답 수 약 6.54% 감소. 99%까지 추가 정답 45건이 필요하다.

| 실제 성별 | 예측 F | 예측 M |
|---|---:|---:|
| F (2,977) | 2,932 | 45 |
| M (2,620) | 55 | 2,565 |

## Train OOF에서 정한 비중

| 후보 | OOF 정확도 | Fold 1 개선 | Fold 2 개선 | Fold 3 개선 | 선택 |
|---|---:|---:|---:|---:|---|
| WavLM 단독 | 98.052528% | — | — | — | 기준 |
| WavLM 75% / Wav2Vec2 25% | **98.173128%** | +0.080397%p | +0.200992%p | +0.080407%p | **채택** |
| WavLM 50% / Wav2Vec2 50% | 98.128462% | −0.066997%p | +0.254589%p | +0.040204%p | 제외 |

비중은 Train 22,388통화의 OOF에서만 선택했다. 선택 후 전체 Train로 다시 학습하고
Internal Validation 5,597통화에 적용했다. 최종 점수를 보고 비중을 변경하지 않았다.

## 시간·환경

- 반환 로그의 세션 시간: **9,054.8116초(2시간 30분 55초)**.
- Python 3.13.15 / NumPy 2.2.2 / SciPy 1.15.3 / scikit-learn 1.6.1.
- L4용 Colab에서 실행한 cache 재사용 실험. LR은 CPU에서 학습한다.
- identity에는 이번 세션 GPU 이름이 기록되지 않았다. GPU 추론 시간이나 단계별 소요시간은 이 ZIP으로 검증할 수 없다.
- 기존 WavLM 정답 수 5,490건 재현. 성별 정확도·혼동행렬도 반환 baseline 기록과 일치.

## 검증 범위

로컬에서 5,597개 unique ID 및 기존 WavLM과 동일한 평가 ID/정답 라벨을 확인했다.
확률 범위·합·argmax, 혼동행렬·정확도·성별 지표, 교정/회귀를 다시 계산했다.
원본 runner의 hash, run identity, 선택 잠금·모델 hash 참조 관계 및 반환된 OOF 선택 요약도 대조했다.
raw OOF 확률과 실제 모델 weight는 ZIP에 없어 재학습·모델 내용까지 독립 검증한 것은 아니다.

이 결과는 이미 관찰한 Internal Validation이며 새로운 독립 test가 아니다.
공식 Validation은 미사용이며 동일 화자 분리도 보장하지 않는다. 소폭 개선이 새 데이터셋에서도
유지된다는 결론이나 통계적으로 유의한 개선이라는 결론은 내리지 않는다.

## 원본·검증 파일

- [completion.json](completion.json): 최종 지표·WavLM baseline·세션 시간.
- [selection.lock.json](selection.lock.json): 후보 OOF 결과·선택 비중.
- [validation_predictions.csv](validation_predictions.csv): 통화별 최종 예측과 확률.
- [identity.json](identity.json): 원본 코드·cache·분할 hash·환경.
- [models.json](models.json), [validation_started.json](validation_started.json): 모델·선택 잠금 참조.
- [REPORT.md](REPORT.md): 실행 코드가 생성한 원본 보고서.
- [verification.json](verification.json): 로컬 재계산·파일 checksum·검증 한계.
- [실험 코드·Colab 실행 방법](../../README.md).
