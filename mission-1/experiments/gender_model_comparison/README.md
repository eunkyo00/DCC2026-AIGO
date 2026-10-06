# 성별 사전학습 모델 비교 · ECAPA Validation

**완료:** Train 200통화 예비 비교 후 ECAPA를 고정해 Internal Validation 5,597통화를 평가했다.
ECAPA는 **96.819725% (5,419정답 / 178오답)**로 기존 Wav2Vec2의 97.248526%보다 낮았다.
기록 기준: 2026-09-22. [Mission 1 전체 이력](../../EXPERIMENT_LOG.md).

## 결과와 결정

| 평가 범위 | 결과 | 검증 상태 |
|---|---|---|
| Train 200통화 | ECAPA 99%, WavLM 95%, 기존 LR/fusion OOF 97.5% | 사용자 전달 출력 기준; 원본 ZIP 독립 검증 미완료 |
| Internal Validation 5,597통화 | ECAPA 96.819725%; Male 97.824427%, Female 95.935506% | 반환 ZIP·모든 ID·확률·샘플 수·hash·identity 로컬 검증 완료 |
| 추가 Train 2,000통화 / 원본 전체 추론 | 생략 | 성능 결과 없음 |

ECAPA는 Wav2Vec2 대비 오답 83개를 교정했지만 정답 107개가 회귀했다.
현재 최고 모델은 Wav2Vec2로 유지하고 Validation 점수에 맞춘 ECAPA 튜닝은 하지 않는다.
후속 학습 실험은 [별도 계획](../../EXPERIMENT_PLAN_99.md)으로 분리했다.

## 먼저 볼 파일

| 목적 | 파일 |
|---|---|
| 최종 성능·오류 비교 | [validation_results/REPORT.md](validation_results/REPORT.md) |
| 정량 지표·실측 시간 | [metrics.json](validation_results/metrics.json) |
| 반환 ZIP·실행 코드 검증 근거 | [verification_provenance.json](validation_results/verification_provenance.json) |
| ECAPA 실행·resume·환경 복구 | [ECAPA_VALIDATION_README.md](ECAPA_VALIDATION_README.md) |
| 전체 Colab notebook | [ECAPA_Internal_Validation.ipynb](ECAPA_Internal_Validation.ipynb) |
| 복사 가능한 전체 셀 코드 | [ECAPA_VALIDATION_COLAB.py](ECAPA_VALIDATION_COLAB.py) |
| Train 200통화 선택·결과 | [SCREEN_200_REPORT.md](SCREEN_200_REPORT.md) |
| 과거 2,000통화 계획·200통화 resume | [docs/TRAIN_SCREENING_ARCHIVE.md](docs/TRAIN_SCREENING_ARCHIVE.md) |

## 코드와 산출물 구조

```text
compare_models.py / vendor/         기존 고정 모델·전처리 (변경하지 않음)
build_validation.py                Validation metadata·bundle 생성
make_validation_notebook.py         notebook와 전체 셀 코드 생성
ecapa_validation.py                L4 preflight·추론·resume·export
validation_common.py               GPU 없이 수행하는 공통 무결성 검사
analyze_ecapa_validation.py         현재 bundle과 일치하는 결과 분석
verify_returned_validation.py       이번 반환 ZIP의 과거 코드 버전 검증·분석
validation_bundle_manifest.json    입력·코드·baseline hash
validation_results/                검증 완료 REPORT·metrics·provenance
artifacts/ / validation_data/       로컬 생성 ZIP·jobs, Git 제외
docs/                              과거 실행 절차
```

기존 Train용 `bundle_manifest.json`, `build_handoff.py`, `screen_200.py`와 두 Train notebook은
과거 실험 재현을 위해 보존한다. Validation에서는 Train 전용 `load_jobs()`를 사용하지 않는다.
`ECAPA_VALIDATION_HANDOFF.md`도 평가 전 인계 기록이며 최신 상태는 이 README와 최종 REPORT다.
코드·데이터 경로 이동으로 기존 hash를 깨뜨리지 않도록 실행 파일은 원래 위치에 유지했다.

## 로컬 재검증

이번 Colab 결과 ZIP은 실행 후 패키지 오류 표시 문구를 보완하기 전 코드에서 생성됐다.
아래 명령은 알려진 진단 문구 변경만 복원하고 모든 실행 파일 hash를 확인한 뒤 분석한다.
GPU·추가 학습·재추론은 필요 없다.

```bash
python3 mission-1/experiments/gender_model_comparison/verify_returned_validation.py \
  /absolute/path/ecapa_validation_results.zip
```

Git에 포함하지 않은 원본 metadata·생성 jobs는 [재현 안내](ECAPA_VALIDATION_README.md)에 따라 준비한다.
현재 코드로 새로 생성한 결과는 `analyze_ecapa_validation.py`로 분석한다.

## 검증과 공유 범위

실제 Colab L4 추론은 사용자가 실행했고, 반환 ZIP을 로컬 검증했다.
13개 CPU 테스트가 통과했으며 합성 테스트 결과는 실제 성능표에 포함하지 않는다.

```bash
.venv/bin/python -m unittest discover \
  -s mission-1/experiments/gender_model_comparison -p 'test_*.py' -v
```

정답은 평가에만 사용했다. ECAPA·WavLM 외부 classifier에 추가 학습·threshold tuning을 하지 않았다.
ECAPA와 Wav2Vec2는 통화 집계도 다르므로 전체 파이프라인 비교다.
원본 WAV·모델 weight·대용량 cache·반환 ZIP·통화별 상세 CSV는 Git에서 제외한다.
