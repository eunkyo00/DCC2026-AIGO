# Mission 1 실험·코드 안내

최종 제출은 [final_submission/](final_submission/README.md)이며, 전체 진행 과정과 최신 결과는 [README](README.md)를 따른다. 아래 폴더는 기존 경로를 보존해 당시 실행과 import 경로를 유지했다. 코드가 존재한다는 사실만으로 전체 데이터 학습이 완료됐다고 보지 않는다.

| 구분 | 폴더·문서 | 역할·상태 |
|---|---|---|
| 최종 제출 | [final_submission/](final_submission/README.md) | 29,200통화 재실험, 50:50, 실행 로그·공식 Validation 결과 포함 |
| 데이터 점검 | [eda/](eda/README.md), [validation/](validation/README.md) | 초기 cohort와 통화 분할·품질 점검 기록 |
| 완료 baseline | [baseline/f0_lr/](baseline/f0_lr/README.md), [baseline/mfcc_svm/](baseline/mfcc_svm/README.md) | 같은 초기 5,597통화 평가 |
| 초기 전처리 예제 | `baseline/feature_extraction.py`, `baseline/train_svm.py`, `train.ipynb` | 5초·MFCC40 예제; 실제 MFCC13 실험 및 최종 모델과 구분. `baseline/results/baseline_results.json`은 당시 미실행 상태 기록 |
| 완료 사전학습 표현 | [ssl/wav2vec2_frozen/](ssl/wav2vec2_frozen/README.md) | 고정 Wav2Vec2 특징 + LR |
| 완료 특징·분류기 비교 | [experiments/embedding_classifiers/](experiments/embedding_classifiers/README.md) | Train CV, MFCC 결합, OOF 오류 분석 |
| 완료 ECAPA 비교 | [experiments/gender_model_comparison/](experiments/gender_model_comparison/README.md) | 예비 모델 비교와 초기 Internal Validation 평가 |
| 완료 WavLM + 후속 설계 | [experiments/wavlm_gender_adaptation/](experiments/wavlm_gender_adaptation/README.md) | Frozen WavLM 완료 결과는 [results/2026-10-07/](experiments/wavlm_gender_adaptation/results/2026-10-07/README.md); adaptation·T4·handoff는 별도 실행·인계 코드 |
| 완료 초기 앙상블 | [experiments/wavlm_wav2vec2_ensemble/](experiments/wavlm_wav2vec2_ensemble/README.md) | 27,985통화 조건의 75:25 결과. GitHub 최신 공유 노트북 보존 |
| 후보·검증 코드 | [experiments/wav2vec2_gender_finetuning/](experiments/wav2vec2_gender_finetuning/README.md) | 부분 fine-tuning 설계·CPU 검증·인계; 최종 모델에는 미사용 |
| 후보·검증 코드 | [experiments/wavlm_window_attention/](experiments/wavlm_window_attention/README.md) | 구간 가중치 실험; 최종 모델에는 미사용 |
| 후보·검증 코드 | [lightweight/logmel_cnn/](lightweight/logmel_cnn/README.md) | 경량 CNN 설계·검증; 최종 모델에는 미사용 |
| 전체 재실험 개발본 | [full29200/](full29200/README.md) | 단일 스크립트·CPU 테스트. 실제 제출용 확정본은 `final_submission/run_full29200.py` |
| 전체 재실험 Colab 대안 | [full29200_colab/](full29200_colab/README.md) | 다중 파일·위젯 기반 실행 구성. 실제 제출 노트북과 구분 |
| 과거 문서 | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), [EXPERIMENT_PLAN_99.md](EXPERIMENT_PLAN_99.md), [PPT_SUPPORT.md](PPT_SUPPORT.md) | 해당 시점의 계획·결과·발표 자료; 최신 설정은 상위 README |

## 저장소 정리 기준

- 누락됐던 실행 코드·검증 테스트·노트북·설계 문서를 기존 실험 폴더에 추가했다. 더 최신 GitHub 문서와 공유 노트북은 로컬 과거본으로 덮어쓰지 않았다.
- 원본 데이터, 가중치, 특징 캐시, ZIP, 가상환경은 제외했다. 29MB `baseline/mfcc_svm/results/call_features.csv`는 재생성 가능한 특징 캐시다.
- `SHARED_README_DRAFT.md`, `PRESENTATION_SCENARIO.md`는 편집용 초안으로 로컬에 보존했다. 초기 앙상블의 과거 노트북 생성기는 GitHub 최신 공유본을 되돌릴 수 있어 제외했다.
- handoff 폴더는 인계 시점의 독립 실행 묶음이다. `handoff_src/`는 그 묶음을 생성하는 원본 보조 파일이며, 실제 결과와는 구분한다.
- 일부 역사적 테스트·생성기는 별도 데이터 매니페스트·모델 의존성을 요구한다. 합성 CPU 검증은 전체 GPU 학습 성능의 재현을 대신하지 않는다.

## 이번 정리의 검증 (2026-10-14)

- 추가 Python 43개 구문 검사, 노트북 6개 JSON·일반 Python 셀 구문 검사 통과. 새 안내 문서의 로컬 링크 확인.
- 전체 29,200 개발본 CPU 테스트 10개, Colab 파이프라인 8개, 구간 가중치 9개, 경량 CNN 6개: **33개 통과**. 합성 데이터와 축소 모델 검증이며 실제 전체 GPU 학습 재실행은 아니다.
- 과거 fine-tuning/인계 통합 테스트는 `data/jobs.json`, `data/baseline_sources.json`, `artifacts/wavlm_teammate_bundle.zip` 등 별도 생성 산출물이 없어 전체 통과를 확인하지 못했다. 해당 코드의 `build.py`/`build_handoff.py`는 당시 매니페스트·EDA 산출물을 요구한다. 가중치·특징 캐시·개인 경로가 포함될 수 있는 생성물을 테스트 통과만을 위해 저장소에 추가하지 않았다.
- 최종 제출 코드·노트북·기록은 앞서 반영한 패키지 그대로 유지했다. 공식 Validation 정확도와 성별별 수치는 실행된 노트북 출력·CSV 건수·README의 기록을 확인했다.
