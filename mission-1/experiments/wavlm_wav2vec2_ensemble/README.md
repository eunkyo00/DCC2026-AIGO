# WavLM + Wav2Vec2 확률 앙상블

**Internal Validation 5,597통화에서 정확도 98.213329%, 오답 100건.**
WavLM 단독보다 정답이 7건 증가했다. 두 backbone은 고정하고 기존 특징으로 LR만 학습했다.

[상세 결과](results/2026-10-09/README.md) · [Colab 노트북](WavLM_Wav2Vec2_Ensemble_L4.ipynb) · [실행 코드](run.py) · [검증 코드](verify_results.py)

## 결과 요약

| 모델 | 정답 | 오답 | 정확도 | Male | Female |
|---|---:|---:|---:|---:|---:|
| Frozen WavLM + LR | 5,490 | 107 | 98.088262% | 97.748092% | 98.387639% |
| **WavLM 75% + Wav2Vec2 25%** | **5,497** | **100** | **98.213329%** | **97.900763%** | **98.488411%** |

- +0.125067%p: 기존 오답 11건 교정, 정답 4건 회귀.
- Train 내부 3개 fold 모두 개선된 75:25를 채택. 50:50은 한 fold에서 하락해 제외.
- 99%에는 최소 5,542정답이 필요하므로 현재보다 45건을 더 맞혀야 한다.
- 실측 세션 시간 **9,054.81초 = 2시간 30분 55초**. 단계별 시간 기록은 없어 지연 원인은 확정할 수 없다.

## 실험 방법

1. Train 22,388통화 / Internal Validation 5,597통화의 기존 고정 분할 사용.
2. WavLM 마지막 층 mean 1,024차원, Wav2Vec2 768차원 cache를 재사용.
3. Train만 stratified 3-fold(seed42)로 나눈 뒤 각 fold의 학습 통화에서 Scaler와 LR을 fit.
4. WavLM LR C=0.1, Wav2Vec2 LR C=1.0을 고정하고 OOF 확률 생성.
5. WavLM 비중 0.75 / 0.5만 비교. 모든 fold에서 단독 대비 하락이 없고 전체 +0.1%p 이상일 때 채택.
6. 비중 잠금 → 전체 Train로 LR 재학습 → Internal Validation 평가. 조건 미달이면 단독 모델 유지.

채택 기준은 보수적인 휴리스틱이며 통계적 유의성 검정이 아니다. OOF는 모델 선택에 사용했고,
과거 WavLM 특징/C 선택에도 같은 Train 일부가 사용되었다. 독립 test 성능으로 해석하지 않는다.
사람 ID가 없어 화자 단위 분리도 보장하지 않는다. 공식 Validation은 사용하지 않았다.

## Colab 실행

1. [노트북](WavLM_Wav2Vec2_Ensemble_L4.ipynb)을 내려받아 Colab에 업로드한다.
2. Drive를 연결하고 셀을 위에서 아래로 실행한다.
3. 다음 실제 경로가 기본값이다. 다른 곳에 저장했다면 경로만 수정한다.

```text
/content/drive/MyDrive/DDC-Colab/wavlm_gender_adaptation_l4_bf16_v2/
/content/drive/MyDrive/DDC-Colab/wav2vec2_frozen_cuda_fp32/wav2vec2_l4_full_embeddings.npz
```

WavLM 폴더에는 `identity.json`, `completion.json`, `pooled/` 및 hash sidecar가 필요하다.
Wav2Vec2는 검증된 전체 embedding NPZ 또는 `call_cache/identity.json`과 `extraction_summary.json`이 있는 출력 폴더를 지원한다.
**원래 Train 특징이 필요하며 결과 ZIP만으로는 재학습할 수 없다.** 캐시가 누락되면 음성을 자동 재추출하지 않고 중단한다.

노트북에는 읽을 수 있는 실행 코드를 포함했고, split CSV는 고정 GitHub commit에서 다운로드해
SHA256을 검사한다. 긴 Base64 문자열이나 전체 Drive 폴더 탐색은 사용하지 않는다.
공유본은 준비 셀을 정리한 버전이며 실행 runner는 실제 결과의 `code_sha256`과 동일하다.
노트북에 실행 출력은 포함되지 않으며 실제 결과는 `results/2026-10-09/`에서 확인한다.

LR은 CPU에서 학습하므로 L4 GPU 사용률이 낮은 것이 정상이다. 512통화별 compact cache와
완료 fold를 재사용한다. 현재 결과 ZIP에는 세션 시간만 있어 cache 읽기/학습 시간은 분리할 수 없다.
실행 전 예상 15–90분을 초과했으므로 이를 완료시간 보장으로 사용하지 않는다.
한 output에는 한 Colab 런타임만 사용한다. 환경·소스가 바뀌면 새 output으로 실행해야 한다.

## 파일 안내

| 파일 | 내용 |
|---|---|
| `run.py` | 결과를 생성한 원본 runner; hash 일치 검증 |
| `WavLM_Wav2Vec2_Ensemble_L4.ipynb` | 업로드용 노트북, 준비 셀 가독성 개선 |
| `requirements.txt` | CPU 학습 환경 버전 |
| `test_run.py` | 분할 누수·Scaler 범위·선택 규칙·재개 합성 테스트 |
| `verify_results.py` | 반환 예측 및 공개된 WavLM baseline 대조, 표준 라이브러리만 사용 |
| `results/2026-10-09/` | 원본 반환 파일·상세 보고서·로컬 검증 기록 |

검증 명령(저장소 루트):

```bash
python3 mission-1/experiments/wavlm_wav2vec2_ensemble/verify_results.py
python -m pip install -r mission-1/experiments/wavlm_wav2vec2_ensemble/requirements.txt
python -m unittest discover -s mission-1/experiments/wavlm_wav2vec2_ensemble -p 'test_*.py' -v
```

원본 WAV·특징 cache·모델 weight는 GitHub에 포함하지 않는다. `models.joblib`는 Drive에 보관하며
반환 ZIP에는 포함되지 않았다. raw OOF 예측·모델 파일도 없어 그 내용 자체를 독립 재계산한 것은 아니다.
현재 코드는 실험용이다. 대회 제출에는 새 음성용 통합 `inference.py`, 지정 CSV와 모델 파일 포장이 별도로 필요하다.
