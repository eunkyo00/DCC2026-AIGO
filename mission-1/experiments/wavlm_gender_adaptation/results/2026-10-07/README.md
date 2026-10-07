# WavLM-Large 성별 분류 실험 결과 · 2026-10-07

**Internal Validation 5,597통화에서 정확도 98.088262%, 정답 5,490 / 오답 107.**
현재 비교한 모델 중 가장 높은 정확도다. 99% 목표에는 정답 52건이 더 필요하다.

## 결과 비교

| 모델 | 정답 | 오답 | 정확도 |
|---|---:|---:|---:|
| MFCC + RBF SVM | 5,321 | 276 | 95.068787% |
| Frozen Wav2Vec2 + LR | 5,443 | 154 | 97.248526% |
| Frozen Gender ECAPA | 5,419 | 178 | 96.819725% |
| **Frozen WavLM-Large + LR** | **5,490** | **107** | **98.088262%** |

Wav2Vec2 대비 **+0.839736%p**, 정답 +47건. 기존 오답 82건을 교정했고 기존 정답 35건은 오답으로 바뀌었다.
이는 전체 파이프라인 비교이며 backbone만의 효과나 통계적 우월성을 입증한 결과는 아니다.

| 실제 성별 | 예측 M | 예측 F | 성별 정확도 |
|---|---:|---:|---:|
| M (2,620) | 2,561 | 59 | 97.748092% |
| F (2,977) | 48 | 2,929 | 98.387639% |

## 실제 수행한 실험

- 모델: `microsoft/wavlm-large`, revision `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`.
- 입력: 신고자(`speaker=1`) 음성, 8kHz → 16kHz, 전체 caller를 최대 12초 창으로 처리.
- 후보: 마지막 층 mean / 6·12·18·24층 mean+std × LR C=0.1/1.0.
- Train 내부 dev 4,478통화로 선택: **last_mean, C=0.1**, dev 97.990174% (4,388 정답).
- 선택 잠금 후 Train 22,388통화 전체로 LR 재학습, Internal Validation 5,597통화 평가.
- 환경: NVIDIA L4, BF16, Python 3.13.15, PyTorch/torchaudio 2.11.0+cu130, CUDA 13.0.
- **Backbone fine-tuning은 수행하지 않았다.** 상위 2층 benchmark는 자원 측정이며 최종 적응 모델 학습 결과가 아니다.
- 공식 Validation 3,640통화는 사용하지 않았다. 이미 관찰한 Internal Validation이며 독립 test가 아니다.

## 파일 안내

| 파일 | 내용 |
|---|---|
| [REPORT.md](REPORT.md) | 로컬 검증 보고서·baseline 교정/회귀 |
| [metrics.json](metrics.json) | 검증된 성능 지표 |
| [call_comparison.csv](call_comparison.csv) | 통화별 예측과 기존 모델 비교 |
| [raw/](raw/) | ZIP에서 추출한 원본 로그·설정·해시·예측 |
| [Colab notebook](../../colab/WavLM_Executed_2026-10-07.ipynb) | 전달받은 실행 코드를 셀별로 정리한 노트북 |
| [Colab Python export](../../colab/wavlm_gender_adaptation.py) | 사용자 전달 원본, 변경 없이 보관 |
| [runner.py](../../runner.py) | 단계별 실험 실행 코드 |
| [실험 설계·재현 안내](../../README.md) | 실행 전 작성된 원본 문서; 결과는 이 문서 기준 |

Python export에는 `!nvidia-smi` 등 Colab 문법이 있으므로 일반 Python CLI로 실행하지 않는다.
노트북은 전달받은 소스를 복원한 것이며 실행 출력은 포함하지 않는다.
실행 당시 수동 `.run.lock` 삭제 셀도 보존되어 있다. 재실행 전에 이전 런타임이 종료되었는지 확인해야 한다.

## 검증과 재현

반환 ZIP의 파일 해시, 코드·모델 identity, 분할, 5,597개 unique ID와 label,
확률·argmax·음성 coverage, 지표 재계산, baseline 비교를 로컬에서 검증했다.
기록된 실패와 미해결 실패는 모두 0건이다.

```bash
python3 mission-1/experiments/wavlm_gender_adaptation/verify.py /path/to/wavlm_results.zip
```

위 엄격 검증에는 실행 당시 bundle의 `data/`와 기존 baseline `call_comparison.csv`가 필요하다.
원본 WAV, 데이터 매니페스트, 모델 weight, 특징 cache, 실행 bundle ZIP은 GitHub에 포함하지 않는다.
Git clone만으로 학습이나 엄격 검증을 즉시 실행할 수 있는 패키지는 아니다.
팀 공유 데이터와 실행 당시 bundle을 준비한 뒤 원본 노트북의 순서대로 실행한다.
실제 패키지 버전은 [identity.json](raw/identity.json), 측정 자원과 ETA는 [benchmark.json](raw/benchmark.json) 참조.
