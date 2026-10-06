# Mission 1 · 99% 목표의 후속 실험 계획

2026-09-22 기준. 이 문서는 **앞으로 검증할 가설과 실행 규칙**이다.
완료된 A1~A3, 오류 분석, ECAPA/WavLM 비교와 ECAPA Validation은
[실험 이력](EXPERIMENT_LOG.md)에 정리했다. 현재 최고는 Wav2Vec2 + LR **97.248526%**다.

## 목표와 남은 격차

| 기준 | 필요한 정답 / 5,597 | 허용 오답 | 현재 Wav2Vec2 대비 추가 정답 |
|---|---:|---:|---:|
| 현재 Wav2Vec2 | 5,443 | 154 | — |
| 98% 이상 | 5,486 이상 | 111 이하 | 43 이상 |
| **99% 이상** | **5,542 이상** | **55 이하** | **99 이상** |

ECAPA는 5,419정답·178오답으로 목표까지 추가 123정답이 필요했다.
99%는 목표이며 아래 어떤 모델의 예상 정확도나 달성 보장도 아니다.

## 지금의 판단

외부 성별 ECAPA를 그대로 적용하는 실험은 완료했고, 기존 Wav2Vec2보다 낮았다.
ECAPA의 Validation 점수에 맞춘 반복적인 창 길이·threshold 조정은 하지 않는다.
다음 방향은 성별 전용 새 checkpoint 검토 또는 우리 Train 데이터에 대한 backbone 적응이다.

기존 Wav2Vec2 본체는 완전히 고정했다. LR·SVM·MFCC 결합 학습을 본체 fine-tuning으로 부르지 않는다.
이전 WavLM 95%도 외부 `tiantiaf/wavlm-large-age-sex`의 Train 200통화 결과이며,
우리 Train으로 학습한 WavLM의 성능을 뜻하지 않는다.

## 후보 조사와 우선순위

아래 순위는 연구 근거와 현재 결과에 따른 **실험 우선순위 제안**이다. 우리 데이터에서의 성능 순위가 아니다.

| 후보 | 제안하는 방법 | 조사 근거와 한계 | 현재 상태 |
|---|---|---|---|
| 1. MERaLiON-GR | 추가 학습 없이 적합성 확인 → 유망하면 우리 Train으로 LoRA 적응 | 2026년 성별 전용 모델. 한국어 8kHz 전화 음성에서의 우월성은 미확인 | 조사만 완료 |
| 2. WavLM-Large | frozen 여러 층 특징 + 분류기 → 제한된 부분 fine-tuning/LoRA | 화자·성별 표현의 연구 근거. 외부 성별 checkpoint와 별개 실험 | 다른 창용 인계 프롬프트 작성 |
| 3. 기존 Wav2Vec2 | frozen 새 head 대조군 → 상위 일부 층 fine-tuning | 우리 Validation 최고 baseline을 출발점으로 사용 | 다른 창용 인계 프롬프트 작성 |

MERaLiON-GR 논문은 Vox-Profile 대비 공개 평가 15개 중 12개 개선·1개 동률·2개 하락을 보고한다.
배포 저장소는 `MERaLiON/MERaLiON-GI-v1`이며 논문의 GR 명칭과 다르다.
한국어 전화 음성에 대한 결과는 없으므로 작은 구현·성능 점검부터 필요하다.
[논문](https://arxiv.org/html/2608.04433v1) · [공식 모델](https://huggingface.co/MERaLiON/MERaLiON-GI-v1).

WavLM 학습에서는 마지막 층만 고정해 쓰는 방식 외에 여러 층 특징 결합을 후보로 삼는다.
Whisper Large-v3 encoder, ReDimNet2, W2v-BERT 2.0도 조사했으나 이번 실행 후보로 확정하지 않았다.
음성 인식·화자 검증 점수를 우리 성별 분류 Accuracy로 환산하지 않는다.
[Vox-Profile 비교](https://arxiv.org/html/2505.14648v1) ·
[WavLM](https://huggingface.co/microsoft/wavlm-large) ·
[ReDimNet2](https://arxiv.org/abs/2603.11841) ·
[W2v-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0).

## 두 학습 실험의 공통 규칙

1. **분할 고정:** 기존 Train 22,388통화 안에서 통화 단위 inner-train/dev를 만들고 같은 ID·seed·hash를 두 실험에서 사용한다. 사람 단위 분리는 식별자가 없으므로 보장하지 않는다.
2. **설정 선택:** 층 수·학습률·학습 길이·checkpoint·집계 선택은 inner-dev에서 수행한다. 기존 Internal Validation 5,597개를 early stopping이나 설정 선택에 사용하지 않는다.
3. **최종 평가:** Train 내부에서 절차를 고정하고 최종 재학습 여부를 명시한 다음 기존 Internal Validation을 평가한다. 이미 관찰한 데이터이므로 새 독립 test라고 하지 않는다.
4. **차이 분리:** frozen backbone + 새 head를 대조군으로 두고, backbone 적응·집계 변경 효과를 가능한 한 따로 확인한다.
5. **음성 보존:** 학습 crop은 별도로 정의할 수 있지만 최종 통화 평가에는 모든 caller 구간을 사용한다. 모델 고유 전처리와 기존 방식의 차이를 기록한다.
6. **오류 분석:** Train에서 라벨 불확실성·상담원 혼입·잡음을 청취 점검할 수 있으나, 근거 없이 재라벨링하거나 평가 통화를 제외하지 않는다.
7. **재현:** revision, 입력 hash, 학습/고정 파라미터, optimizer·scheduler·AMP·RNG·step 상태를 기록하고 resume 수준을 명시한다.
8. **역할:** 사용자가 Colab L4에서 GPU 학습·추론, 로컬에서 코드·결과 검증·분석을 수행한다. 한 GPU에서 두 실험을 동시에 돌리지 않는다.

## L4 시간과 자원

아직 새 학습 실험의 실측값은 없다. 대화에서 제시한 아래 범위는 예산 가정으로만 보존한다.
**Train 22,388통화, 3 epochs, 통화당 약 10초 crop, 혼합정밀도·작은 batch·gradient accumulation,
최종 Validation 전체 평가**를 가정했으며, 이 설정들은 확정되지 않았다.

| 후보 | 구현 복잡도 판단 | 학습 + 최종 평가의 미검증 시간 예산 |
|---|---|---:|
| MERaLiON-GR + LoRA | 높음 | 10~30시간 |
| WavLM-Large 부분 적응 | 중상 | 6~20시간 |
| Wav2Vec2 상위 층 fine-tuning | 중간 | 3~12시간 |

환경 복구·코드 수정·여러 설정 재실험은 포함하지 않는다. 매 epoch 전체 caller를 사용하거나
Drive 읽기가 느리면 더 길어질 수 있다. 모델 크기만으로 VRAM 적합성·속도를 확정하지 않는다.
각 모델의 대표 길이로 100~200 training step과 평가 표본을 먼저 측정해 이 범위를 교체한다.
현재의 11.19시간 Wav2Vec2 추출이나 2시간 8분 41초 ECAPA 추론을 새 학습시간으로 일반화하지 않는다.

## 작업 분리와 완료 기준

권장 폴더는 `experiments/wavlm_gender_adaptation/`, `experiments/wav2vec2_gender_finetuning/`이다.
폴더 이름은 제안이며 이 문서가 두 실험의 구현 완료를 의미하지 않는다.
각 실험은 독립된 venv·코드·Drive output을 사용하고 기존 결과·identity를 덮어쓰지 않는다.

새 결과는 5,597 unique IDs exact coverage, F/M 확률·argmax·finite·합=1, hash·identity,
완료/실패 기록을 로컬 검증한 뒤 [현재 성능표](README.md)에 반영한다.
Overall/Male/Female Accuracy와 기존 baseline 대비 교정·회귀·공통 오답을 함께 보고한다.
