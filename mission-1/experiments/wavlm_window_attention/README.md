# Frozen WavLM · 구간 가중치 실험

`WavLM_Window_Attention_L4.ipynb` 하나를 Colab에 업로드하고 L4를 선택한다.
노트북은 읽을 수 있는 코드 셀로 구성되며, 긴 Base64 문자열·별도 코드 ZIP·저장소 clone이 필요 없다.

## 준비물

- Drive `DDC-Colab/wavlm_gender_inputs.json`: 기존 검증된 입력 목록.
- 이전에 사용한 원본 WAV. `DATA_ROOT` 바로 아래 `Training/`이 있어야 한다.
- `DDC-Colab/wavlm_gender_adaptation_l4_bf16_v2/identity.json`이 있으면 데이터 경로를 자동으로 읽는다.
- 새 출력: `DDC-Colab/wavlm_window_attention_l4_v1`. 기존 결과를 덮어쓰지 않는다.
- 여유 Drive 공간 최소 4GiB. 인터넷은 최초 공개 모델·패키지 다운로드에 필요하다.

입력 JSON의 전체 SHA256과 내부 jobs SHA256을 확인한다. 총 27,985개 unique ID,
Train 22,388 / Internal Validation 5,597 및 모든 음성 경로가 Training인지 검사한다.
원본 split은 서울 여부를 검증한 기존 분할이다. 이번 실험은 새로운 지역 필터를 만들지 않는다.

## 검증할 가설

통화 전체를 평균하면 일부 구간의 유용한 정보가 희석될 수 있다. 구간 가중치를 학습하는 작은
모델이 같은 특징의 평균 집계보다 개선되는지 검증한다. 오답 원인이 잡음·화자 혼입이라는 사실은
아직 청취로 확정하지 않았으며, 가중치가 높은 구간을 자동으로 '깨끗한 신고자'라고 해석하지 않는다.

1. 기존과 같은 caller 구간 crop, 8→16kHz resampling, 연결, 최대 12초 균등 창.
2. 고정 `microsoft/wavlm-large@c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`에서 각 창의 마지막 층 mean 1,024차원 추출.
3. 실제 sample 수 가중 평균 + LR(C=.1), 같은 평균 + 작은 linear head, 학습 가중치 + 같은 head 비교.
4. Attention은 1,024→32→1 scorer. 초기에는 길이 가중 평균과 같고, KL=.01로 과도한 집중 억제.
5. 두 neural head의 classifier 초기값, dropout=.1, AdamW lr=.001, weight decay=.001, batch64, seed42를 동일하게 설정.

WavLM 본체는 `requires_grad=False`, eval/inference mode로 실행한다. 구간 정보가 없는 기존
통화 평균 cache를 역으로 복원할 수 없어 특징을 새로 추출한다. 약 92,103개 창, 순수 float32
특징은 약 0.35GiB이며 압축·인덱스·모델/로그 용량은 별도다.

## 과적합 방지와 선택

- Train 내부 stratified 3-fold(call 단위). 각 outer 학습 fold 중 15%를 내부 early stopping에 사용한다.
- 최대 12 epochs, patience3, inner stop log-loss로 epoch 선택. 그 epoch 수로 outer 학습 전체에서 초기화 후 재학습.
- Scaler는 각 학습 subset에만 fit. outer 평가 fold로 early stopping하거나 LR을 학습하지 않는다.
- OOF에서 세 fold 모두 LR 대비 하락이 없고 전체 +0.1%p 이상인 neural 후보만 채택한다.
- Attention은 같은 mean head보다 전체 정답 수도 많아야 채택. 동률이면 mean, 조건 미달이면 LR.
- 선택 잠금 후 neural epoch 수는 outer fold에서 선택된 값의 중앙값으로 고정하고 전체 Train에서 새로 학습한다.
- 그 다음에만 Internal Validation 특징을 추출하고 선택 모델·같은 파이프라인 LR을 평가한다.

이는 모델 선택용 OOF이며 독립 test 성능 추정이 아니다. 화자 ID가 없어 같은 사람이 다른 fold에
들어갈 수 있다. 기존 WavLM/C 선택에 사용된 Train을 재사용하므로 과거 선택까지 완전히 nested
검증하는 것도 아니다. 공식 Validation은 미사용. 기존 Internal Validation은 이미 관찰한 평가셋이다.
과거 앙상블 98.213329%는 참고값으로만 보고하며 이번 후보 선택에 사용하지 않는다.
99% 목표는 5,542 정답 이상이며 달성 보장은 없다.

정답 성별에 따라 전처리를 바꾸지 않는다. 성별 라벨은 supervision/stratification/평가용이며
음성 입력에는 사용하지 않는다. 추가 라벨·텍스트·상용 AI API를 모델 입력으로 사용하지 않는다.

## 실행 시간과 중단

첫 200통화는 Train 음성 길이 분포를 등간격으로 대표하는 표본이다. 실제 WAV 읽기·resampling·
BF16 forward·Drive 저장을 측정해 전체 Train/평가 추출 시간을 보여 준다. 모델 로드·워밍업은 별도다.
미리 몇 시간이라고 단정하지 않는다. 전체 추출은 시간 단위 작업이며, Drive throttling으로 ETA를
초과할 수 있다. benchmark 특징도 전체 추출에서 재사용한다.

특징은 64통화 블록으로 atomic 저장하고 내부 배열 checksum·ID·실제 sample coverage를 검사한다.
재개 시 완료 블록은 GPU 재추출하지 않는다. 미저장 블록은 최대 64통화 재시도할 수 있다.
Head는 매 epoch 모델·optimizer·RNG·best epoch를 저장한다. 미완료 epoch는 다시 실행한다.
완료 fold는 재사용한다. 첫 완료 fold부터 남은 개발 시간 ETA를 표시한다.
표준 Stop은 자식 process group을 종료한다. 런타임 로컬 lock은 다른 VM까지 잠그지 않으므로
한 output에는 한 런타임만 사용한다. 환경·코드·설정이 달라지면 재개를 거부한다.

## 결과

- `window_attention_results.zip`: 설정/benchmark/OOF 요약/최종 지표/통화별 예측/로그.
- `selection.lock.json`: 채택 후보, OOF 성능, fold별 개선, 선택 epoch.
- `model.pt`: 분류기와 Scaler, frozen backbone의 식별 정보. backbone weight는 별도 HF cache.
- `completion.json`: 선택 모델과 LR 성능, 교정/회귀, 성별·길이별 결과, 99% 달성 여부.
- `logs/`: 실행 stdout, `events.jsonl`: 단계별 상태 및 실행 시간.

실험용 산출물이다. 대회 제출용 새 음성 `inference.py`, backbone 파일 포장, 지정 CSV 형식은 별도
구성이 필요하다. 실제 실행 로그가 포함된 Colab `.ipynb`도 저장한다.

## 로컬 검증

```bash
python -m unittest discover -s mission-1/experiments/wavlm_window_attention -p 'test_*.py' -v
python mission-1/experiments/wavlm_window_attention/make_notebook.py
```

합성 특징 CPU 테스트와 실제 입력 manifest 검증을 수행했다. 실제 L4 forward/전체 원본 음성
추출/Colab Drive I/O는 로컬에서 실행하지 않았으므로 첫 benchmark가 실제 환경 검증 단계다.
