# Mission 1 · WavLM-Large Train 성별 분류

상태: **실행 코드 준비; 실제 L4 학습·성능·ETA는 아직 미측정**. Colab은 사용자가 실행한다. **2026-09-26 수정본: T4(FP32) / L4(BF16) 자동 구분.**
99%는 목표이며 보장하지 않는다. 이전 `tiantiaf/wavlm-large-age-sex` Train 200통화 95%와
이번 `microsoft/wavlm-large` 자기지도 사전학습 backbone + 우리 Train 학습을 구분한다.
기존 Wav2Vec2는 frozen + LR, backbone fine-tuning 이력이 없다.
ECAPA 과거 인계 지시는 실행하지 않는다. 최신 검증 결과는 96.819725%, 178오답이다.

## 바로 실행

1. `artifacts/wavlm_gender_adaptation_t4_v2_bundle.zip` → Drive `DDC-Colab/` 업로드.
2. `WavLM_Gender_Adaptation_T4.ipynb`을 Colab T4 또는 L4에서 열고 위에서 아래로 실행.
3. 기존 config의 실제 data_root/output_dir 확인 → 기존 Drive identity 교차검증.
4. 별도 venv → preflight → benchmark. 실제 비용을 확인한 뒤 Train 추출·dev 선택.
5. 조건부 적응 비교 또는 frozen-only → 설정 잠금 → 전체 Train 재학습 → Internal Validation.
6. `wavlm_results.zip` 반환 후 로컬 검증:

```bash
python3 mission-1/experiments/wavlm_gender_adaptation/verify.py /absolute/path/wavlm_results.zip
```

기존 파일·환경·cache identity·공용 README는 변경하지 않는다. 코드 `/content/wavlm_gender_adaptation_v2`,
venv `/content/wavlm_adaptation_env_v2`, Drive output은 기존 source output의 형제 `wavlm_gender_adaptation_t4_fp32_v2` 또는 `wavlm_gender_adaptation_l4_bf16_v2`. 기존 L4 v1 결과와 섞지 않는다.
HF cache `/content/gender_model_hf_cache`는 같은 내용 주소 기반 다운로드 캐시를 공유하며 기존 identity를 덮어쓰지 않는다.
실제 torch/torchaudio·CUDA·Python 버전을 출력·기록한다. 이전 `2.11.0+cu128`을 가정하지 않으며 정상 torch를 재설치하지 않는다.
비 torch 패키지만 별도 venv에서 고정한다. wheel 불일치 시 자동 복구하지 않고 먼저 오류를 확인한다.

## 사전 등록한 제한된 후보

| 단계 | 후보/선택 | 학습 범위와 비용 |
|---|---|---|
| F0 | 마지막 24층 mean → Scaler/LR, C=.1/1 | backbone 전체 고정, 1,024차원 |
| F1 | 6·12·18·24층 mean/std concat → Scaler/LR, C=.1/1 | backbone 전체 고정, 8,192차원; F0와 캐시 공유 |
| N0 | 4층 학습 가중합 mean/std → LayerNorm → Linear(2) | 동일 초기 frozen backbone, head만 학습; 적응 효과 대조군 |
| A1 | N0와 같은 head + Transformer 상위 2층(인덱스 22,23) | CNN·projection·하위 22층·나머지 norm 고정; 최대 4 epochs |

중간층 음색/화자 정보와 상위층 정보를 함께 시험하기 위해 등간격 4층을 선택했다.
전체 25 hidden states의 frame-level 캐시는 만들지 않는다. forward 중 hidden states는 임시로 존재한다.
mean/std는 하나의 평균이 숨길 수 있는 시간 변동도 반영한다. F0/F1은 4개 후보만 비교한다.
선택: 고정 dev Overall 정답 수 최대, 동률이면 last_mean → 작은 C 순. Scaler는 inner-train만 fit.
정규화/학습률/threshold/집계는 Internal Validation을 보며 고르지 않는다.

N0/A1은 F0/F1 최고 dev <99%인 경우에만 후보가 된다. 비용 및 Train 청취 검토 후 notebook에서
명시적으로 활성화한다. frozen-only를 끝내도 backbone 적응 효과를 확인했다고 주장하지 않는다.
N0/A1의 head seed=42, crop·순서·4 epochs·head LR=3e-4·backbone LR=1e-5,
AdamW decay=.01, gradient clip=1, accumulation=4, microbatch=1을 일치시킨다.
선형 decay는 4 epochs horizon, warmup 없음. 각 epoch의 **전체 caller dev** 정확도로 epoch를 선택하며 동률은 먼저 나온 epoch.
A1이 N0보다 최소 0.2%p 좋고 frozen LR보다 나쁘지 않을 때만 A1 최종 평가도 수행한다.
동일 dev 반복 선택의 불확실성이 있으며 미세한 개선을 확정적인 우월성으로 해석하지 않는다.
LoRA는 이번 후보에서 제외했다. 상위 2층만으로 제한된 적응 가설을 먼저 검증해 별도 라이브러리/하이퍼파라미터 탐색 비용을 줄인다.

## 분할·최종 재학습·평가 잠금

원본 metadata 3개 hash는 `common.py`에 고정하며 bundle 빌드 때 대조한다.
정상 27,985 = Train 22,388 + Internal Validation 5,597(M 2,620/F 2,977).
공식 Validation 3,640은 manifest 및 오디오 입력에 포함하지 않는다.

Train 내에서 성별마다 `SHA256("wavlm-inner-v1|42|" + call_id)` 오름차순(동률 ID순),
처음 `round(n/5)`를 dev로 지정한다. sklearn 버전에 의존하지 않는 80:20 stratified call 분할이다.
`data/inner_split.csv`, `inner_train_ids.txt`, `dev_ids.txt`를 다른 Wav2Vec2 창에도 **그대로 전달**하면 동일 분할을 쓸 수 있다.
`design_manifest.json`에 seed·규칙·개수·hash를 기록한다. 기존 Wav2Vec2 3-fold 결과와 같다고 주장하지 않는다.
같은 call의 모든 segments는 같은 partition이며 사람 ID가 없어 사람 단위 독립성은 보장할 수 없다.

`selection.lock.json`이 생기면 runner는 개발 단계를 차단한다. 최종 LR은 고른 설정으로 전체 Train 22,388개에 fit.
선택된 A1은 pinned backbone과 seed42의 새 head로 **처음부터** 전체 Train에서 선택 epoch 수만큼 학습한다.
N0는 dev에서의 적응 대조군이며 최종 Validation에는 LR 및 채택된 A1만 넣는다.
최종 scheduler도 4-epoch horizon이며 실제 steps/epoch는 전체 Train 크기에 맞춘다.
`final_models.json`에 weight hash를 잠그고 `validation_started.json` 후 재학습을 금지한다.
Validation 점수를 보고 설정 변경·checkpoint 교체·threshold tuning을 하지 않는다.
기존에 관찰한 Internal Validation이며 새로운 독립 test가 아니다. F/M argmax, 동률 F.

## 오디오·normalization·pooling

기존 `compare_models.py:load_wave`와 Wav2Vec2 crop/resample을 확인한 뒤 같은 알고리즘을 독립 모듈에 옮겼다.
실제 8kHz mono float32 WAV, speaker=1의 모든 시간순 구간에 `round(t*8000)`.
각 구간에 `scipy.signal.resample_poly(2,1,window=('kaiser',5.0))`, float32, caller 전체 연결.
겹친 caller 주석 구간도 기존 정의처럼 각각 보존한다. 연결 경계와 12초 창 경계는 문맥 단절의 근사 오차가 있다.

최종/개발 평가와 frozen 추출은 전체 caller를 `np.array_split`으로 **최대 12초 균등 창**에 나눈다.
어떤 구간이나 tail도 버리지 않는다. 각 창을 checkpoint feature extractor의 zero-mean/unit-variance로
정규화한 뒤 CNN 최소 입력 400 samples 미만만 0-pad한다. window 크기와 짧은 입력 처리를 모델 고유 정규화와 구분한다.
창마다 4층의 시간축 1차·2차 moment를 계산한 후 **실제 음성 sample 수**로 가중 평균해 통화 mean/std를 얻는다. std의 variance 하한은 1e-7이며 짧은 입력의 0-variance 미분을 안정화한다.
각 창 내부는 hidden frame 동일 가중이며, 창 사이 actual sample 가중은 frame 평균과 완전히 동일하지 않을 수 있다.
LR/신경망 모두 이 통화 특징으로 판정한다. 확률 평균·majority vote·Validation threshold 후보는 없다.

신경망 학습만 caller 연결 파형에서 무작위 연속 crop을 사용한다. 기본 12초는 사전 계획값이며
Colab 자원 측정 전 최종 실행 확정값은 아니다. OOM/비용 때문에 8/16초로 바꾸면 **새 output identity에서**
preflight/benchmark를 다시 수행하고 개발 전에 확정한다. Train/dev 점수나 Validation을 보고 길이를 탐색하지 않는다.
crop 시작은 seed/epoch/call ID로 결정하며 모든 call을 매 epoch 한 번 포함한다. 짧은 통화는 전체를 쓴다.
checkpoint의 dropout/LayerDrop/SpecAugment는 모두 비활성화, eval mode에서도 상위 2층 gradient는 흐른다.

Train metadata의 speaker 라벨 충돌·주석 겹침·길이 위험을 빌드 시 점검한다.
`data/train_audio_review.csv`에 겹침 고위험 24개·최단 12개·해시 표본 12개(중복 제거)를 저장한다.
`audit`는 원본 caller 및 동일 방식 학습 crop을 만들어 실제 상담원 혼입·화자 교체·성별 라벨 불일치를 청취하도록 한다.
`review_status=reviewed`, `actual_speaker_matches_label=yes/no/uncertain`, 문제 설명을 기록해야 적응 학습이 열린다.
no/uncertain은 자동 제거/라벨 변경하지 않는다. 실제 화자와 통화 label의 불일치는 주석 겹침만으로 확정할 수 없다.
현재 로컬에는 WAV 볼륨이 없어 청취 상태는 pending이다.

## revision·자원·시간

초기 모델: `microsoft/wavlm-large@c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`.
HF API에서 확인한 immutable revision이며 공식 processor는 do_normalize=true, 16kHz다.
로드된 초기 파일 hash·loading missing/unexpected keys·전체/학습/고정 parameter 이름 및 수를 저장한다.
missing/mismatched/error key는 중단, 외부 분류 head의 unexpected key는 로드 기록에 남긴다.
random head 초기 seed·초기 tensor hash와 F/M 순서를 기록한다. 성별 외부 checkpoint를 초기화에 쓰지 않는다.
근거: [공식 모델](https://huggingface.co/microsoft/wavlm-large/tree/c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c),
[WavLM 구현/hidden states 문서](https://huggingface.co/docs/transformers/model_doc/wavlm).

통화 cache float32 `(27985,4,2,1024)`는 약 **0.85 GiB** + 작은 파일/메타데이터 overhead.
전체 frame 캐시 및 LR RAM 계산은 실제 metadata 길이 기반 `design_manifest.json`에 기록한다.
LR은 입력·float64 변환·Scaler 복사 여유를 포함해 available RAM 5 GiB 이상, Drive free 8 GiB 이상을 사전 요구한다.
모델 다운로드는 약 1.2 GiB급 별도 로컬 HF cache이며 초기 가중치 해시 때 일시적인 I/O가 든다.
checkpoint는 frozen backbone 중복 저장 없이 trainable delta/head/optimizer를 저장한다. 복원은 pinned 초기 모델이 필요하다.
상위 2층 Adam 상태와 임시 atomic 저장을 포함한 실제 용량은 benchmark에서 측정한다.

T4는 FP32 기본이며 BF16을 명시적으로 거부한다. L4는 기존 BF16 기본, TF32 금지. FP16 경로는 추가하지 않았다. 실제 Train 길이/성별 표본에서 FP32 기준과 pooled feature 상대 L2 오차<.05 및 finite를 확인한다. T4 FP32에서는 동일 정밀도 반복 검사다.
이는 최종 정확도 동일성 보장이 아니다. BF16/FP32 둘 다 scaler 비활성 상태를 checkpoint에 저장한다.
기본 gradient checkpointing은 사용하지 않는다. 하위층 no-grad와 1개 crop microbatch로 메모리를 제한한다.
OOM 시 같은 identity에서 길이를 조용히 바꾸지 않는다: 먼저 다른 GPU 프로세스 제거 → 새 output에서 8초 crop으로
preflight/benchmark → 그래도 불가하면 중단하고 설정 검토. accumulation 증가는 effective batch만 키우며 단일 crop OOM을 해결하지 않는다.

대표 Train 길이/성별 120통화에서 frozen/top2 각각 **100 optimizer steps**(+2 warmup),
전체 caller 60통화 평가를 실제 Colab에서 수행한다. frozen-only 선택이어도 benchmark는 frozen/top2 둘 다 측정하며 가중치는 폐기한다. 읽기/resample, forward/backward, optimizer,
Drive checkpoint 쓰기·fsync, 평가 결과 저장을 포함해 epoch 및 4-epoch 개발/최종 학습 ETA를 계산한다.
모델 load·다운로드는 별도이며 반복 읽기는 warm cache일 수 있다. 진행 중 실제 wall time으로 갱신한다.
RAM RSS·available RAM·peak/reserved VRAM·Drive free·checkpoint bytes를 저장한다. 실측 전 시간은 확정하지 않는다.

## 지속 저장·재개·실패

GPU 프로세스 목록 + 런타임 공용 lock + Drive output lock으로 중복 실행을 점검한다.
다른 실험이 이 lock을 사용하지 않거나 동시에 시작하는 경쟁/서로 다른 VM의 Drive 동기화까지 배타 보장하지 않으므로
한 L4 및 같은 output을 여러 notebook에서 동시에 실행하지 않는다. Stop은 subprocess group TERM→10초 후 KILL,
부모 사망 signal을 사용한다. 강제 종료 후 남은 lock은 owner/token/이전 실행 종료 확인 후 수동 해제한다.

학습 checkpoint는 **100 optimizer steps마다 및 epoch 끝** atomic replace한다.
model trainable delta + head, optimizer, scheduler, AMP scaler, Python/NumPy/Torch/CUDA RNG,
epoch·다음 batch·global step·sampler seed·epoch ID 순서 hash를 저장한다. workers=0, gradient accumulation 경계에서만 저장.
중간 microbatch gradient 복원은 하지 않는다. 첫 checkpoint 이전 중단도 step 0부터의 재실행 범위를 기록한다. 재개 시 마지막 저장 step 이후 시도한 step 범위를 출력·기록하고 재실행한다.
즉 **committed-step resume**이며 모든 물리적 실행이 exactly-once라는 주장은 하지 않는다.
이전의 저장되지 않은 계산은 최대 100 optimizer step 재시도할 수 있으며 조용히 중복/누락하지 않는다.
같은 환경·알고리즘 deterministic 설정을 쓰지만 다른 CUDA/환경을 섞는 resume는 거부한다.

통화별 cache는 atomic NPZ + hash sidecar, 평가 결과는 append/flush/fsync JSONL이다.
shape·finite·identity·audio coverage·hash를 확인한 뒤 완료 통화만 건너뛴다.
NPZ 저장 직후 sidecar 전 중단된 orphan 및 JSONL truncated tail은 **자동 성공 처리하지 않고 중단**한다.
해당 파일을 보존·점검 후 미완료 항목을 복구해야 한다. stage events와 실패 이력을 지우지 않는다.
모든 예정 모델이 5,597 unique ID exact coverage를 충족해야 completion/ZIP을 생성한다.
로컬 검증은 source/bundle/model identity, export hash, 확률 NaN/inf·합·argmax,
샘플/창 coverage, label, 실패 후 단계 복구, baseline ID/hash 및 정답 수를 확인한다.
Overall/Male/Female accuracy, M/F confusion, 세 baseline별 corrected/regressed/both_wrong/both_correct를 출력한다.
단계 완료 기록이 없는 실패는 로컬에서 거부한다. 실패 통화를 뺀 점수는 없다.

## 로컬 빌드·검사

```bash
python3 mission-1/experiments/wavlm_gender_adaptation/build.py
python3 -m unittest discover -s mission-1/experiments/wavlm_gender_adaptation -p 'test_*.py' -v
```

`build.py`는 이 폴더에만 쓰고 원본 metadata는 읽기만 한다. notebook과 bundle hash가 함께 갱신된다.
`tests`는 합성 fixture이며 실제 학습 결과가 아니다. GPU forward/backward·OOM·실제 ETA는 Colab에서 확인해야 한다. T4 FP32 실행시간은 L4 수치를 재사용하지 않고 별도로 측정한다.
원본 WAV/모델/큰 cache/ZIP은 Git 제외. split ID 목록과 재현 코드·설계 hash는 공유한다.
공용 README 갱신 초안은 이 폴더의 `SHARED_README_DRAFT.md`에만 둔다.
