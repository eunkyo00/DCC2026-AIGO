# Mission 1 · Wav2Vec2 부분 fine-tuning

**상태: 실행 코드 준비 및 로컬 검증 단계. L4 학습·새 Validation 결과는 아직 없음.**
[Colab notebook](Wav2Vec2_Gender_Finetuning.ipynb)을 Colab에서 열고,
`wav2vec2_finetuning_bundle.zip`을 `MyDrive/DDC-Colab/`에 업로드한다.
실행 순서: 환경 확인 → benchmark → search → lock → refit → evaluate → export.
원본 WAV가 필요하다. 저장된 embedding만으로 backbone을 학습할 수 없다.

## 기존 완료 결과와 이번 변경

| 실험 | 실제 완료 범위 | Internal Validation 정답 / 오답 |
|---|---|---:|
| MFCC+SVM | 고정 split 평가 | 5,321 / 276 (95.068787%) |
| Frozen Wav2Vec2+LR | 본체 전부 frozen, Train-only Scaler+LR | 5,443 / 154 (97.248526%) |
| Frozen Gender ECAPA | 고정 직접 분류기 평가 | 5,419 / 178 (96.819725%) |
| 이번 실험 | 코드 준비; GPU 실행 대기 | 미측정 |

ECAPA Train 200통화 198정답(99%)은 위 결과와 다른 예비 비교다.
이전 embedding LR/SVM/MFCC 결합은 저장된 특징의 실험이고 본체 fine-tuning이 아니다.
과거 계획의 `D 보류`, `추가 학습 금지` 등은 이번 추가 학습 허용 요청으로 대체한다.
과거 10초 crop·3 epochs·3~12시간은 확정 설정/실측이 아니다.
**99%는 최소 5,542/5,597 정답(55오답 이하)이라는 목표이며 보장하지 않는다.**

기존 checkpoint `facebook/wav2vec2-base`, revision
`0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8`을 사용한다.

| 항목 | 기존 baseline | 새 두 arm |
|---|---|---|
| 본체 | `requires_grad_(False)`, eval, inference_mode | 대조군 전부 frozen / 후보 상위 2 또는 4층만 trainable |
| 분류기 | Train-only Scaler+LR | 같은 seed의 FP32 Linear(768,2), AdamW |
| 고정 범위 | 본체 전체 | convolutional encoder, feature projection, 위치 convolution, encoder 전체 공통 norm, 하위 Transformer |
| pooling | frame 평균 → segment 동일 가중 call 평균 | 동일한 정의; 학습 가능 집계 없음 |
| 학습 음성 | 특징 추출 때 모든 caller 구간 | 모든 caller 구간; crop·증강 없음 |
| 연산 | backbone FP32 | 기본 backbone bf16, pooling/head/loss FP32; TF32 off |

기존 NumPy float64 누적 후 float32 변환과 달리 새 pooling은 FP32 tensor 연산이다.
기존 LR와 차이는 head·optimizer·precision도 포함하므로 그 차이를 backbone 효과로 단정하지 않는다.
**본체 효과의 직접 비교는 새 frozen+head 대조군과 fine-tuned+같은 head 사이**다.
두 arm의 학습 길이는 각각 같은 dev 규칙으로 선택한다. 층 수 비교는 최대 2개다.

## 분할과 선택 규칙

- 원본 정상 Training 27,985, 기존 Train 22,388 / Internal Validation 5,597을 유지한다.
- Internal Validation은 Male 2,620 / Female 2,977. 공식 Validation 3,640은 제외한다.
- 모든 source CSV는 요청의 SHA256과 정확히 일치해야 build한다.
- 다른 창의 WavLM `inner_assign` 규칙과 일치: 각 성별에서
  `SHA256("wavlm-inner-v1|42|" + call_id)` 오름차순(ID로 동률 처리), 앞 `(n+2)//5`개 dev.
  **inner-train 17,910 / dev 4,478**, seed 42.
- [split_identity.json](data/split_identity.json)에 ID mapping hash, CSV hash, 발견한 WavLM 생성 파일의 hash를 기록한다.
  build는 다른 폴더를 읽기만 한다. 생성 파일이 발견되면 ID별 배정 일치를 검증한다.
- 통화 ID로 partition을 정하므로 같은 통화의 모든 segment는 같은 partition이다.
  ID, 성별 정답, partition은 모델 입력에 포함하지 않는다. 정답은 loss와 평가에만 사용한다.
- 자원 pilot: top 0/2/4 각각 **100 optimizer steps**, 통화 4개 accumulation,
  성별·길이 분위수·최단/최장 통화를 포함한 400통화. 평가 표본은 inner-dev 40통화다.
- 사전 자원 기준: peak reserved memory < GPU 총 메모리 85%, 평균 step ≤ 120초.
  OOM은 기록하고 후보를 제외한다. 대조군 또는 모든 fine-tuning 후보가 탈락하면 탐색을 중단한다.
  더 작은 crop이나 다른 precision으로 조용히 바꾸지 않는다.
- 적격 후보만 최대 **4 epochs** 학습하고 각 epoch 전체 dev의 정답 수를 측정한다.
  후보별 최고 dev epoch(동률 이른 epoch); fine 후보 간 동률은 적은 층 수, 이른 epoch 순이다.
  이 4-epoch 상한은 이번 제한 탐색 설계이며 과거 3 epochs 가정과 별개다.
- backbone LR `1e-5`, head LR `1e-3`, AdamW weight decay `0.01`, gradient norm clip `1.0`.
  constant LR scheduler, class weight 없음, 클래스 순서 F/M, argmax(동률 F) 고정.
  threshold 탐색·학습 가능한 pooling ablation은 이번 bundle에 포함하지 않는다.
- 선택을 `selection.json`에 잠근 뒤 **두 arm 모두 pretrained weights부터 새로 전체 Train 22,388 재학습**한다.
  각 arm이 dev에서 선택한 정수 epoch만 학습한다. 같은 seed와 constant LR 사용.
  전체 Train의 epoch당 step 수 증가와 총 optimizer step 수를 기록한다.
- 고정된 두 arm을 Internal Validation에서 한 번 비교한다. checkpoint/층/LR/epoch/집계를
  Internal Validation으로 선택하지 않는다. 이미 관찰된 자료이며 새 독립 test라고 주장하지 않는다.

## 음성 처리와 통화별 loss

실제 WAV를 읽을 때마다 finite 8kHz mono와 구간 경계를 확인한다.
caller `speaker=1`의 각 구간은 `round(start_s*8000):round(end_s*8000)`으로 자른다.
SciPy Kaiser-5 `resample_poly(up=2,down=1)`로 16kHz 변환,
400 sample 미만이면 400까지 zero pad, feature extractor로 **segment 전체를 먼저 normalization**한다.
15초 초과 segment는 `tensor_split(ceil(length/240000))`로 모두 처리한다.
chunk의 hidden-frame 수 가중 평균이 segment 표현, segment 동일 가중 평균이 call 표현이다.
단일 무배치 chunk를 처리해 인공 batch padding과 attention mask가 필요 없다.
최소 길이 zero pad까지 평균하는 기존 방식도 유지한다. 조용한 절단이나 실패 segment 제외는 없다.
chunk 경계의 문맥 단절은 기존 baseline에도 있던 근사다.

학습/평가 음성·집계 차이는 **없다**. training loss는 call 평균 표현에 적용한 CE 하나다.
한 epoch에서 통화를 seed+epoch의 permutation으로 한 번씩 방문하고 4통화 loss 평균을 누적한다.
마지막 accumulation은 실제 잔여 통화 수로 나눈다. 길거나 segment가 많은 통화는 더 많은
연산을 쓰지만 loss 가중치는 다른 통화와 같다.

학습에서 full call의 모든 그래프를 보유하면 긴 통화가 OOM을 일으킬 수 있어 2-pass chain rule을 사용한다.
첫 pass(no_grad)로 call 표현을 계산하고 head CE에서 표현 gradient를 얻은 뒤,
두 번째 pass에서 각 chunk에 그 gradient × frame/segment 가중치를 역전파한다.
첫 pass에서도 backbone forward가 필요하므로 처리량 비용이 있다.
backbone은 eval 모드로 dropout·LayerDrop·SpecAugment를 끈다. **eval은 autograd를 끄지 않으므로
상위층은 두 번째 pass에서 학습된다.** GPU 그래프는 chunk 하나만 유지한다.
통화의 waveform/chunk는 CPU RAM에 유지하므로 극단적으로 긴 통화의 RAM 비용은 남는다.

gradient checkpointing은 이 구현에서 꺼져 있고 켜면 중단한다. 2-pass streaming을 대신 사용한다.
bf16은 loss scaling이 필요 없어 disabled GradScaler를 사용하되 상태는 저장한다.
매 step loss/hidden/gradient/parameter finite 검사와 clipping을 수행한다.
optimizer의 실제 parameter ID 집합이 trainable 집합과 같고 중복이 없는지 확인한다.
첫 step에서 head와 각 열린 층이 실제 변경됐는지 확인하며, frozen 파라미터의 SHA256을
시작·첫 step·매 epoch·완료 시 비교한다. 감사는 `parameters.json`, `update_audit.json`, `training.json`에 남긴다.

## Colab 환경, 측정, 재개

현재 Colab 환경은 로컬에서 접근할 수 없다. notebook이 실제 환경을 검사한다.
`/content/gender_model_comparison/config.json`의 실제 `data_root`를 우선 읽고,
없으면 `MyDrive/DDC-Colab/gender_comparison_v1/identity.json`에서 읽는다.
두 파일이 있으면 경로 일치를 검사한다. 원본이 없으면 추정 경로로 진행하지 않는다.
기존 HF cache `/content/gender_model_hf_cache`를 같은 revision으로 사용한다.
기존 cache를 삭제/교체하지 않으며 필요한 snapshot 파일만 HF가 추가할 수 있다.

전용 venv `/content/wav2vec2_finetuning_env`, 코드 `/content/wav2vec2_finetuning`,
Drive output `DDC-Colab/wav2vec2_finetuning_v1`을 사용한다.
현재 torch/CUDA/L4와 torchaudio import·wheel 버전을 먼저 검사하고 정상 torch를 재설치하지 않는다.
non-torch requirements 설치는 전용 venv에만 수행하며 torch constraint를 적용한다.
실행 identity에 코드/bundle/데이터/split/config/패키지/GPU/CUDA/driver를 고정한다.

같은 GPU에서 WavLM을 동시에 실행하지 않는다. 시작 시 `nvidia-smi`의 CUDA PID와
로컬 GPU lock을 확인한다. Drive output에는 exclusive `ACTIVE_SESSION.lock`을 사용한다.
이 점검 뒤 외부 코드가 GPU를 점유하는 race까지 통제하지 못하므로 두 notebook을 함께 실행하지 말아야 한다.
Stop은 자식 process group을 종료하고, 실패/중단/완료 event를 기록한다.
SIGKILL/런타임 종료의 미완료 event와 stale lock은 실제 프로세스 종료 확인 후 notebook 복구 셀로 기록·해제한다.

checkpoint는 기본 25 optimizer steps마다 및 epoch 끝에 fsync+atomic replace한다.
model, optimizer, scheduler, disabled AMP scaler, Python/NumPy/torch/CUDA RNG,
epoch/step, 다음 call cursor, sampler seed, 이력, frozen hash를 함께 저장한다.
accumulation 도중 상태는 저장하지 않는다. 중단 후 마지막 commit부터 최대 24개의 완료 미커밋
step과 진행 중 step을 **다시 계산**할 수 있지만, optimizer state를 함께 되돌리므로
최종 학습 궤적에서 committed step을 중복 적용하거나 건너뛰지 않는다.
num_workers=0이고 학습 stochastic augmentation은 없다. 동일 환경 CPU 작은 모델 테스트에서
재개와 연속 학습의 가중치 일치를 검사한다. L4의 bitwise 재현은 실제 GPU 검증 전에는 보장하지 않는다.
환경 identity가 바뀌면 resume을 거부한다. checkpoint는 신뢰하는 자기 실험 파일만 로드한다.

benchmark는 데이터 읽기+전처리, forward/backward, optimizer, checkpoint 저장, 평가 시간을 구분한다.
Train 탐색/dev 반복/refit epoch/최종 5,597 평가 ETA를 별도로 출력한다.
Drive 읽기 지연과 캐시 온도, 감사/optimizer 오버헤드, 세션 중단으로 실측은 달라질 수 있다.
평가 pilot은 재개 시에도 새 진단 폴더에서 실측해 cache skip 시간을 ETA로 사용하지 않는다.
최종 평가 sample은 40통화이며 전체 길이 분포의 완벽한 대표는 아니다. 과거 추출 11.19시간이나
계획 3~12시간을 이 학습의 확정 시간으로 사용하지 않는다.
checkpoint는 frozen weights까지 포함해 약 수백 MB이므로 후보별 Drive 저장 비용과 여유 공간을 확인한다.

## 결과 전달과 로컬 검증

notebook 마지막 셀에서 weights 없는 결과 ZIP을 내려받아 새 디렉터리에 해제한다.
전체 stage JSON, 실패 이력, 선택 증거, checkpoint hash, 2×5,597 통화별 JSON,
export file hash manifest를 포함한다. 모든 결과가 정상 완료되지 않으면 성공 보고를 거부한다.
과거 실패가 복구된 경우에는 숨기지 않고 보고서에 별도로 포함한다.

```bash
python3 mission-1/experiments/wav2vec2_gender_finetuning/verify.py \
  /absolute/path/to/extracted_results --output /absolute/path/to/new_local_report
```

검증기는 stdlib만 사용한다. unique IDs exact coverage, 정답·identity·파일 hash,
두 클래스 확률 finite/범위/합=1/argmax, 모든 segment와 원본 sample 수,
완료·미완료·실패 이력, Train 선택 증거를 검사한다.
checkpoint 자체를 전달하면 재해시하며, 없으면 Colab 기록 hash의 연결과 export hash를 확인하고
로컬 weights 재검증을 하지 않았음을 명시한다.
Overall/Male/Female Accuracy, confusion matrix(M/F), 기존 각 baseline 및 새 대조군 대비
corrected/regressed/both_wrong/both_correct와 5,443정답 대비 순증감을 보고한다.
독립 검증이 아닌 파일 내부 무결성 검사이며 악의적인 위조를 막는 전자서명은 아니다.

## 로컬 개발·검증

```bash
python3 mission-1/experiments/wav2vec2_gender_finetuning/build.py
python3 mission-1/experiments/wav2vec2_gender_finetuning/make_notebook.py
# torch/transformers/numpy/scipy/soundfile가 있는 별도 환경에서 CPU synthetic tests
python -m unittest discover -s mission-1/experiments/wav2vec2_gender_finetuning -p 'test_*.py' -v
```

`build.py`는 다른 실험 파일을 변경하지 않는다. 공용 README 수정 초안은
[SHARED_README_DRAFT.md](SHARED_README_DRAFT.md)에만 남긴다.
원본 WAV, weights, cache, venv, ZIP은 Git에서 제외한다. 큰 파생 `data/jobs.json`도
build로 재생성하며 Git에서 제외한다. bundle과 notebook은 변경 후 반드시 함께 재생성한다.
