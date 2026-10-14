# DDC Mission 1 — 사전 설계 v1

이 문서는 CNN 구현 전에 고정한 실험 정의다. 작업 브랜치는 mission1/eunkyo,
독립 경로는 mission-1/lightweight/logmel_cnn이다. Frozen Wav2Vec2의 코드,
config, cache, 실험 정의 및 현재 Colab L4 런타임은 수정·실행 대상이 아니다.
현재 런타임에서 이 실험을 동시에 실행하지 않는다. 별도 환경 또는 기존 작업 종료 후 실행한다.

## 기존 결과·누수 검토

MFCC 구현은 baseline/mfcc_svm/run_mfcc_baseline.py이며 범용 baseline/train_svm.py와
혼동하지 않는다. 8kHz, 13 MFCC, 26 mel, FFT 256, Hamming 200, hop 80,
pre-emphasis .97, segment mean/std → call mean/std + 수/길이 = 54차원이다.
SVC(C=1, gamma=scale, rbf), Train median imputer와 StandardScaler를 쓴다.
성능 튜닝은 없으며 앞 5,000 Train의 fit은 자원 판단용이다.
저장된 예측의 정확도 및 fixed split 정합성을 새 audit 명령으로 다시 계산한다.

고정 split SHA256: 04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06.
27,985 calls, Train 22,388 / 최종 Internal Validation 5,597.
공식 Validation 3,640은 이 실험에 사용하지 않는다. 기존 기록은 5,321/5,597
=95.068787%, 오답 276, [[2484,136],[140,2837]] (M,F)이다.
기존 전체 시간 1994.817684초, full SVC fit 약 7.97초. WAV 읽기와 MFCC 계산,
특징 쓰기 시간 및 실제 peak RSS는 분리 기록되지 않았으므로 역산·창작하지 않는다.
512MB SVC cache는 peak RAM 측정값이 아니다. 다른 하드웨어 시간의 직접 우열도 금지한다.

Training 정상 JSON+WAV 매칭 27,985개가 대상이며 EDA file_ref와 call_id를 연결한다.
speaker=1의 442,639개 구간만 사용한다. 동일 call의 segment 양쪽 배정,
경로/ID 겹침은 기존 검사에서 0. 그러나 role speaker는 사람 ID가 아니므로
동일 사람 반복·사건·재편집 녹음의 교차 누수는 미확인이다. 상담원 annotation overlap도
약 96% 통화에 있어 caller 구간이 순수 caller 분리음을 뜻하지 않는다.
이미 SVM/F0 결과를 본 split이므로 완전히 새로운 blind test라고 부르지 않는다.
metadata의 성별/파일명/경로/텍스트는 CNN 입력에 넣지 않는다.

## 입력·전체 데이터 처리

원본 8kHz mono를 그대로 읽고 형식·구간 오류는 즉시 실패한다. 임의 call 제한,
segment 선택, VAD 제거, 길이 상한, 무작위 crop은 없다. 겹치는 caller annotation도
MFCC와 동일하게 각각 유지한다. 각 구간별 FFT 256 / window 200 / hop 80,
40 Slaney mel (50–3800Hz), log(max(power,1e-10)), pre-emphasis 없음.
끝 잔여 샘플을 버리지 않도록 ceil frame과 zero padding을 사용한다.
256-frame 비중첩 chunk로 모든 frame을 보존하고 마지막 chunk는 zero pad+mask.
구간 경계는 섞지 않는다. chunk 길이는 계산 분할 단위이지 발화 절삭 상한이 아니다.

log-mel float32를 call 256개 단위 비압축 NPZ shard에 저장한다(약 110개).
유효 frame 자체는 14.87GB이나 현 구현은 마지막 chunk의 padding도 저장하므로
mel 배열은 약 25.01GB다.
한 shard에는 연속 mel 배열과 segment/call offsets를 저장하며 allow_pickle=False.
원본 WAV SHA256, source manifest/segments/split/code/config hash 및 coverage를 기록한다.
완료 shard는 임시 파일→fsync→rename, 영구 경로로 복사 후 SHA256 대조한다.
JSON commit index에 반영된 shard만 재사용하며 복구 시 재검증한다.
로컬 cache를 지워도 영구 shard에서 복구한다. raw feature에는 학습 통계를 섞지 않는다.
구현은 한 shard를 메모리에 유지하며 실제 크기와 RSS를 기록한다.

## 모델·학습 (단일 사전 지정 후보)

입력 B×1×40×256. Conv2d(3×3,pad=1,stride=2)+ReLU 3층:
1→8 (80 params), 8→16 (1,168), 16→32 (4,640).
각 층 뒤 시간 padding을 mask하고 마지막 유효 위치의 주파수·시간 평균을 낸다.
Linear(32,2)=66. 총 **5,954 trainable parameters**, FP32 weight 23,816 bytes.
출력 공간은 20×128, 10×64, 5×32. Conv forward MAC/chunk=1,658,880,
head=64, 총 **1,658,944 MAC/chunk** (bias/ReLU/mask/reduction 제외).
활성값 합은 chunk당 35,840 floats=140KiB, 학습 autograd/workspace는 별도다.

고정 Train 내 gender별 SHA256 순서로 약 10%를 inner-dev로 분리한다
(M 1,048 + F 1,191 = 2,239; inner-fit 20,149).
mel별 평균/표준편차는 inner-fit의 모든 frame에서만 계산한다.
AdamW lr=.001, weight_decay=.0001, seed=42, FP32, 최대 30 epoch,
patience=5, inner-dev call cross entropy 최소값으로 epoch 선택(동률은 앞 epoch).
각 epoch 모든 inner-fit call과 모든 caller chunk를 방문한다.
학습 loss는 call 내 유효 frame 비율로 가중한 chunk CE를 call 사이 동일 가중한다.
이는 call 평균 logit CE와 다른 학습 목적이며 명시적으로 선택한 약한 라벨 학습이다.
8 calls마다 optimizer step, chunk microbatch=32; 긴 call도 모두 gradient 누적한다.
추론은 유효 frame 수로 chunk logits를 평균하여 call당 하나의 예측을 낸다.
chunk/segment가 많은 call에 학습 가중치를 더 주지 않는다.

선택 epoch를 잠그고 새 seed=42 초기화로 Train 22,388 전체를 그 epoch 수만큼
재학습한다. 이때만 normalization을 전체 Train에 다시 fit한다.
최종 모델·epoch·normalization hash를 고정한 뒤 5,597개를 단 한 번 평가한다.
최종 평가 명령은 소비 표식을 먼저 영구 저장한다. 중단 시 동일 모델로만 복구하며,
다른 checkpoint로 재평가하지 않는다. 평가 결과를 보고 후보를 바꾸지 않는다.

## 계산 비용과 측정

실제 segment 길이로 frame/chunk 총수를 전수 계산하는 audit를 먼저 실행한다.
raw valid-frame bytes=frames×40×4, forward MAC=chunks×1,658,944.
대략 100만초라면 약 1억 frame=16GB raw feature이며, segment별 padding 때문에
단순 시간/2.56보다 chunk 수가 많다. 실제 audit 값을 계획 기준으로 사용한다.
학습 forward+backward는 forward의 약 3배를 예산용 가정으로만 둔다.
선택 및 재학습의 최대 60 epoch, inner-dev 최대 30회, 최종 평가 1회 비용을 모두 센다.
작은 모델이어도 많은 작은 chunk와 Python/IO 비용으로 빨라진다는 보장은 없다.

WAV read/decode, 메모리 내 SHA256, log-mel 계산, local shard write,
permanent copy/verify, normalization, inner selection, full fit, final inference의
wall seconds를 기록한다. 학습 시간에는 feature cache read, H2D, forward/backward,
optimizer가 포함된다. GPU stage 경계에서는 synchronize한다.
RSS high-water와 GPU allocated/reserved peak 및 환경/장치를 기록한다.
OS cold cache를 임의로 주장하지 않는다. 최초 전체 추출의 실제 wall time과
full epoch 시간을 보고하고, 짧은 warm-cache sample로 전체 wall time을 추정하지 않는다.
현재 L4 benchmark는 실행하지 않는다. 시간·성능·메모리 비교의 미측정 셀은 null이다.

최종 비교는 call_id+gender 일대일 결합: 정확도, 성별 정확도, confusion matrix,
둘 다 정답 / CNN만 정답 / MFCC만 정답 / 둘 다 오답, 오류 Jaccard,
paired discordance 및 정확도 차이 %p를 기록한다. 향상·무손실은 보장하지 않는다.

측정 참고: https://docs.pytorch.org/docs/stable/notes/cuda.html
저장 참고: https://numpy.org/doc/stable/reference/generated/numpy.savez.html
