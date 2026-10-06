# Mission 2 — Small-stem ResNet18

> 이 폴더는 앞 1.5초 특징을 사용하는 **초기 90.248957% 단독 모델**의 재현 기록이다. 현재 유지하는 모델은 `first_plus_whole`을 사용하는 **ResNet18+34 결합, Accuracy 91.419%**이며 [Mission 2 개요](../README.md)와 [최신 결과 보고서](../results/resnet_ensemble_fixed_05.md)를 참고한다. 이 폴더의 단일 모델 실행 명령으로 최신 결합 성능이 자동 재현되는 것은 아니다.

## 1. 실험 한눈에 보기

서울 지역 Training **889,358개 발화**를 학습하고 공식 Validation **111,947개 발화**에서 역할을 예측한다. `speaker=0`은 119대원, `speaker=1`은 신고자다. 전사 텍스트, 정답 화자, 발화 순서·파일명·통화 ID는 모델 입력에 사용하지 않는다.

```text
한 발화의 WAV → 앞 1.5초 64-band log-Mel → Small-stem ResNet18
    → 신고자 확률 → 고정 threshold 0.5 → speaker 0/1
```

## 2. 데이터와 전처리

저장소 루트의 공통 파이프라인을 `inspect_schema` → `io` → `filter_seoul` → `audio_check` → `eda_all` → `build_manifests` 순으로 실행해 서울 데이터만 추린다. [Mission 2 EDA](../eda/REPORT.md)에 실제 집계 결과를 기록했다. 생성된 매니페스트 `mission2_speaker_Training.csv`, `mission2_speaker_Validation.csv`는 로컬 절대경로가 들어 있어 커밋하지 않는다.

`github_preprocessing.py`의 핵심 함수는 공통 전처리 commit `c837272`의 구현과 대조했다. `precompute_features.py`는 JSON의 `startAt`/`endAt`(ms)으로 발화를 자르고 끝에 0을 붙이거나 앞 1.5초를 취해 64-band log-Mel을 `[0,1]`로 만든다. 입력 특징은 `[64,24]` float16 shard로 저장했다가 학습 때 float32로 복원한다. **이 90.25% 실험은 `first_1p5` 특징만 사용했다.** 코드의 `first_plus_whole` 분기는 후속 실험용이다.

원본 WAV를 보유한 경우 이 디렉터리에서 특징을 생성한다. `../../manifests/`는 공통 스크립트를 저장소 루트에서 실행했을 때의 경로다.

```bash
python -m pip install -r requirements_precompute.txt
python precompute_features.py \
  --data-root "/path/to/대학부 데이터" \
  --train-manifest ../../manifests/mission2_speaker_Training.csv \
  --valid-manifest ../../manifests/mission2_speaker_Validation.csv \
  --output-dir "/path/to/mission02_features" \
  --workers 6 --shard-size 50000
python verify_features.py "/path/to/mission02_features"
```

이미 `metadata.json`, `Training/`, `Validation/`을 갖춘 `mission02_features/`가 있다면 원본 WAV 없이 아래 학습부터 실행할 수 있다.

## 3. 모델과 학습

작은 Mel 입력에 맞춰 일반 ResNet18의 `7×7, stride=2` stem을 `3×3, stride=1`로 바꾸고 초기 max-pooling을 제거했다. ImageNet 사전학습 가중치로 시작하며 SpecAugment, Mixup, label smoothing, AdamW와 cosine scheduler를 사용한다. 파라미터 수는 **11,169,858개**다.

```bash
python -m pip install -r requirements.txt
python train.py \
  --feature-root "/path/to/mission02_features" \
  --output-dir "/path/to/persistent/mission02_experiment_smallstem" \
  --model-name resnet18_smallstem \
  --epochs 16 --batch-size 512 --num-workers 2 \
  --learning-rate 2e-4 --dropout 0.25 \
  --label-smoothing 0.03 --mixup-alpha 0.10 \
  --patience 4 --resume
```

출력 경로는 Google Drive처럼 런타임 종료 후에도 남는 곳으로 지정한다. 매 epoch `history.json`과 재개용 `last_checkpoint.pt`를 저장하고, 최고 Validation 모델을 `best_model.pt`로 갱신한다. 중간에 끊기면 **같은 명령**을 다시 실행해 완료된 epoch 다음부터 재개한다.

`DCC_mission2_improve.ipynb`의 Small-stem 셀은 위 명령과 같지만, 노트북 전체는 기존 baseline·hybrid까지 비교한다. 또한 Drive에 기존 `mission02.zip`, 특징 폴더, baseline 체크포인트가 있어야 하므로 **이 저장소 파일만으로 곧바로 노트북 전체를 실행할 수는 없다**. Small-stem만 재학습할 때는 위 명령을 사용한다.

## 4. 기록된 Validation 결과

사용자가 제공한 실험 ZIP의 `best_model.pt`와 [전체 epoch 기록](results/history.json)이 일치한다. epoch 16에서 **101,031 / 111,947개 정답 = 90.248957%**였으며 Validation에서 선택한 threshold는 **0.535**다. 기본 threshold 0.5에서는 **90.139977%**였다.

| 실제 역할 | 예측 119대원 | 예측 신고자 | 역할별 정답률 |
|---|---:|---:|---:|
| 119대원 (`0`) | 46,396 | 7,449 | 86.17% |
| 신고자 (`1`) | 3,467 | 54,635 | 94.03% |

[결과 보고서](results/REPORT.md)와 [metrics.json](results/metrics.json)에 수치와 확인 범위를 남겼다. 이는 같은 Validation에서 epoch와 threshold를 선택한 결과이며 비공개 Test 성능은 아니다.

## 5. 결과 기록 확인

저장소에 포함된 공개 기록끼리 일치하는지 검사할 수 있다. 별도로 보관한 `best_model.pt`가 있으면 해시와 체크포인트 안의 수치도 비교한다.

```bash
python verify_results.py
python verify_results.py --checkpoint "/path/to/mission02_experiment_smallstem/best_model.pt"
```

이 검사는 **원본 Validation 음원을 다시 추론하는 절차가 아니다**. 원본 음원과 체크포인트를 통한 독립 재평가는 별도로 해야 한다.

## 6. 최종 추론

테스트 WAV/JSON과 별도로 보관한 `best_model.pt`가 준비되면 한 번의 명령으로 CSV를 생성한다.

```bash
python inference.py \
  --audio_dir "/path/to/test/1.원천데이터" \
  --label_dir "/path/to/test/2.라벨링데이터" \
  --ckpt_path "/path/to/mission02_experiment_smallstem/best_model.pt" \
  --output "/path/to/outputs/mission2.csv"
```

출력 컬럼은 `audio file name,startAt,endAt,speaker`다. 추론 시 JSON에서 발화의 `startAt`·`endAt`만 읽고 `speaker`·`text`는 읽지 않는다. 전처리 종류와 모델명은 체크포인트에서 읽지만 **threshold는 저장값과 관계없이 0.5**로 고정한다.

### 2026-10-06 규정 반영

현재 학습 코드는 임계값 탐색을 제거하고 Accuracy@0.5로 best epoch와 early stopping을 결정한다. 과거 튜닝 실행의 `--resume`은 거부한다. 새 output-dir에서 실행해야 하며, 과거 가중치를 초기값으로 사용하는 것은 과거 모델 선택 이력을 없애지 않는다. `results/`와 `verify_results.py`는 원본 과거 실험 기록·일치 검사이며 현재 제출 성능 검증이 아니다. 과거 0.535의 90.248957% 대신, 같은 체크포인트의 0.5 결과는 90.139977%다.

## 7. 파일과 한계

| 파일 | 역할 |
|---|---|
| `precompute_features.py`, `verify_features.py` | 원본 WAV에서 특징 생성·검사 |
| `train.py`, `model.py`, `feature_dataset.py` | 학습·모델·특징 로더 |
| `inference.py` | 제출 형식 CSV 생성 |
| `verify_results.py`, `results/` | 공개 성능 기록의 내부 대조 |
| `DCC_mission2_training.ipynb`, `DCC_mission2_improve.ipynb` | 기존 Drive 작업용 Colab 노트북 |
| `test_*.py` | 합성·작은 데이터 기반 코드 검사 |

원본 WAV/JSON, 개별 발화 매니페스트, 약 3GB의 특징, `best_model.pt`(약 45MB), `last_checkpoint.pt`(약 134MB)는 저장소에 넣지 않는다. 현재 작성자의 체크포인트는 Drive의 `MyDrive/DCC/mission02_experiment_smallstem/best_model.pt`에 별도 보관돼 있다. 따라서 **저장소만 내려받아서는 90.25%를 재평가하거나 제출 추론을 실행할 수 없다**. 실제 성능은 실행 환경·학습 재실행에 따라 달라질 수 있다.
