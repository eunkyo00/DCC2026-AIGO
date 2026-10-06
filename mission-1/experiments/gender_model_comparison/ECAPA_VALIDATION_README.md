# ECAPA · Internal Validation 실행 안내

상태: **L4 추론 및 로컬 검증 완료: 96.819725% (5,419/5,597), 오답 178개.**
[검증 REPORT](validation_results/REPORT.md) · [Metrics](validation_results/metrics.json). 아래 실행 절차는 재현용이며 재추론할 필요가 없다.
브랜치 `mission1/eunkyo`. 추가 2,000통화, WavLM 재실행, 학습, 전처리 변경은 하지 않는다.

## Colab에서 실행

1. 로컬 `artifacts/ecapa_validation_bundle.zip`을 Drive의 `내 드라이브/DDC-Colab/`에 업로드한다.
2. [ECAPA_Internal_Validation.ipynb](ECAPA_Internal_Validation.ipynb)를 Colab에서 연다.
3. 기존 Train 추론을 중지하고 **Python 3 · L4 GPU** 런타임에서 1–3번 셀을 실행한다.
4. preflight 출력과 실측 기반 예상 시간을 확인한 뒤 4번 셀을 실행한다.
5. 완료되면 5번 셀에서 `ecapa_validation_results.zip`을 내려받아 로컬에 전달한다.

[ECAPA_VALIDATION_COLAB.py](ECAPA_VALIDATION_COLAB.py)는 같은 내용을 담은 전체 Colab 셀 코드다.
ZIP 안에 runner와 기존 ECAPA 구현이 모두 포함된다. 경로를 임의로 추정하지 않고
`/content/gender_model_comparison/config.json`의 실제 `data_root`를 재사용한다.
런타임이 새로 생성되어 config가 없으면 기존 Drive `identity.json`의 경로를 사용한다.

| 항목 | 경로 |
|---|---|
| 기존 Python | `/content/gender_compare_env/bin/python` |
| 기존 Train 코드/config | `/content/gender_model_comparison/` (보존) |
| 새 Validation 코드 | `/content/ecapa_gender_validation/` |
| HF 캐시 | `/content/gender_model_hf_cache` |
| 기존 Train provenance | `/content/drive/MyDrive/DDC-Colab/gender_comparison_v1/` |
| 새 결과 | `/content/drive/MyDrive/DDC-Colab/ecapa_gender_validation_v1/` |

L4 전용이다. `torch.cuda.get_device_name(0)`이 `NVIDIA L4` 또는 `L4`인지 환경 셀과 runner 양쪽에서 검사하며, 다른 GPU에서는 추론을 시작하지 않는다.
정상 환경은 재설치하지 않는다. Python/GPU/기존 패키지 버전이 Train identity와 다르면 중단한다.
CUDA 13 torchaudio 오류가 있을 때만 기존 torch `2.11.0+cu128`을 확인하고 공식 cu128
index의 동일 torchaudio wheel을 `--no-deps`로 복구한다. 패키지 버전 검사를 끄지 않는다.
기존 identity·ECAPA load 기록이 필요하며 기존 Train 추론을 다시 실행해 만들지 않는다.

## 고정 입력·모델 검증

원본 세 metadata SHA256은 인계 문서와 모두 일치한다. 빌드 시 전체 ID·성별·partition,
모든 caller 구간·길이·segment 수를 검사한다. 정답은 모델 입력에 전달하지 않는다.

| 확인 항목 | 값 |
|---|---:|
| Internal Validation | 5,597통화 |
| Male / Female | 2,620 / 2,977 |
| caller 구간 | 88,290 |
| crop/resample 후 caller 총 길이 | 187,393.638초 (52.0538시간) |
| 최대 15초 균등 창 | 15,347 |
| 3초 미만 단일 입력 padding | 1통화 |

이 음성 길이는 추론 실행시간이 아니다. 겹치는 caller 구간도 기존 정의대로 각각 보존한다.
공식 Validation 3,640개는 포함하지 않는다.

기존 `compare_models.py`의 `load_wave`, `windows`, `load_model`, `predict`를 변경 없이 호출한다.
checkpoint는 `JaesungHuh/voice-gender-classifier`, revision은
`db1222153bd60337e900be22add7af180452adc0`이다. vendored 코드와 strict loading을 유지한다.
FP32 / eval / Frozen / batch 1, autocast·TF32 비활성화. 8kHz crop → 기존 Kaiser
resample_poly → 모든 caller 연결 → 균등 창 → 실제 샘플 수 가중 softmax 평균,
M/F에서 F/M 순서로 변환 후 argmax한다. 추가 학습·threshold 조정은 없다.

## 시간 추정

이번 완료 실행의 L4 추론·Drive 저장 시간은 **7,720.671초 (2시간 8분 41초)**였다.
아래는 실행 전에 사용한 추정 방법이다.
preflight는 Drive의 검증된 ECAPA screen200 기록을 우선 사용하고 없으면 benchmark를 사용한다.

- 통화 수 기준: `기존 실측 초 / 기존 통화 수 × 남은 통화 수`.
- 음성량 기준: `기존 실측 초 / 기존 caller 샘플 수 × 남은 caller 샘플 수`.
- 초기 수치는 모델 로딩·Drive flush 제외다. 두 수치는 신뢰구간이 아니다.
- 첫 통화 및 이후 매 10통화에 현재 세션의 읽기·추론·Drive flush 포함 wall time으로 갱신한다.
- 모델 로딩 시간은 별도 출력한다. 초기 및 갱신 추정은 `eta.json`, `progress.json`에 기록한다.

## 저장·중단·재개

`predictions.jsonl`에 통화별 ID·identity hash·F/M 확률·argmax·성별·segment 수·실제 샘플 수·
창/padding 수·시간을 저장하고 **매 통화 flush/fsync**한다. Drive 서비스 자체의 동기화 지연은 있을 수 있다.
실패하면 즉시 중단하고 `failure_attempts.jsonl`에 남긴다. 재시도 성공 후에도 과거 실패 기록을 보존한다.
완료된 ID는 재추론하지 않으며 identity·확률·샘플/창 수를 검증한 뒤 재개한다.
손상된 마지막 미완성 JSONL 조각만 원본을 보관하고 복구한다. 중간 손상·중복 ID는 거부한다.

notebook Stop은 subprocess group에 TERM 후 필요 시 KILL을 보내 자식도 종료한다.
프로세스 부모가 종료되면 Linux parent-death signal도 사용한다. SIGKILL/런타임 소실은
실패 로그를 기록할 수 없으므로 완료 ID coverage와 남은 lock으로 식별한다.

출력 폴더의 `.validation.lock`으로 중복 실행을 차단한다. 강제 종료 후 lock이 남으면
notebook 하단에서 owner를 읽고 **이전 실행 종료를 확인한 후** token을 지정해 수동 해제한다.
자동 시간 만료는 없다. Drive의 여러 VM 간 동기화는 분산 lock 보장을 제공하지 않으므로
한 OUTPUT을 여러 Colab 런타임에서 동시에 실행하지 않는다.

전부 완료되어야 `completion.json`과 결과 ZIP을 생성한다. 실패 통화를 제외한 점수는 생성하지 않는다.
종료 후 동일 4번 셀을 실행해도 완료 통화는 다시 추론하지 않고 검증·export만 한다.

## 로컬 검증·분석

저장소 루트에서 실행한다(GPU나 torch 없이 stdlib만 필요). **이번 완료 ZIP은 과거 실행 버전이므로**
`verify_returned_validation.py`를 사용한다. 이 명령은 현재 bundle과 당시 실행 bundle의 차이를
hash로 검증한다. 아래는 기존 원본 metadata와 생성 jobs가 준비된 로컬 기준이다.

```bash
python3 mission-1/experiments/gender_model_comparison/verify_returned_validation.py \
  /absolute/path/ecapa_validation_results.zip
```

검사: ZIP/file hash, bundle·source provenance·환경·모델 identity, 5,597 unique ID exact coverage,
성별, `(5597, 2)` 확률, NaN/inf·범위·합=1, argmax, 모든 caller 샘플·창·padding,
실패 이력 및 미해결 실패. 로컬의 기존 MFCC·Wav2Vec2 Validation 예측 hash와 동일 ID·정답을 확인한다.

현재 bundle로 새로 생성한 결과에만 `analyze_ecapa_validation.py`를 사용한다.

기본 출력 `validation_results/`:

- `metrics.json`: Overall/Male/Female Accuracy, 정답·오답, confusion matrix (행=정답/열=예측 M/F),
  MFCC·Wav2Vec2 각각의 corrected/regressed/both_wrong/both_correct, 99% 판정.
- `call_comparison.csv`: 5,597개 ID별 ECAPA 확률·정답·세 모델 예측.
- `REPORT.md`: 검증된 점수·오류 비교와 해석 제한.

Train fusion OOF는 비교하지 않는다. 99%는 최소 5,542 정답/최대 55 오답이다.
로컬 검증을 완료해 상위 README 성능표에 ECAPA 96.819725%를 반영했다.
200통화 ZIP 원본이 있으면 함께 전달해 독립 검증할 수 있지만 200통화 재추론은 필요하지 않다.
기존에 관찰한 Internal Validation이며 새로운 독립 test가 아니다. 후속 개선은 Train 실험으로 분리한다.

## 재현·로컬 검증 이력

```bash
python3 mission-1/experiments/gender_model_comparison/build_validation.py
.venv/bin/python -m unittest discover -s mission-1/experiments/gender_model_comparison -p 'test_*.py' -v
```

빌드는 원본 metadata가 있는 로컬에서 실행한다. 새 checkout에는 Git 제외된
`data/jobs.json`과 `validation_data/jobs.json`이 없다. 기존 Train bundle의 `data/jobs.json`을 복원하거나
원본 CV OOF 자료를 갖춘 환경에서 `build_handoff.py`로 재생성한 뒤 Validation을 빌드한다.
원본 metadata·CV 자료가 없으면 공개 REPORT·metrics만 확인할 수 있다. 별도 manifest·jobs·ZIP·notebook을 재생성하고
기존 Train bundle은 검증만 한다. 이번 검증에서 13개 CPU 테스트가 통과했다.
실패 후 resume에서 이미 완료된 첫 통화를 재실행하지 않는지, lock/손상/중복 거부,
5,597개 **합성 임시 결과**의 export→로컬 비교와 변조 거부를 확인했다.
합성 테스트는 실제 성능 결과가 아니며 저장소에 결과를 남기지 않는다. CUDA 실동작은 사용자의 Colab 실행과 반환 ZIP에서 확인했다.

원본 WAV·weight·대용량 캐시는 Git에 추가하지 않는다. ZIP과 생성 metadata/통화별 결과는 로컬에 보존하고,
재현 코드·manifest·검증 완료 metrics·REPORT를 공유한다.

## 이번 반환 ZIP 재검증

실행 당시 bundle은 패키지 오류 상세 표시를 추가하기 전 버전이다. 검사를 끄지 않고,
알려진 문구 변경만 되돌린 코드가 반환 manifest의 모든 hash와 일치하는지 확인했다.
현재 bundle·원본 ZIP은 수정하지 않았다. 다음 명령이 당시 코드 복원과 분석을 재현한다.

```bash
python3 mission-1/experiments/gender_model_comparison/verify_returned_validation.py \
  /absolute/path/ecapa_validation_results.zip
```

반환 ZIP SHA256: `2a26d6d23a15d182c9e7fbf7085a0eb6a4bf409f0bb2e6b7b7aad67bc835271e`.
