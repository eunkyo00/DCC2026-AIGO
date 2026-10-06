# Train 성별 모델 예비 비교 · 과거 실행 절차

**보관 문서:** 아래는 당시 2,000통화 계획과 200통화 resume 절차다.
전체 2,000통화 실험은 생략했고 ECAPA Validation까지 완료했다. 아래 명령을 다시 실행할 필요가 없다.
[현재 결과·파일 안내](../README.md) · [최종 ECAPA 결과](../validation_results/REPORT.md).

## 완료된 200통화 비교 절차

2,000통화 pilot이 210통화까지 진행된 시점에 실행 시간을 줄이기 위해 단계적으로 비교하도록 변경했다.
[Screen_200_Resume.ipynb](../Screen_200_Resume.ipynb)의 코드 셀 하나를 기존 Colab 노트북에 복사해서 실행한다.
복사용 Python 파일은 [SCREEN_200_COLAB_CELL.py](../SCREEN_200_COLAB_CELL.py)이다.
기존 pilot 셀을 먼저 중지한다. 복구 셀은 남아 있는 동일 pilot 자식 프로세스도 종료 확인한다.
기존 8번 `run('pilot')`와 9번 `run('compare')` 셀은 다시 실행하지 않는다.

- 원래 call_id 정렬 순서의 첫 200통화를 두 모델의 공통 표본으로 고정한다(Male 101 / Female 99).
  결과·정오답·처리 속도로 대상을 선택하지 않는다. 전체 모집단을 정밀하게 대표하는 표본이라고 주장하지 않는다.
- 원래 서명된 bundle과 2,000통화용 cache는 수정하지 않는다. 호환되는 완료 결과를 별도 `_screen200` 폴더로 재사용한다.
- 모델 revision, 가중치, GPU, 패키지, 전처리와 추론 조건을 확인한다. 이미 완료한 200통화는 모델 로딩도 생략한다.
- 두 모델의 동일 200통화가 완료되면 정확도, 성별 정확도, 혼동행렬, 기존 fusion OOF 오답 교정/회귀를 비교한다.
- 결과는 `_screen200/comparison.json`과 `screen200_results.zip`에 저장한다. 시간은 실행 시작 시 남은 양과 실측값으로 추정한다.
- 후속 결정은 ECAPA의 고정 Internal Validation 평가다. 추가 2,000통화 비교는 생략한다. 200통화에서는 오답 1개가 0.5%p이므로 작은 차이로 승자를 정하거나 99% 달성을 주장하지 않는다.
  최종 후보는 별도의 충분한 검증과 기존 fixed Validation 평가가 필요하다.

복구·선택·손상 캐시 거부는 로컬 테스트로 확인했다. 새 복구 경로의 CUDA 실행은 Colab 결과로 확인한다.

## 기존 전체 notebook

[Gender_Model_Comparison.ipynb](../Gender_Model_Comparison.ipynb)에 Drive 연결 → 환경 설치 →
두 모델 로딩 → 6-call smoke → 20-call benchmark → Train 2,000통화 비교 → 결과 ZIP이 모두 들어 있다.
**기존 절차 기록:** 두 모델의 Colab 실행과 200통화 비교까지 완료했다. 아래 2,000통화 단계는 현재 실행 대상이 아니다.

## 실행

1. `artifacts/gender_comparison_bundle.zip`을 **내 드라이브/DDC-Colab/**에 업로드한다.
2. notebook을 Colab에 업로드하고 Python 3 / L4 GPU를 설정한다.
3. 1~9번을 위에서 아래로 실행한다. 오류 셀에서 중단하고 아래 셀은 실행하지 않는다.
4. 마지막 `comparison.json` 또는 `gender_comparison_results.zip`을 전달한다.

기존 원본 WAV 폴더 바로가기를 사용한다. embedding NPZ·원본 WAV·모델 weight는 새로 업로드하지 않는다.
ZIP에는 코드와 sample index, 기존 OOF 예측만 있다. 다운로드한 모델 weight는 Colab 임시 디스크에 둔다.

| 단계 | 시간 안내 |
|---|---|
| Drive·ZIP | 30초–2분, 로그인 별도 |
| 환경 설치 | 2–5분, 네트워크에 따라 증가 |
| Preflight | 1–5분, Drive 상태에 따라 증가 |
| 최초 모델 다운로드·로딩 | 3–10분, 네트워크에 따라 증가 |
| Smoke / Benchmark | 각각 1–5분 예상, 실제 값 기록 |
| 2,000-call 비교 | 직전 benchmark의 모델별 예상치 합계를 참고 |

위 시간은 GPU 실측값이 아니다. benchmark projection에는 모델 로딩과 cache flush 등이 빠져 있으며,
Drive 변동도 있으므로 전체 소요시간을 보장하지 않는다.

## 고정 정의

- 모델: `tiantiaf/wavlm-large-age-sex`, `JaesungHuh/voice-gender-classifier`.
- 코드·checkpoint revision 고정. 원본과 수정 코드 hash는 `model_sources.json`에 기록.
- FP32·Frozen·batch size 1, autocast·TF32 끔. 학습·threshold tuning 없음.
- Train 22,388개에서 seed 42, 성별 층화로 같은 2,000개 선정: Male 936 / Female 1,064.
- caller segments 31,269개 전부 사용. Internal Validation 추론 없음.
- 기존 6-call fixture에는 holdout이 포함되어 새 Train 전용 smoke 6개·benchmark 20개를 고정했다.
- 원본 8kHz mono, round 기반 caller crop, 기존 polyphase 8→16kHz resampling 유지.
- **새 입력:** caller crop을 시간순 연결 → 최대 15초의 균등 창 → 모델별 확률.
  3초 미만 전체 caller 음성만 단일 입력을 3초까지 zero-pad하고 개수를 기록한다.
- 원본 caller 샘플을 버리지 않으며 기존 구간 중복도 임의 제거하지 않는다.
- 창별 Female/Male 확률을 실제 샘플 수로 가중 평균한다. ECAPA의 원본 M/F 순서는 F/M으로 바꾼다.
- 기존 Wav2Vec2는 segment 동일 가중 embedding 평균이다. **입력·집계도 다른 파이프라인 비교**이며,
  차이를 backbone 하나의 효과로 해석하지 않는다.

Vox-Profile 공식 구현은 3초 미만 입력의 신뢰성과 15초 초과 길이에 주의를 명시한다.
caller 연결은 이 조건을 고려한 새 설계다. 연결 경계와 초단기 입력 padding의 한계가 있으며,
한국어 8kHz 음성과 사전학습 데이터의 차이가 없어지는 것은 아니다.

## 환경과 원본 코드 수정

새 subprocess와 system-site-packages 가상환경을 사용하여 기존 NumPy import 잔류 문제를 피한다.
Colab 기본 CUDA torch를 유지하고 torchaudio 버전을 맞춘다. 모델은 순서대로 GPU에 올린다.

WavLM 공식 코드에 맞춰 Transformers 4.46.3을 사용한다. 불필요한 SpeechBrain mask 함수를
batch-one full-length mask로 대체하고 processor에 CPU NumPy를 전달한다. backbone 중복 다운로드를
피하기 위해 고정 config로 초기화한 뒤 성별 checkpoint의 전체 safetensors를 strict=True로 읽는다.
누락 weight는 무시하지 않는다. LoRA는 weight 로딩 후 eval에서 병합된다. 이 checkpoint에서 꺼진
RevGrad import를 제거했다. 원본 라이선스는 vendor에 보존한다.

원본: [Vox-Profile](https://github.com/tiantiaf0627/vox-profile-release),
[ECAPA gender](https://github.com/JaesungHuh/voice-gender-classifier).

## 저장·복구

Drive의 `DDC-Colab/gender_comparison_v1`에 통화별 JSONL을 저장하고 10통화마다 fsync한다.
동일 환경 재실행 시 완료 통화는 건너뛴다. 마지막 불완전 JSON 행만 복구하며 내부 손상은 중단한다.
GPU·패키지·코드·data_root identity가 다르면 cache 혼합을 거부한다. 실패는 로그를 남기고 즉시 중단한다.
문제 해결 후 실패 통화를 재시도하며, 실패 통화를 제외하고 정확도를 계산하지 않는다.
같은 output에 두 notebook을 동시에 실행하지 않는다.

새 Colab 세션에서는 1~3번을 다시 실행한 뒤 이전에 완료한 단계 다음부터 이어갈 수 있다.
GPU와 패키지 조건이 다르면 새 output을 사용하거나 원래 환경을 복원해야 한다.

## 로컬 검증

Train-only coverage, fixed split, OOF hash, 구간 수, bundle hash, crop/resampling,
창 샘플 보존, 불완전 로그 복구, notebook JSON/Python 문법을 검사한다.
**실제 모델 CUDA 로딩·forward 검증은 Colab의 5번 셀에서 수행한다.**

```bash
.venv/bin/python mission-1/experiments/gender_model_comparison/build_handoff.py
.venv/bin/python -m unittest discover -s mission-1/experiments/gender_model_comparison -p 'test_*.py'
```

2,000통화는 예비 비교다. 작은 소수점 차이로 최종 승자를 확정하지 않는다.
전체 27,985-call 추출·추가 학습·최종 Validation 평가를 자동 실행하지 않는다.
