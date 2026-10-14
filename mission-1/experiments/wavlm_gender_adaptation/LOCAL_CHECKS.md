# 로컬 검증 기록 · 2026-09-22

실제 데이터 학습·추론 결과가 아니다. L4 GPU, 실제 음성 청취, 성능, VRAM, ETA는 미측정이다.

- 원본 split/calls/segments의 요청 SHA256 3개 일치.
- 27,985 unique calls, Train 22,388 / Internal Validation 5,597 및 성별 수 검증.
- 고정 inner-train 17,910 / dev 4,478. Train segment 성별과 call 성별의 metadata 충돌 0건.
- Train 21,578통화에서 caller/operator 주석 시간 겹침 위험. 실제 동시 발성이나 화자 오류라는 뜻은 아니다.
- Python compile 및 notebook 모든 코드 셀 compile 확인.
- stdlib 무결성 5개 + 선택 의존성 CPU 검사 5개 = **10개 통과**.
- 5,597개 합성 결과의 local verify/metrics/report 및 hash·argmax·실패 상태 변조 거부.
- 중복/누락 ID, NaN/inf, 잘못된 확률 합, 오디오 coverage, lock 충돌, truncated journal 거부.
- 실제 HF WavLM 구조를 축소한 랜덤 모델(24 layers, hidden 16)의 상위 2층만 gradient 발생 확인.
- 합성 WAV의 caller crop/resample·전체 창 샘플 보존, 고정 crop 재현성 확인.
- checkpoint optimizer/scheduler/RNG 복원 뒤 다음 update의 tensor가 uninterrupted CPU 실행과 정확히 일치.
- dev 평가 도중 실패 → 완료 dev rows 및 epoch-end checkpoint 재개 → 저장 완료 학습 step 중복 없이 4 epochs 완료;
  uninterrupted 실행과 최종 tensor 일치.

CPU 검사는 저장소 `.venv`를 변경하지 않고 별도 `/tmp/aigo-wavlm-cpu-check`에서 실행했다.
Python 3.12 / torch 2.6.0 CPU / transformers 4.46.3. 이는 Colab torch 2.11 CUDA 호환 검증을 대체하지 않는다.
원본 모델 weights 다운로드/전체 학습은 로컬에서 수행하지 않았다.

```bash
/tmp/aigo-wavlm-cpu-check/bin/python -m unittest discover \
  -s mission-1/experiments/wavlm_gender_adaptation -p 'test_*.py' -v
```

Colab에서는 고정 revision 실제 weights로 preflight 수치 검사 및 실제 forward/backward 100 steps,
데이터 읽기·checkpoint 저장·전체 caller 평가 표본의 wall time을 확인해야 한다.

## T4 수정본 · 2026-09-26

T4 FP32 자동 선택, T4 BF16 거부, 기존 L4 BF16 경로 유지,
GPU별 config/output 분리와 생성 notebook의 설정 셀 실행을 추가 확인했다.
동일 CPU 검사 환경에서 **12개 테스트 통과**, Python compile 및 새 ZIP 전체 파일 hash와 notebook hash 일치.
T4 실제 GPU 실행·VRAM·시간은 아직 미측정이며 Colab preflight/benchmark가 필요하다.
새 업로드 파일: `artifacts/wavlm_gender_adaptation_t4_v2_bundle.zip`.
