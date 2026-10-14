# 로컬 검증 기록 · 2026-09-22

브랜치 `mission1/eunkyo`. 이번 작업의 쓰기는 `wav2vec2_gender_finetuning/` 안에서만 수행했다.
다른 창의 WavLM split은 읽고 검증·복사했으며 원본은 변경하지 않았다. Git commit은 하지 않았다.

**5개 테스트 통과**, 최종 실행 13.609초. CPU synthetic model과 실제 CSV metadata 검증이며,
실제 원본 WAV 전체 읽기, pretrained checkpoint 로딩, L4 bf16 학습/메모리/속도 검증은 아니다.
GPU 단계는 제공한 notebook에서 실행해야 한다. 새 정확도·ETA는 아직 없다.

| 검증 | 결과 |
|---|---|
| frozen/new head 및 top-layer의 2-pass gradient vs full graph | 허용 오차 내 일치 |
| convolution/lower-layer frozen hash 보존, head/upper-layer 실제 업데이트 | 통과 |
| optimizer checkpoint 후 새 모델에서 resume vs 연속 2 epoch | 모든 가중치 bitwise 일치(CPU 작은 모델) |
| 누락/중복 ID, NaN, 확률 합 오류, argmax/label/identity/segment/sample 오류 | 전부 거부 |
| 2×5,597개의 합성 평가 JSON export → 로컬 verifier → 비교 보고서 | 통과 |
| export 파일 변경 | SHA256 오류로 거부 |
| 실제 CSV source SHA256, 27,985통화, 442,639 caller segments | 통과 |
| 기존 baseline 정답 수 재계산 | Wav2Vec2 5,443 / MFCC 5,321 / ECAPA 5,419 |
| WavLM inner split의 ID별 partition과 복사본 byte hash | 일치 |
| notebook 모든 code cell Python 구문, ZIP CRC, 포함 파일 hash, notebook ZIP hash binding | 통과 |

로컬 검증 환경: Python 3.12.2, torch 2.14.0, transformers 5.17.0,
NumPy 2.5.3, SciPy 1.18.1, soundfile 0.14.0. 테스트용 별도 venv만 만들었다.
Colab requirements의 NumPy 2.2.2/soundfile 0.13.1 및 현재 CUDA torch 환경과 다르므로
실제 실행 환경 확인·pilot을 대체하지 않는다.

Inner split:
- inner-train 17,910 / dev 4,478
- seed 42, gender-stratified `wavlm-inner-v1|42|call_id` SHA256 order
- CSV SHA256 `b6664bee9b72dd14a53743388274577d6c5b73e5ef5f51ab4d53da6fa3321e17`
- mapping SHA256 `43ed10cb53bbe9a7d8335a5e5edf463eafdfcac88f67f8b94981a4c0bf4da945`

최종 upload ZIP: 7,697,018 bytes.
SHA256 `418a39dbe1f28dddcfa03fcd5ff357fa2243562ab0c8cc8bc564d56c54788abe`.
ZIP·WAV·weights·venv·대형 파생 jobs index의 gitignore 적용을 확인했다.

공용 README 변경안은 `SHARED_README_DRAFT.md`에만 저장했다.
