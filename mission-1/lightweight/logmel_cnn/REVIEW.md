# 기존 MFCC + RBF SVM 및 누수 검토

기준 결과는 고정 split SHA256 `04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06`의
Train 22,388 call / Internal Validation 5,597 call에서 나온 95.068787%다. MFCC 코드는
`mission-1/baseline/mfcc_svm/run_mfcc_baseline.py`이며, 8 kHz, 13 MFCC, 26 mel band,
FFT 256, 25 ms Hamming window, 10 ms hop, 50–3,800 Hz, pre-emphasis 0.97을 사용한다.
각 caller segment의 MFCC mean/std를 다시 call 단위 mean/std로 집계해 54차원을 만들고,
Train-only median imputation과 StandardScaler를 적용한 `SVC(C=1, gamma=scale, rbf)`를
학습한다. 저장된 confusion matrix는 `[[2484,136],[140,2837]]` (행 true M/F, 열 pred M/F),
오답은 276개다. Male accuracy는 94.809160%, Female accuracy는 95.297279%다.

같은 Validation에서 F0 + acoustic + Logistic Regression은 91.209577%였고, 두 모델 모두
정답 4,967개, MFCC만 정답 354개, F0만 정답 138개, 둘 다 오답 138개였다. 이 문서의 CNN
실험은 이 비교를 참조하되 MFCC 결과를 보고 Validation 설정을 조정하지 않는다.

## 확인된 누수와 한계

- split은 call 단위이므로 같은 call의 segment가 Train과 Validation으로 나뉘는 누수는 막는다.
- EDA에서 call_id, recordId, audioPath, filename stem과 exact WAV bytes의 cross-split 중복은
  확인되지 않았다. 다만 exact byte 검사에는 header failure 제외와 re-encoding/cropping 미검출
  한계가 있다.
- person ID, 전화번호 hash, 사건 단위 식별자가 없어 동일 신고자의 다른 통화가 양 split에
  반복되는 speaker-level leakage는 확인할 수 없다.
- caller/operator annotation overlap이 약 96%인 것으로 조사되어 `speaker=1` 구간에
  상담원 음향이 실제로 섞였을 가능성이 있다. annotation overlap만으로 동시 발화 자체를
  확정하지 않는다.
- MFCC와 CNN 모두 `speaker=1` 구간 및 성별 label을 사용하므로 이 known risk를 해결하지
  않는다. 공식 Validation 3,640개도 사용하지 않는다.

## CNN 비교의 측정 원칙

CNN의 시간은 WAV decode/read, source hash, log-mel 계산, local shard write, permanent copy
및 verify, inner selection, full Train fit, Validation inference로 나눠 기록한다. RSS high-water와
장치 동기화 후의 GPU 측정값을 저장한다. 전수 audit에서 계산한 92,916,809 valid frame,
610,649 chunk, raw float32 14,866,689,440 bytes(약 13.85 GiB), 1,013,032,494,656
forward MAC을 기준으로 예산을 잡는다. 짧은 warm-cache sample로 전체 실행 시간을 추정하지
않는다. 실제 CNN 실행 전에는 성능 향상, 오류 무손실, MFCC보다 짧은 실행시간을 보장하지 않는다.
