# DDC Mission 1 STEP 3 — 고정 Call-level Validation

모델 학습·추론·음향 특징 추출 없이 Training 정상 파싱·매칭 통화만으로 내부 split을 생성했다. 작업 브랜치는 `mission1/eunkyo`이며 기존 EDA·baseline·공통 코드는 수정하지 않았다.

## 1. 설정과 데이터 출처

- split 버전: `m1_internal_call_v1`; 단위: `call_id`; 비율: Train 80% / Internal Validation 20%.
- 고정 seed: **42**; stratification: **gender (M=남, F=여)**.
- gender별 `SHA256(seed + NUL + gender + NUL + call_id)` 순으로 정렬하고 앞 20%를 Internal Validation으로 배정한다. 해시 순서는 의사무작위 순열로 사용한다. 동률이면 call_id로 정렬한다. seed 탐색이나 모델 점수 기반 선택은 하지 않았다.
- `call_id`는 EDA의 `file_ref`와 동일하며 `SHA256(NFC(파일 stem))`이다. 실제 인물 ID가 아니다.
- EDA `calls.csv`에서 `split=Training`, `matched=True`만 선택했다. 공식 Validation 3,640개는 배정·비율·majority 결정에 사용하지 않았다.
- EDA `segments.csv`의 발화 경계·화자·gender를 재사용했다. WAV 경로만 Training 파일 목록에서 연결했다. WAV/JSON 내용을 다시 읽지 않았다.
- 기존 EDA에서 확정한 Training 27,985통화를 v1의 고정 분석 대상으로 사용한다.

**컬럼 정의:** `n_segments`는 신고자(speaker=1) segment 수, `total_duration`은 신고자 segment 길이의 단순 합(초)이다. 겹침 제거 시간은 EDA의 `reporter_union_duration`으로 별도 보존한다. `n_all_segments`와 `all_segment_duration`은 상담원을 포함한다. 음성을 결합하는 aggregation이나 전처리는 수행하지 않았다.

## 2. 통화 및 성별 분포

전체 정상 통화 **27,985개**를 빠짐없이 한 번씩 배정했다.

| partition | call 수 | 남 M | 남 비율 | 여 F | 여 비율 |
| --- | --- | --- | --- | --- | --- |
| train | 22388 | 10480 | 46.8108% | 11908 | 53.1892% |
| internal_validation | 5597 | 2620 | 46.8108% | 2977 | 53.1892% |

## 3. 누수 및 정합성 검증

| 검사 | 결과 |
| --- | --- |
| call_id_overlap | 0 |
| wav_path_overlap | 0 |
| relative_wav_path_overlap | 0 |
| filename_prefix_overlap | 0 |
| segment_leakage_calls | 0 |
| unassigned_segments | 0 |
| duplicate_call_ids | 0 |
| missing_gender | 0 |
| conflicting_gender | 0 |
| official_validation_calls_in_split | 0 |

기존 인덱스에서 신고자 segment **442,639개**를 대조했다. 상담원을 포함한 모든 대상 segment도 동일한 부모 배정을 상속했다. 통화별 전체/신고자 segment 수와 신고자 duration을 EDA 통화 표에 다시 대조해 일치함을 확인했다.

| partition | 부모 call 배정을 상속한 전체 화자 segment |
| --- | --- |
| train | 681604 |
| internal_validation | 170246 |

파일 stem에서 날짜 형태 suffix를 제외한 앞부분을 추가 식별자 후보로 비교했다. prefix는 전체 call_id와 마찬가지로 두 partition 사이 교차가 0이었다. 실제 사건 식별자라는 의미는 확인 필요다.

| 기존 JSON ID | 내부 교차 | 근거 |
| --- | --- | --- |
| _id | 0 | EDA에서 정상 파싱 전체 32,838개의 값이 고유 |
| recordId | 0 | EDA에서 정상 파싱 전체 32,838개의 값이 고유 |
| audioPath | 0 | EDA에서 정상 파싱 전체 32,838개의 값이 고유 |

추가 JSON ID 결과는 기존 EDA의 전역 고유성 결과를 재사용했다. 고유한 집합의 부분집합에서는 중복이 새로 생길 수 없다. 원본 JSON을 재파싱해 ID를 새로 추출한 검사는 아니다. 해당 근거 파일의 SHA256을 metadata에 고정했다. EDA 이후 원본 내용이 변경됐는지 확인하는 전수 checksum 검사는 이번 범위에서 수행하지 않았다.

segment를 무작위로 나누면 같은 통화의 신고자 음색·녹음 채널·배경음·겹친 음성 구간이 양쪽으로 들어갈 수 있다. 이번 배정은 통화의 모든 segment가 한 partition을 상속하므로 이런 통화 내부 누수를 방지한다. Random segment split과의 모델 성능 비교는 하지 않았다.

## 4. 분포 비교

| partition | 신고자 segment 총수 | call당 평균 | 중앙값 | 신고자 duration 총합 s | call당 평균 s | 중앙값 s |
| --- | --- | --- | --- | --- | --- | --- |
| train | 354349 | 15.8276 | 15.0 | 749744.974 | 33.4887 | 30.219 |
| internal_validation | 88290 | 15.7745 | 15 | 187393.638 | 33.4811 | 30.144 |

duration 구간은 `[0,0.5)`, `[0.5,1)`, `[1,2)`, `[2,4)`, `[4,∞)`초로 서로 배타적이다. 원래 정수 밀리초 경계에서 차이를 계산해 정확히 0.5/1/2/4초인 구간의 오분류를 막았다. 신고자 segment 기준이다.

| 길이 구간 | Train 수 | Train 비율 | Internal Validation 수 | Internal Validation 비율 |
| --- | --- | --- | --- | --- |
| <0.5s | 28480 | 8.0373% | 7029 | 7.9613% |
| 0.5–1s | 90479 | 25.5339% | 22420 | 25.3936% |
| 1–2s | 101284 | 28.5831% | 25197 | 28.5389% |
| 2–4s | 84563 | 23.8643% | 21256 | 24.0752% |
| >=4s | 49543 | 13.9814% | 12388 | 14.0310% |

여성 비율 차이는 0.000000 percentage point, 발화 길이 구간 비율의 최대 차이는 0.2109 percentage point다. call당 segment 수 SMD는 -0.0087, duration SMD는 -0.0005다.

SMD는 (Validation 평균 − Train 평균) / sqrt((Train 분산 + Validation 분산)/2)다. 사전에 정한 기술적 점검 기준은 |SMD|>0.1 또는 구간 비율 차이>2 percentage points이며, 현재 점검 플래그는 **False**다. 통계적 동등성이나 동일 인물 분리를 보장하는 검정은 아니다. 분포를 맞추려고 seed를 바꾸지 않는다.

## 5. Baseline 0: Majority class

내부 Train에서 가장 많은 gender는 **F (여성)**이다. 이를 모든 Internal Validation 통화에 예측하면 Accuracy는 **2,977 / 5,597 = 53.189209%**이다. 정확한 비율 0.5318920850455601를 split_summary.json에 저장했다.

정답 분포를 이용한 산술 계산이며 어떤 분류기나 음향 특징도 학습·추출하지 않았다. 다수 class 선택은 내부 Train에서만 수행했다. Accuracy의 분모는 segment 수가 아닌 call 수다.

## 6. 고정 사용 계약 및 확인 필요

- 모든 후속 모델은 `split_assignments.csv`와 해당 통화 목록을 사용한다. 모델별로 다시 split하거나 seed를 바꾸지 않는다.
- 기존 파일이 있으면 생성 명령은 실패한다. `--verify`는 입력·출력 SHA256, 배정 재현성, 누수, 분포를 읽기 전용으로 재검증한다.
- 절대 경로가 있는 통화 목록은 `manifests/`에 저장하며 기존 저장소 ignore 규칙을 따른다. 경로 없는 배정표와 metadata·보고서는 공유 대상으로 보존한다.
- EDA segment 인덱스의 `file_ref`를 배정표 `call_id`에 many-to-one으로 연결한다. 미매칭 call은 제외 사유를 명시하고, partition별 독립 segment split은 금지한다.
- 누락/손상 파일 복구, 라벨 수정, cohort 변경은 새 split 버전으로 검토해야 한다. v1을 덮어써 비교 기준을 바꾸지 않는다.
- 동일 인물의 여러 통화 여부: **확인 필요**. speaker는 역할 ID이며 사람 ID가 아니다.
- 같은 사건 또는 편집·재인코딩 원본 재사용 여부: **확인 필요**. ID/파일명 고유성이 이를 증명하지 않는다.
- EDA의 통화 내 화자 구간 겹침과 660Hz 톤, 기존 파일 누락·JSON 이상은 이번 작업에서 수리하거나 전처리하지 않았다.
- 내부 Validation은 모델 비교용이며 fit에 넣지 않는다. 공식 Validation은 내부 split과 구분해 별도 평가 정책으로 유지한다.

**STEP 4 준비 상태:** 고정 split과 검증 기준은 준비 완료. 이는 데이터의 모든 품질 문제가 해소됐다는 뜻은 아니다. 이번 작업에서는 STEP 4 실험을 실행하지 않았다.
