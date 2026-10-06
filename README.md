# 전처리 파이프라인 (A: 인프라 / EDA / 공통 전처리)

119 응급 신고 음성 데이터 분류 대회 — Mission 1(성별) / Mission 2(화자) / Mission 3(증상 인식) 공용
데이터 스캔·EDA·전처리 파이프라인 문서. 자세한 대회 규정은 `0.중요/문제_분석.md`,
역할 분담은 `0.중요/권장역할분담.md` 참고.

## 0. 코드 위치

이 문서에서 다루는 코드는 전부 레포 루트의 `common/` 패키지 밑에 있다 (미션별 코드는
`mission-1/`, `mission-2/`, `mission-3/`에 각자 작성). Python import/실행은 항상 레포
루트에서 `common.` 접두사를 붙여서 한다 (예: `python -m common.eda.eda_all`,
`from common.preprocessing.preprocessing_utils import ...`).

```
common/
├── config.py
├── utils/            (io, inspect_schema, filter_seoul, audio_check)
├── preprocessing/    (preprocessing_utils, build_manifests)
└── eda/              (eda_all)
```

## 1. 데이터 위치 & 구조

로컬 데이터 루트는 `common/config.py`의 `DATA_ROOT`로 관리한다 (환경변수
`EMERGENCY_CALL_DATA_ROOT`로 팀원마다 다른 경로 오버라이드 가능).

실제 확인된 폴더 구조 (`common/utils/inspect_schema.py`, `common/utils/io.py`로 확인):

```
{DATA_ROOT}/
└── 대학부 데이터/
    ├── Training/
    │   ├── 1.원천데이터/TS_서울_구급/*.wav      (29,200개)
    │   └── 2.라벨링데이터/TL_서울_구급/*.json    (29,200개)
    └── Validation/
        ├── 1.원천데이터/VS_서울_구급/*.wav      (3,640개)
        └── 2.라벨링데이터/VL_서울_구급/*.json    (3,640개)
```

wav/json은 **파일명(stem)이 완전히 동일**해서 그 기준으로 1:1 매칭된다 (json 내부의
`audioPath` 필드는 원본 배포 서버 경로라 로컬 매칭에 쓰지 않음). 매칭률은 Training/Validation
둘 다 100% (누락 0건, `common/utils/io.py` 확인).

### 라벨 json 스키마 (실측)

```json
{
  "gender": "M",                     // Mission 1 라벨
  "address": "서울특별시 영등포구 대림동",  // 서울 필터링 기준
  "symptom": ["호흡곤란"],             // Mission 3 라벨 (9개 클래스 외 값도 섞여 있음)
  "startAt": 0, "endAt": 48240,       // 통화 전체 길이 (ms)
  "utterances": [
    {"id": "...", "startAt": 990, "endAt": 2055, "text": "네 119입니다.", "speaker": 0}
    // speaker 0 = 119대원(상담원), speaker 1 = 신고자  (Mission 2 라벨)
  ]
}
```

`startAt`/`endAt`은 전부 **밀리초(ms)** 단위.

## 2. 실행 순서

```bash
# 0) 최초 1회: json 스키마 실측 (키/타입/고유값 분포 확인)
python -m common.utils.inspect_schema

# 1) wav/json 매칭 상태 확인
python -m common.utils.io

# 2) 서울 지역 필터링 확인 (학습은 서울만 사용해야 함)
python -m common.utils.filter_seoul

# 3) 오디오 샘플레이트/채널/길이 일관성 확인
python -m common.utils.audio_check

# 4) 라벨 분포 EDA (csv + png를 eda_outputs/에 저장)
python -m common.eda.eda_all

# 5) 미션별 전처리 매니페스트 생성 (manifests/에 저장)
python -m common.preprocessing.build_manifests
```

모든 명령은 **레포 루트에서** 실행해야 한다 (`common`을 모듈로 찾아야 하므로).

모든 스크립트는 `--data-root`로 데이터 경로를, `--split Training/Validation/both`로 대상
split을 지정할 수 있다 (기본값은 `both`).

## 3. EDA 결과 요약 (전수 조사, 샘플링 없음)

| 항목 | Training (29,200) | Validation (3,640) |
|---|---|---|
| 오디오 샘플레이트 | 8,000Hz (100%) | 8,000Hz (100%) |
| 채널 | mono (100%) | mono (100%) |
| 길이 | 30초 ~ 180초 (평균 72초) | 30초 ~ 180초 (평균 73초) |
| 손상 파일 | 0건 | 0건 |
| 서울 비율 | 100% | 100% |
| gender | F 15,542 / M 13,658 | F 1,959 / M 1,681 |
| speaker(발화 단위) | 상담원 427,161 / 신고자 462,197 | 상담원 53,845 / 신고자 58,102 |

symptom(9개 클래스, Training 기준, 통화당 0~9개 멀티라벨):

| 클래스 | 복통 | 어지러움 | 전신쇠약 | 고열 | 구토 | 호흡곤란 | 열상 | 오심 | 두통 |
|---|---|---|---|---|---|---|---|---|---|
| 등장 횟수 | 6,771 | 6,334 | 5,310 | 5,186 | 4,463 | 3,927 | 3,465 | 3,341 | 2,905 |

최다/최소 클래스 비율 약 2.3배 — 심한 불균형은 아니지만 Mission 3에서 class weighting이나
threshold 조정을 고려할 만한 수준. 오디오가 이미 8kHz/mono로 통일돼 있어 **리샘플링이나
채널 변환이 필요 없다는 것**이 가장 중요한 발견.

전체 plot(gender/urgencyLevel/disasterMedium/sentiment/symptom 분포)은 `eda_outputs/*.png`,
원본 집계는 `eda_outputs/labels_{split}.csv` 참고.

## 4. 미션별 전처리 로직

오디오 원본은 미리 잘라서 별도 파일로 저장하지 않는다. **wav는 그대로 두고, "어디를 어떻게 잘라 쓸지"에 대한 인덱스(매니페스트)만 미리 만들어두고, 실제 crop/pad/feature 추출은 각 미션의 dataloader에서 그때그때 수행**하는 구조다. 발화 단위로 실제 오디오 파일을 다 잘라두면 통화당 수십 개씩 파일이 늘어나 디스크 용량이 감당되지 않기 때문.

공통 처리 흐름은 **자르기(crop) → 길이 맞추기(pad/trim) → 숫자로 변환(MFCC/Mel 또는 텍스트)**
3단계이고, 미션마다 "뭘 자르고 뭘 라벨로 쓰는지"만 다르다.

### Mission 1 — 성별 분류

- **입력 근거**: 라벨의 `startAt`/`endAt`/`speaker`만 학습 시 사용 가능 (대회 규정)
- **현재 실험**: 신고자(`speaker == 1`) 구간을 사용하고 모델별 특징을 통화 단위로 집계한다.
  F0 + Acoustic + LR, MFCC(13계수) + RBF SVM, Frozen Wav2Vec2 + LR, 고정 Gender ECAPA 평가를 완료했다.
- **라벨**: 통화 json의 `gender`
- **결과**: 동일 Internal Validation 5,597통화에서 각각 91.209577%, 95.068787%,
  97.248526%, 96.819725%. 현재 최고는 Frozen Wav2Vec2 + LR이다.
  전체 L4 Wav2Vec2 embedding 추출 및 감사는 실제 11.19시간 걸렸다.
- **초기 공통 유틸리티 예제**: 5초 pad/trim, MFCC 40계수와 100건 속도 측정은 초기
  전처리 예제이며 현재 Mission 1 실험 설정·전체 소요시간을 나타내지 않는다.
- **진행 상태**: Train CV·OOF 정량 오류 분석, ECAPA/WavLM 200통화 비교, ECAPA Validation 검증까지 완료.
  본체 fine-tuning·표본 청취·새 입력용 통합 추론은 완료 결과에 포함하지 않는다. [Mission 1 README](mission-1/README.md)에서 설정·결과·폴더를 확인한다.

### Mission 2 — 화자 분류 (신고자 vs 119대원)

- **입력 근거**: 발화 하나(`startAt`/`endAt`)만 사용, **텍스트·발화 순서 일체 사용 불가**
  (순수 음향 정보만으로 판단해야 하는 대회 규정 중 가장 까다로운 미션)
- **처리**: 발화 1개 = 샘플 1개. 1.5초로 고정 → Mel-spectrogram(64 bands, log-scale) 추출
- **라벨**: `speaker` (0=119대원, 1=신고자)
- **관련 함수**: `crop_segment` → `pad_or_trim_to_length` → `extract_melspectrogram`
- **주의**: 매니페스트(`mission2_speaker_*.csv`)에도 `text` 컬럼을 아예 넣지 않았다 —
  실수로라도 텍스트/순서 정보가 모델 입력에 흘러들어가는 것을 원천 차단하기 위함
- **실측 성능**: 통화 1건당 wav 1번만 로드해서 그 안의 발화들을 재사용 → 발화당 평균 1.6ms
  (통화당 평균 41ms) → 전체 100만 건 이상도 30분 이내 처리 가능

### Mission 3 — 증상 인식 (9개 클래스 멀티라벨)

- **입력 근거**: 전사 텍스트(`utterances[].text`)만 사용 가능, 오디오 불필요
- **처리**: 통화의 모든 발화 텍스트를 시간순 리스트로 추출. `symptom` 리스트에서 9개
  타겟 클래스(`config.SYMPTOM_CLASSES`)에 속하지 않는 라벨만 제거
  (예: `['두통','복통','찰과상']` → `['두통','복통']`, **샘플 전체가 아니라 라벨만 제외**)
- **라벨**: 필터링된 `symptom` 리스트 (0~9개)
- **관련 함수**: `get_utterances`, `extract_utterance_text`, `filter_symptom_labels` /
  `get_symptom_labels`
- **실측**: Training 29,200건 전부 필터링 후에도 symptom이 1개 이상 남음 (9개 클래스 밖으로만
  구성된 통화 0건)

### 서울 필터링

`common/utils/filter_seoul.py`가 `address` 필드(알려진 키 우선, 실패 시 "서울" 키워드를 포함하는
문자열을 재귀 탐색하는 fallback)로 서울 여부를 판정한다. `build_manifests.py`는 매니페스트
생성 시 이 필터를 항상 적용한다 (현재 배포 데이터는 Training/Validation 모두 100% 서울이라
실질적으로 걸러지는 데이터는 없지만, 향후 다른 지역 데이터가 섞여도 안전하도록 방어 코드로 유지).

## 5. 매니페스트 파일 스펙 (`manifests/`)

로컬 절대경로를 담고 있고 언제든 재생성 가능해서 `.gitignore`에 포함되어 있다. 각자
`python -m common.preprocessing.build_manifests --data-root <자기 경로>`로 재생성해서 쓰면 된다.

| 파일 | 컬럼 | 설명 |
|---|---|---|
| `mission1_gender_{split}.csv` | stem, wav_path, gender, reporter_segments_ms, num_segments | `reporter_segments_ms`는 `[[start,end], ...]` 형태의 JSON 문자열 |
| `mission2_speaker_{split}.csv` | stem, wav_path, utterance_id, startAt, endAt, speaker | 발화 1행 = 샘플 1개 |
| `mission3_symptom_{split}.csv` | stem, json_path, utterance_texts, symptom_labels, num_symptoms | `utterance_texts`는 JSON 문자열 리스트, `symptom_labels`는 `;`로 join |

| 매니페스트 | Training rows | Validation rows |
|---|---|---|
| mission1_gender | 29,200 | 3,640 |
| mission2_speaker | 889,358 | 111,947 |
| mission3_symptom | 29,200 | 3,640 |

## 6. preprocessing_utils.py 함수 요약

| 함수 | 역할 |
|---|---|
| `load_audio(path, target_sr=None, mono=True)` | wav 로드 (필요 시 리샘플링) |
| `crop_segment(waveform, sr, start_ms, end_ms)` | startAt/endAt(ms) 구간 자르기 |
| `pad_to_length` / `trim_to_length` / `pad_or_trim_to_length` | 고정 길이로 맞추기 |
| `sliding_window(waveform, sr, window_ms, stride_ms)` | 긴 오디오를 겹치는 윈도우들로 분할 |
| `extract_mfcc(waveform, sr, n_mfcc=40)` | MFCC 추출 |
| `extract_melspectrogram(waveform, sr, n_mels=128, to_db=True)` | Mel-spectrogram 추출 (이미지 모델용) |
| `get_utterances(label)` | 라벨에서 발화 리스트 추출 (키 이름 방어적 탐색) |
| `get_speaker_id` / `get_speaker_role` | speaker id ↔ 0/1, "dispatcher"/"reporter" 변환 |
| `guess_speaker_role_from_text(text)` | 키워드 기반 화자 추정 — **QA/EDA 전용, Mission 2 모델 입력 금지** |
| `extract_utterance_text(label, speaker_id=None)` | 발화 텍스트 리스트 (화자 필터링 가능) |
| `filter_symptom_labels` / `get_symptom_labels` | symptom을 9개 타겟 클래스로 필터링 |

모든 함수는 `common/preprocessing/preprocessing_utils.py`에 docstring과 함께 정의되어 있으니
미션별 코드에서 필요한 것만 `from common.preprocessing.preprocessing_utils import ...`로
가져다 쓰면 된다.

## 7. 재현 방법 (다른 팀원 환경)

```bash
pip install -r requirements.txt

# 자기 로컬 데이터 경로가 다르면 환경변수로 지정
export EMERGENCY_CALL_DATA_ROOT="/path/to/emergency_call_dataset"   # mac/linux
# 또는
set EMERGENCY_CALL_DATA_ROOT=D:\data\emergency_call_dataset          # windows

python -m common.eda.eda_all
python -m common.preprocessing.build_manifests
```

## 8. 지켜야 할 제약 (재확인)

- Training은 서울 지역 데이터만 사용 (자동 필터링됨)
- Validation은 하이퍼파라미터 튜닝/early stopping/모델 선택에만 사용, **backprop 편입 금지**
- Mission 2는 텍스트·발화 순서 정보를 모델 입력으로 절대 사용 금지 (순수 음향만)
- Mission 3은 전사 텍스트 외 라벨링데이터 사용 불가
- symptom은 9개 클래스 외 라벨 제외 시 **해당 라벨만 제거**, 샘플 전체를 버리면 안 됨
