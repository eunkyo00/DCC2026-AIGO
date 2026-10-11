# Mission 1 데이터 구조 및 EDA

이 디렉터리만 이번 작업에서 추가했다. `common/`, 기존 baseline·노트북·README,
원본 WAV/JSON은 수정하지 않는다. 모델 학습이나 성별 예측은 실행하지 않는다.

## 재실행

Python 3.12 환경에서 저장소 루트 기준:

```bash
python3 -m venv /tmp/ddc-eda
/tmp/ddc-eda/bin/python -m pip install -r mission-1/eda/requirements.txt
/tmp/ddc-eda/bin/python mission-1/eda/run_eda.py \
  --data-root "/Volumes/STORYLiNK/대학부 데이터"
/tmp/ddc-eda/bin/python mission-1/eda/build_report.py
```

`--data-root`는 `Training/`, `Validation/`을 직접 포함하는 폴더다.
`--out-dir`로 결과 위치를 바꿀 수 있지만 원본 데이터 내부 출력은 거부한다.
기본 결과는 `mission-1/eda/eda_outputs/`이며 저장소 기존 `.gitignore`의
`eda_outputs/` 규칙으로 제외된다. README, 코드, REPORT는 공유할 수 있다.
새 실행은 동일 출력 위치의 **생성 결과만** 덮어쓴다.

`--skip-acoustics`는 JSON·헤더·중복 검사만 실행한다.
`--samples-per-stratum 5 --seed 42`가 기본 표본 설정이다.
표본 수를 바꾸면 별도 `--out-dir` 사용을 권장한다.

## 범위와 해석

- JSON을 읽고 전체 WAV 헤더 및 RIFF chunk 경계를 확인한다. 작은 파일 읽기는
  최대 8개 스레드·32개 대기 결과로 제한한다. PCM 헤더는 `wave`로 읽고 미지원 형식만
  `soundfile`로 확인한다. 전체 파형 디코딩은 표본에서 한 통화씩 진행한다.
- macOS `._*`와 `__MACOSX`는 데이터로 세지 않는다.
- split 안에서 유니코드 NFC 정규화 stem으로 매칭한다. 중복 stem을 임의로 선택하지 않는다.
- `all_labels`는 누락 WAV가 있는 JSON도 포함한다. `matched`는 WAV/JSON 각각 하나인 쌍만 포함한다.
- 시간은 실측 스키마에 맞춰 ms에서 초로 변환한다. 비정상 시간은 문제로 기록하고 길이 통계에서 제외한다.
- `gender`는 **통화 신고자 라벨**이다. 상담원 segment에 이 gender를 부여하지 않는다.
- 신고자 수는 통화 안의 역할 `speaker=1` 기준이며, 서로 다른 실제 인물 수를 뜻하지 않는다.
- 총 신고자 시간은 단순 합과 같은 역할 내 겹침을 제거한 구간 합집합을 모두 기록한다.
  신고자·상담원 구간 교집합도 기록한다. 이는 annotation상 겹침이며 실제 동시 발성을 확정하지 않는다.
- WAV 전체 바이트 중복은 크기 → 앞뒤 4KiB → 후보 전체 SHA256 순으로 확인한다.
  같은 WAV 바이트라면 이 필터를 통과하므로 헤더가 읽힌 파일 내에서 정확한 바이트 중복 검사는 전수다.
  재인코딩·편집·일부분 재사용·동일 사람 재등장은 검출하지 못한다.
- 헤더 및 RIFF 검사는 전체 음성 디코딩이나 청취를 대체하지 않는다.
- 오디오 표본은 split × gender × 신고자 합집합 시간 사분위별 같은 수의 통화를
  고정 seed로 뽑는다. 사분위는 정렬 후 순위로 나눠 동률을 처리한다.
  최대 80통화이고 한 통화씩만 로드한다. 표본 요약은 모집단 비율로 가중하지 않는다.
- 전체 통화에서 peak/RMS/무음 대용치/클리핑 대용치를 계산한다. 신고자 RMS도 별도 기록한다.
- 무음 대용치: 20ms 프레임 RMS < 0.001 (-60 dBFS). 끝의 불완전 프레임은 제외한다.
  VAD가 아니므로 조용한 발화/잡음을 잘못 구분할 수 있다.
- 클리핑 대용치: 정규화 진폭 절댓값 >= 0.999. 재스케일된 클리핑이나 모든 왜곡을 검출하지 못한다.
- 음향 특징은 0.5초 이상 신고자 segment 중 무작위 최대 3개, 각 최대 3초의 무작위 crop이다.
  전 통화/전 신고자 발화를 대표한다고 단정할 수 없으며 제외된 초단발화의 F0는 평가하지 않는다.
- pYIN 65–650 Hz, frame 512, hop 160, center=False를 사용한다. voiced ratio는 pYIN 판정이다.
  범위를 벗어난 고음·저음, 울음·비명·잡음·상대 화자 혼입 및 octave error가 있을 수 있다.
- spectral centroid, RMS, MFCC 13개(mean/std, mel 40개)를 동일한 표본 창에서 계산한다.
  기본 음향 특징만 추출하며 학습·스케일러 fit·분류·Accuracy 측정을 하지 않는다.
- 텍스트·주소·실제 ID 원문은 출력하지 않는다. `file_ref`는 stem의 전체 SHA256이다.
  이 해시는 익명성 보증이 아니라 로컬 감사를 위한 가명 식별자이며 모델 입력이 아니다.

## 산출물

| 파일 | 내용 |
|---|---|
| `REPORT.md` | 이번 실행의 핵심 결과, 문제, 모델링 전 결정 사항 |
| `eda_outputs/inventory.json` | 파일 수, 매칭, 중복 stem, 제외한 보조 파일 |
| `eda_outputs/schema.json` | 전체 JSON 키 경로와 타입별 등장 횟수 |
| `eda_outputs/schema_examples_redacted.json` | split별 실제 JSON 3개의 비식별 구조 예시(발화는 앞 3개) |
| `eda_outputs/summary.json` | 전체/매칭 표본별 분포, 분위수, 음향 요약, 문제 수 |
| `eda_outputs/calls.csv` | 통화별 발화 수·시간·구간 겹침 |
| `eda_outputs/segments.csv` | 유효 segment별 구간·길이; 상담원 gender는 not_applicable |
| `eda_outputs/audio_headers.csv` | 전체 WAV 헤더 및 RIFF 구조 검사 |
| `eda_outputs/issues.csv` | 누락·시간 오류·검사 실패 등 가명 참조 목록 |
| `eda_outputs/leakage.json` | 식별자·바이트 중복 검사 범위와 결과 |
| `eda_outputs/sample_selection.json` | 표본 선택·계층 크기 |
| `eda_outputs/sample_acoustics.csv` | 표본 음향 특징 및 실제 선택 구간 |
| `eda_outputs/run_metadata.json` | 실행 경로·옵션·버전·소요 시간·방법 |
| `eda_outputs/*.png` | gender, segment 길이, 통화 발화량, 표본 음향 특징 |

통계 CSV에는 정답 gender 및 감사용 정보가 포함된다. 이후 모델 입력은 별도 allowlist로
WAV와 허용된 startAt/endAt/speaker만 구성해야 한다.

## 검증 및 압축본 교차 확인

```bash
/tmp/ddc-eda/bin/python mission-1/eda/test_eda.py
/tmp/ddc-eda/bin/python mission-1/eda/audit_archive.py "/Users/joeunkyo/Downloads/대학부 데이터.zip"
/tmp/ddc-eda/bin/python mission-1/eda/audit_pitch_peak.py \
  --data-root "/Volumes/STORYLiNK/대학부 데이터" \
  --archive "/Users/joeunkyo/Downloads/대학부 데이터.zip"
```

`test_eda.py`는 임시 합성 WAV/JSON으로 누락·구간 겹침·시간 오류·범위 초과·
split 간 바이트 중복·비식별 출력을 검증한다. 실제 데이터와 별개다.
`audit_archive.py`는 압축 해제 없이 ZIP 목록 및 JSON만 읽어 누락 분포와 ID 교차를
`eda_outputs/archive_crosscheck.json`에 기록한다. WAV payload/CRC 검사 결과는 아니다.
이 보조 확인 결과를 외장 드라이브 파일과 합쳐 표본 수를 늘리지 않는다.
`audit_pitch_peak.py`는 반복된 329Hz F0 추정값을 점검하기 위한 이번 데이터 전용 진단이다.
반복 구간을 가진 남녀 각 1통화와 낮은 F0 남성 대조 1통화의 실제 파형을 검사한다.
ZIP은 가명 참조를 로컬 파일명으로 연결하는 목록으로만 쓰고, 오디오는 외장 드라이브에서 읽는다.
`pitch_peak_audit.json`과 `pitch_peak_diagnostics.png`에 스펙트럼·F0 범위 민감도를 저장한다.
80통화 전체 또는 모집단의 톤 비율을 추정한 결과는 아니다. 톤의 생성 목적은 확인 필요다.
압축본 교차 결과도 보고서에 포함하려면 `audit_archive.py` 다음에 `build_report.py`를 실행한다.
보고서 생성기는 `--results`와 `--out`으로 입력 결과 폴더와 보고서 위치를 지정할 수 있다.
