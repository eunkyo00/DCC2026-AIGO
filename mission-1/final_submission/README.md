# Mission 1 제출 자료 — 신고자 성별 분류 (29,200통화 전체 재실험)

WavLM-Large + Wav2Vec2-base 고정 특징 + 로지스틱 회귀(LR) 확률 결합(50:50). 노트북은 **새 결과 폴더에서 처음부터 끝까지 실제로 실행**했고(Jupyter 커널, 18셀 오류 0, 총 163.7분), 모든 출력은 그 실행 결과입니다.

## 제출 항목 대응
| 제출 항목 | 파일 |
|---|---|
| `inference.py` | `inference.py` |
| 학습 코드 (학습 로그 포함 .ipynb) | `Mission1_full29200_train.ipynb` (학습 로그는 5번 셀 출력, 1,575줄) |
| 학습 완료 모델 가중치 | `model.pt` (1.64GB) |
| 모델 코드 (.py) | `run_full29200.py` (데이터 검사·특징 추출·학습·내보내기·예측 함수), `inference.py`가 이 파일의 예측 함수를 사용 |
| 필요 패키지 | `requirements.txt` (스크립트가 만든 목록), `requirements-lock.txt` (실행 환경의 `pip freeze` 전체) |

### model.pt
- 크기 1,639,528,359 bytes, SHA-256 `abbfae10784130564362c0e2d6153f9458d154d9261abdf300c30407d29a17a7`
- WavLM-Large·Wav2Vec2-base 가중치와 LR이 모두 들어 있어 추론 때 인터넷 연결이나 별도 다운로드가 필요 없습니다.
- 원본은 실행 결과 폴더 `.work/export/model.pt`이며, 결과 ZIP `Mission1_full29200_result.zip` 안에도 같은 파일이 있습니다.

## 실행 명령
```bash
python -m pip install -r requirements.txt     # GPU용 PyTorch는 먼저 pytorch.org 안내대로 설치

# 추론 (명령 1회로 CSV 생성)
python inference.py --audio_dir {wav folder} --label_dir {json folder} --ckpt_path ./model.pt --output ./outputs/mission1.csv

# 학습 (원본 Training 폴더 지정)
python -u run_full29200.py --source {Training folder} --run-dir {결과 폴더}
```
- 출력 CSV 열: `audio file name`, `gender` (값은 F 또는 M). 입력 JSON에서는 신고자(speaker=1) 구간의 startAt/endAt만 사용합니다.

## 실행 환경
| 항목 | 값 |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| CPU / RAM | Intel Core i7-14650HX / 23.7GB |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU 8GB, 드라이버 592.82 |
| Python / PyTorch | 3.11.9 / 2.8.0+cu128 (CUDA 12.8) |
| 주요 패키지 | transformers 4.46.3, tokenizers 0.20.3, huggingface-hub 0.31.4, numpy 2.0.2, scipy 1.15.3, scikit-learn 1.6.1, soundfile 0.13.1 |
| 정밀도 | BF16 (자동) |
| 사전학습 모델 | `microsoft/wavlm-large` (rev c1423ed9…), `facebook/wav2vec2-base` (rev 0b5b8e86…) |

## 결과
| 항목 | 값 |
|---|---|
| 데이터 전수 검사 | WAV 29,200 / JSON 29,200 / 정상 29,200, 오류 0 |
| 분할 | 학습 23,360 / 내부 검증 5,840 (통화 단위, 성별 층화, seed 42) |
| 결합 비율 선택 | 학습 3-fold OOF 정답 수 최대 → WavLM 50 : Wav2Vec2 50 (OOF 22,937, 75:25는 22,933) |
| 내부 검증 5,840 | 앙상블 98.116% (5,730), WavLM 단독 98.014%, Wav2Vec2 단독 96.935% |
| **공식 Validation 3,640** | **98.242%** (정답 3,576 / 오답 64), 여 98.47% (1,929/1,959), 남 97.98% (1,647/1,681) |
| 혼동행렬 (행=실제 F,M / 열=예측) | `[[1929, 30], [34, 1647]]` |
| 공식 Validation 추론 시간 | 약 17분 (3,640통화, 노트북 7번 셀 실행 구간, GPU) |

- 같은 코드를 앞서 터미널에서 한 번 더 실행했고 두 실행의 결과가 같았습니다(분할·OOF·내부 검증·공식 Validation 모두 일치).
- 공식 Validation은 학습·결합 비율 선택에 쓰지 않았고, 학습이 끝난 뒤 채점에만 썼습니다. 비공개 Test 성능은 아닙니다.
- 이전 실험(27,985통화, 내부 5,597통화)과는 데이터 수·분할·패딩 방식이 달라 점수를 직접 비교하지 않습니다.

## 알아둘 점
- 학습 대상은 두 LR 분류기뿐이라(사전학습 본체 동결) epoch·loss 로그가 없고, 데이터 검사·특징 저장 진행·OOF·내부 검증 결과가 학습 기록입니다.
- 결합 비율 선택 규칙이 이전 원고(모든 fold 하락 없음 + OOF +0.1%p)와 다릅니다. 이 스크립트는 OOF 정답 수 최대로 고릅니다. 이전 규칙을 적용하면 75:25·50:50 모두 기준 미달이라 100:0이 남습니다.
- 분할은 통화 단위로 겹치지 않지만, 같은 화자가 양쪽에 있을 가능성까지 배제한 것은 아닙니다.
- 노트북의 `!python ...` 셀은 커널과 같은 Python(`sys.executable`)으로 실행했습니다. 로그에 나오는 경고(UserWarning)는 라이브러리 버전 안내이며 결과에 영향이 없습니다.

## 폴더 내용
`run_records/`: `metrics.json`, `selection.json`, `dataset_check.json`, `identity.json`, `export_verification.json`, `REPORT.md`, `split_assignments.csv` (실행 산출 기록) / `outputs/mission1.csv`: 제출 명령으로 만든 공식 Validation 예측 결과
