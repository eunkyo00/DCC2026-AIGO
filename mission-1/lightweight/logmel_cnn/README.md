# Mission 1 경량 후보 — 8 kHz log-mel + 소형 CNN

이 디렉터리는 `mission1/eunkyo` 브랜치의 독립 실험이다. 현재 실행 중인
`mission-1/ssl/wav2vec2_frozen`의 코드, 설정, cache, 결과, Colab 런타임은 읽거나
수정하거나 실행하지 않는다.

## 고정 실험 정의

기존 MFCC + RBF SVM은 고정 call split의 Train 22,388개와 Internal Validation
5,597개에서 95.068787%를 기록했다. 8 kHz mono 원본과 `speaker=1` caller 구간
442,639개를 그대로 사용한다. call을 임의로 줄이거나 caller 구간을 버리거나 무작위
crop/VAD/길이 상한을 적용하지 않는다. 구간 경계는 보존하고, 마지막 FFT frame만
zero padding한다.

입력은 50–3,800 Hz, 40 mel band, FFT 256, 200-sample Hamming window, 80-sample
hop의 natural-log mel power다. 발화 하나가 256 frame을 넘으면 발화 내부에서만
여러 chunk로 나누며, chunk는 계산 단위일 뿐 음성을 절삭하지 않는다. 모델은
`Conv2d(1,8) → Conv2d(8,16) → Conv2d(16,32)` (각 3×3, stride 2, ReLU)와
`Linear(32,2)`다. 학습 파라미터는 5,954개, FP32 weight는 23,816 bytes,
chunk당 forward MAC은 1,658,944다.

Train 내부에서 성별별 SHA256 순위로 10%를 inner-dev로 고정한다. 정규화 통계는
inner-fit의 유효 frame에서만 계산하고, AdamW(lr 0.001, weight decay 0.0001),
최대 30 epoch, patience 5로 inner-dev loss에서 epoch를 선택한다. 선택한 epoch만큼
새 모델을 Train 전체에 재학습한 뒤, 최종 Internal Validation을 한 번만 읽는다.
최종 Validation은 모델 선택·early stopping·정규화에 사용하지 않는다.

## 실행

먼저 전수 비용을 계산한다.

```bash
python mission-1/lightweight/logmel_cnn/run_logmel_cnn.py audit \
  --out mission-1/lightweight/logmel_cnn/artifacts/audit.json
```

그 다음 모든 caller segment를 읽어 call 256개 단위의 비압축 NPZ shard로 저장한다.
완료된 shard는 임시 파일을 fsync한 뒤 rename하고, `--permanent-store`를 지정하면
별도 영구 경로에 복사한 뒤 SHA256을 대조한다. `index.json`에 등록된 shard와 hash가
복구 기준이다. 작은 파일 수만 개를 만들지 않으며, 각 shard는 mel 배열, 유효 frame
수와 call membership을 보존한다.
전수 audit의 유효 frame은 약 14.87GB이지만, 현재 NPZ에는 마지막 chunk의 padding도
포함되므로 실제 mel 배열은 약 25.01GB다. Colab에서는 `/content`에 로컬 cache를
두고 별도 Drive 폴더를 영구 복구본으로 사용한다.

```bash
python mission-1/lightweight/logmel_cnn/run_logmel_cnn.py extract \
  --data-root '/content/drive/MyDrive/실제_데이터_루트' \
  --store /content/logmel_store \
  --permanent-store '/content/drive/MyDrive/logmel_cnn_exp/features'
```

마지막으로 별도 환경에서 학습한다. 기본 device는 CPU이며, 검증된 별도 GPU 환경에서
`--device cuda`를 사용할 수 있다.

```bash
python mission-1/lightweight/logmel_cnn/run_logmel_cnn.py train \
  --store /content/logmel_store \
  --out '/content/drive/MyDrive/logmel_cnn_exp/run' \
  --device cuda \
  --mfcc-predictions mission-1/baseline/mfcc_svm/results/val_predictions.csv
```

학습은 매 epoch 작은 진행 checkpoint를 Drive에 원자적으로 저장한다. 런타임이 끊기면
동일 `--store`와 `--out`으로 재실행해 완료된 epoch부터 이어간다. 최종 모델은
Validation 평가 전에 저장하며, 평가가 중단되면 같은 모델만 재평가한다.

결과 JSON은 WAV 읽기/디코드, log-mel 계산, shard 쓰기·영구 복사, inner 선택, 전체
Train 재학습, Validation 추론 시간을 분리한다. audit의 전수 frame/chunk 수와 실제
전체 추출 시간을 기록하며, 짧은 warm-cache 표본으로 전체 시간을 외삽하지 않는다.
오류 비교는 call_id와 gender를 일대일로 맞춰 정확도, 혼동행렬, 두 모델 모두 정답/
각 모델만 정답/둘 다 오답, 오류 Jaccard와 정확도 차이를 기록한다. 결과가 MFCC보다
좋거나 모든 샘플에서 손실이 없다고 가정하지 않는다.

## 검토한 위험

call 단위 split은 같은 call의 발화가 Train과 Validation으로 갈라지는 누수를 막지만,
person ID나 phone hash가 없어 동일 신고자의 다른 통화가 양쪽에 있는지는 확인할 수
없다. EDA에서 caller/operator annotation overlap이 약 96%였으므로 `speaker=1`이
순수 caller 음성이라는 보장도 없다. 기존 MFCC 결과를 본 fixed split이므로 blind test나
독립 화자 일반화 결과로 해석하지 않는다. Frozen Wav2Vec2 결과와의 시간·메모리 비교도
같은 하드웨어·측정 범위가 아니면 우열을 주장하지 않는다.
