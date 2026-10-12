# ResNet18 + ResNet34 + WavLM4: 최고 Validation 91.972%

## 결과의 출처와 범위

최초 작성은 사용자가 공유한 Colab 출력의 전사이며, 2026-10-12에 저장된 발화별 확률을 ID로 대조해 최종 혼동행렬까지 재계산했다. 공식 Validation 111,947발화를 사용했고 모든 최종 임계값은 0.5다. 여러 실험과 비율 선택에 같은 Validation을 사용했으므로 독립 Test 결과가 아니다. 실제 세 가중치를 통합 재추론한 결과는 아니다. [최신 확인 기록](verification_20261012.md)에 파라미터·환경·미측정 항목을 구분했다.

- Drive 기록: `DCC_mission02/mission02_ensemble_ratio_comparison/ratios.json`
- 확장 비교: `DCC_mission02/mission02_ensemble_ratio_comparison_extended/ratios.json`
- 원래 모델 비교: `DCC_mission02/mission02_resnet_wavlm4_comparison/comparison.json`
- [집계 수치 JSON](reported_metrics.json)은 원본 JSON 복사본이 아니라 공유 로그 전사본이다. 단위는 percent다.

## 선택한 구성

```python
p_caller = 0.30 * p_resnet18 + 0.30 * p_resnet34 + 0.40 * p_wavlm4
speaker = int(p_caller >= 0.5)
```

| 지표 | 결과 |
|---|---:|
| Accuracy | 91.972% |
| 대원 Recall | 87.990% |
| 신고자 Recall | 95.663% |

분모는 전체 111,947, 실제 대원 53,845, 실제 신고자 58,102다. 저장된 확률로 재계산한 정답은 102,960개이며 혼동행렬은 `[[47378,6467],[2520,55582]]`다(행=정답, 열=예측, 순서=대원/신고자). 반올림 비율에서 개수를 역산한 값이 아니다.

세 모델을 1/3씩 사용한 결과가 아니다. 30:30:40은 Validation 비교 이후 선택했으며, 사전 확정한 최종 비율이라고 주장하지 않는다.

## 실험 흐름과 해석

1. ResNet18+34 결합은 91.419%였으나 대원 Recall이 낮았다.
2. WavLM4 단독은 90.426%였지만 대원 Recall은 89.59%로 더 높았다. 단순 교체보다 보완 가능성을 검토했다.
3. 동일 ID로 대조한 결과 ResNet18+34 오답·WavLM 정답 3,573개, 반대 4,685개, 둘 다 오답 6,033개였다. 확률 결합이 모든 보완 사례를 맞힌다는 의미는 아니다.
4. 25% 결합은 91.819%로 높아졌다. 주변 비율과 추가 범위를 비교한 결과 40%가 시험 범위 내 최고였다.
5. 45%는 대원 Recall이 더 높지만 신고자 Recall이 낮아 정답이 20개 줄었다. 더 높은 WavLM 비중이 항상 개선으로 이어지지 않았다.

## 비교한 전체 비율

나머지 비중은 ResNet18과 ResNet34에 동일하게 배분했다. 예: WavLM 40%일 때 각각 30%.

| WavLM | ResNet18 | ResNet34 | Accuracy | 대원 Recall | 신고자 Recall |
|---|---|---|---:|---:|---:|
| 15% | 42.5% | 42.5% | 91.675% | 87.297% | 95.732% |
| 20% | 40% | 40% | 91.771% | 87.464% | 95.763% |
| 25% | 37.5% | 37.5% | 91.819% | 87.540% | 95.785% |
| 30% | 35% | 35% | 91.883% | 87.653% | 95.802% |
| 35% | 32.5% | 32.5% | 91.942% | 87.815% | 95.766% |
| **40%** | **30%** | **30%** | **91.972%** | **87.990%** | **95.663%** |
| 45% | 27.5% | 27.5% | 91.954% | 88.179% | 95.453% |
| 50% | 25% | 25% | 91.659% | 88.218% | 94.849% |
| 55% | 22.5% | 22.5% | 91.358% | 88.181% | 94.301% |
| 60% | 20% | 20% | 91.181% | 88.160% | 93.980% |

40%와 45%의 차이는 작으며 새로운 데이터에서의 우열을 보장하지 않는다. 비율을 Validation으로 선택했으므로 이 수치에 모델 선택 편향이 포함될 수 있다. 94%에는 도달하지 않았다.

## 사용한 모델 식별

| 모델 | SHA-256 |
|---|---|
| ResNet18 small-stem | `f72bfb9151a2a6bfd23de104d4c94676997c3fd4fe3853dd6c8e4ca140ee64ad` |
| ResNet34 small-stem | `d92093905a386f06dabf595089c52331e466a18d5827e0f9fdde31b0926a49ed` |
| WavLM4, 팀원 평가 보고값 | `706567c77bf1cc7cf196e2cd55760cb4f2f4045fcbf61687731b23f0b186c05c` |

WavLM은 10만 발화/내부 dev 4천/6 epoch 중 내부 dev 최고인 epoch 5, step 30,000 체크포인트다. 새 30만 발화 실험은 이 결과에 포함되지 않는다.

WavLM 원본 평가 보고서: Accuracy 0.9042582650718644, 혼동행렬 `[[48241,5604],[5114,52988]]`, threshold 0.5. total/active parameters 각각 38,170,066, RTX 5060 Laptop GPU, batch 16, 약 408.609초. 시간 범위는 파형 shard 읽기·리샘플·forward로 원본 WAV/JSON I/O를 제외하며 **앙상블 시간·파라미터 수가 아니다**.

## 캐시를 이용한 비율 비교 실행

`compare_ensemble_ratios.py`는 numpy만 사용하는 CPU 비교 코드다. private Drive 폴더에 기존 `mission02_ensemble_eval.zip`과 `mission02_resnet_wavlm4_comparison/{resnet18_probs.npz,resnet34_probs.npz}`가 있어야 한다. 실제 모델 추론 코드나 공개 데이터 번들이 아니다.

```sh
python compare_ensemble_ratios.py --drive-dir /content/drive/MyDrive/DCC_mission02
python compare_ensemble_ratios.py --drive-dir /content/drive/MyDrive/DCC_mission02 --extended
```

첫 명령은 15~40%, 두 번째는 40~60%를 평가한다. 각 실행의 JSON 경로는 위와 같으며 같은 명령을 다시 실행하면 해당 보고서만 갱신한다. ID·정답·캐시 서명과 WavLM NPZ 해시를 확인하고 threshold를 탐색하지 않는다. 기존 모델을 자동 교체하지 않는다.

## 제출 전 남은 확인

- WavLM CSV는 소수점 6자리다. 실제 가중치에서 생성하는 전체 정밀도 확률과 통합 추론을 다시 평가해야 한다.
- ResNet의 과거 epoch 선택에 튜닝 점수가 사용됐다. 현재 최종 threshold 0.5가 이 선택 이력을 바꾸지는 않으며 주최 측 확인이 필요하다.
- 실제 WAV/JSON 입력, 두 전처리, 세 로컬 가중치, 단일 inference.py 실행, 결과 CSV 규격과 전체 추론시간을 확인해야 한다.
- 데이터/가중치/개별 발화 확률은 이 공개 폴더에 포함하지 않았다. 최신 최고 모델을 실행 검증한 최종 제출 번들은 아직 아니다.

[Mission 2 개요](../../README.md) · [발표 원고](../../mission02_ppt.md)
