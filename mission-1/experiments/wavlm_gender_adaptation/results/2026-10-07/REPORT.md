# WavLM · Internal Validation 검증 결과

Previously observed Internal Validation; not independent test; official Validation unused

| 모델 | 정답 | 오답 | Overall | Male | Female |
|---|---:|---:|---:|---:|---:|
| frozen_lr | 5490 | 107 | 98.088262% | 97.748092% | 98.387639% |

frozen_lr confusion (true/pred M,F): `[[2561, 59], [48, 2929]]`.

- mfcc: corrected 207, regressed 38, both_wrong 69, both_correct 5283
- wav2vec2: corrected 82, regressed 35, both_wrong 72, both_correct 5408
- ecapa: corrected 121, regressed 50, both_wrong 57, both_correct 5369

99%는 5,542 정답 이상. 결과에 맞춘 재튜닝은 하지 않는다.
과거 실패 기록 0건, 미해결 실패 0건. 실제 L4 시간은 반환 benchmark.json 및 실행 기록 참조.
