# Mission 1 full 29,200 fresh experiment

- Source: 29,200 calls; Train 23,360; Internal Validation 5,840.
- Internal Validation Accuracy: 98.116438%.
- WavLM weight: 0.5; precision: bf16.
- Final submission LR refit: all 29,200 calls AFTER metrics/weight were locked. The reported accuracy belongs to the evaluation model, not an independent test of the final refit model.
- Frozen pretrained encoders; LR heads are freshly fitted; no old experiment features.
- Protocol: full29200-fixed-buckets-v1. Fixed bucket padding differs from previous experiment.
- Official Validation never used. Split is call-disjoint, not guaranteed person-disjoint.
- Verification: CPU tests, exported LR parity and checkpoint loading. Actual-device inference roundtrip: see export_verification.json.
- Run: python run_full29200.py --predict --audio-dir WAV --label-dir JSON --checkpoint model.pt --predictions predictions.csv
