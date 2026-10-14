# Mission 1: fresh full-29,200 Colab experiment

This pipeline verifies and uses all 29,200 Training calls. It does not import the old 27,985-call split, features, or LR weights. Public pretrained WavLM-Large and Wav2Vec2-base are frozen, as in the existing final model family; feature extraction and classifier fitting run afresh. This is not encoder fine-tuning from random initialization.

## Colab use

Download `Mission1_full29200_Colab_Package.zip` from `output/mission1_full29200_colab/` and unzip it on your computer. Upload `Mission1_full29200.ipynb` into Colab, choose a CUDA GPU, and run the six short form cells in order. In cell 1, upload the accompanying `mission1_runtime.zip`. Long implementation code lives in that ZIP; paths are chosen through widgets, not pasted into cells.

Mount Drive and select the Training ZIP/folder. Separate WAV and JSON ZIPs can both be added. Nested ZIPs (up to three layers) are supported; official Validation/VS_/VL_ and macOS resource files are ignored. Shared items must be accessible through a shortcut in My Drive. Never select overlapping sources. Data is copied/extracted to VM disk so GPU extraction avoids tens of thousands of small Drive reads. The original source is never repaired or changed.

The complete dataset must contain exactly 29,200 unique WAV/JSON pairs, valid F/M labels, Seoul addresses, valid caller boundaries, and 8kHz/mono/PCM16 headers. Missing, duplicate, or invalid files stop the run rather than silently reducing the dataset. Full file hashes are computed; waveform decode/finite values are verified when each call is extracted. This code cannot recreate missing original audio: a complete supplied dataset is required first. Diagnostics are saved to `dataset_check.json` or `last_failure.json`.

## Training and metrics

1. Seed-42 stratified call split: 23,360 Training and 5,840 internal Validation calls. Official Validation is not used.
2. Fresh full feature extraction for both models. No call/segment subsampling or duration cap: all caller segments are used, and long sequences are split into bounded windows.
3. Three-fold OOF on the 23,360 Training calls selects an ensemble weight among 1.0, 0.75, 0.5, 0.25, and 0.0. LR C is fixed at 0.1 (WavLM) and 1.0 (Wav2Vec2). Standardization is fitted inside each fold.
4. Fit the evaluation classifiers on 23,360 calls; save accuracy and predictions for the separate 5,840 calls.
5. Lock evaluation results and selected weight, then refit submission LR classifiers on **all 29,200 Training calls**. `model.pt` contains this full-data refit. The recorded internal accuracy belongs to the earlier evaluation model; it is not an independent evaluation of the full-data refit model.

Call split does not prove person-disjointness because a stable person ID is unavailable. The old 98.213329% is not directly comparable: both dataset/split and the following numerical protocol have changed. New accuracy is measured at runtime, not promised in advance.

## Efficiency and numerical protocol

- Local WAV reads, four CPU preprocessing workers, bounded 64-call blocks, fixed length buckets, multiple windows per GPU forward, inference mode, and no collection of all transformer hidden layers.
- BF16 autocast on compatible GPUs (e.g. L4/A100); FP32 otherwise (e.g. T4). GPU model weights remain FP32. TF32 is allowed. Precision and versions are saved as run identity.
- WavLM: concatenate resampled caller segments, evenly split into <=12s windows, individually normalize, fixed-bucket right padding with attention mask, valid-frame mean, then real-sample-weighted window mean.
- Wav2Vec2: individually normalize caller segments, split into <=15s windows, fixed-bucket right padding **without attention mask** as required for group-norm Wav2Vec2-base; pool valid frames, frame-weight within segment, equal-weight across segments.
- Fixed bucket sizes are part of the new protocol. Wav2Vec2 padding can affect model features; this is intentionally a new experiment, not an exact replay of the old single-segment implementation. Training and exported inference share the same `core.py`.
- GPU OOM halves the batch size and retries without changing fixed padding buckets. A single-window OOM stops with an actionable error.

## Saving, ETA, and recovery

Drive outputs are under `MyDrive/AIGO_Mission1_full29200/<run name>`. Every 64 completed calls, both models' embeddings and a checksum sidecar are saved before the completed counter advances. Interruption may require recalculating the current unsaved block. OOF folds, selection, classifiers, metrics, predictions, progress, events, and failures are saved separately. Checkpoints must match the data hashes, split, source code, software versions, and precision; mismatches stop instead of mixing runs.

Use a new name for a new experiment. After disconnection, run cells 1–4 again, select the existing run in the restart menu, and run cell 5. Local dataset staging and inspection may run again on a new VM, but completed new-run feature blocks are reused. Old 27,985-run caches cannot be imported. Run only one runtime per output folder. If changing GPU changes FP32/BF16, use a new run or the original compatible GPU.

ETA starts after the first 64-call block for **both** models and is updated by measured extraction time per caller-audio second, including checkpoint write time. The first estimate is noisy; 512+ calls are more informative. It excludes data copying/inspection, initial model download/loading, CPU LR fitting, and final ZIP export. No reliable whole-run wall time can be promised before GPU/data I/O are measured. Earlier single-utterance Wav2Vec2 timing is not a benchmark for this new batched implementation. Plan for hours, not minutes; this code has no measured full-L4 timing yet.

The final ZIP is roughly the FP32 encoder weights (~1.6GB) plus small reports/classifiers. It is assembled on local disk, copied to Drive, hash-verified, and offered with a download button. Browsers may block large downloads; the Drive copy is always available after successful completion. Intermediate record ZIP downloads exclude large feature shards; the shards remain in Drive for recovery. No original audio/transcripts are placed into the result ZIP. Colab VM disk needs space for source ZIP, extracted audio, model cache, and export (at least several GB beyond the data size).

## Final ZIP and inference

Contains `model.pt`, `inference.py`, `model.py`, `core.py`, `requirements.txt`, `train.ipynb`, `training_source.zip`, `identity.json`, dataset check, split assignments, OOF selection, evaluation/full-refit classifiers, validation predictions, metrics, report, progress, and export verification. The source setup keeps Colab's CUDA PyTorch; exported requirements record the actual PyTorch version.

```bash
python inference.py --audio_dir WAV_FOLDER --label_dir JSON_FOLDER --ckpt_path model.pt --output predictions.csv
```

Inference reads only caller speaker/startAt/endAt from JSON; gender/text/address are not model inputs. Pretrained weights/config are embedded, so inference never downloads a model. CUDA is required. A BF16 checkpoint requires a BF16-capable GPU.

## Verification

`test_pipeline.py` checks complete-or-stop inventory, nested ZIP/Validation exclusion, invalid labels/Seoul filter, deterministic split, checksum/identity/interrupted-write handling, OOF/final-full-data refit, exported LR parity, real tiny WavLM/Wav2Vec2 CPU batching independence and state reload, OOM batch reduction, and crop/short-audio behavior. These synthetic tests do not establish dataset accuracy or full GPU performance.

At actual run time, final export additionally loads the serialized checkpoint and reruns three real audio calls on CUDA. It checks probability tolerance and exact predicted-class agreement against the extracted features before marking completion and releasing the final ZIP. The roundtrip result is in `export_verification.json`.

Model padding guidance: https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/wav2vec2
Colab local-data/Drive guidance: https://research.google.com/colaboratory/faq.html
