#!/usr/bin/env python3
"""Mission 2 제출용 단일 실행 추론 스크립트."""

from __future__ import annotations

import argparse
import csv
import json
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from github_preprocessing import crop_segment, extract_melspectrogram, load_audio, pad_or_trim_to_length
from model import build_model, count_parameters
from multiview_preprocessing import make_multiview_feature, quantize_feature


def normalized_stem(path: Path) -> str:
    return unicodedata.normalize("NFC", path.stem)


def index_files(root: Path, suffix: str) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob(f"*{suffix}"):
        if path.is_file() and not path.name.startswith("._") and "__MACOSX" not in path.parts:
            indexed[normalized_stem(path)].append(path)
    return dict(indexed)


def make_feature(waveform: np.ndarray, sr: int, start_at: float, end_at: float) -> np.ndarray:
    segment = crop_segment(waveform, sr, start_at, end_at)
    segment = pad_or_trim_to_length(segment, int(round(1.5 * sr)))
    mel = extract_melspectrogram(segment, sr, n_mels=64, to_db=True)
    feature = np.clip((mel.astype(np.float32) + 80.0) / 80.0, 0.0, 1.0)
    # 학습 shard의 저장 정밀도(float16)와 동일하게 맞춘 뒤 모델 입력은 float32로 복원한다.
    return feature.astype(np.float16).astype(np.float32)


def make_feature_for_variant(
    waveform: np.ndarray, sr: int, start_at: float, end_at: float, feature_variant: str
) -> np.ndarray:
    if feature_variant == "first_1p5":
        return make_feature(waveform, sr, start_at, end_at)
    if feature_variant == "first_plus_whole":
        feature = make_multiview_feature(waveform, sr, start_at, end_at)
        # 학습에 저장된 uint8 특징과 정확히 같은 양자화 경로를 사용한다.
        return quantize_feature(feature).astype(np.float32) / 255.0
    raise ValueError(f"지원하지 않는 체크포인트 전처리: {feature_variant}")


@torch.inference_mode()
def predict_batch(model, features: list[np.ndarray], device: torch.device, threshold: float) -> list[int]:
    tensor = torch.from_numpy(np.stack(features)).unsqueeze(1).to(device)
    probabilities = model(tensor).softmax(dim=1)[:, 1]
    return (probabilities >= threshold).long().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mission 2 화자 역할 추론")
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    model_name = str(checkpoint.get("model_name", "resnet18"))
    feature_variant = str(checkpoint.get("feature_variant", "first_1p5"))
    if feature_variant not in ("first_1p5", "first_plus_whole"):
        raise ValueError(f"지원하지 않는 체크포인트 전처리: {feature_variant}")
    model = build_model(
        pretrained=False,
        dropout=float(checkpoint.get("dropout", 0.30)),
        spec_augment=False,
        model_name=model_name,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    threshold = 0.5  # 저장된 과거 튜닝 임계값을 사용하지 않는다.

    wav_index = index_files(args.audio_dir, ".wav")
    json_index = index_files(args.label_dir, ".json")
    stems = sorted(wav_index.keys() & json_index.keys())
    rows: list[dict] = []
    pending_features: list[np.ndarray] = []
    pending_meta: list[tuple[str, object, object]] = []
    started = time.perf_counter()

    def flush() -> None:
        if not pending_features:
            return
        predictions = predict_batch(model, pending_features, device, threshold)
        for (audio_name, start_at, end_at), prediction in zip(pending_meta, predictions):
            rows.append(
                {
                    "audio file name": audio_name,
                    "startAt": start_at,
                    "endAt": end_at,
                    "speaker": prediction,
                }
            )
        pending_features.clear()
        pending_meta.clear()

    for stem in stems:
        if len(wav_index[stem]) != 1 or len(json_index[stem]) != 1:
            continue
        wav_path = wav_index[stem][0]
        waveform, sr = load_audio(str(wav_path))
        with json_index[stem][0].open("r", encoding="utf-8") as handle:
            label = json.load(handle)
        utterances = label.get("utterances", [])

        for utterance in utterances if isinstance(utterances, list) else []:
            if not isinstance(utterance, dict):
                continue
            start_at, end_at = utterance.get("startAt"), utterance.get("endAt")
            try:
                start_value, end_value = float(start_at), float(end_at)
            except (TypeError, ValueError):
                continue
            if end_value <= start_value:
                continue
            pending_features.append(
                make_feature_for_variant(waveform, sr, start_value, end_value, feature_variant)
            )
            pending_meta.append((wav_path.name, start_at, end_at))
            if len(pending_features) >= args.batch_size:
                flush()
    flush()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio file name", "startAt", "endAt", "speaker"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - started
    print(f"output={args.output}")
    print(f"rows={len(rows):,} threshold={threshold:.3f} device={device}")
    print(f"model_name={model_name}")
    print(f"feature_variant={feature_variant}")
    print(f"parameters={count_parameters(model):,}")
    print(f"elapsed_seconds={elapsed:.3f} ms_per_sample={(elapsed / max(len(rows), 1) * 1000):.3f}")


if __name__ == "__main__":
    main()
