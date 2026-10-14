"""Mission 1 manifest에서 MFCC mean/std 특징을 생성하고 캐시한다."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from common.preprocessing.preprocessing_utils import (
    crop_segment,
    extract_mfcc,
    load_audio,
    pad_or_trim_to_length,
)

TARGET_SECONDS = 5.0
N_MFCC = 40
FEATURE_DIM = N_MFCC * 2
VALID_GENDERS = ("M", "F")


@dataclass(frozen=True)
class FeatureSet:
    X: np.ndarray
    y: np.ndarray
    stems: np.ndarray
    elapsed_seconds: float
    from_cache: bool


def parse_segments(value: str) -> list[tuple[float, float]]:
    """manifest의 JSON 구간 문자열을 검증하고 시간순 구간으로 반환한다."""
    try:
        raw = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid reporter_segments_ms JSON: {value!r}") from exc

    segments: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        raise ValueError("reporter_segments_ms must be a JSON list")
    for segment in raw:
        if not isinstance(segment, list) or len(segment) != 2:
            raise ValueError(f"Invalid segment: {segment!r}")
        start_ms, end_ms = float(segment[0]), float(segment[1])
        if end_ms > start_ms:
            segments.append((start_ms, end_ms))
    if not segments:
        raise ValueError("No valid reporter segment remains")
    return sorted(segments)


def aggregate_mfcc(mfcc: np.ndarray) -> np.ndarray:
    """(40, frames) MFCC를 coefficient별 mean/std의 80차원 벡터로 만든다."""
    if mfcc.ndim != 2 or mfcc.shape[0] != N_MFCC or mfcc.shape[1] == 0:
        raise ValueError(f"Expected MFCC shape ({N_MFCC}, frames), got {mfcc.shape}")
    feature = np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1)))
    return feature.astype(np.float32, copy=False)


def extract_feature(wav_path: str, segments: Sequence[tuple[float, float]]) -> np.ndarray:
    """WAV를 한 번 읽고 신고자 발화를 연결한 뒤 공통 함수로 5초/MFCC 처리한다."""
    waveform, sr = load_audio(wav_path)
    crops = [crop_segment(waveform, sr, start, end) for start, end in segments]
    crops = [crop for crop in crops if crop.size]
    if not crops:
        raise ValueError(f"All reporter segments are empty: {wav_path}")
    reporter_audio = np.concatenate(crops)
    fixed_audio = pad_or_trim_to_length(reporter_audio, int(round(TARGET_SECONDS * sr)))
    return aggregate_mfcc(extract_mfcc(fixed_audio, sr, n_mfcc=N_MFCC))


def read_manifest(manifest_path: Path, limit: Optional[int] = None) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"stem", "wav_path", "gender", "reporter_segments_ms"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")
        rows = []
        for row in reader:
            if row["gender"] not in VALID_GENDERS:
                raise ValueError(f"Invalid gender {row['gender']!r} for {row['stem']}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest_path}")
    return rows


def _cache_paths(cache_dir: Path, split: str) -> tuple[Path, Path]:
    return cache_dir / f"{split}_features.npz", cache_dir / f"{split}_metadata.json"


def _cache_is_current(metadata_path: Path, manifest_path: Path, limit: Optional[int]) -> bool:
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_mtime_ns": manifest_path.stat().st_mtime_ns,
        "limit": limit,
        "target_seconds": TARGET_SECONDS,
        "n_mfcc": N_MFCC,
        "aggregation": "temporal_mean_std",
    }


def build_or_load_features(
    manifest_path: Path,
    cache_dir: Path,
    split: str,
    *,
    force: bool = False,
    limit: Optional[int] = None,
) -> FeatureSet:
    """특징 캐시를 읽거나 manifest 전체에서 다시 생성한다."""
    manifest_path = manifest_path.resolve()
    feature_path, metadata_path = _cache_paths(cache_dir, split)
    if not force and feature_path.exists() and _cache_is_current(metadata_path, manifest_path, limit):
        with np.load(feature_path, allow_pickle=False) as data:
            return FeatureSet(data["X"], data["y"], data["stems"], 0.0, True)

    rows = read_manifest(manifest_path, limit)
    started = time.perf_counter()
    features = []
    for index, row in enumerate(rows, start=1):
        try:
            segments = parse_segments(row["reporter_segments_ms"])
            features.append(extract_feature(row["wav_path"], segments))
        except Exception as exc:
            raise RuntimeError(f"Feature extraction failed at row {index}, stem={row['stem']}") from exc
        if index % 500 == 0 or index == len(rows):
            print(f"[{split}] extracted {index}/{len(rows)}")
    elapsed = time.perf_counter() - started
    X = np.stack(features).astype(np.float32, copy=False)
    y = np.asarray([row["gender"] for row in rows])
    stems = np.asarray([row["stem"] for row in rows])

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_path, X=X, y=y, stems=stems)
    metadata = {
        "manifest_path": str(manifest_path),
        "manifest_mtime_ns": manifest_path.stat().st_mtime_ns,
        "limit": limit,
        "target_seconds": TARGET_SECONDS,
        "n_mfcc": N_MFCC,
        "aggregation": "temporal_mean_std",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return FeatureSet(X, y, stems, elapsed, False)
