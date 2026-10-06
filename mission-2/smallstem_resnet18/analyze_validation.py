#!/usr/bin/env python3
"""Validation 오류를 발화 길이별로 분석한다. 길이는 모델 입력으로 사용하지 않는다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from feature_dataset import PrecomputedFeatureDataset
from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Mission 2 길이별 Validation 진단")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = PrecomputedFeatureDataset(args.feature_root, "Validation")
    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    variant = str(checkpoint.get("feature_variant", "first_1p5"))
    if variant != dataset.feature_variant:
        raise ValueError(f"checkpoint={variant}, feature dataset={dataset.feature_variant}")
    model = build_model(
        pretrained=False,
        dropout=float(checkpoint.get("dropout", 0.3)),
        spec_augment=False,
        model_name=str(checkpoint.get("model_name", "resnet18")),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    manifest = pd.read_csv(
        args.valid_manifest, encoding="utf-8-sig",
        usecols=["stem", "startAt", "endAt", "speaker"],
    )
    ordered = pd.concat([part for _, part in manifest.groupby("stem", sort=False)], ignore_index=True)
    if len(ordered) != len(dataset):
        raise ValueError(f"manifest rows={len(ordered)}, features={len(dataset)}")
    expected_labels = ordered["speaker"].to_numpy(dtype=np.int64)
    durations = (ordered["endAt"] - ordered["startAt"]).to_numpy(dtype=np.float32) / 1000.0

    actual_labels = []
    predicted = []
    threshold = 0.5  # 대회 규정: 임계값 탐색 및 저장된 튜닝 값 사용 금지.
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for features, labels in loader:
            logits = model(features.to(device))
            probabilities = logits.softmax(dim=1)[:, 1].cpu().numpy()
            actual_labels.append(labels.numpy())
            predicted.append(probabilities >= threshold)
    y = np.concatenate(actual_labels)
    p = np.concatenate(predicted)
    if not np.array_equal(y, expected_labels):
        raise ValueError("특징 순서가 검증 매니페스트와 다릅니다.")

    ranges = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 3), (3, 6), (6, float("inf"))]
    by_duration = []
    for low, high in ranges:
        mask = (durations >= low) & (durations < high)
        by_duration.append({
            "seconds": f"{low:g}–{high:g}",
            "samples": int(mask.sum()),
            "accuracy": float(np.mean(p[mask] == y[mask])) if mask.any() else None,
            "speaker_0_accuracy": (
                float(np.mean(p[mask & (y == 0)] == 0)) if (mask & (y == 0)).any() else None
            ),
            "speaker_1_accuracy": (
                float(np.mean(p[mask & (y == 1)] == 1)) if (mask & (y == 1)).any() else None
            ),
        })
    result = {
        "model_name": str(checkpoint.get("model_name", "resnet18")),
        "feature_variant": variant,
        "threshold": threshold,
        "overall_accuracy": float(np.mean(p == y)),
        "by_duration": by_duration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
