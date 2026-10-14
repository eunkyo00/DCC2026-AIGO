"""공식 Training으로만 scaler/SVM을 fit하고 Validation accuracy를 기록한다."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
else:
    project_root = Path(__file__).resolve().parents[2]

try:
    from .feature_extraction import FEATURE_DIM, build_or_load_features
except ImportError:
    from feature_extraction import FEATURE_DIM, build_or_load_features


def _optional_limit(value: str) -> Optional[int]:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 1 MFCC + RBF SVM baseline")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=project_root / "manifests/mission1_gender_Training.csv",
    )
    parser.add_argument(
        "--valid-manifest",
        type=Path,
        default=project_root / "manifests/mission1_gender_Validation.csv",
    )
    parser.add_argument("--cache-dir", type=Path, default=project_root / "mission-1/cache")
    parser.add_argument("--artifact-dir", type=Path, default=project_root / "mission-1/artifacts")
    parser.add_argument(
        "--result-path",
        type=Path,
        default=project_root / "mission-1/baseline/results/baseline_results.json",
    )
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--limit-train", type=_optional_limit, default=None, help="smoke test only")
    parser.add_argument("--limit-valid", type=_optional_limit, default=None, help="smoke test only")
    return parser.parse_args()


def _class_metrics(y_true, y_pred) -> tuple[dict[str, float], dict[str, int]]:
    accuracies, correct = {}, {}
    for label in ("M", "F"):
        mask = y_true == label
        correct[label] = int((y_pred[mask] == label).sum())
        accuracies[label] = float(correct[label] / mask.sum()) if mask.any() else 0.0
    return accuracies, correct


def run(args: argparse.Namespace) -> dict:
    train = build_or_load_features(
        args.train_manifest,
        args.cache_dir,
        "train",
        force=args.force_features,
        limit=args.limit_train,
    )
    valid = build_or_load_features(
        args.valid_manifest,
        args.cache_dir,
        "valid",
        force=args.force_features,
        limit=args.limit_valid,
    )
    if train.X.shape[1] != FEATURE_DIM or valid.X.shape[1] != FEATURE_DIM:
        raise ValueError("Unexpected feature dimension")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ])
    train_started = time.perf_counter()
    model.fit(train.X, train.y)  # scaler와 SVM 모두 Training에만 fit
    training_seconds = time.perf_counter() - train_started

    inference_started = time.perf_counter()
    predictions = model.predict(valid.X)
    inference_seconds = time.perf_counter() - inference_started
    class_accuracy, class_correct = _class_metrics(valid.y, predictions)
    matrix = confusion_matrix(valid.y, predictions, labels=["M", "F"]).tolist()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifact_dir / "m1_001_mfcc_svm.joblib"
    joblib.dump(model, model_path)
    result = {
        "experiment_id": "M1-001",
        "preprocessing": "reporter segments -> concatenate -> 5 sec pad/trim -> MFCC(40) -> temporal mean/std",
        "model": {"type": "SVC", "kernel": "rbf", "C": 1.0, "gamma": "scale"},
        "training_samples": int(len(train.y)),
        "validation_samples": int(len(valid.y)),
        "training_gender_distribution": dict(Counter(train.y.tolist())),
        "validation_gender_distribution": dict(Counter(valid.y.tolist())),
        "feature_shape": {"training": list(train.X.shape), "validation": list(valid.X.shape)},
        "validation_accuracy": float(accuracy_score(valid.y, predictions)),
        "per_gender_accuracy": class_accuracy,
        "per_gender_correct": class_correct,
        "confusion_matrix": {"labels": ["M", "F"], "values": matrix},
        "timing_seconds": {
            "feature_extraction_training": train.elapsed_seconds,
            "feature_extraction_validation": valid.elapsed_seconds,
            "training": training_seconds,
            "validation_inference": inference_seconds,
        },
        "feature_cache_used": {"training": train.from_cache, "validation": valid.from_cache},
        "model_path": str(model_path.resolve()),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "is_full_official_run": args.limit_train is None and args.limit_valid is None,
    }
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    args = parse_args()
    for path in (args.train_manifest, args.valid_manifest):
        if not path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {path}\nGenerate it with: python -m common.preprocessing.build_manifests"
            )
    run(args)


if __name__ == "__main__":
    main()
