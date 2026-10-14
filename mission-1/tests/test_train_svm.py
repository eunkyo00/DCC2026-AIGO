import sys
from argparse import Namespace
from pathlib import Path

import numpy as np


BASELINE_DIR = Path(__file__).resolve().parents[1] / "baseline"
sys.path.insert(0, str(BASELINE_DIR))
import train_svm  # noqa: E402
from feature_extraction import FeatureSet  # noqa: E402


def test_run_fits_and_persists_pipeline(monkeypatch, tmp_path):
    rng = np.random.default_rng(42)
    train = FeatureSet(
        X=np.vstack((rng.normal(-2, 0.1, (5, 80)), rng.normal(2, 0.1, (5, 80)))).astype("float32"),
        y=np.asarray(["M"] * 5 + ["F"] * 5),
        stems=np.asarray([f"train-{i}" for i in range(10)]),
        elapsed_seconds=1.25,
        from_cache=False,
    )
    valid = FeatureSet(
        X=np.vstack((rng.normal(-2, 0.1, (2, 80)), rng.normal(2, 0.1, (2, 80)))).astype("float32"),
        y=np.asarray(["M", "M", "F", "F"]),
        stems=np.asarray([f"valid-{i}" for i in range(4)]),
        elapsed_seconds=0.5,
        from_cache=False,
    )

    def fake_features(_manifest, _cache, split, **_kwargs):
        return train if split == "train" else valid

    monkeypatch.setattr(train_svm, "build_or_load_features", fake_features)
    args = Namespace(
        train_manifest=tmp_path / "train.csv",
        valid_manifest=tmp_path / "valid.csv",
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        result_path=tmp_path / "result.json",
        force_features=False,
        limit_train=None,
        limit_valid=None,
    )
    result = train_svm.run(args)

    assert result["training_samples"] == 10
    assert result["validation_samples"] == 4
    assert result["validation_accuracy"] == 1.0
    assert result["confusion_matrix"]["values"] == [[2, 0], [0, 2]]
    assert (args.artifact_dir / "m1_001_mfcc_svm.joblib").is_file()
    assert args.result_path.is_file()
