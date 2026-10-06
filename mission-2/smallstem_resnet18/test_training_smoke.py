#!/usr/bin/env python3
"""두 뷰 특징으로 학습·체크포인트·재개가 동작하는지 작은 데이터로 확인한다."""

import json
import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import torchvision
    import soundfile
    import pandas
    DEPENDENCY_ERROR = None
except (ModuleNotFoundError, RuntimeError) as exc:
    DEPENDENCY_ERROR = exc


@unittest.skipIf(DEPENDENCY_ERROR is not None, str(DEPENDENCY_ERROR))
class TrainingSmokeTest(unittest.TestCase):
    def test_one_epoch_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features = root / "features"
            splits = {}
            for split in ("Training", "Validation"):
                folder = features / split
                folder.mkdir(parents=True)
                rng = np.random.default_rng(42)
                np.save(folder / "features_0000.npy", rng.integers(0, 256, (4, 64, 48), dtype=np.uint8))
                np.save(folder / "labels_0000.npy", np.array([0, 1, 0, 1], dtype=np.int8))
                splits[split] = {
                    "complete": True,
                    "samples": 4,
                    "shards": [{"features": "features_0000.npy", "labels": "labels_0000.npy", "samples": 4}],
                }
            (features / "metadata.json").write_text(json.dumps({
                "format_version": 1, "feature_variant": "first_plus_whole", "splits": splits,
            }), encoding="utf-8")
            output = root / "output"
            cmd = [
                sys.executable, str(Path(__file__).with_name("train.py")),
                "--feature-root", str(features), "--output-dir", str(output),
                "--model-name", "resnet18_smallstem", "--epochs", "1",
                "--batch-size", "2", "--num-workers", "0", "--no-pretrained",
                "--mixup-alpha", "0", "--resume",
            ]
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            first = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["feature_variant"], "first_plus_whole")
            self.assertEqual(checkpoint["validation"]["samples"], 4)
            self.assertEqual(checkpoint["threshold"], 0.5)
            self.assertEqual(checkpoint["selection_policy"], "accuracy_at_fixed_0.5")
            self.assertNotIn("accuracy_tuned", checkpoint["validation"])

            second = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("completed_epoch=1", second.stdout)

            last_path = output / "last_checkpoint.pt"
            state = torch.load(last_path, map_location="cpu", weights_only=True)
            del state["selection_policy"]
            torch.save(state, last_path)
            legacy = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            self.assertNotEqual(legacy.returncode, 0)
            self.assertIn("이전 임계값 튜닝 실행은 재개할 수 없습니다", legacy.stderr)

            fine_tune_cmd = list(cmd)
            fine_tune_cmd[fine_tune_cmd.index("--output-dir") + 1] = str(root / "fine_tune")
            fine_tune_cmd.extend(["--init-checkpoint", str(output / "best_model.pt")])
            fine_tune = subprocess.run(fine_tune_cmd, capture_output=True, text=True, timeout=120, env=env)
            self.assertEqual(fine_tune.returncode, 0, fine_tune.stdout + fine_tune.stderr)
            self.assertIn("initialized_from=", fine_tune.stdout)

            manifest = root / "validation.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["stem", "startAt", "endAt", "speaker"])
                writer.writeheader()
                for index, (length, label) in enumerate(zip((400, 800, 2000, 4000), (0, 1, 0, 1))):
                    writer.writerow({"stem": "one_call", "startAt": index * 5000,
                                     "endAt": index * 5000 + length, "speaker": label})
            diagnosis = root / "diagnosis.json"
            inspect_cmd = [
                sys.executable, str(Path(__file__).with_name("analyze_validation.py")),
                "--feature-root", str(features), "--valid-manifest", str(manifest),
                "--ckpt-path", str(output / "best_model.pt"), "--output", str(diagnosis),
            ]
            inspected = subprocess.run(inspect_cmd, capture_output=True, text=True, timeout=120, env=env)
            self.assertEqual(inspected.returncode, 0, inspected.stdout + inspected.stderr)
            self.assertEqual(sum(row["samples"] for row in json.loads(diagnosis.read_text())["by_duration"]), 4)


if __name__ == "__main__":
    unittest.main()
