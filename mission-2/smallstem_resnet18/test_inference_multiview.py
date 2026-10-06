#!/usr/bin/env python3
"""새 체크포인트로 1회 명령 추론 CSV가 생성되는지 구조를 검사한다."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import soundfile as sf
    import torch
    import torchvision
    import inference
    from model import build_model
    DEPENDENCY_ERROR = None
except (ModuleNotFoundError, RuntimeError) as exc:
    DEPENDENCY_ERROR = exc


@unittest.skipIf(DEPENDENCY_ERROR is not None, str(DEPENDENCY_ERROR))
class InferenceMultiViewTest(unittest.TestCase):
    def test_csv_from_audio_and_start_end_only(self):
        def fake_mel(waveform, sr, n_mels, to_db):
            del sr, to_db
            frames = 1 + len(waveform) // 512
            return np.repeat(np.linspace(-80, 0, frames, dtype=np.float32)[None, :], n_mels, axis=0)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "wav"
            labels = root / "json"
            audio.mkdir()
            labels.mkdir()
            sf.write(audio / "sample.wav", np.zeros(4 * 8000, dtype=np.float32), 8000)
            (labels / "sample.json").write_text(json.dumps({"utterances": [
                {"startAt": 0, "endAt": 3000, "speaker": 7, "text": "IGNORE"},
                {"startAt": 3100, "endAt": 3800, "speaker": 7, "text": "IGNORE"},
            ]}), encoding="utf-8")
            model = build_model(pretrained=False, spec_augment=False, model_name="resnet18_smallstem")
            checkpoint = root / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(), "model_name": "resnet18_smallstem",
                "feature_variant": "first_plus_whole", "dropout": 0.3, "threshold": 0.99,
            }, checkpoint)
            output = root / "mission2.csv"
            with mock.patch("multiview_preprocessing.extract_melspectrogram", side_effect=fake_mel):
                with mock.patch.object(sys, "argv", [
                    "inference.py", "--audio_dir", str(audio), "--label_dir", str(labels),
                    "--ckpt_path", str(checkpoint), "--output", str(output),
                ]):
                    with contextlib.redirect_stdout(io.StringIO()) as captured:
                        inference.main()
                    self.assertIn("threshold=0.500", captured.getvalue())
            with output.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, ["audio file name", "startAt", "endAt", "speaker"])
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["speaker"] in {"0", "1"} for row in rows))


if __name__ == "__main__":
    unittest.main()
