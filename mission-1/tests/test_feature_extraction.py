import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "baseline/feature_extraction.py"
SPEC = importlib.util.spec_from_file_location("m1_features", MODULE_PATH)
features = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = features
SPEC.loader.exec_module(features)


def test_parse_segments_sorts_and_drops_empty_ranges():
    value = json.dumps([[200, 300], [100, 150], [400, 400]])
    assert features.parse_segments(value) == [(100.0, 150.0), (200.0, 300.0)]


def test_aggregate_mfcc_is_mean_then_std():
    mfcc = np.tile(np.asarray([[1.0, 3.0]], dtype=np.float32), (40, 1))
    result = features.aggregate_mfcc(mfcc)
    assert result.shape == (80,)
    np.testing.assert_allclose(result[:40], 2.0)
    np.testing.assert_allclose(result[40:], 1.0)
