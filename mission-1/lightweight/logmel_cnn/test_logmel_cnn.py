from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("logmel", HERE / "run_logmel_cnn.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frame_count_preserves_tail():
    assert MODULE.frame_count(256) == 1
    assert MODULE.frame_count(257) == 2
    assert MODULE.frame_count(8000) == int(np.ceil((8000 - 256) / 80)) + 1


def test_chunking_preserves_all_frames_and_boundaries():
    mel = np.arange(300 * 40, dtype=np.float32).reshape(300, 40)
    chunks, valid = MODULE.chunks_for_segment(mel)
    assert [x.shape for x in chunks] == [(40, 256), (40, 256)]
    assert valid == [256, 44]
    np.testing.assert_array_equal(chunks[0][:, :256].T, mel[:256])
    np.testing.assert_array_equal(chunks[1][:, :44].T, mel[256:])


def test_inner_dev_is_train_only_and_gender_stratified():
    jobs = [{"call_id": f"{i:03d}", "gender": "M" if i < 10 else "F", "partition": "train"} for i in range(20)]
    jobs += [{"call_id": "heldout", "gender": "M", "partition": "internal_validation"}]
    dev, fit = MODULE.inner_dev_ids(jobs)
    assert "heldout" not in dev | fit
    assert len([x for x in dev if int(x) < 10]) == 1
    assert len([x for x in dev if int(x) >= 10]) == 1


def test_model_parameter_count_is_fixed():
    import pytest
    torch = pytest.importorskip("torch")
    _, count = MODULE.model_and_count(torch)
    assert count == 5954


def test_chunk_batches_zero_normalized_padding():
    import pytest
    torch = pytest.importorskip("torch")
    class Store:
        def call(self, _):
            x = np.full((2, 40, 256), 3.0, dtype=np.float32)
            return x, np.array([256, 7], dtype=np.int32)
    batches = list(MODULE.chunk_batches(Store(), "a", np.full(40, 2.0, dtype=np.float32),
                                        np.ones(40, dtype=np.float32), torch.device("cpu"), torch))
    x, valid, denominator = batches[0]
    assert x.shape == (2, 1, 40, 256)
    assert denominator == 263
    assert float(x[1, 0, :, 7:].abs().sum()) == 0.0
    assert valid.tolist() == [256, 7]


def test_shard_recovers_from_permanent_copy(tmp_path):
    import soundfile as sf
    wav = tmp_path / "one.wav"
    sf.write(wav, np.zeros(4000, dtype=np.float32), 8000)
    job = {"call_id": "one", "gender": "F", "partition": "train", "wav_path": str(wav),
           "intervals": [(0.0, 0.5)], "n_segments": 1, "duration_s": 0.5}
    local, permanent = tmp_path / "local", tmp_path / "permanent"
    MODULE.extract_all([job], local, MODULE.EXPECTED_SPLIT, permanent, {"fixture": "one"})
    first, valid = MODULE.ShardStore(local).call("one")
    assert valid.sum() == MODULE.frame_count(4000)
    assert (permanent / "index.json").is_file()
    (local / "index.json").unlink()
    (local / "shard_00000.npz").unlink()
    wav.unlink()
    MODULE.extract_all([job], local, MODULE.EXPECTED_SPLIT, permanent, {"fixture": "one"})
    recovered, recovered_valid = MODULE.ShardStore(local).call("one")
    np.testing.assert_array_equal(first, recovered)
    np.testing.assert_array_equal(valid, recovered_valid)
