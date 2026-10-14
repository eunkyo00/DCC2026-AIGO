"""Full-call 8 kHz log-mel CNN experiment for Mission 1.

The command is deliberately staged. ``audit`` only counts the frozen cohort;
``extract`` reads every caller interval and creates recoverable feature shards;
``train`` selects an epoch on a deterministic Train-only inner split, refits on
all Train calls, and evaluates the frozen Internal Validation exactly once.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import time

import numpy as np
import soundfile as sf

SEED = 42
SR = 8000
N_MELS = 40
N_FFT = 256
WIN = 200
HOP = 80
FMIN = 50.0
FMAX = 3800.0
CHUNK_FRAMES = 256
SHARD_CALLS = 256
MACS_PER_CHUNK = (20 * 128 * 8 * 1 * 9 + 10 * 64 * 16 * 8 * 9 +
                  5 * 32 * 32 * 16 * 9 + 32 * 2)
EXPECTED_SPLIT = "04b5018778321f775a0bf95949a37636090d4df4e3710b16627dfee309408f06"
EXPECTED_CALLS = 27985
EXPECTED_SEGMENTS = 442639
BASELINES = {"majority": 0.5318920850455601, "mfcc": 0.9506878685009826}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_csv(path: Path):
    import csv
    with path.open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if platform.system() == "Darwin" else 1024)


@contextmanager
def timed(timings: dict, name: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started


def load_jobs(split_path: Path, manifest_path: Path, segments_path: Path,
              data_root: Path | None = None) -> tuple[list[dict], str]:
    split_hash = sha256_file(split_path)
    if split_hash != EXPECTED_SPLIT:
        raise ValueError(f"Frozen split checksum changed: {split_hash}")
    split_rows = list(read_csv(split_path))
    if len(split_rows) != EXPECTED_CALLS or len({r["call_id"] for r in split_rows}) != EXPECTED_CALLS:
        raise ValueError("Frozen split must contain 27,985 unique calls")
    split = {r["call_id"]: (r["gender"], r["partition"]) for r in split_rows}
    counts = Counter(v[1] for v in split.values())
    if counts != {"train": 22388, "internal_validation": 5597}:
        raise ValueError(f"Unexpected split sizes: {counts}")
    manifests = {r["call_id"]: r for r in read_csv(manifest_path)}
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in read_csv(segments_path):
        if row["split"] == "Training" and row["file_ref"] in split and row["speaker"] == "1":
            if row.get("matched") != "True" or row.get("exceeds_wav") != "False":
                raise ValueError(f"Invalid caller segment: {row['file_ref']}")
            intervals[row["file_ref"]].append((float(row["start_s"]), float(row["end_s"])))
    if set(split) != set(manifests) or set(split) != set(intervals):
        raise ValueError("Split, manifest, and caller segment coverage differ")
    if sum(map(len, intervals.values())) != EXPECTED_SEGMENTS:
        raise ValueError("Expected 442,639 caller segments")
    jobs = []
    for call_id in sorted(split):
        gender, partition = split[call_id]
        m = manifests[call_id]
        if (m["gender"], m["partition"]) != (gender, partition):
            raise ValueError(f"Manifest differs from split: {call_id}")
        if int(m["n_segments"]) != len(intervals[call_id]):
            raise ValueError(f"Manifest caller count differs: {call_id}")
        relative = Path(m["wav_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative WAV path for {call_id}")
        wav_path = data_root / relative if data_root is not None else Path(m["wav_path"])
        jobs.append({"call_id": call_id, "gender": gender, "partition": partition,
                     "wav_path": str(wav_path), "intervals": intervals[call_id],
                     "n_segments": len(intervals[call_id]),
                     "duration_s": sum(b - a for a, b in intervals[call_id])})
    return jobs, split_hash


def frame_count(n: int) -> int:
    return 1 + (max(0, n - N_FFT) + HOP - 1) // HOP


def interval_samples(start_s: float, end_s: float) -> int:
    return round(end_s * SR) - round(start_s * SR)


def extract_logmel(samples: np.ndarray, mel_filter: np.ndarray, window: np.ndarray) -> np.ndarray:
    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("empty or non-mono segment")
    count = frame_count(len(samples))
    padded = np.pad(np.asarray(samples, dtype=np.float32), (0, max(0, (count - 1) * HOP + N_FFT - len(samples))))
    shape = (count, N_FFT)
    frames = np.lib.stride_tricks.as_strided(padded, shape=shape,
                                              strides=(padded.strides[0] * HOP, padded.strides[0])).copy()
    power = np.abs(np.fft.rfft(frames * window, n=N_FFT, axis=1)) ** 2 / N_FFT
    return np.log(np.maximum(power @ mel_filter.T, 1e-10)).astype(np.float32)


def make_mel_filter() -> tuple[np.ndarray, np.ndarray]:
    # Librosa's Slaney-style scale with area normalization, implemented here
    # so the feature extractor does not depend on a hidden global cache.
    def hz_to_mel(hz):
        hz = np.asarray(hz, dtype=np.float64)
        return np.where(hz < 1000.0, hz / (200.0 / 3.0),
                        15.0 + 15.0 * np.log(hz / 1000.0) / np.log(6.4))
    def mel_to_hz(mel):
        mel = np.asarray(mel, dtype=np.float64)
        return np.where(mel < 15.0, mel * (200.0 / 3.0),
                        1000.0 * np.exp((mel - 15.0) * np.log(6.4) / 15.0))
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2))
    filt = np.zeros((N_MELS, len(freqs)), dtype=np.float32)
    for i, (left, center, right) in enumerate(zip(edges[:-2], edges[1:-1], edges[2:])):
        filt[i] = np.maximum(0, np.minimum((freqs - left) / (center - left), (right - freqs) / (right - center)))
        filt[i] *= 2.0 / max(right - left, np.finfo(float).eps)
    return filt, np.pad(np.hamming(WIN).astype(np.float32), ((N_FFT - WIN) // 2, (N_FFT - WIN) - (N_FFT - WIN) // 2))


def chunks_for_segment(mel: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    chunks, valid = [], []
    for start in range(0, len(mel), CHUNK_FRAMES):
        part = mel[start:start + CHUNK_FRAMES]
        n = len(part)
        chunks.append(np.pad(part, ((0, CHUNK_FRAMES - n), (0, 0))).T)
        valid.append(n)
    return chunks, valid


def extract_call(job: dict, mel_filter: np.ndarray, window: np.ndarray, timings: dict) -> dict:
    with timed(timings, "wav_read_seconds"):
        raw_wav = Path(job["wav_path"]).read_bytes()
    with timed(timings, "wav_decode_seconds"):
        audio, rate = sf.read(io.BytesIO(raw_wav), dtype="float32", always_2d=False)
    if rate != SR or audio.ndim != 1:
        raise ValueError(f"Unexpected WAV for {job['call_id']}: {rate} Hz, shape={audio.shape}")
    with timed(timings, "wav_sha256_seconds"):
        source_hash = hashlib.sha256(raw_wav).hexdigest()
    chunks, valid, segment_chunk_offsets = [], [], [0]
    with timed(timings, "logmel_compute_seconds"):
        for start_s, end_s in job["intervals"]:
            start, end = round(start_s * rate), round(end_s * rate)
            if start < 0 or end <= start or end > len(audio):
                raise ValueError(f"Invalid bounds for {job['call_id']}: {start_s}, {end_s}")
            mel = extract_logmel(audio[start:end], mel_filter, window)
            a, b = chunks_for_segment(mel)
            chunks.extend(a); valid.extend(b); segment_chunk_offsets.append(len(chunks))
    if not chunks:
        raise ValueError(f"No feature chunks for {job['call_id']}")
    return {"call_id": job["call_id"], "gender": job["gender"], "partition": job["partition"],
            "mel": np.stack(chunks), "valid": np.asarray(valid, dtype=np.int32),
            "segment_chunk_offsets": np.asarray(segment_chunk_offsets, dtype=np.int64),
            "n_segments": job["n_segments"], "duration_s": job["duration_s"], "wav_sha256": source_hash}


def audit(jobs: list[dict], out: Path, split_hash: str) -> None:
    total_frames = sum(frame_count(interval_samples(a, b)) for j in jobs for a, b in j["intervals"])
    chunks = sum((frame_count(interval_samples(a, b)) + CHUNK_FRAMES - 1) // CHUNK_FRAMES
                 for j in jobs for a, b in j["intervals"])
    payload = {"dataset": {"calls": len(jobs), "segments": sum(j["n_segments"] for j in jobs),
                           "train_calls": sum(j["partition"] == "train" for j in jobs),
                           "validation_calls": sum(j["partition"] == "internal_validation" for j in jobs),
                           "split_sha256": split_hash},
               "feature_plan": {"sample_rate": SR, "n_mels": N_MELS, "n_fft": N_FFT,
                                "win_length": WIN, "hop_length": HOP, "frequency_range_hz": [FMIN, FMAX],
                                "chunk_frames": CHUNK_FRAMES, "total_valid_frames": total_frames,
                                "total_chunks": chunks, "raw_float32_bytes": total_frames * N_MELS * 4,
                                "forward_macs": chunks * MACS_PER_CHUNK,
                                "forward_macs_per_chunk": MACS_PER_CHUNK},
               "model": {"trainable_parameters": 5954, "fp32_weight_bytes": 23816}}
    write_json(out, payload)


class ShardStore:
    def __init__(self, root: Path):
        self.root = root
        self.index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        self.jobs = {x["call_id"]: x for x in self.index["calls"]}
        self._loaded_shard: int | None = None
        self._loaded: dict = {}

    def _load(self, shard: int) -> dict:
        if shard != self._loaded_shard:
            p = self.root / f"shard_{shard:05d}.npz"
            with np.load(p, allow_pickle=False) as data:
                self._loaded = {name: data[name] for name in data.files}
            self._loaded_shard = shard
        return self._loaded

    def call(self, call_id: str) -> tuple[np.ndarray, np.ndarray]:
        row = self.jobs[call_id]; data = self._load(row["shard"])
        a, b = row["chunk_start"], row["chunk_end"]
        return data["mel"][a:b], data["valid"][a:b]


def extract_all(jobs: list[dict], out: Path, split_hash: str, permanent: Path | None,
                source_hashes: dict[str, str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    mel_filter, window = make_mel_filter()
    existing = {}
    index_path = out / "index.json"
    if permanent and not index_path.exists() and (permanent / "index.json").exists():
        shutil.copy2(permanent / "index.json", index_path)
    if index_path.exists():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing.get("split_sha256") != split_hash:
            raise ValueError("Existing feature store belongs to another frozen split")
        if existing.get("source_hashes") != source_hashes:
            raise ValueError("Existing feature store source CSVs differ")
        config = existing.get("config", {})
        if config != {"sr": SR, "n_mels": N_MELS, "n_fft": N_FFT, "win": WIN, "hop": HOP,
                      "fmin": FMIN, "fmax": FMAX, "chunk_frames": CHUNK_FRAMES}:
            raise ValueError("Existing feature store config differs")
    calls_by_id = {row["call_id"]: row for row in existing.get("calls", [])}
    shard_hashes = dict(existing.get("shards", {}))
    timings: dict[str, float] = dict(existing.get("timings", {})); rss_peak = float(existing.get("rss_peak_mb", 0.0)); started = time.perf_counter()
    for shard_start in range(0, len(jobs), SHARD_CALLS):
        shard_jobs = jobs[shard_start:shard_start + SHARD_CALLS]; arrays = []; valids = []; segment_offsets = []; cursor = 0; segment_cursor = 0
        shard_number = shard_start // SHARD_CALLS
        shard = out / f"shard_{shard_number:05d}.npz"
        expected_hash = shard_hashes.get(shard.name)
        if permanent and expected_hash and not shard.exists():
            source = permanent / shard.name
            if source.is_file():
                with timed(timings, "cache_restore_seconds"):
                    shutil.copy2(source, shard)
                    if sha256_file(shard) != expected_hash:
                        shard.unlink()
                        raise IOError(f"Restored shard hash differs: {shard.name}")
        if shard.exists() and all(j["call_id"] in calls_by_id for j in shard_jobs):
            if expected_hash and sha256_file(shard) == expected_hash:
                print(f"restored shard {shard_number:05d}", flush=True)
                continue
        shard_calls = []
        for job in shard_jobs:
            row = extract_call(job, mel_filter, window, timings); rss_peak = max(rss_peak, rss_mb())
            arrays.append(row["mel"]); valids.append(row["valid"])
            segment_offsets.append(row["segment_chunk_offsets"][:-1] + cursor)
            shard_calls.append({"call_id": job["call_id"], "gender": job["gender"], "partition": job["partition"],
                                "shard": shard_number, "chunk_start": cursor,
                                "chunk_end": cursor + len(row["mel"]), "n_segments": job["n_segments"],
                                "segment_start": segment_cursor, "segment_end": segment_cursor + job["n_segments"],
                                "duration_s": job["duration_s"], "wav_sha256": row["wav_sha256"]})
            cursor += len(row["mel"])
            segment_cursor += job["n_segments"]
        payload = {"mel": np.concatenate(arrays), "valid": np.concatenate(valids),
                   "segment_chunk_offsets": np.concatenate(segment_offsets)}
        tmp = shard.with_suffix(f".tmp-{os.getpid()}.npz")
        with timed(timings, "local_shard_write_seconds"):
            np.savez(tmp, **payload)
            with tmp.open("rb") as f: os.fsync(f.fileno())
            os.replace(tmp, shard)
        if permanent:
            dest = permanent / shard.name; dest.parent.mkdir(parents=True, exist_ok=True)
            with timed(timings, "permanent_copy_verify_seconds"):
                shutil.copy2(shard, dest)
                if sha256_file(shard) != sha256_file(dest):
                    raise IOError(f"Permanent shard verification failed: {dest}")
        calls_by_id.update({row["call_id"]: row for row in shard_calls})
        index = {"schema": 1, "split_sha256": split_hash, "source_hashes": source_hashes,
             "created_utc": existing.get("created_utc", datetime.now(timezone.utc).isoformat()),
             "config": {"sr": SR, "n_mels": N_MELS, "n_fft": N_FFT, "win": WIN, "hop": HOP,
                        "fmin": FMIN, "fmax": FMAX, "chunk_frames": CHUNK_FRAMES},
             "calls": sorted(calls_by_id.values(), key=lambda x: x["call_id"]),
             "shards": {**shard_hashes, shard.name: sha256_file(shard)},
             "timings": {**timings, "total_extraction_seconds": time.perf_counter() - started},
             "rss_peak_mb": rss_peak}
        shard_hashes[shard.name] = index["shards"][shard.name]
        atomic_json(index_path, index)
        if permanent:
            with timed(timings, "permanent_index_commit_seconds"):
                atomic_json(permanent / "index.json", index)
        print(f"extracted {min(shard_start + len(shard_jobs), len(jobs)):,}/{len(jobs):,}", flush=True)


def inner_dev_ids(jobs: list[dict]) -> tuple[set[str], set[str]]:
    groups = {"M": [], "F": []}
    for j in jobs:
        if j["partition"] == "train": groups[j["gender"]].append(j["call_id"])
    dev = set()
    for gender, ids in groups.items():
        ranked = sorted(ids, key=lambda x: sha256_bytes(f"{SEED}:{gender}:{x}"))
        dev.update(ranked[:round(len(ids) * 0.1)])
    return dev, set(groups["M"] + groups["F"]) - dev


def require_torch():
    import torch
    return torch


def sync_device(device, torch) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def model_and_count(torch):
    class TinyCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = torch.nn.ModuleList([torch.nn.Conv2d(a, b, 3, stride=2, padding=1)
                                              for a, b in [(1, 8), (8, 16), (16, 32)]])
            self.head = torch.nn.Linear(32, 2)
        def forward(self, x, valid):
            z = x
            lengths = valid
            for conv in self.convs:
                z = torch.relu(conv(z))
                lengths = (lengths + 1) // 2
                mask = torch.arange(z.shape[-1], device=z.device)[None, :] < lengths[:, None]
                z = z * mask[:, None, None, :]
            z = z.sum(dim=(2, 3)) / (z.shape[2] * lengths).to(z.dtype).unsqueeze(1)
            return self.head(z)
    model = TinyCNN()
    return model, sum(p.numel() for p in model.parameters())


def call_ids_for(jobs: list[dict], ids: set[str]) -> list[str]:
    return [j["call_id"] for j in jobs if j["call_id"] in ids]


def moments(store: ShardStore, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(N_MELS, dtype=np.float64)
    total_sq = np.zeros(N_MELS, dtype=np.float64)
    count = 0
    for cid in ids:
        mel, valid = store.call(cid)
        for x, n in zip(mel, valid):
            v = x[:, :n].astype(np.float64, copy=False)
            total += v.sum(axis=1); total_sq += np.square(v).sum(axis=1); count += n
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(total_sq / count - mean * mean, 0)), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def chunk_batches(store, cid, mean, std, device, torch, batch_size=32):
    mel, valid = store.call(cid)
    for start in range(0, len(valid), batch_size):
        n = valid[start:start + batch_size]
        x = ((mel[start:start + batch_size] - mean[None, :, None]) /
             std[None, :, None]).astype(np.float32)
        for i, length in enumerate(n):
            x[i, :, int(length):] = 0
        yield (torch.from_numpy(x).unsqueeze(1).to(device),
               torch.from_numpy(n.astype(np.int64)).to(device),
               int(valid.sum()))


def train_pass(store: ShardStore, ids: list[str], mean: float, std: float, device, epochs: int,
               max_epochs: int, patience: int, torch, validation_ids: list[str] | None = None,
               optimizer=None, model=None, progress_path: Path | None = None):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    if model is None: model, _ = model_and_count(torch)
    model.to(device)
    if optimizer is None: optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_loss, bad, chosen = None, float("inf"), 0, 0
    identity = sha256_bytes("\n".join(ids) + ":" + sha256_file(store.root / "index.json") +
                            ":" + hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest() +
                            f":{epochs}:{max_epochs}:{patience}")
    first_epoch = 1
    if progress_path and progress_path.exists():
        state = torch.load(progress_path, map_location=device, weights_only=True)
        if state["identity"] != identity:
            raise ValueError(f"Training progress belongs to another experiment: {progress_path}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first_epoch = state["epoch"] + 1
        best_state, best_loss, bad, chosen = (state["best_state"], state["best_loss"],
                                              state["bad"], state["chosen"])
        print(f"resuming epoch {first_epoch} from {progress_path}", flush=True)
    def evaluate(eval_ids):
        model.eval(); losses = []
        with torch.no_grad():
            for cid in eval_ids:
                label = 0 if store.jobs[cid]["gender"] == "M" else 1
                weighted = 0.0
                for x, valid, denominator in chunk_batches(store, cid, mean, std, device, torch):
                    y = torch.full((len(valid),), label, device=device, dtype=torch.long)
                    weighted += float((criterion(model(x, valid), y) * valid).sum().item())
                losses.append(weighted / denominator)
        return float(np.mean(losses))
    if validation_ids is not None and bad >= patience:
        first_epoch = max_epochs + 1
    for epoch in range(first_epoch, (epochs if epochs else max_epochs) + 1):
        model.train()
        for group_start in range(0, len(ids), 8):
            group = ids[group_start:group_start + 8]
            optimizer.zero_grad(set_to_none=True)
            for cid in group:
                label = 0 if store.jobs[cid]["gender"] == "M" else 1
                for x, valid, denominator in chunk_batches(store, cid, mean, std, device, torch):
                    y = torch.full((len(valid),), label, device=device, dtype=torch.long)
                    ((criterion(model(x, valid), y) * valid).sum() / denominator / len(group)).backward()
            optimizer.step()
        sync_device(device, torch)
        print(f"epoch {epoch} fit complete", flush=True)
        if validation_ids is not None:
            val_loss = evaluate(validation_ids)
            if val_loss < best_loss - 1e-9:
                best_loss, bad, chosen = val_loss, 0, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else: bad += 1
        if progress_path:
            tmp = progress_path.with_suffix(".tmp")
            torch.save({"identity": identity, "epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "best_state": best_state,
                        "best_loss": best_loss, "bad": bad, "chosen": chosen}, tmp)
            os.replace(tmp, progress_path)
        if bad >= patience: break
    if validation_ids is not None:
        model.load_state_dict(best_state); return model, chosen, best_loss
    return model, epochs, None


def predict(store: ShardStore, ids: list[str], model, mean: float, std: float, device, torch):
    model.eval(); result = []
    with torch.no_grad():
        for cid in ids:
            weighted_logits = torch.zeros(2, device=device)
            for x, valid, denominator in chunk_batches(store, cid, mean, std, device, torch):
                weighted_logits += (model(x, valid) * valid.unsqueeze(1)).sum(0) / denominator
            pred = int(weighted_logits.argmax().item())
            result.append({"call_id": cid, "gender": store.jobs[cid]["gender"], "prediction": "M" if pred == 0 else "F"})
    return result


def compare(predictions: list[dict], mfcc_path: Path | None) -> dict:
    y = np.array([p["gender"] for p in predictions]); p = np.array([p["prediction"] for p in predictions])
    out = {"accuracy": float(np.mean(y == p)), "gender_accuracy": {g: float(np.mean(p[y == g] == g)) for g in ("M", "F")},
           "confusion_matrix_labels": ["M", "F"],
           "confusion_matrix": [[int(np.sum((y == a) & (p == b))) for b in ("M", "F")] for a in ("M", "F")]}
    if mfcc_path:
        baseline_rows = list(read_csv(mfcc_path))
        mfcc = {r["call_id"]: r for r in baseline_rows}
        if len(mfcc) != len(baseline_rows): raise ValueError("Duplicate MFCC call IDs")
        if set(mfcc) != {x["call_id"] for x in predictions}: raise ValueError("MFCC prediction membership differs")
        if any(mfcc[x["call_id"]]["gender"] != x["gender"] for x in predictions):
            raise ValueError("MFCC gender labels differ")
        m = np.array([mfcc[x["call_id"]]["prediction"] for x in predictions])
        e1, e2 = set(np.where(y != p)[0]), set(np.where(y != m)[0])
        out["comparison"] = {"both_correct": int(np.sum((y == p) & (y == m))), "cnn_only_correct": int(np.sum((y == p) & (y != m))),
                              "mfcc_only_correct": int(np.sum((y != p) & (y == m))), "both_wrong": int(np.sum((y != p) & (y != m))),
                              "error_jaccard": len(e1 & e2) / len(e1 | e2) if e1 | e2 else 1.0,
                              "accuracy_difference_pp": float((np.mean(y == p) - np.mean(y == m)) * 100)}
    return out


def train_all(jobs: list[dict], store: ShardStore, out: Path, mfcc_path: Path | None, device_name: str) -> None:
    torch = require_torch(); torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(device_name); dev, fit_ids = inner_dev_ids(jobs)
    if store.index.get("split_sha256") != EXPECTED_SPLIT or len(store.jobs) != EXPECTED_CALLS:
        raise ValueError("Feature store does not cover the frozen cohort")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model.pt"
    claim = out / "evaluation_claim.json"
    if (out / "metrics.json").exists():
        raise FileExistsError("Final Validation result already exists; refusing another evaluation")
    if checkpoint.exists() != claim.exists():
        raise RuntimeError("Incomplete checkpoint or evaluation claim; inspect output directory")
    train_ids = call_ids_for(jobs, fit_ids); dev_ids = call_ids_for(jobs, dev)
    timings = {}; started = time.perf_counter()
    full_train = call_ids_for(jobs, {j["call_id"] for j in jobs if j["partition"] == "train"})
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        claim_data = json.loads(claim.read_text())
        if sha256_file(checkpoint) != claim_data["model_sha256"]:
            raise ValueError("Claimed model checksum differs")
        timings.update(claim_data.get("timings", {}))
        final_model, count = model_and_count(torch)
        final_model.load_state_dict(saved["state_dict"]); final_model.to(device)
        full_mean = np.asarray(saved["mean"], dtype=np.float32)
        full_std = np.asarray(saved["std"], dtype=np.float32)
        chosen_epoch, inner_loss = saved["epoch"], saved["inner_dev_loss"]
        print("Resuming final inference with the identical frozen model", flush=True)
    else:
        with timed(timings, "inner_normalization_seconds"):
            mean, std = moments(store, train_ids)
        model, count = model_and_count(torch)
        with timed(timings, "inner_selection_seconds"):
            _, chosen_epoch, inner_loss = train_pass(store, train_ids, mean, std, device, 0, 30, 5,
                                                     torch, dev_ids, model=model,
                                                     progress_path=out / "inner_progress.pt")
            sync_device(device, torch)
        with timed(timings, "full_train_normalization_seconds"):
            full_mean, full_std = moments(store, full_train)
        torch.manual_seed(SEED)
        with timed(timings, "full_train_seconds"):
            final_model, _, _ = train_pass(store, full_train, full_mean, full_std, device,
                                           chosen_epoch, chosen_epoch, 0, torch,
                                           progress_path=out / "full_progress.pt")
            sync_device(device, torch)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
                    "mean": full_mean.tolist(), "std": full_std.tolist(), "epoch": chosen_epoch,
                    "inner_dev_loss": inner_loss}, temporary)
        os.replace(temporary, checkpoint)
        atomic_json(claim, {"model_sha256": sha256_file(checkpoint), "split_sha256": EXPECTED_SPLIT,
                            "epoch": chosen_epoch, "timings": timings,
                            "created_utc": datetime.now(timezone.utc).isoformat()})
    val_ids = call_ids_for(jobs, {j["call_id"] for j in jobs if j["partition"] == "internal_validation"})
    with timed(timings, "validation_inference_seconds"):
        predictions = predict(store, val_ids, final_model, full_mean, full_std, device, torch)
        sync_device(device, torch)
    metrics = {"dataset": {"split_sha256": store.index["split_sha256"], "train_calls": len(full_train), "internal_validation_calls": len(val_ids),
                            "inner_fit_calls": len(train_ids), "inner_dev_calls": len(dev_ids)},
               "model": {"name": "tiny_logmel_cnn", "parameters": count, "input": [1, N_MELS, CHUNK_FRAMES],
                         "conv_channels": [1, 8, 16, 32], "macs_per_chunk": MACS_PER_CHUNK},
               "training": {"seed": SEED, "inner_dev_rule": "10% per gender by SHA256 rank", "normalization": "Train-only valid frames",
                            "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)", "selected_epoch": chosen_epoch,
                            "inner_dev_loss": inner_loss, "normalization_mean": full_mean.tolist(),
                            "normalization_std": full_std.tolist()},
               "metrics": compare(predictions, mfcc_path),
               "timings": {**timings, "current_process_seconds": time.perf_counter() - started},
               "memory": {"peak_rss_mb": rss_mb(),
                          "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                          "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None},
               "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device)}}
    with (out / "val_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["call_id", "gender", "prediction"]); w.writeheader(); w.writerows(predictions)
    atomic_json(out / "metrics.json", metrics)


def main() -> None:
    here = Path(__file__).resolve().parent; mission = here.parents[1]
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["audit", "extract", "train"])
    p.add_argument("--split", type=Path, default=mission / "validation/split_assignments.csv")
    p.add_argument("--manifest", type=Path, default=mission / "validation/manifests/calls.csv")
    p.add_argument("--segments", type=Path, default=mission / "eda/eda_outputs/segments.csv")
    p.add_argument("--data-root", type=Path, help="Root containing Training/...; uses manifest wav_relative_path")
    p.add_argument("--store", type=Path, default=here / "artifacts/logmel_store")
    p.add_argument("--permanent-store", type=Path)
    p.add_argument("--out", type=Path, default=here / "artifacts/run")
    p.add_argument("--mfcc-predictions", type=Path)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(); jobs, split_hash = load_jobs(a.split.resolve(), a.manifest.resolve(),
                                                     a.segments.resolve(), a.data_root)
    if a.stage == "audit": audit(jobs, a.out, split_hash)
    elif a.stage == "extract":
        extract_all(jobs, a.store, split_hash, a.permanent_store,
                    {"split": split_hash, "manifest": sha256_file(a.manifest),
                     "segments": sha256_file(a.segments),
                     "extractor": sha256_bytes("".join(inspect.getsource(fn) for fn in
                         (frame_count, extract_logmel, make_mel_filter,
                          chunks_for_segment, extract_call)))})
    else: train_all(jobs, ShardStore(a.store), a.out, a.mfcc_predictions, a.device)


if __name__ == "__main__": main()
