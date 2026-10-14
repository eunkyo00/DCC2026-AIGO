"""Shared, versioned signal processing for fresh training and offline inference."""
import contextlib
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from scipy.special import expit
from transformers import WavLMConfig, WavLMModel, Wav2Vec2Config, Wav2Vec2Model

MODELS = {
    'wavlm': ('microsoft/wavlm-large', 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'),
    'wav2vec2': ('facebook/wav2vec2-base', '0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8'),
}
# Fixed padding buckets make features independent of other calls in the batch.
# Wav2Vec2-base uses group norm: do not pass attention_mask (HF model guidance).
# This is a new batching/padding protocol, not bit-identical to the old experiment.
BUCKETS = (8000, 16000, 32000, 48000, 64000, 96000, 128000, 192000, 240000)
PROTOCOL = 'full29200-fixed-buckets-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def intervals(doc):
    result = []
    for u in doc['utterances']:
        speaker = u.get('speaker', u.get('Speaker'))
        require(str(speaker) in ('0', '1'), 'Invalid speaker')
        if str(speaker) != '1':
            continue
        a, b = u['startAt'], u['endAt']
        require(all(type(v) in (int, float) and math.isfinite(v) for v in (a, b)), 'Invalid boundary')
        require(0 <= a < b, 'Invalid caller boundary')
        result.append((float(a), float(b)))
    require(bool(result), 'No caller segments')
    return result


def read_segments(path, bounds):
    audio, rate = sf.read(path, dtype='float32')
    require(rate == 8000 and audio.ndim == 1 and np.isfinite(audio).all(), 'Expected finite 8kHz mono WAV')
    segments = []
    for start, end in bounds:
        a, b = round(start * rate / 1000), round(end * rate / 1000)
        require(0 <= a < b <= len(audio), 'Caller boundary exceeds audio')
        segments.append(resample_poly(audio[a:b], 2, 1, window=('kaiser', 5.0)).astype(np.float32))
    require(bool(segments), 'No caller audio')
    return segments


def units(segments, kind):
    """Yield (normalized samples, segment index, aggregation weight)."""
    if kind == 'wavlm':
        audio = np.concatenate(segments)
        chunks = np.array_split(audio, math.ceil(len(audio) / 192000))
        for x in chunks:
            n = len(x)
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
            yield np.pad(x, (0, max(0, 400 - len(x)))), 0, n
    else:
        for segment_index, x in enumerate(segments):
            x = np.pad(x, (0, max(0, 400 - len(x))))
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
            for chunk in np.array_split(x, math.ceil(len(x) / 240000)):
                yield chunk, segment_index, 1


def configure_torch():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class Encoder:
    def __init__(self, kind, precision, cache_dir=None, entry=None, device='cuda'):
        self.kind, self.precision, self.device = kind, precision, torch.device(device)
        config_type, model_type = ((WavLMConfig, WavLMModel) if kind == 'wavlm'
                                   else (Wav2Vec2Config, Wav2Vec2Model))
        if entry is None:
            name, revision = MODELS[kind]
            self.model = model_type.from_pretrained(name, revision=revision, cache_dir=cache_dir)
        else:
            self.model = model_type(config_type.from_dict(entry['config']))
            self.model.load_state_dict(entry['weights'], strict=True)
        self.model.requires_grad_(False).eval().to(self.device)
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def _forward(self, batch, size):
        try:
            return self._forward_once(batch, size)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise RuntimeError('GPU memory too small even for one window; choose a larger GPU') from None
        torch.cuda.empty_cache()
        middle = len(batch) // 2
        a, na = self._forward(batch[:middle], size)
        b, nb = self._forward(batch[middle:], size)
        return np.concatenate([a, b]), np.concatenate([na, nb])

    @torch.inference_mode()
    def _forward_once(self, batch, size):
        # Retry smaller batches on OOM. Fixed padding is retained, so math does not change.
        lengths = torch.tensor([len(r[0]) for r in batch], device=self.device)
        x = np.zeros((len(batch), size), dtype=np.float32)
        for i, r in enumerate(batch):
            x[i, :len(r[0])] = r[0]
        values = torch.from_numpy(x).to(self.device)
        kwargs = {}
        if self.kind == 'wavlm':
            kwargs['attention_mask'] = (torch.arange(size, device=self.device)[None] < lengths[:, None]).long()
        amp = (torch.autocast('cuda', dtype=torch.bfloat16)
               if self.device.type == 'cuda' and self.precision == 'bf16' else contextlib.nullcontext())
        with amp:
            h = self.model(values, **kwargs).last_hidden_state
        frame_lengths = self.model._get_feat_extract_output_lengths(lengths).long()
        mask = torch.arange(h.shape[1], device=self.device)[None] < frame_lengths[:, None]
        mean = (h.float() * mask[:, :, None]).sum(1) / frame_lengths[:, None]
        result = mean.cpu().numpy()
        require(np.isfinite(result).all(), 'Nonfinite encoder feature')
        return result, frame_lengths.cpu().numpy()

    def encode(self, prepared, batch_seconds):
        buckets = {}
        counts = []
        for call, segments in enumerate(prepared):
            counts.append(1 if self.kind == 'wavlm' else len(segments))
            for x, segment, weight in units(segments, self.kind):
                size = next(n for n in BUCKETS if n >= len(x))
                buckets.setdefault(size, []).append((x, call, segment, weight))
        sums = [np.zeros((n, self.dim), np.float64) for n in counts]
        weights = [np.zeros(n, np.float64) for n in counts]
        for size, records in sorted(buckets.items()):
            batch_size = min(64, max(1, int(batch_seconds * 16000 / size)))
            for offset in range(0, len(records), batch_size):
                batch = records[offset:offset + batch_size]
                features, frames = self._forward(batch, size)
                for r, f, nf in zip(batch, features, frames):
                    _, call, segment, weight = r
                    weight = weight if self.kind == 'wavlm' else int(nf)
                    sums[call][segment] += f * weight
                    weights[call][segment] += weight
        return np.stack([(s / w[:, None]).mean(0) for s, w in zip(sums, weights)]).astype(np.float32)

    def entry(self):
        return {'config': self.model.config.to_dict(),
                'weights': {k: v.detach().cpu() for k, v in self.model.state_dict().items()}}


def probability(x, state):
    value = np.asarray(x, dtype=np.float32).copy()
    value -= state['mean'].numpy()
    value /= state['scale'].numpy()
    p = expit(value @ state['coef'].numpy().reshape(-1) + state['intercept'].item())
    return np.stack([1-p, p], axis=-1)


class GenderEnsemble:
    def __init__(self, checkpoint):
        configure_torch()
        state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
        require(state['format'] == PROTOCOL and state['classes'] == ['F', 'M'], 'Unsupported checkpoint')
        require(torch.cuda.is_available(), 'CUDA GPU required')
        require(state['precision'] != 'bf16' or torch.cuda.is_bf16_supported(), 'BF16 checkpoint requires L4/A100 or compatible GPU')
        self.state = state
        self.encoders = {k: Encoder(k, state['precision'], entry=state[k]) for k in MODELS}

    def predict_segments(self, prepared):
        p = 0
        for kind, encoder in self.encoders.items():
            x = encoder.encode(prepared, self.state['batch_seconds'][kind])
            w = self.state['wavlm_weight'] if kind == 'wavlm' else 1-self.state['wavlm_weight']
            p = p + w * probability(x, self.state[kind]['lr'])
        require(np.isfinite(p).all(), 'Nonfinite predictions')
        return p
