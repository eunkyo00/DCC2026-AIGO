"""고정 WavLM/Wav2Vec2 특징과 학습된 Scaler/LR의 75:25 앙상블.

체크포인트에 본체 가중치·설정·전처리 설정·Scaler/LR 파라미터를 포함한다.
추론은 네트워크 다운로드나 학습 데이터 캐시에 의존하지 않는다.
"""
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from scipy.special import expit
from transformers import WavLMConfig, WavLMModel, Wav2Vec2Config, Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor


def lr_probability(feature, state):
    """sklearn의 float32 Scaler 출력과 classes_=[F,M] 이진 LR 계산을 재현한다."""
    value = np.asarray(feature, dtype=np.float32).copy()
    value -= state['mean'].numpy()
    value /= state['scale'].numpy()
    p_male = expit(value @ state['coef'].numpy().reshape(-1) + state['intercept'].item())
    return np.array([1 - p_male, p_male], dtype=np.float64)


class GenderEnsemble:
    def __init__(self, checkpoint):
        state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
        if state['format'] != 'mission1-ensemble-v1' or state['classes'] != ['F', 'M']:
            raise ValueError('Unsupported checkpoint format/class order')
        if state['wavlm_weight'] != 0.75 or state['wavlm_precision'] != 'bf16':
            raise ValueError('Unexpected ensemble settings')
        # 원 실험은 WavLM BF16/CUDA, Wav2Vec2 FP32였다. 정밀도를 조용히 변경하지 않는다.
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('This checkpoint requires a BF16-capable CUDA GPU (original run: L4)')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device('cuda')
        self.state = state
        self.encoders, self.processors = {}, {}
        for name, config_type, model_type in [('wavlm', WavLMConfig, WavLMModel),
                                              ('wav2vec2', Wav2Vec2Config, Wav2Vec2Model)]:
            entry = state[name]
            encoder = model_type(config_type.from_dict(entry['config']))
            encoder.load_state_dict(entry['weights'], strict=True)
            encoder.requires_grad_(False).eval().to(self.device)
            self.encoders[name] = encoder
            self.processors[name] = Wav2Vec2FeatureExtractor(**entry['processor'])

    @torch.inference_mode()
    def wavlm_features(self, segments):
        """각 구간 8→16kHz 후 연결, 균등 12초 창의 마지막 층 평균을 실샘플 수로 가중."""
        waveform = np.concatenate(segments)
        moments, lengths = [], []
        for window in np.array_split(waveform, math.ceil(len(waveform) / 192000)):
            # 원 실험: 실제 샘플을 정규화한 다음 최소 convolution 길이로 패딩.
            value = self.processors['wavlm'](window, sampling_rate=16000,
                return_tensors='pt', padding=False).input_values
            if value.shape[1] < 400:
                value = torch.nn.functional.pad(value, (0, 400 - value.shape[1]))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = self.encoders['wavlm'](value.to(self.device), output_hidden_states=True).hidden_states[-1]
            moments.append(hidden[0].float().mean(0))
            lengths.append(len(window))
        weights = torch.tensor(lengths, dtype=torch.float32, device=self.device)
        result = (torch.stack(moments) * weights[:, None]).sum(0) / weights.sum()
        return result.cpu().numpy()

    @torch.inference_mode()
    def wav2vec2_features(self, segments):
        """발화별 정규화·최대 15초 균등 분할, 프레임 가중 평균 후 발화 동일 가중 평균."""
        embeddings = []
        for segment in segments:
            # 원 Wav2Vec2 실험은 정규화 전에 패딩하므로 WavLM과 순서를 구분한다.
            if len(segment) < 400:
                segment = np.pad(segment, (0, 400 - len(segment)))
            value = self.processors['wav2vec2'](segment, sampling_rate=16000,
                return_tensors='pt').input_values[0]
            total, frames = np.zeros(768, dtype=np.float64), 0
            for chunk in torch.tensor_split(value, math.ceil(len(value) / 240000)):
                hidden = self.encoders['wav2vec2'](chunk[None].to(self.device)).last_hidden_state
                mean = hidden.mean(1)[0].float().cpu().numpy()
                total += mean * hidden.shape[1]
                frames += hidden.shape[1]
            embeddings.append((total / frames).astype(np.float32))
        return np.stack(embeddings).mean(0, dtype=np.float64).astype(np.float32)

    def predict(self, audio_path, intervals):
        audio, rate = sf.read(audio_path, dtype='float32', always_2d=False)
        if rate != 8000 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError(f'Expected finite 8kHz mono audio: {audio_path}')
        segments = []
        for start, end in intervals:
            a, b = round(start * rate), round(end * rate)
            if not 0 <= a < b <= len(audio):
                raise ValueError(f'Caller bounds exceed audio: {audio_path}')
            segments.append(resample_poly(audio[a:b], 2, 1, window=('kaiser', 5.0)).astype(np.float32))
        if not segments:
            raise ValueError('No caller audio')
        p1 = lr_probability(self.wavlm_features(segments), self.state['wavlm']['lr'])
        p2 = lr_probability(self.wav2vec2_features(segments), self.state['wav2vec2']['lr'])
        probability = 0.75 * p1 + 0.25 * p2
        if not np.isfinite(probability).all():
            raise ValueError('Non-finite probability')
        return self.state['classes'][int(probability.argmax())]
