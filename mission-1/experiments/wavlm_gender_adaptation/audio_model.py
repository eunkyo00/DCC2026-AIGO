"""Same crop/polyphase resampling as existing baselines; bounded WavLM contexts."""
import sys
sys.modules['torchvision'] = None
sys.modules['librosa'] = None
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, WavLMModel
from common import *


def load_wave(j, root):
    # Adapted from gender_model_comparison/compare_models.py:load_wave.
    audio,sr=sf.read(Path(root)/j['wav_relative_path'],dtype='float32',always_2d=False)
    require(sr==8000 and audio.ndim==1 and np.isfinite(audio).all(),'Expected finite 8kHz mono WAV')
    parts=[]
    for start,end in j['intervals']:
        a,b=round(start*sr),round(end*sr)
        require(0<=a<b<=len(audio),'Invalid caller bounds')
        parts.append(resample_poly(audio[a:b],2,1,window=('kaiser',5.0)).astype(np.float32))
    result=np.concatenate(parts)
    require(len(result)==samples(j),'Caller sample coverage')
    return result


def windows(x):
    # Balanced windows, no dropped tail; disjoint intervals can have boundary discontinuities.
    return np.array_split(x, math.ceil(len(x)/(WINDOW*16000)))


def crop(x, seconds, seed):
    n=int(seconds*16000)
    start=int(np.random.default_rng(seed).integers(0,max(1,len(x)-n+1)))
    return x[start:start+n], start


class Encoder(torch.nn.Module):
    def __init__(self, precision='bf16', mode='frozen'):
        super().__init__()
        self.precision=precision
        self.processor=AutoFeatureExtractor.from_pretrained(MODEL,revision=REVISION)
        self.backbone, self.loading=WavLMModel.from_pretrained(MODEL,revision=REVISION,output_loading_info=True)
        require(not self.loading['missing_keys'] and not self.loading['mismatched_keys'] and not self.loading['error_msgs'], 'Incomplete pretrained backbone load')
        self.backbone.config.apply_spec_augment=False
        self.backbone.config.layerdrop=0.0
        self.backbone.requires_grad_(False)
        if mode=='top2':
            for layer in self.backbone.encoder.layers[-2:]:layer.requires_grad_(True)
        self.backbone.eval()  # fixed dropout for controlled frozen/top2 comparison; gradients still enabled
        require(self.backbone.config.num_hidden_layers==24 and self.backbone.config.hidden_size==1024,'Unexpected backbone')
        require(self.processor.do_normalize and self.processor.sampling_rate==16000,'Normalization changed')
        self.cuda()

    def forward(self,x):
        require(len(x)>0,'Empty input')
        # Normalize real waveform first, then minimum conv-length padding. Batch size one: no temporal batching pad.
        inp=self.processor(x,sampling_rate=16000,return_tensors='pt',padding=False).input_values
        if inp.shape[1]<400: inp=torch.nn.functional.pad(inp,(0,400-inp.shape[1]))
        inp=inp.cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.precision=='bf16'):
            hs=self.backbone(inp,output_hidden_states=True).hidden_states
        return torch.stack([torch.stack([hs[k][0].float().mean(0),hs[k][0].float().square().mean(0)]) for k in LAYERS])

    def pooled(self,x):
        moments=[];weights=[]
        with torch.no_grad():
            for part in windows(x):
                moments.append(self(part));weights.append(len(part))
        w=torch.tensor(weights,device='cuda',dtype=torch.float32)
        stats=(torch.stack(moments)*w[:,None,None,None]).sum(0)/w.sum()
        return torch.stack([stats[:,0],(stats[:,1]-stats[:,0].square()).clamp_min(1e-7).sqrt()],1)

    def parameter_record(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    trainable_names=[n for n,p in self.named_parameters() if p.requires_grad],
                    frozen_names=[n for n,p in self.named_parameters() if not p.requires_grad],
                    loading=self.loading, processor=self.processor.to_dict())


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mix=torch.nn.Parameter(torch.zeros(4))
        self.norm=torch.nn.LayerNorm(2048)
        self.classifier=torch.nn.Linear(2048,2)

    def forward(self,stats):
        # stats [4,2,1024]: learned convex combination of layer mean/std.
        z=(stats*self.mix.softmax(0)[:,None,None]).sum(0).flatten()
        return self.classifier(self.norm(z))


def crop_stats(m):
    return torch.stack([m[:,0],(m[:,1]-m[:,0].square()).clamp_min(1e-7).sqrt()],1)
