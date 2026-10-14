"""Small synthetic CPU checks; no pretrained downloads or real performance claims."""
import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
try:
    import torch
    import numpy as np
    import soundfile as sf
    import audio_model as am
    from transformers import WavLMConfig, WavLMModel, Wav2Vec2FeatureExtractor
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'Optional CPU torch/transformers/scipy environment required')
class CPUModelTests(unittest.TestCase):
    def test_crop_resample_preserves_all_samples_and_window_tail(self):
        from common import samples,WINDOW
        with tempfile.TemporaryDirectory() as td:
            x=np.sin(np.arange(80000)*.1).astype(np.float32)
            sf.write(Path(td)/'x.wav',x,8000,subtype='FLOAT')
            j=dict(call_id='a',wav_relative_path='x.wav',intervals=[[.000125,1.5],[3.,4.], [3.5,9.999875]])
            y=am.load_wave(j,td);self.assertEqual(len(y),samples(j))
            z=np.arange(WINDOW*16000*2+1,dtype=np.float32)
            parts=am.windows(z);np.testing.assert_array_equal(np.concatenate(parts),z)
            self.assertLessEqual(max(map(len,parts)),WINDOW*16000)
            a,start=am.crop(z,8,42);b,other=am.crop(z,8,42)
            self.assertEqual(start,other);np.testing.assert_array_equal(a,b)

    def tiny_encoder(self,mode):
        # Real HF WavLM implementation with reduced width, 24 layers for actual selected-layer code.
        cfg=WavLMConfig(hidden_size=16,num_hidden_layers=24,num_attention_heads=2,intermediate_size=24,
            conv_dim=(8,8,8),conv_stride=(5,4,4),conv_kernel=(10,8,8),num_conv_pos_embedding_groups=2,
            num_conv_pos_embeddings=8,feat_extract_norm='layer',do_stable_layer_norm=True,
            apply_spec_augment=False,layerdrop=0.,hidden_dropout=0.,attention_dropout=0.,activation_dropout=0.)
        e=am.Encoder.__new__(am.Encoder);torch.nn.Module.__init__(e)
        e.precision='fp32';e.processor=Wav2Vec2FeatureExtractor(do_normalize=True,sampling_rate=16000)
        e.backbone=WavLMModel(cfg).eval();e.backbone.requires_grad_(False)
        if mode=='top2':
            for l in e.backbone.encoder.layers[-2:]:l.requires_grad_(True)
        return e

    def test_real_wavlm_upper_layers_receive_gradients(self):
        torch.set_num_threads(1)
        enc=self.tiny_encoder('top2')
        with patch.object(torch.Tensor,'cuda',lambda t,*a,**k:t):
            stats=am.crop_stats(enc(np.random.default_rng(42).normal(size=1600).astype(np.float32)))
            self.assertEqual(stats.shape,(4,2,16));self.assertTrue(torch.isfinite(stats).all())
            (stats**2).sum().backward()
        gradients={n:p.grad for n,p in enc.named_parameters() if p.grad is not None}
        self.assertTrue(gradients)
        self.assertTrue(all('encoder.layers.22.' in n or 'encoder.layers.23.' in n for n in gradients))
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients.values()))
        self.assertTrue(any(g.abs().sum()>0 for g in gradients.values()))

    def test_checkpoint_resume_rng_optimizer_sampler(self):
        import random
        import runner
        torch.manual_seed(42);np.random.seed(42);random.seed(42)
        enc=torch.nn.Linear(2,2);head=torch.nn.Linear(2,2)
        opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters()),lr=.01)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1-s/20)
        scaler=torch.amp.GradScaler('cpu',enabled=False)
        def step():
            opt.zero_grad();x=torch.randn(3,2)+np.random.random()+random.random()
            head(enc(x)).square().sum().backward();opt.step();sched.step()
        step()
        with tempfile.TemporaryDirectory() as td, patch('torch.cuda.get_rng_state_all',return_value=[]),patch('torch.cuda.set_rng_state_all'):
            p=Path(td)/'checkpoint.pt';state=dict(epoch=1,next_batch=2,step=7,sampler_seed=42,order_sha256='test')
            runner.checkpoint(p,enc,head,opt,sched,scaler,state,'identity')
            step();expected={n:t.detach().clone() for n,t in enc.named_parameters()}
            restored=runner.restore(p,enc,head,opt,sched,scaler,'identity');self.assertEqual(restored,state)
            step()
            for n,t in enc.named_parameters():torch.testing.assert_close(t,expected[n],rtol=0,atol=0)
            with self.assertRaises(RuntimeError):runner.restore(p,enc,head,ih='other')

    def test_interrupted_dev_resumes_without_repeating_committed_training(self):
        import runner
        from common import read,journal,digest
        class FakeEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.randn(4,1024)*.01)
            def forward(self,x):
                mean=self.w*float(np.mean(x))
                return torch.stack([mean,mean.square()+.2],1)
            def pooled(self,x):return am.crop_stats(self(x))
            def parameter_record(self):return {'synthetic':True}
        def make_train(mode,precision):
            runner.seed_all();e=FakeEncoder();h=am.Head()
            o=torch.optim.AdamW(list(e.parameters())+list(h.parameters()),lr=.001)
            return e,h,o,torch.amp.GradScaler('cpu',enabled=False)
        jobs=[dict(call_id=f'id{i:03}',gender='FM'[i%2],partition='train',intervals=[[0,.005]]) for i in range(20)]
        ident=dict(config=dict(accumulation=4,crop_seconds=12,data_root='synthetic'),precision='fp32')
        original_tensor=torch.tensor
        def cpu_tensor(*a,**k):
            if k.get('device')=='cuda':k['device']='cpu'
            return original_tensor(*a,**k)
        def wave(*a):return np.ones(80,dtype=np.float32)
        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            root=Path(td);broken=root/'interrupted';broken.mkdir();clean=root/'clean';clean.mkdir()
            for target,value in [('runner.make_train',make_train),('audio_model.load_wave',wave),
                                 ('torch.tensor',cpu_tensor),('runner.memory',lambda *_:{}),
                                 ('torch.cuda.synchronize',lambda:None),('torch.cuda.get_rng_state_all',lambda:[]),
                                 ('torch.cuda.set_rng_state_all',lambda _:None)]:stack.enter_context(patch(target,value))
            # Interrupt the first dev pass after some durable predictions have been appended.
            real_eval=runner.evaluate_neural
            def interrupted_eval(*args,**kwargs):
                count=[0]
                def fail_read(*a):
                    count[0]+=1
                    if count[0]==3:raise RuntimeError('synthetic mid-dev interruption')
                    return wave()
                with patch('audio_model.load_wave',fail_read):return real_eval(*args,**kwargs)
            with patch('runner.evaluate_neural',interrupted_eval):
                with self.assertRaisesRegex(RuntimeError,'synthetic'):runner.training(broken,ident,jobs,'top2')
            before=journal(broken/'develop_top2/steps.jsonl')
            self.assertEqual(len(before),4)
            runner.training(broken,ident,jobs,'top2')
            runner.training(clean,ident,jobs,'top2')
            after=journal(broken/'develop_top2/steps.jsonl')
            self.assertEqual([r['step'] for r in after],list(range(1,17)))
            a=torch.load(broken/'develop_top2/latest.pt',weights_only=False)
            b=torch.load(clean/'develop_top2/latest.pt',weights_only=False)
            for part in ('model','head'):
                for name in a[part]:torch.testing.assert_close(a[part][name],b[part][name],atol=0,rtol=0)
            self.assertEqual(read(broken/'develop_top2/completion.json')['steps'],16)

    def test_cache_tamper_rejected(self):
        import runner,io
        from common import atomic,save,sha,samples
        j=dict(call_id='abc',intervals=[[0,1]])
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);p=runner.cache_path(out,j);p.parent.mkdir(parents=True)
            buf=io.BytesIO();np.savez(buf,features=np.zeros((4,2,1024),np.float32),identity='ih',call_id='abc',samples=samples(j))
            atomic(p,buf.getvalue());save(p.with_suffix('.hash.json'),dict(sha256=sha(p)))
            self.assertEqual(runner.cached(out,j,'ih').shape,(4,2,1024))
            with open(p,'ab') as f:f.write(b'x')
            with self.assertRaises(RuntimeError):runner.cached(out,j,'ih')


if __name__=='__main__':unittest.main()
