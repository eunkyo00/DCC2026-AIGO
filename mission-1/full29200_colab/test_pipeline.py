"""CPU acceptance tests: complete inventory, split, durable checkpoints, batching, export math."""
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WavLMConfig, WavLMModel, Wav2Vec2Config, Wav2Vec2Model
import core
import pipeline as p


class FullRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.out=self.root/'out';self.out.mkdir()

    def dataset(self,n=30):
        raw=self.root/'Training';raw.mkdir()
        for i in range(n):
            audio=np.sin(np.arange(2400)*(.03+i*.001)).astype(np.float32)*.2
            sf.write(raw/f'call{i:03d}.wav',audio,8000,subtype='PCM_16')
            (raw/f'call{i:03d}.json').write_text(json.dumps({'gender':'F' if i%2 else 'M',
                'address':'서울특별시 강남구','utterances':[{'speaker':1,'startAt':0,'endAt':250}]}))
        return raw

    def test_inventory_full_or_stop_and_seoul_filter(self):
        raw=self.dataset()
        data=p.stage([raw],self.root/'staged',lambda _:None)
        jobs=p.validate_dataset(data,self.out,lambda _:None,expected=30)
        jobs=p.split_jobs(jobs)
        self.assertEqual(sum(r['partition']=='train' for r in jobs),24)
        again=p.split_jobs([dict(j) for j in reversed(sorted(jobs,key=lambda j:j['call_id'],reverse=True))])
        self.assertEqual([(j['call_id'],j['partition']) for j in jobs],[(j['call_id'],j['partition']) for j in again])
        (data/'wav'/'call000.wav').unlink()
        with self.assertRaises(ValueError):
            p.validate_dataset(data,self.out,lambda _:None,expected=30)
        self.assertEqual(p.read(self.out/'dataset_check.json')['valid_calls'],29)
        label=data/'json'/'call001.json';doc=json.loads(label.read_text());doc['address']='부산광역시';label.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):
            p.validate_dataset(data,self.out,lambda _:None,expected=30)
        self.assertTrue(p.read(self.out/'dataset_check.json')['failures'])

    def test_zip_validation_excluded_nested_and_no_silent_corruption(self):
        raw=self.dataset(4)
        memory=io.BytesIO()
        with zipfile.ZipFile(memory,'w') as z:
            for f in raw.iterdir():z.write(f,'Training/'+f.name)
            z.writestr('Validation/never.json','bad')
            z.writestr('__MACOSX/._junk.json','bad')
        archive=self.root/'data.zip'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('Training.zip',memory.getvalue())
        data=p.stage([archive],self.root/'zip_staged',lambda _:None)
        jobs=p.validate_dataset(data,self.out,lambda _:None,expected=4)
        self.assertEqual(len(jobs),4)
        # Broken labels cause failure, not partial training.
        (data/'json'/'call001.json').write_bytes(b'\xff\xff')
        with self.assertRaises(ValueError):p.validate_dataset(data,self.out,lambda _:None,expected=4)

    def test_checkpoint_identity_checksum_and_interrupted_save(self):
        path=p.block_path(self.out,0);ids=np.array(['a','b']);ident={'test':1}
        x={'wavlm':np.ones((2,1024),np.float32),'wav2vec2':np.ones((2,768),np.float32)}
        p.save_block(path,ids,ident,x)
        self.assertTrue(np.array_equal(p.load_block(path,ids,ident)['wavlm'],x['wavlm']))
        with self.assertRaises(ValueError):p.load_block(path,ids,{'test':2})
        path.with_suffix('.json').unlink()
        self.assertIsNone(p.load_block(path,ids,ident))
        p.save_block(path,ids,ident,x)
        path.write_bytes(path.read_bytes()+b'x')
        with self.assertRaises(ValueError):p.load_block(path,ids,ident)

    def test_training_oof_and_final_full_refit(self):
        raw=self.dataset()
        data=p.stage([raw],self.root/'staged',lambda _:None)
        jobs=p.split_jobs(p.validate_dataset(data,self.out,lambda _:None,expected=30))
        rng=np.random.default_rng(42)
        features={k:rng.normal(size=(30,d)).astype(np.float32) for k,d in [('wavlm',1024),('wav2vec2',768)]}
        for x in features.values():
            x[:,0]=np.array([5 if j['gender']=='M' else -5 for j in jobs])
        models,metrics=p.train(jobs,features,self.out,{'test':True},lambda _:None)
        self.assertEqual(metrics['train_calls'],24)
        self.assertEqual(metrics['internal_validation_calls'],6)
        self.assertEqual(metrics['final_refit_calls'],30)
        self.assertTrue((self.out/'evaluation_classifiers.joblib').exists())
        self.assertTrue((self.out/'classifiers.joblib').exists())
        for kind in core.MODELS:
            state=p.lr_state(models[kind])
            self.assertTrue(np.allclose(core.probability(features[kind],state),models[kind].predict_proba(features[kind]),atol=1e-6))
            scaler=models[kind].steps[0][1]
            self.assertEqual(int(scaler.n_samples_seen_),30)
        models2,metrics2=p.train(jobs,features,self.out,{'test':True},lambda _:None)
        self.assertEqual(metrics,metrics2)

    def tiny_encoder(self,kind):
        opts=dict(hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,
                  conv_dim=(8,8,8),conv_stride=(5,2,2),conv_kernel=(10,3,3),
                  num_conv_pos_embeddings=16,num_conv_pos_embedding_groups=2,
                  hidden_dropout=0.,attention_dropout=0.,feat_proj_dropout=0.,
                  layerdrop=0.,mask_time_prob=0.,mask_feature_prob=0.)
        cls,model=(WavLMConfig,WavLMModel) if kind=='wavlm' else (Wav2Vec2Config,Wav2Vec2Model)
        c=cls(**opts);torch.manual_seed(42);m=model(c)
        return core.Encoder(kind,'fp32',entry={'config':c.to_dict(),'weights':m.state_dict()},device='cpu')

    def test_real_tiny_transformers_batch_independence_and_reload(self):
        core.configure_torch()
        rng=np.random.default_rng(7)
        samples=[[rng.normal(size=800).astype(np.float32),rng.normal(size=1200).astype(np.float32)],
                 [rng.normal(size=600).astype(np.float32)]]
        for kind in core.MODELS:
            e=self.tiny_encoder(kind)
            batch=e.encode(samples,2.)
            singles=np.concatenate([e.encode([s],.5) for s in samples])
            self.assertTrue(np.allclose(batch,singles,rtol=2e-4,atol=2e-5),kind)
            path=self.root/(kind+'.pt');torch.save(e.entry(),path)
            entry=torch.load(path,map_location='cpu',weights_only=True)
            e2=core.Encoder(kind,'fp32',entry=entry,device='cpu')
            self.assertTrue(np.allclose(batch,e2.encode(samples,2.),atol=1e-6))

    def test_oom_reduces_batch_without_changing_buckets(self):
        e=self.tiny_encoder('wav2vec2')
        original=e._forward_once
        def limited(batch,size):
            if len(batch)>1:raise torch.cuda.OutOfMemoryError('synthetic')
            return original(batch,size)
        records=[(np.ones(500,np.float32),0,0,1),(np.zeros(600,np.float32),1,0,1)]
        expected=original(records,8000)
        e._forward_once=limited
        actual=e._forward(records,8000)
        self.assertTrue(np.allclose(expected[0],actual[0],atol=1e-5))

    def test_extract_interruption_resumes_committed_blocks(self):
        raw=self.dataset(100)
        data=p.stage([raw],self.root/'staged',lambda _:None)
        jobs=p.split_jobs(p.validate_dataset(data,self.out,lambda _:None,expected=100))
        identity={'precision':'fp32','batch_seconds':{'wavlm':2,'wav2vec2':2}}
        calls=[]
        class FakeEncoder:
            fail=True
            def __init__(self,kind,*args,**kwargs):self.kind=kind
            def encode(self,prepared,batch_seconds):
                calls.append((self.kind,len(prepared)))
                if self.fail and len(prepared)==36:raise RuntimeError('simulated disconnect')
                return np.ones((len(prepared),1024 if self.kind=='wavlm' else 768),np.float32)
        with patch.object(core,'Encoder',FakeEncoder), patch.object(torch.cuda,'is_available',return_value=True), patch.object(torch.cuda,'get_device_name',return_value='CPU test double'):
            with self.assertRaisesRegex(RuntimeError,'disconnect'):
                p.extract(jobs,self.out,identity,lambda _:None)
            self.assertEqual(p.read(self.out/'progress.json')['completed_calls'],64)
            FakeEncoder.fail=False;calls.clear()
            result=p.extract(jobs,self.out,identity,lambda _:None)
        self.assertEqual(calls,[('wavlm',36),('wav2vec2',36)])
        self.assertEqual(result['wavlm'].shape,(100,1024))
        self.assertEqual(p.read(self.out/'progress.json')['resumed_calls'],64)

    def test_signal_bounds_and_nonempty(self):
        raw=self.dataset(2)
        with self.assertRaises(ValueError):core.read_segments(raw/'call000.wav',[(0,10000)])
        with self.assertRaises(ValueError):core.intervals({'utterances':[]})
        for kind in core.MODELS:
            for x,_,_ in core.units([np.ones(2,np.float32)],kind):
                self.assertGreaterEqual(len(x),400)
                self.assertTrue(np.isfinite(x).all())


if __name__=='__main__':unittest.main(verbosity=2)
