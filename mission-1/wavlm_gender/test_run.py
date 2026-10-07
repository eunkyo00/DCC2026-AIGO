# 검증용 파일입니다. 실행할 때는 run.py만 사용하세요.
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import run
from run import append, save


class LaunchTests(unittest.TestCase):
    def test_blank_config_fails_before_environment(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with patch.object(run,'INPUT_JSON',''),self.assertRaisesRegex(RuntimeError,'Fill INPUT_JSON'):
                run.load_settings()

    def test_code_identity_allows_paths_but_detects_code_changes(self):
        source=Path(run.__file__).read_text()
        before=run.code_identity()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'run.py'
            changed=source.replace('INPUT_JSON = ""', 'INPUT_JSON = "/실제 경로/inputs.json"',1)
            path.write_text(changed)
            with patch.object(run,'__file__',str(path)):
                self.assertEqual(run.code_identity(),before)
                path.write_text(changed.replace('WINDOW = 12','WINDOW = 8',1))
                self.assertNotEqual(run.code_identity(),before)

    def test_environment_installs_embedded_requirements_without_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);envdir=out/'.venv';envdir.mkdir()
            hardware=dict(torch='synthetic')
            save(envdir/'.wavlm_environment.json',dict(base_python=str(Path(run.sys.executable).resolve()),torch='synthetic',requirements_sha256=run.digest(run.REQUIREMENTS)))
            calls=[];checks=[0]
            def command(args,**kwargs):
                calls.append(list(map(str,args)))
                if '-c' in args:
                    checks[0]+=1
                    if checks[0]==1:raise RuntimeError('missing dependency')
                return ''
            with patch.object(run,'command',command),patch.object(run,'probe',return_value=hardware):
                run.environment(out,hardware)
            install=next(c for c in calls if 'pip' in c)
            self.assertEqual(install[5:],run.REQUIREMENTS)
            self.assertFalse(any('requirements.txt' in a for c in calls for a in c))
            self.assertFalse(any(a.startswith(('torch==','torchaudio==')) for a in install))

    def test_input_json_integrity_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);path=root/'inputs.json';body='{"synthetic": true}'
            save(path,dict(format='wavlm-inputs-v1',files={'data/fixture.json':body}))
            manifest=dict(input_json_sha256=run.sha(path),files={'data/fixture.json':hashlib.sha256(body.encode()).hexdigest()})
            with patch.object(run,'ROOT',root),patch.object(run,'bundle'),patch.object(run,'runtime_manifest',return_value=manifest):
                run.prepare_metadata(path);run.prepare_metadata(path)
                self.assertEqual((root/'data/fixture.json').read_text(),body)
                (root/'data/fixture.json').write_bytes(b'corrupt')
                with self.assertRaisesRegex(RuntimeError,'Existing metadata changed'):run.prepare_metadata(path)
                path.write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'Wrong input JSON'):run.prepare_metadata(path)

    def test_full_command_sequence_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);stages=[];verified=[]
            config=dict(output_dir=str(out),hf_cache_dir=str(out/'cache'))
            def fake_command(argv,**kwargs):
                args=list(map(str,argv))
                if '--worker' in args and args[args.index('--worker')+1]!='check-data':
                    stage=args[args.index('--worker')+1];stages.append(stage)
                    append(out/'events.jsonl',dict(stage=stage,status='complete'))
                    if stage=='benchmark':save(out/'benchmark.json',dict(synthetic=True))
                if '--verify' in args:verified.append(True)
                return ''
            with patch.object(run,'command',fake_command):
                run.pipeline('python','config',config)
                self.assertEqual(stages,['preflight','benchmark','extract','frozen-select','lock-frozen','final-fit','validation'])
                stages.clear();run.pipeline('python','config',config)
                self.assertEqual(stages,['preflight','validation'])
                self.assertEqual(len(verified),2)

    def test_benchmark_only_does_not_start_full_training(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);stages=[]
            def fake_command(argv,**kwargs):
                args=list(map(str,argv))
                if '--worker' in args and args[args.index('--worker')+1]!='check-data':
                    stage=args[args.index('--worker')+1];stages.append(stage)
                    append(out/'events.jsonl',dict(stage=stage,status='complete'))
                    if stage=='benchmark':save(out/'benchmark.json',{})
                return ''
            with patch.object(run,'command',fake_command):
                run.pipeline('python','config',dict(output_dir=str(out),hf_cache_dir=str(out/'hf')),True)
            self.assertEqual(stages,['preflight','benchmark'])




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
    import run as am
    am.load_audio()
    from transformers import WavLMConfig, WavLMModel, Wav2Vec2FeatureExtractor
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'Optional CPU torch/transformers/scipy environment required')
class CPUModelTests(unittest.TestCase):
    def test_crop_resample_preserves_all_samples_and_window_tail(self):
        from run import samples,WINDOW
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
        import run as runner
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
        import run as runner
        from run import read,journal,digest
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
            for target,value in [('run.make_train',make_train),('run.load_wave',wave),
                                 ('torch.tensor',cpu_tensor),('run.memory',lambda *_:{}),
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
                with patch('run.load_wave',fail_read):return real_eval(*args,**kwargs)
            with patch('run.evaluate_neural',interrupted_eval):
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
        import run as runner,io
        from run import atomic,save,sha,samples
        j=dict(call_id='abc',intervals=[[0,1]])
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);p=runner.cache_path(out,j);p.parent.mkdir(parents=True)
            buf=io.BytesIO();np.savez(buf,features=np.zeros((4,2,1024),np.float32),identity='ih',call_id='abc',samples=samples(j))
            atomic(p,buf.getvalue());save(p.with_suffix('.hash.json'),dict(sha256=sha(p)))
            self.assertEqual(runner.cached(out,j,'ih').shape,(4,2,1024))
            with open(p,'ab') as f:f.write(b'x')
            with self.assertRaises(RuntimeError):runner.cached(out,j,'ih')




import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from run import *


@unittest.skipUnless((run.ROOT/'data/jobs.json').exists(), 'Prepare the supplied input JSON before metadata tests')
class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.jobs=bundle()

    def test_t4_and_l4_precision_guards(self):
        for name in ('T4','Tesla T4','NVIDIA T4'):
            self.assertEqual(gpu_profile(name)['precision'],'fp32')
            self.assertEqual(gpu_profile(name)['output_tag'],'t4_fp32_v2')
            with self.assertRaisesRegex(RuntimeError,'T4 requires FP32'):
                gpu_profile(name,'bf16')
        self.assertEqual(gpu_profile('NVIDIA L4')['precision'],'bf16')
        self.assertEqual(gpu_profile('L4','fp32')['precision'],'fp32')
        for name in ('CPU','NVIDIA A100',''):
            with self.assertRaises(RuntimeError):gpu_profile(name)



    def prediction(self,j):
        return dict(call_id=j['call_id'],gender=j['gender'],probabilities_F_M=[.5,.5],prediction='F',identity='test',
                    samples=samples(j),windows=math.ceil(samples(j)/(WINDOW*16000)))

    def test_reject_incomplete_duplicate_nonfinite_argmax_audio(self):
        jj=self.jobs[:2];rr=[self.prediction(j) for j in jj]
        check_predictions(rr,jj,'test')
        for bad in [rr[:1],rr+[rr[0]],rr+[dict(rr[0],call_id='unknown')]]:
            with self.assertRaises(RuntimeError):check_predictions(bad,jj,'test')
        for change in [dict(probabilities_F_M=[float('nan'),0]),dict(probabilities_F_M=[float('inf'),0]),
                       dict(probabilities_F_M=[.2,.2]),dict(prediction='M'),dict(samples=1),dict(windows=0),dict(identity='other')]:
            bad=copy.deepcopy(rr);bad[0].update(change)
            with self.assertRaises(RuntimeError):check_predictions(bad,jj,'test')

    def test_lock_and_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'lock'
            with lock(p):
                with self.assertRaises(FileExistsError):
                    with lock(p):pass
            self.assertFalse(p.exists())
            p=Path(td)/'rows';append(p,dict(call_id='a'));self.assertEqual(len(journal(p)),1)
            with open(p,'ab') as f:f.write(b'{"call_id":')
            with self.assertRaises(RuntimeError):journal(p)


    def test_full_return_and_tamper(self):
        from run import verify
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);out=p/'result';out.mkdir()
            ih_record=dict(config={'data_root':'synthetic-fixture'},bundle_sha256=bundle_identity(),model=MODEL,revision=REVISION,source_hashes=SOURCE_HASHES,
                           inner_split_sha256=sha(ROOT/'data/inner_split.csv'),class_order=['F','M'],layers=LAYERS,window_seconds=WINDOW)
            save(out/'identity.json',ih_record);ih=digest(ih_record)
            save(out/'dataset_check.json',dict(status='pass',calls=27985,failures=[],data_root='synthetic-fixture',bundle_sha256=ih_record['bundle_sha256']))
            save(out/'preflight.json',dict(identity=ih,status='pass',initial_files={'synthetic':'not-a-real-model'}))
            save(out/'initial_parameters.json',{})
            save(out/'benchmark.json',dict(identity=ih,**{m:dict(optimizer_steps=100,evaluation_sample_calls=60) for m in ('frozen','top2')}))
            save(out/'frozen_selection.json',dict(identity=ih))
            save(out/'selection.lock.json',dict(identity=ih,adaptation=None))
            save(out/'final_models.json',dict(identity=ih,selection_sha256=sha(out/'selection.lock.json'),models={'frozen_lr':dict(sha256='fake')}))
            save(out/'validation_started.json',dict(identity=ih,final_models_sha256=sha(out/'final_models.json')))
            pi=digest(dict(run=ih,model='fake',tag='validation_frozen_lr'))
            val=[j for j in self.jobs if j['partition']=='internal_validation']
            rr=[dict(self.prediction(j),identity=pi) for j in val]
            path=out/'validation_frozen_lr.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rr))
            save(out/'completion.json',dict(identity=ih,status='complete',models={'frozen_lr':dict(prediction_identity=pi,path=path.name,sha256=sha(path),metrics=metrics(rr))}))
            append(out/'events.jsonl',dict(stage='validation',status='complete'))
            def hashes():save(out/'export_manifest.json',{q.name:sha(q) for q in out.iterdir() if q.name!='export_manifest.json'})
            hashes();result=verify(out,p/'verified');self.assertEqual(result['models']['frozen_lr']['metrics']['calls'],5597)
            path.write_text(path.read_text().replace('"prediction": "F"','"prediction": "M"',1));hashes()
            with self.assertRaises(RuntimeError):verify(out,p/'bad')
            path.write_text(''.join(json.dumps(r)+'\n' for r in rr));append(out/'events.jsonl',dict(stage='validation',status='failed'));hashes()
            with self.assertRaises(RuntimeError):verify(out,p/'bad')




if __name__=='__main__':unittest.main()
