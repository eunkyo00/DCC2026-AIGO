"""CPU synthetic tests: mathematical gradients, freeze/update, exact resume, integrity rejection."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model
from common import *
from engine import Classifier,backward_call,optimizer_for,frozen_hash,group_hashes,train

torch.set_num_threads(1)


class Extractor:
    def __call__(self,x,**kwargs):
        x=torch.as_tensor(x).float()
        return SimpleNamespace(input_values=((x-x.mean())/torch.sqrt(x.var(unbiased=False)+1e-7))[None])


class Tiny(Classifier):
    def __init__(self,layers=2):
        torch.nn.Module.__init__(self)
        self.device=torch.device('cpu'); self.precision='fp32'; self.layers=layers
        config=Wav2Vec2Config(hidden_size=16,num_hidden_layers=4,num_attention_heads=2,intermediate_size=24,
            conv_dim=(8,8,8),conv_stride=(5,2,2),conv_kernel=(10,3,3),num_conv_pos_embedding_groups=2,
            num_conv_pos_embeddings=8,hidden_dropout=0.,attention_dropout=0.,activation_dropout=0.,feat_proj_dropout=0.,layerdrop=0.,mask_time_prob=0.)
        self.backbone=Wav2Vec2Model(config); self.backbone.requires_grad_(False)
        if layers:
            for layer in self.backbone.encoder.layers[-layers:]: layer.requires_grad_(True)
        self.head=torch.nn.Linear(16,2); self.extractor=Extractor(); self.eval()


class Tests(unittest.TestCase):
    def setUp(self): torch.manual_seed(42)

    def test_two_pass_exact_gradient_and_frozen_control(self):
        for layers in (0,2):
            model=Tiny(layers); ref=copy.deepcopy(model)
            segments=[[torch.randn(430),torch.randn(490)],[torch.randn(600)]]
            _,_,scaler,_=optimizer_for(model,dict(head_lr=1e-3,backbone_lr=1e-5,weight_decay=.01))
            frozen=frozen_hash(model)
            loss=backward_call(model,segments,1,4,scaler)
            means=[]
            for chunks in segments:
                values=[ref.encode(c) for c in chunks]; frames=sum(n for _,n in values)
                means.append(sum(z*(n/frames) for z,n in values))
            z=torch.stack(means).mean(0)
            direct=torch.nn.functional.cross_entropy(ref.logits(z)[None],torch.tensor([1]))/4
            direct.backward()
            self.assertAlmostEqual(loss,float(direct.detach())*4,places=6)
            for (name,p),(_,q) in zip(model.named_parameters(),ref.named_parameters()):
                if p.requires_grad:
                    self.assertIsNotNone(p.grad,name)
                    torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-7)
                else: self.assertIsNone(p.grad,name)
            self.assertEqual(frozen,frozen_hash(model))

    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); jobs=[]
            for i in range(6):
                name=f'{i}.wav'; sf.write(root/name,np.sin(np.arange(800)*(.1+i*.01)).astype(np.float32),8000)
                jobs.append(dict(call_id=str(i),gender='M' if i%2 else 'F',wav_relative_path=name,intervals=[[0.,.04],[.05,.09]]))
            cfg=dict(data_root=td,head_lr=1e-3,backbone_lr=1e-5,weight_decay=.01,seed=42,accumulation_calls=2,checkpoint_steps=1)
            initial=Tiny(); a=copy.deepcopy(initial); b=copy.deepcopy(initial)
            with patch('torch.cuda.synchronize'),patch('torch.cuda.max_memory_allocated',return_value=0),patch('torch.cuda.max_memory_reserved',return_value=0):
                train(a,jobs,cfg,root/'full','same',2)
                train(b,jobs,cfg,root/'resume','same',2,max_steps=1)
                restarted=copy.deepcopy(initial)
                train(restarted,jobs,cfg,root/'resume','same',2)
            for (n,p),(_,q) in zip(a.named_parameters(),restarted.named_parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0,msg=n)
            self.assertEqual(frozen_hash(initial),frozen_hash(a))
            self.assertTrue(all(v!=group_hashes(a)[k] for k,v in group_hashes(initial).items()))
            state=torch.load(root/'resume/last.pt',weights_only=False)
            self.assertEqual((state['epoch'],state['cursor'],state['step']),(2,0,6))
            self.assertEqual(state['sampler']['next_call'],0)

    def test_result_fail_closed(self):
        job=dict(call_id='a',gender='M',intervals=[[0.,1.],[2.,3.]])
        good=dict(call_id='a',gender='M',prediction='M',probabilities_F_M=[.2,.8],identity='id',segments=2,source_samples=16000)
        validate([good],[job],'id')
        for change in (dict(probabilities_F_M=[float('nan'),.8]),dict(probabilities_F_M=[.2,.7]),dict(prediction='F'),dict(identity='other'),dict(segments=1),dict(source_samples=8000),dict(gender='F')):
            with self.assertRaises(ValueError): validate([dict(good,**change)],[job],'id')
        with self.assertRaises(ValueError): validate([],[job],'id')
        with self.assertRaises(ValueError): validate([good,good],[job],'id')

    def test_full_export_verifier_and_tamper_rejection(self):
        import subprocess, sys
        _,jobs=bundle(); val=[j for j in jobs if j['partition']=='internal_validation']
        baselines=read(ROOT/'data/baselines.json')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); run=root/'run'; run.mkdir()
            identity=dict(bundle_manifest_sha256=sha(ROOT/'bundle_manifest.json'),split=read(ROOT/'data/split_identity.json'),config=dict(epochs=4))
            ih=digest(identity); save(run/'identity.json',identity)
            candidates=[]
            for k in (0,2):
                report=dict(identity=digest([ih,'search',k]),epochs=4,history=[dict(epoch=e,correct=4000+k,n=4478) for e in range(1,5)])
                path=run/f'search_top{k}'/'training.json'; save(path,report)
                candidates.append(dict(layers=k,epochs=1,dev_correct=4000+k,dev_n=4478,training_sha256=sha(path)))
            selection=dict(identity=ih,status='locked',candidates=candidates,control=candidates[0],finetuned=candidates[1]); save(run/'selection.json',selection)
            save(run/'benchmark.json',dict(identity=ih,status='complete',candidates=[dict(layers=k,eligible=True) for k in (0,2)]))
            save(run/'events/evaluate.json',dict(identity=ih,stage='evaluate',status='complete'))
            for arm in ('control','finetuned'):
                folder=run/f'final_{arm}'; sid=digest([ih,'refit',arm,digest(selection)]); ck='synthetic-checkpoint-hash'; eid=digest([sid,ck])
                save(folder/'evaluation_identity.json',dict(identity=eid,run_identity=ih,refit_identity=sid,checkpoint_sha256=ck,selection_sha256=sha(run/'selection.json')))
                save(folder/'training.json',dict(status='complete',identity=sid,epochs=1,trainable_groups_changed=True,frozen_sha256='frozen'))
                keys=['head']+[f'layer{i}' for i in range(selection[arm]['layers'])]
                save(folder/'update_audit.json',dict(passed=True,frozen_sha256='frozen',initial={k:'before' for k in keys},after_first_step={k:'after' for k in keys}))
                rr=[]
                for j in val:
                    pred=baselines['wav2vec2'][j['call_id']]
                    r=dict(call_id=j['call_id'],gender=j['gender'],prediction=pred,probabilities_F_M=[.8,.2] if pred=='F' else [.2,.8],identity=eid,segments=len(j['intervals']),source_samples=sum(round(b*8000)-round(a*8000) for a,b in j['intervals']))
                    save(folder/'validation_calls'/f"{j['call_id']}.json",r); rr.append(r)
                save(folder/'evaluation.json',dict(status='complete',identity=eid,expected=5597,completed=5597,pending=0,failed=0,metrics=metric(rr)))
            save(run/'export_manifest.json',dict(files={str(p.relative_to(run)):sha(p) for p in run.rglob('*.json')}))
            command=[sys.executable,str(ROOT/'verify.py'),str(run),'--output',str(root/'report')]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=read(root/'report/comparison.json')
            self.assertEqual(report['results']['finetuned']['metrics']['correct'],5443)
            self.assertEqual(report['results']['finetuned']['comparison']['wav2vec2']['both_wrong'],154)
            victim=next((run/'final_finetuned/validation_calls').glob('*.json'))
            obj=read(victim); obj['probabilities_F_M']=[.3,.7]; save(victim,obj)
            result=subprocess.run(command[:-1]+[str(root/'bad_report')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Export file hash mismatch',result.stderr)

    def test_real_bundle_partition_and_baselines(self):
        _,jobs=bundle(); inner=inner_assign(jobs)
        self.assertEqual(len(inner),22388)
        self.assertEqual(sum(v=='dev' for v in inner.values()),4478)
        self.assertFalse(set(inner)&{j['call_id'] for j in jobs if j['partition']=='internal_validation'})
        base=read(ROOT/'data/baselines.json'); truth={j['call_id']:j['gender'] for j in jobs if j['partition']=='internal_validation'}
        for name,n in [('wav2vec2',5443),('mfcc',5321),('ecapa',5419)]:
            self.assertEqual(set(base[name]),set(truth))
            self.assertEqual(sum(p==truth[c] for c,p in base[name].items()),n)

if __name__=='__main__': unittest.main()
