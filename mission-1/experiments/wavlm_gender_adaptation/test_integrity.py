import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from common import *


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

    def test_notebook_gpu_profiles_are_isolated(self):
        import shutil
        from types import SimpleNamespace
        nb=read(ROOT/'WavLM_Gender_Adaptation_T4.ipynb')
        setup=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and 'GPU_NAME =' in ''.join(c['source']))
        code=setup[setup.index('# Query hardware'):setup.index("print('GPU:'")]
        with tempfile.TemporaryDirectory() as td:
            work=Path(td)/'work';work.mkdir()
            shutil.copyfile(ROOT/'common.py',work/'common.py')
            outputs=[]
            for gpu in ('Tesla T4','NVIDIA L4'):
                scope=dict(Path=Path,WORK=work,SOURCE=Path(td)/'existing_l4_v1',DATA_ROOT='original_data',json=json,
                           subprocess=SimpleNamespace(check_output=lambda *a,_gpu=gpu,**kw:_gpu+'\n'))
                exec(compile(code,'notebook_gpu_setup','exec'),scope)
                config=read(scope['CONFIG']);outputs.append(scope['OUTPUT'])
                self.assertEqual(config['expected_gpu'],gpu)
                self.assertEqual(config['data_root'],'original_data')
                self.assertEqual(config['precision'],'fp32' if gpu=='Tesla T4' else 'bf16')
            self.assertNotEqual(*outputs)
        self.assertEqual((ROOT/'WavLM_Gender_Adaptation_T4.ipynb').read_bytes(),(ROOT/'WavLM_Gender_Adaptation.ipynb').read_bytes())

    def test_source_and_split(self):
        mission=ROOT.parents[1]
        for n,h in SOURCE_HASHES.items():self.assertEqual(sha(mission/n),h)
        a=inner_assign(self.jobs);b=inner_assign(list(reversed(self.jobs)))
        self.assertEqual(a,b)
        self.assertEqual(sum(v=='dev' for v in a.values()),4478)
        self.assertEqual(sum(v=='inner_train' for v in a.values()),17910)
        self.assertFalse(set(a)&{j['call_id'] for j in self.jobs if j['partition']=='internal_validation'})
        actual={r['call_id']:r['partition'] for r in rows(ROOT/'data/inner_split.csv')}
        self.assertEqual(a,actual)
        self.assertEqual(sha(ROOT/'data/inner_split.csv'),read(ROOT/'design_manifest.json')['inner_split_sha256'])

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

    def test_notebook_code_compiles(self):
        nb=read(ROOT/'WavLM_Gender_Adaptation.ipynb')
        for c in nb['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
        self.assertGreater(len([c for c in nb['cells'] if c['cell_type']=='markdown']),8)

    def test_full_return_and_tamper(self):
        from verify import verify
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);out=p/'result';out.mkdir()
            ih_record=dict(bundle_sha256=sha(ROOT/'bundle_manifest.json'),model=MODEL,revision=REVISION,source_hashes=SOURCE_HASHES,
                           inner_split_sha256=sha(ROOT/'data/inner_split.csv'),class_order=['F','M'],layers=LAYERS,window_seconds=WINDOW)
            save(out/'identity.json',ih_record);ih=digest(ih_record)
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
