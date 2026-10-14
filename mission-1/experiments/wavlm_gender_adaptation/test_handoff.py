"""Relocate the distribution and run CPU regression checks without the parent repository."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent


class HandoffTests(unittest.TestCase):
    def test_relocated_bundle_paths_data_checks_and_existing_cpu_tests(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'relocated-code';root.mkdir()
            with zipfile.ZipFile(HERE/'artifacts/wavlm_teammate_bundle.zip') as z:z.extractall(root)
            # Reuse actual CPU model/checkpoint tests against the portable modules.
            (root/'test_cpu_model.py').write_bytes((HERE/'test_cpu_model.py').read_bytes())
            # Existing synthetic 5597-row verifier fixture, extended only for new dataset provenance.
            s=(HERE/'test_integrity.py').read_text()
            s=s.replace("ih_record=dict(bundle_sha256=", "ih_record=dict(config={'data_root':'synthetic-fixture'},bundle_sha256=")
            s=s.replace("save(out/'identity.json',ih_record);ih=digest(ih_record)", "save(out/'identity.json',ih_record);ih=digest(ih_record)\n            save(out/'dataset_check.json',dict(status='pass',calls=27985,failures=[],data_root='synthetic-fixture',bundle_sha256=ih_record['bundle_sha256']))")
            (root/'test_return.py').write_text(s)
            (root/'test_portable.py').write_text(PORTABLE_TESTS)
            result=subprocess.run([sys.executable,'-m','unittest','-v','test_portable','test_cpu_model',
                  'test_return.IntegrityTests.test_full_return_and_tamper',
                  'test_return.IntegrityTests.test_reject_incomplete_duplicate_nonfinite_argmax_audio'],
                  cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            print(result.stdout+result.stderr)

    def test_notebook_and_archive(self):
        import hashlib
        p=HERE/'handoff/WavLM_Teammate.ipynb';nb=json.loads(p.read_text())
        source=''.join(''.join(c['source']) for c in nb['cells'])
        h=hashlib.sha256((HERE/'artifacts/wavlm_teammate_bundle.zip').read_bytes()).hexdigest()
        self.assertIn(h,source)
        for key in ('BUNDLE_PATH','WORK_DIR','BASE_PYTHON','VENV_DIR','DATA_ROOT','OUTPUT_DIR','HF_CACHE_DIR','GPU_LOCK_PATH','EXPECTED_GPU','PRECISION','RETURNED_ZIP','VERIFIED_OUTPUT_DIR'):
            self.assertIn(key+' = ""',source)
        self.assertNotIn('/content/',source);self.assertNotIn('/Users/',source)
        for c in nb['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'handoff-notebook','exec')
        for name in ('runner.py','portable.py','verify.py'):
            s=(HERE/'handoff'/name).read_text()
            self.assertNotIn('/content/',s);self.assertNotIn('/Users/',s);self.assertNotIn("config['source_output']",s)
        self.assertEqual(json.loads((HERE/'handoff/config.template.json').read_text())['data_root'],'')


PORTABLE_TESTS=r'''
import json, tempfile, wave, unittest
from pathlib import Path
from unittest.mock import patch
import portable
from common import ROOT,read,bundle

class PortableTests(unittest.TestCase):
    def config(self,t):
        root=Path(t).resolve();data=root/'data';(data/'Training').mkdir(parents=True)
        (root/'shared').mkdir()
        return dict(data_root=str(data),output_dir=str(root/'results'),hf_cache_dir=str(root/'hf'),
                    gpu_lock_path=str(root/'shared/gpu.lock'),expected_gpu='Tesla T4',precision='fp32')
    def test_empty_and_relative_paths_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as td:
            c=self.config(td)
            portable.validate_paths(c)
            for key in portable.PATH_FIELDS:
                for value in ('','relative-folder'):
                    bad=dict(c);bad[key]=value
                    with self.assertRaises(RuntimeError):portable.validate_paths(bad)
            bad=dict(c);bad['output_dir']=str(Path(c['data_root'])/'overwrite')
            with self.assertRaises(RuntimeError):portable.validate_paths(bad)
            self.assertFalse(Path(c['output_dir']).exists())
    def test_header_checks_and_failure_are_durable(self):
        with tempfile.TemporaryDirectory() as td:
            c=self.config(td);p=Path(c['data_root'])/'Training/a.wav'
            with wave.open(str(p),'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(8000);f.writeframes(bytes(160))
            j=dict(call_id='a',wav_relative_path='Training/a.wav',intervals=[[0,.01]])
            inv=[dict(call_id='a',wav_relative_path='Training/a.wav',size_bytes=str(p.stat().st_size),frames='80')]
            with patch('portable.bundle',return_value=({},[j])),patch('portable.rows',return_value=inv):
                self.assertEqual(portable.check_dataset(c)['status'],'pass')
                p.unlink()
                with self.assertRaises(RuntimeError):portable.check_dataset(c)
                self.assertEqual(read(Path(c['output_dir'])/'dataset_check.json')['status'],'failed')
    def test_package_and_baseline_are_self_contained(self):
        m,jobs=bundle();self.assertEqual(len(jobs),27985)
        self.assertTrue((ROOT/'data/baseline_predictions.csv').is_file())
        self.assertTrue(all(not Path(j['wav_relative_path']).is_absolute() for j in jobs))
'''


if __name__=='__main__':unittest.main()
