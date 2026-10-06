"""CPU-only corruption/resume tests. Synthetic results never leave temporary directories."""
import contextlib
import copy
import io
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import validation_common as common
import ecapa_validation as runner
import analyze_ecapa_validation as analysis


def row(job, ih='test'):
    n = common.samples(job)
    return dict(call_id=job['call_id'], gender=job['gender'], identity_sha256=ih,
                prediction=job['gender'], probabilities_F_M=[.9,.1] if job['gender']=='F' else [.1,.9],
                n_segments=job['n_segments'], caller_samples_16k=n, chunks=math.ceil(n/240000),
                padded_chunks=int(n<48000), read_resample_seconds=.01, elapsed_seconds=.02)


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.jobs = common.bundle()

    def test_metadata_and_original_bundle_unchanged(self):
        self.assertEqual(sum(j['n_segments'] for j in self.jobs), 88290)
        self.assertEqual(sum(common.samples(j)<48000 for j in self.jobs), 1)
        runner.base.verify_bundle()
        self.assertFalse(set(j['call_id'] for j in self.jobs) & set(j['call_id'] for j in runner.base.load_jobs()))

    def test_invalid_cache_rejected(self):
        j = self.jobs[0]
        valid = row(j)
        for change in [dict(probabilities_F_M=[float('nan'),.5]), dict(probabilities_F_M=[[.5,.5]]),
                       dict(probabilities_F_M=[.8,.8]), dict(prediction='F' if j['gender']=='M' else 'M'),
                       dict(identity_sha256='other'), dict(caller_samples_16k=1), dict(chunks=999),
                       dict(padded_chunks=99), dict(read_resample_seconds=-1)]:
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                common.validate([{**valid, **change}], [j], 'test')
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            common.validate([valid,valid], [j], 'test')
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            common.validate([], [j], 'test', complete=True)

    def test_partial_tail_repair_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'predictions.jsonl'
            p.write_bytes(b'{"call_id":"a"}\n{"call_id":')
            self.assertEqual(common.jsonl(p, repair=True), [{'call_id':'a'}])
            self.assertEqual(len(list(Path(tmp).glob('*.interrupted-*'))), 1)
            p.write_bytes(b'broken\n')
            with self.assertRaises(RuntimeError):
                common.jsonl(p, repair=True)
            p.write_bytes(b'broken\n{"call_id":"a"}\n')
            with self.assertRaises(RuntimeError):
                common.jsonl(p, repair=True)

    def test_duplicate_run_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with runner.lock(Path(tmp)):
                with self.assertRaisesRegex(RuntimeError, 'lock'):
                    with runner.lock(Path(tmp)):
                        pass
            self.assertFalse((Path(tmp)/'.validation.lock').exists())

    def test_failed_second_call_resumes_without_repeating_first(self):
        jobs = self.jobs[:2]
        identity = {'synthetic_test':True}
        ih = common.digest(identity)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            common.save(out/'identity.json', identity)
            common.save(out/'preflight.json', {'identity_sha256':ih})
            cfg = dict(output_dir=str(out), source_output=str(out/'source'), data_root=str(out))
            fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:None))
            model = types.SimpleNamespace(parameters=lambda:[])
            calls = []
            def wave(j, root):
                calls.append(j['call_id'])
                self.assertNotIn('gender', j)
                if len(calls)==2:
                    raise RuntimeError('simulated Drive failure')
                return np.zeros(common.samples(j), dtype=np.float32)
            def predict(model, name, audio, device):
                return np.array([.6,.4]), math.ceil(len(audio)/240000), int(len(audio)<48000)
            identity['model'] = {'weight_sha256':'test'}
            ih = common.digest(identity)
            common.save(out/'identity.json', identity)
            common.save(out/'preflight.json', {'identity_sha256':ih})
            with patch.dict('sys.modules', {'torch':fake_torch}), patch.object(runner,'bundle',return_value=({},jobs)), \
                 patch.object(runner,'environment',return_value=identity), \
                 patch.object(runner.base,'load_model',return_value=(model,{'weight_sha256':'test'})), \
                 patch.object(runner.base,'load_wave',side_effect=wave), patch.object(runner.base,'predict',side_effect=predict), \
                 patch.object(runner,'export') as export, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    runner.execute(cfg,'run')
                self.assertEqual(len(common.jsonl(out/'predictions.jsonl')),1)
                self.assertFalse((out/'completion.json').exists())
                runner.execute(cfg,'run')
                export.assert_called_once()
            self.assertEqual(calls,[jobs[0]['call_id'],jobs[1]['call_id'],jobs[1]['call_id']])
            common.validate(common.jsonl(out/'predictions.jsonl'), jobs, ih, complete=True)
            self.assertEqual(len(common.jsonl(out/'failure_attempts.jsonl')),1)

    def test_full_synthetic_export_and_local_comparison(self):
        # Tests plumbing only; these are not predictions or scientific results.
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)
            fixed = dict(dtype='float32',autocast=False,tf32=False,batch_size=1,
                         pipeline='caller_concat_balanced_3_to_15s_duration_weighted_probability_v1')
            source = dict(bundle_sha256=self.manifest['original_bundle_sha256'], gpu='synthetic',python='synthetic',
                          data_root='synthetic',versions={},**fixed)
            load = dict(checkpoint='JaesungHuh/voice-gender-classifier',revision='db1222153bd60337e900be22add7af180452adc0',weight_sha256='synthetic')
            common.save(out/'source_identity.json',source)
            common.save(out/'source_ecapa_load.json',load)
            identity = dict(source, bundle_sha256=common.sha(common.ROOT/'validation_bundle_manifest.json'),
                            source_identity_sha256=common.sha(out/'source_identity.json'),
                            source_model_load_sha256=common.sha(out/'source_ecapa_load.json'), model=load)
            common.save(out/'identity.json',identity)
            common.save(out/'preflight.json',dict(identity_sha256=common.digest(identity)))
            import json
            (out/'predictions.jsonl').write_text(''.join(json.dumps(row(j,common.digest(identity)))+'\n' for j in self.jobs))
            runner.export(out,self.jobs,identity)
            import zipfile
            with zipfile.ZipFile(out/'ecapa_validation_results.zip') as z:
                z.extractall(out/'returned')
            report = analysis.analyze(out/'returned',out/'analysis')
            self.assertEqual(report['versus_baselines']['mfcc']['corrected'],276)
            self.assertEqual(report['versus_baselines']['wav2vec2']['corrected'],154)
            p = out/'returned/predictions.jsonl'
            p.write_text(p.read_text()+'{}\n')
            with self.assertRaisesRegex(RuntimeError,'hash'):
                analysis.analyze(out/'returned',out/'analysis')


if __name__ == '__main__':
    unittest.main()
