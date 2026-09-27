import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import run
from common import append, save


class LaunchTests(unittest.TestCase):
    def test_blank_config_fails_before_environment(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'config.py').write_text('INPUT_ZIP=""\n')
            with patch.object(run,'ROOT',root),self.assertRaisesRegex(RuntimeError,'Fill INPUT_ZIP'):
                run.load_settings()

    def test_input_archive_integrity_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);zpath=root/'inputs.zip';body=b'{"synthetic": true}'
            with zipfile.ZipFile(zpath,'w') as z:z.writestr('data/fixture.json',body)
            save(root/'metadata_manifest.json',dict(archive_sha256=run.sha(zpath),files={'data/fixture.json':hashlib.sha256(body).hexdigest()}))
            with patch.object(run,'ROOT',root),patch.object(run,'bundle'):
                run.prepare_metadata(zpath);run.prepare_metadata(zpath)
                self.assertEqual((root/'data/fixture.json').read_bytes(),body)
                (root/'data/fixture.json').write_bytes(b'corrupt')
                with self.assertRaisesRegex(RuntimeError,'Existing metadata changed'):run.prepare_metadata(zpath)

    def test_full_command_sequence_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);stages=[];verified=[]
            config=dict(output_dir=str(out),hf_cache_dir=str(out/'cache'))
            def fake_command(argv,**kwargs):
                args=list(map(str,argv))
                if str(run.ROOT/'runner.py') in args:
                    stage=args[args.index(str(run.ROOT/'runner.py'))+1];stages.append(stage)
                    append(out/'events.jsonl',dict(stage=stage,status='complete'))
                    if stage=='benchmark':save(out/'benchmark.json',dict(synthetic=True))
                if str(run.ROOT/'verify.py') in args:verified.append(True)
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
                if str(run.ROOT/'runner.py') in args:
                    stage=args[args.index(str(run.ROOT/'runner.py'))+1];stages.append(stage)
                    append(out/'events.jsonl',dict(stage=stage,status='complete'))
                    if stage=='benchmark':save(out/'benchmark.json',{})
                return ''
            with patch.object(run,'command',fake_command):
                run.pipeline('python','config',dict(output_dir=str(out),hf_cache_dir=str(out/'hf')),True)
            self.assertEqual(stages,['preflight','benchmark'])


if __name__=='__main__':unittest.main()
