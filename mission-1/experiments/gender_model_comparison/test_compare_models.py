import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf
from compare_models import windows, recover, load_wave, load_jobs, verify_bundle


class PipelineTests(unittest.TestCase):
    def test_windows_preserve_every_sample_and_weight(self):
        for n in (1, 47999, 48000, 240000, 240001, 480001):
            x = np.arange(n, dtype=np.float32)
            parts = list(windows(x))
            self.assertEqual(sum(length for _, length in parts), n)
            np.testing.assert_array_equal(np.concatenate([part[:length] for part, length in parts]), x)
            self.assertTrue(all(48000 <= len(part) <= 240000 for part, _ in parts))

    def test_resume_discards_only_incomplete_last_line(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'results.jsonl'
            good = json.dumps({'call_id': 'a'}) + '\n'
            p.write_text(good + '{"call_id":')
            self.assertEqual(set(recover(p)), {'a'})
            self.assertEqual(p.read_text(), good)
            p.write_text('broken\n' + good)
            with self.assertRaisesRegex(RuntimeError, 'interior'):
                recover(p)

    def test_crop_resample_keeps_all_selected_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            sf.write(p / 'sample.wav', np.ones(8000, dtype=np.float32) * .1, 8000)
            job = {'call_id': 'test', 'wav_relative_path': 'sample.wav',
                   'intervals': [[0, .25], [.5, 1]], 'n_segments': 2}
            result = load_wave(job, p)
            self.assertEqual(result.shape, (12000,))
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(np.isfinite(result).all())
            job['intervals'][1][1] = 2
            with self.assertRaisesRegex(RuntimeError, 'Invalid crop'):
                load_wave(job, p)

    def test_bundle_has_no_holdout_and_exact_fixtures(self):
        verify_bundle()
        jobs = load_jobs()
        self.assertEqual(len(jobs), 2000)
        self.assertEqual(sum(j['gender'] == 'M' for j in jobs), 936)
        self.assertEqual(sum(j['n_segments'] for j in jobs), 31269)
        self.assertTrue(all(not j['smoke'] or j['benchmark'] for j in jobs))


if __name__ == '__main__':
    unittest.main()
