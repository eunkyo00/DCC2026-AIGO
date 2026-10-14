import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

import features as f
import heads as h


def fixtures(n=60, dim=8):
    rng = np.random.default_rng(13)
    y = np.arange(n) % 2
    values = []
    for i in range(n):
        x = rng.normal(size=(1+i%4, dim)).astype(np.float32)
        x[:, 0] += (2*y[i]-1)*2
        values.append((x, np.full(len(x), 100, dtype=np.int64)))
    return values, y.astype(np.int64)


class ExperimentTests(unittest.TestCase):
    def test_balanced_windows_preserve_tail(self):
        for n in (1, 399, 192000, 192001, 480001):
            x = np.arange(n, dtype=np.float32); windows = f.split_windows(x)
            np.testing.assert_array_equal(np.concatenate(windows), x)
            self.assertLessEqual(max(map(len, windows)), 192000)
            self.assertLessEqual(max(map(len, windows))-min(map(len, windows)), 1)

    def test_attention_initially_matches_mean_and_ignores_padding(self):
        torch.manual_seed(42); mean = h.Head('mean', 8).eval()
        attention = h.Head('attention', 8).eval()
        attention.classifier.load_state_dict(mean.classifier.state_dict())
        x = torch.randn(2, 4, 8); weights = torch.tensor([[1., 2., 0., 0.], [1., 2., 3., 0.]])
        with torch.no_grad():
            a, kl = attention(x, weights)
            torch.testing.assert_close(a, mean(x, weights)[0])
            self.assertAlmostEqual(kl.item(), 0., places=6)
            x[weights == 0] = 100000
            torch.testing.assert_close(a, attention(x, weights)[0])

    def test_nested_folds_have_no_leakage(self):
        y = np.arange(120) % 2; seen = []
        for train, test, fit, stop in h.split_plan(y):
            self.assertFalse(set(train) & set(test))
            self.assertFalse(set(fit) & set(stop))
            self.assertEqual(set(fit) | set(stop), set(train))
            seen.extend(test.tolist())
        self.assertEqual(sorted(seen), list(range(120)))

    def test_checksum_and_audio_coverage(self):
        j = dict(call_id='abc', intervals=[[0., .1]])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'x.npz'
            values = [(np.zeros((1,1024), np.float32), np.array([1600], np.int64))]
            f.write_block(p, [j], values, 'test')
            f.read_block(p, [j], 'test')
            with self.assertRaisesRegex(ValueError, 'identities'): f.read_block(p, [j], 'other')
            with self.assertRaisesRegex(ValueError, 'samples'): f.read_block(p, [dict(j, intervals=[[0., .2]])], 'test')
            with np.load(p) as z: arrays = {k: z[k] for k in z.files}
            arrays['x'][0, 0] = 1.
            np.savez(p, **arrays)
            with self.assertRaisesRegex(ValueError, 'checksum'): f.read_block(p, [j], 'test')

    def test_baseline_fallback_and_attention_gate(self):
        y = np.arange(60) % 2; fold = np.arange(60) % 3
        good = np.eye(2)[y]*.8+.1
        result = h.select(y, dict(lr=good, mean=good, attention=good), fold)
        self.assertEqual(result['selected'], 'lr')
        weak = np.eye(2)[1-y]*.02+.49
        result = h.select(y, dict(lr=weak, mean=good, attention=good), fold)
        self.assertEqual(result['selected'], 'mean')
        result = h.select(y, dict(lr=weak, mean=weak, attention=good), fold)
        self.assertEqual(result['selected'], 'attention')

    def test_real_heads_train_resume_and_scaler_scope(self):
        values, y = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('mean', 'attention'):
                p = Path(tmp)/(kind+'.pt')
                result = h.train_head(values[:40], y[:40], kind, p, 'cpu', epochs=2)
                np.testing.assert_allclose(result['mean'], h.pooled(values[:40]).mean(0), atol=1e-6)
                a = h.predict(result, values[40:], 'cpu')
                self.assertTrue(np.isfinite(a).all()); h.metrics(y[40:], a)
                with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('must resume completed')):
                    again = h.train_head(values[:40], y[:40], kind, p, 'cpu', epochs=2)
                np.testing.assert_array_equal(a, h.predict(again, values[40:], 'cpu'))

    def test_outer_development_end_to_end_and_resume(self):
        values, y = fixtures(60)
        jobs = [dict(call_id=str(i), gender='FM'[y[i]]) for i in range(60)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(h, 'MAX_EPOCHS', 2):
            result = h.develop(jobs, values, Path(tmp), 'cpu')
            self.assertIn(result['selected'], ['lr', 'mean', 'attention'])
            oof = h.load(Path(tmp)/'development/oof.pt')
            self.assertEqual(len(oof['ids']), 60)
            self.assertTrue((oof['folds'] >= 0).all())
            with patch.object(h, 'train_head', side_effect=AssertionError('must resume folds')):
                self.assertEqual(h.develop(jobs, values, Path(tmp), 'cpu'), result)

    def test_extract_resume_excludes_validation_and_preserves_every_sample(self):
        jobs = [dict(call_id=f'{i:064x}', intervals=[[0., .1]], partition='train' if i < 66 else 'internal_validation') for i in range(70)]
        class FakeEncoder:
            def __init__(self): self.calls = 0
            def encode(self, x):
                self.calls += 1
                return np.zeros((1, 1024), np.float32), np.array([len(x)], np.int64)
        encoder = FakeEncoder()
        with tempfile.TemporaryDirectory() as tmp, patch.object(f, 'waveform', return_value=np.zeros(1600, np.float32)):
            out = Path(tmp)
            f.extract(jobs, 'unused', out, 'fixture', 'train', encoder)
            self.assertEqual(encoder.calls, 66)
            self.assertFalse((out/'features/internal_validation').exists())
            f.extract(jobs, 'unused', out, 'fixture', 'train', encoder)
            self.assertEqual(encoder.calls, 66)
            selected, values = f.read_partition(jobs, out, 'fixture', 'train')
            self.assertEqual(len(selected), 66)
            self.assertEqual(sum(w.sum() for _, w in values), 66*1600)

    def test_final_evaluation_hash_lock_and_export(self):
        import run as runner
        values, y = fixtures(24)
        jobs = [dict(call_id=str(i), gender='FM'[y[i]], intervals=[[0., .1]]) for i in range(24)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            f.save(out/'selection.lock.json', dict(identity='fixture', selected='lr'))
            model = h.fit_lr(values[:12], y[:12])
            h.torch_save(out/'model.pt', dict(selected=model, baseline=model))
            f.save(out/'final_models.json', dict(identity='fixture', selection_sha256=f.sha(out/'selection.lock.json'), checkpoint_sha256=f.sha(out/'model.pt')))
            def extract_after_lock(*args):
                self.assertTrue((out/'validation_started.json').exists())
            with patch.object(runner, 'extract', side_effect=extract_after_lock), patch.object(runner, 'read_partition', return_value=(jobs[12:], values[12:])):
                runner.evaluate(jobs, dict(data_root='unused'), out, 'fixture')
            result = f.read(out/'completion.json')
            self.assertEqual(result['metrics']['n'], 12)
            self.assertTrue((out/'validation_predictions.csv').exists())
            for name in ('identity.json', 'benchmark.json'): f.save(out/name, {})
            (out/'events.jsonl').write_text('{}\n')
            runner.export_results(out)
            self.assertTrue((out/'window_attention_results.zip').exists())
            f.save(out/'selection.lock.json', dict(identity='fixture', selected='attention'))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                runner.evaluate(jobs, dict(data_root='unused'), out, 'fixture')


if __name__ == '__main__': unittest.main()
