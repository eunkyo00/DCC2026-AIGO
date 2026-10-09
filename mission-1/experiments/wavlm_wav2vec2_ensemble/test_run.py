import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

spec = importlib.util.spec_from_file_location('ensemble', Path(__file__).with_name('run.py'))
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


class EnsembleTests(unittest.TestCase):
    def test_conservative_fallback(self):
        y = np.array(['F', 'M'] * 30)
        p = np.eye(2)[(y == 'M').astype(int)] * .8 + .1
        result = e.choose(y, p, p, np.arange(len(y)) % 3)
        self.assertEqual(result['wavlm_weight'], 1)

    def test_consistent_improvement(self):
        y = np.array(['F', 'M'] * 30)
        p1 = np.eye(2)[(y != 'M').astype(int)] * .02 + .49
        p2 = np.eye(2)[(y == 'M').astype(int)] * .8 + .1
        result = e.choose(y, p1, p2, np.arange(len(y)) % 3)
        self.assertEqual(result['wavlm_weight'], .75)

    def test_one_bad_fold_rejected(self):
        y = np.array(['F', 'M'] * 30); folds = np.arange(60) % 3
        good = np.eye(2)[(y == 'M').astype(int)] * .8 + .1
        a = good.copy(); b = good.copy()
        a[folds != 0] = .51 - .02 * (good[folds != 0] > .5)
        a[folds == 0] = .49 + .02 * (good[folds == 0] > .5)
        b[folds == 0] = 1 - good[folds == 0]
        self.assertEqual(e.choose(y, a, b, folds)['wavlm_weight'], 1)

    def test_oof_fit_never_sees_dev_and_resume(self):
        y = np.array(['F', 'M'] * 30); ids = np.array([str(i) for i in range(60)])
        x = np.arange(60, dtype=np.float32)[:, None]
        calls = []
        class Model:
            def __init__(self, seen): self.seen = set(seen[:, 0])
            def predict_proba(self, dev):
                self_outer.assertFalse(self.seen & set(dev[:, 0]))
                return np.tile([.4, .6], (len(dev), 1))
        self_outer = self
        def fake_fit(x, y, c):
            calls.append(len(y)); return Model(x)
        with tempfile.TemporaryDirectory() as tmp, patch.object(e, 'fit', fake_fit):
            a, b, folds = e.oof(x, x, y, ids, Path(tmp))
            self.assertEqual(calls, [40] * 6)
            np.testing.assert_array_equal(np.bincount(folds), [20, 20, 20])
            e.oof(x, x, y, ids, Path(tmp))
            self.assertEqual(len(calls), 6)
            p = Path(tmp) / 'oof_fold_0.npz'; p.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                e.oof(x, x, y, ids, Path(tmp))

    def test_scaler_only_training_and_class_order(self):
        rng = np.random.default_rng(42); x = rng.normal(size=(80, 5)); y = np.array(['F', 'M'] * 40)
        model = e.fit(x[:60], y[:60], .1)
        np.testing.assert_allclose(model[0].mean_, x[:60].mean(0))
        self.assertEqual(model[0].n_samples_seen_, 60)
        e.probabilities(model.predict_proba(x[60:]), 20)

    def test_wrong_cache_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root / 'pooled/ab/abcd.npz'
            e.npz(p, call_id='abcd', identity='wrong', samples=10, features=np.zeros((4, 2, 1024), np.float32))
            e.save(p.with_suffix('.hash.json'), dict(sha256=e.sha(p)))
            with self.assertRaisesRegex(ValueError, 'identity'):
                e.load_one(dict(call_id='abcd'), 'wavlm', root, {})

    def test_end_to_end_lock_before_validation_and_completed_resume(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/'result'
            rows = [dict(call_id=f'{i:064x}', gender='FM'[i % 2],
                         partition='train' if i < 60 else 'internal_validation') for i in range(72)]
            ident = dict(test=True)
            for row in rows:
                cid = row['call_id']
                a = rng.normal(size=(4, 2, 1024)).astype(np.float32)
                a[-1, 0, 0] = 10000 if row['partition'] != 'train' else (1 if row['gender'] == 'M' else -1)
                p = root/'w1/pooled'/cid[:2]/(cid+'.npz')
                e.npz(p, call_id=cid, identity=e.digest(ident), samples=1000, features=a)
                e.save(p.with_suffix('.hash.json'), dict(sha256=e.sha(p)))
                p = root/'w2/call_cache'/cid[:2]/(cid+'.npz')
                b = rng.normal(size=768).astype(np.float32); b[0] = a[-1, 0, 0]
                e.npz(p, call_id=cid, identity_sha256=e.digest(ident), embedding=b,
                      quality='{"failed_segments":0,"successful_segments":1}')
            original_fit, original_matrix = e.fit, e.matrix
            def checked_fit(x, y, c):
                self.assertTrue((x[:, 0] < 10000).all())
                return original_fit(x, y, c)
            def checked_matrix(rows, kind, source, identity, out, partition):
                if partition == 'validation':
                    self.assertTrue((out/'selection.lock.json').exists())
                    self.assertTrue((out/'validation_started.json').exists())
                return original_matrix(rows, kind, source, identity, out, partition)
            args = SimpleNamespace(wavlm=str(root/'w1'), wav2vec2=str(root/'w2'), output=str(out), split='unused')
            with patch.object(e, 'metadata', return_value=rows), patch.object(e, 'sources', return_value=(ident, dict(kind='cache', identity=ident))), patch.object(e, 'fit', checked_fit), patch.object(e, 'matrix', checked_matrix):
                e.run(args)
                self.assertEqual(e.read(out/'completion.json')['selected']['n'], 12)
                (out/'ensemble_results.zip').unlink()
                with patch.object(e, 'fit', side_effect=AssertionError('completed resume must not fit')):
                    e.run(args)
                self.assertTrue((out/'ensemble_results.zip').exists())


if __name__ == '__main__': unittest.main()
