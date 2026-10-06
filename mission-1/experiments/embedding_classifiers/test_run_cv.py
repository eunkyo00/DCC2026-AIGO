"""Checks for leakage prevention and incomplete candidate handling."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_cv import evaluate_fold, index_unique, make_model, summarize


class CVChecks(unittest.TestCase):
    def test_scaler_excludes_heldout_extreme_values(self):
        x = np.array([[-3., 0.], [-2., 1.], [2., 0.], [3., 1.],
                      [-1000., 500.], [1000., 500.]])
        y = np.array(['F', 'F', 'M', 'M', 'F', 'M'])
        model = make_model('lr', 1.)
        _, _, report = evaluate_fold(model, x, y, np.arange(4), np.array([4, 5]))
        np.testing.assert_allclose(model.named_steps['standardscaler'].mean_, [0., .5])
        self.assertEqual(report['scaler_fit_calls'], 4)

    def test_overlapping_fold_rejected(self):
        with self.assertRaisesRegex(ValueError, 'leakage'):
            evaluate_fold(make_model('lr', 1.), np.ones((4, 2)),
                          np.array(['F', 'M', 'F', 'M']), np.array([0, 1]), np.array([1, 2]))

    def test_duplicate_call_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            index_unique([{'call_id': 'a'}, {'call_id': 'a'}], 'MFCC')

    def test_no_completed_candidates_means_partial(self):
        import json
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            summarize(Path(directory))
            report = json.loads((Path(directory) / 'cv_summary.json').read_text())
            self.assertEqual(report['status'], 'partial')
            self.assertEqual(report['candidates'], [])
            self.assertFalse(report['validation_evaluated'])


if __name__ == '__main__':
    unittest.main()
