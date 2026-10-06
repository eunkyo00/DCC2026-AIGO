import unittest

import numpy as np

from analyze_oof import overlap


class OverlapChecks(unittest.TestCase):
    def test_changes_are_directional_and_partition_calls(self):
        old = np.array([False, True, False, True, True])
        new = np.array([False, False, True, True, False])
        result = overlap(old, new)
        self.assertEqual(result, {'both_correct': 1, 'corrected': 2,
                                  'regressed': 1, 'both_wrong': 1})
        self.assertEqual(sum(result.values()), len(old))
        self.assertEqual(result['corrected'] - result['regressed'], int(old.sum() - new.sum()))


if __name__ == '__main__':
    unittest.main()
