import json
from pathlib import Path
import tempfile
import unittest

import compare_models as base
from screen_200 import select_jobs, migrate, read_source, validate


class ScreenTests(unittest.TestCase):
    def test_selection_ignores_predictions(self):
        jobs = base.load_jobs()
        expected = [j['call_id'] for j in select_jobs(jobs)]
        for j in jobs:
            j['baseline_fusion_prediction'] = 'F'
        self.assertEqual(expected, [j['call_id'] for j in select_jobs(jobs)])

    def test_partial_pilot_reused_without_altering_original(self):
        jobs = base.load_jobs()
        rows = []
        for j in jobs[:210]:
            rows.append({'call_id': j['call_id'], 'gender': j['gender'],
                         'prediction': 'F', 'probabilities_F_M': [.8, .2],
                         'n_segments': j['n_segments'], 'elapsed_seconds': 1,
                         'caller_samples_16k': 2 * sum(round(b * 8000) - round(a * 8000)
                                                       for a, b in j['intervals'])})
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d) / 'source', Path(d) / 'screen'
            source.mkdir()
            output.mkdir()
            p = source / 'ecapa_pilot.jsonl'
            raw = ''.join(json.dumps(r) + '\n' for r in rows) + '{"call_id":'
            p.write_text(raw)
            selected = select_jobs(jobs)
            result, imported = migrate('ecapa', selected, source, output)
            self.assertEqual(len(result), 200)
            self.assertEqual(imported, 200)
            self.assertEqual(p.read_text(), raw)
            again, _ = migrate('ecapa', selected, source, output)
            self.assertEqual(result, again)
            self.assertEqual(len((output / 'ecapa_pilot.jsonl').read_text().splitlines()), 200)
            result[selected[0]['call_id']]['probabilities_F_M'] = [float('nan'), 0]
            with self.assertRaisesRegex(RuntimeError, 'Invalid probabilities'):
                validate(result, selected)

    def test_interior_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.jsonl'
            p.write_text('broken\n{"call_id":"ok"}\n')
            with self.assertRaisesRegex(RuntimeError, 'Corrupt interior'):
                read_source(p)


if __name__ == '__main__':
    unittest.main()
