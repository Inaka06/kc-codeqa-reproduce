"""CPU-only report checks using synthetic artifacts; no model or dataset access."""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import tempfile
import unittest
import make_report_figures as report


class ReportTests(unittest.TestCase):
    def test_pending_is_not_zero_and_partial_test_is_rejected(self):
        pending = report.result_rows('CodeBERT–BART', None)[0]
        self.assertIsNone(pending['BLEU'])
        self.assertEqual(pending['status'], 'Test belum tersedia')
        with self.assertRaises(ValueError):
            report.result_rows('Seq2seq', {'n': 500, 'training_complete_for_profile': True,
                                          'evaluation_scope': 'medium_subset_only'})

    def test_duplicate_epochs_rejected(self):
        row = {'epoch': 1, 'dev': {'BLEU': 1, 'F1': 2}}
        with self.assertRaises(ValueError):
            report.validate_history([row, row])

    def test_exports_and_completed_hybrid_refresh(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = dict(n=7000, training_complete_for_profile=True, evaluation_scope='full_test',
                          **{k: 20.0 for k in report.METRICS})
            for name, pattern, _, _, _, dev_n in report.SPECS:
                run = root / pattern.replace('*', 'fixture')
                run.mkdir(parents=True)
                key = 'train_mean_sequence_loss_this_session' if name == 'Seq2seq' else 'train_token_loss'
                history = [{'epoch': 1, key: 5.0, 'dev': {'n': dev_n, 'BLEU': 10., 'F1': 12.}},
                           {'epoch': 2, key: 4.0, 'dev': {'n': dev_n, 'BLEU': 9., 'F1': 13.}}]
                (run / 'history.json').write_text(json.dumps(history))
                if name != 'CodeBERT–BART':
                    (run / 'results.json').write_text(json.dumps(result))
                else:
                    hybrid = run
            report.main(['--root', str(root)])
            out = root / 'report_figures'
            for stem in ['dev_bleu', 'dev_f1', 'training_loss', 'test_comparison']:
                for suffix in ['png', 'pdf']:
                    self.assertGreater((out / f'{stem}.{suffix}').stat().st_size, 100)
            self.assertIn('Test belum tersedia', (out / 'test_comparison.csv').read_text())
            self.assertIn('best dev BLEU 10.00 pada epoch 1', (out / 'report.md').read_text())
            (hybrid / 'results.json').write_text(json.dumps(result))
            report.main(['--root', str(root)])
            self.assertNotIn('Test belum tersedia', (out / 'test_comparison.csv').read_text())
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(len(manifest['input_sha256']), 6)


if __name__ == '__main__':
    unittest.main()
