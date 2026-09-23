import unittest
import gzip
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from evaluate import Pool, replay, sigmoid, q99, stop_count, load_alignment, load_coding, evaluate


class StoppingTests(unittest.TestCase):
    def test_positive_ties(self):
        self.assertEqual(stop_count(np.ones(20), np.ones(20), .001, lambda x: x), 4)

    def test_zeros_guard(self):
        x = np.zeros(17)
        self.assertEqual(stop_count(x, np.ones(17), .001, lambda x: x), 17)
        self.assertEqual(stop_count(x, np.ones(17), .001, lambda x: x, 'sequential'), 17)

    def test_formula_and_checkpoints(self):
        x = np.array([0, 1, 2, 3, 3, 3, 3, 3], float)
        self.assertEqual(stop_count(x, np.ones(8), 1.5, lambda x: x), 4)
        self.assertEqual(stop_count(x, np.ones(8), 1.49, lambda x: x), 8)

    def test_fixed_cost(self):
        x = np.arange(8, dtype=float)
        self.assertEqual(stop_count(x, np.ones(8) * 1e9, 10, lambda x: x, fixed_cost=.8), 8)

    def test_sequential_changes_only_checkpoint_schedule(self):
        x = np.array([0, 1, 2, 3, 3, 3, 3, 3], float)
        self.assertEqual(stop_count(x, np.ones(8), 1.2, lambda x: x), 8)
        self.assertEqual(stop_count(x, np.ones(8), 1.2, lambda x: x, 'sequential'), 5)

    def test_caps(self):
        for n in (1, 2, 3, 5, 7):
            self.assertEqual(stop_count(np.arange(n), np.ones(n), 1e-9, lambda x: x), n)

    def test_future_reward_and_cost_cannot_change_stop(self):
        x = np.r_[np.ones(4), np.ones(12)*1000]
        lengths = np.r_[np.ones(4), np.ones(12)*1e8]
        observed = []
        def transform(prefix):
            observed.append(len(prefix))
            return prefix
        self.assertEqual(stop_count(x, lengths, .01, transform), 4)
        self.assertEqual(observed, [4])

    def test_alignment_reference_prefix_only(self):
        x = np.r_[np.ones(4), np.ones(12)*1000]
        transform = lambda prefix: sigmoid(prefix-q99(prefix))
        self.assertEqual(stop_count(x, np.ones(16), .01, transform), 4)

    def test_correctness_changes_scoring_not_stopping(self):
        x = np.arange(8, dtype=float)
        args = ([np.arange(8)], [.1], [1, 4, 8], lambda x: x, False)
        a = replay(Pool('a', x, np.ones(8), np.zeros(8)), *args)
        b = replay(Pool('a', x, np.ones(8), np.ones(8)), *args)
        for method in a[.1]:
            self.assertEqual(a[.1][method][2], b[.1][method][2])
            self.assertAlmostEqual(b[.1][method][3]-a[.1][method][3], 1)

    def test_argmax_tie_breaks_on_raw_reward(self):
        p = Pool('a', np.array([1., 2., 3., 4.]), np.ones(4), np.array([0, 0, 0, 1]))
        r = replay(p, [np.arange(4)], [.01], [1, 4], lambda x: np.ones(len(x))*.5, False)
        self.assertEqual(r[.01]['dmrl_doubling'][0], 1)

    def test_native_alignment_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.jsonl.gz'
            with gzip.open(path, 'wt') as handle:
                handle.write(json.dumps(dict(JSON_idx=7, generations=[dict(text='abc', score=1.5)]))+'\n')
            pool = load_alignment(path, 'score', 'text_chars')[0]
            self.assertEqual(pool.id, '7')
            np.testing.assert_equal(pool.lengths, [3])
            np.testing.assert_equal(pool.rewards, [1.5])

    def test_native_coding_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            path, cache = Path(folder)/'data.jsonl', Path(folder)/'lengths.npz'
            path.write_text(json.dumps(dict(id='p', samples=[dict(idx=1, correct=True, r_score=2), dict(idx=0, correct=False, r_score=1)]))+'\n')
            np.savez(cache, ids=np.array(['p']), indices=np.array([[0, 1]]), chars=np.array([[10, 20]]))
            pool = load_coding(path, cache, 'chars')[0]
            np.testing.assert_equal(pool.rewards, [1, 2])
            np.testing.assert_equal(pool.correct, [0, 1])
            np.testing.assert_equal(pool.lengths, [10, 20])

    def test_test_labels_do_not_change_coding_calibration_or_fixed_n(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest('scikit-learn not installed')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder/'input.jsonl'
            source.write_text('synthetic unit test fixture\n')
            pools = [Pool(str(i), np.arange(8, dtype=float), np.ones(8),
                          np.array([0, 0, 0, 0, 1, 1, 1, 1], float)) for i in range(6)]
            args = SimpleNamespace(task='coding', cap=8, seed=75, prices=[.1], permutations=2,
                                   bootstrap=20, output=folder/'first', data=source,
                                   reward_key='r_score', length_key='chars', length_cache=None,
                                   length_unit='characters')
            first = evaluate(pools, args)
            meta = json.loads((args.output/'METHOD.json').read_text())
            for p in pools:
                if p.id in meta['test_ids']:
                    p.correct = 1-p.correct
            args.output = folder/'second'
            second = evaluate(pools, args)
            meta2 = json.loads((args.output/'METHOD.json').read_text())
            self.assertEqual(meta['calibration'], meta2['calibration'])
            self.assertEqual(first[0]['train_selected_fixed'], second[0]['train_selected_fixed'])
            self.assertEqual(first[0]['samples'], second[0]['samples'])


if __name__ == '__main__':
    unittest.main()
