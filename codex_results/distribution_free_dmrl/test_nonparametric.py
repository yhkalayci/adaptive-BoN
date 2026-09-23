import unittest

import numpy as np

from evaluate import q99, sigmoid, stop_count
from nonparametric_study import (Policy, policies, prefix_statistics,
                                select_on_train, stopping_counts)
from nonparametric_controls import collect_budget
from evaluate import Pool


class NonparametricTests(unittest.TestCase):
    def test_future_values_do_not_change_features(self):
        x = np.arange(40, dtype=float)/10
        original = prefix_statistics(x)
        x[20:] = 1000
        changed = prefix_statistics(x)
        for key in original:
            np.testing.assert_array_equal(original[key][:20], changed[key][:20])

    def test_spacing_formula(self):
        x = np.array([0., 1., 2., 3.])
        u = sigmoid(x-q99(x))
        stats = prefix_statistics(x)
        self.assertAlmostEqual(stats['spacing', 1][-1], (u[-1]-u[-2])/4)
        self.assertAlmostEqual(stats['spacing', 2][-1], ((u[-1]+u[-2])/2-u[-3])/4)

    def test_recent_gain_uses_same_reference(self):
        x = np.arange(9, dtype=float)
        stats = prefix_statistics(x)
        ref = q99(x)
        self.assertAlmostEqual(stats['recent_gain', 4][-1],
                               (sigmoid(8-ref)-sigmoid(4-ref))/4)

    def test_original_rule_is_reproduced(self):
        rng = np.random.default_rng(55)
        for _ in range(20):
            x = rng.normal(size=64)
            lengths = rng.uniform(1, 100, size=64)
            price = .0001
            n = stop_count(x, lengths, price, lambda r: sigmoid(r-q99(r)), 'sequential')
            actual = stopping_counts(prefix_statistics(x), lengths,
                                     [Policy('spacing', 2, 4., 4)], np.array([price]))
            self.assertEqual(n, actual[0, 0])

    def test_ties_and_cap(self):
        configs = policies()
        stops = stopping_counts(prefix_statistics(np.ones(16)), np.ones(16), configs)
        for i, p in enumerate(configs):
            if p.family in ('spacing', 'recent_gain'):
                self.assertTrue(np.all(stops[:, i] == min(p.minimum, 16)))
        self.assertTrue(np.all((stops>=1) & (stops<=16)))

    def test_selection_ignores_test_outcomes(self):
        configs = policies()
        rng = np.random.default_rng(12)
        a = rng.random((8, 6, len(configs), 4))
        f = rng.random((8, 6, 32, 4))
        first, fixed_first = select_on_train(a, f, np.arange(4), configs)
        a[4:] *= 10000
        f[4:] *= 10000
        second, fixed_second = select_on_train(a, f, np.arange(4), configs)
        for method in first:
            np.testing.assert_array_equal(first[method], second[method])
        np.testing.assert_array_equal(fixed_first, fixed_second)

    def test_budget_control_does_not_stop_on_rewards(self):
        p = Pool('p', np.arange(16, dtype=float), np.arange(1, 17, dtype=float))
        a = collect_budget([p], 16, 3, 9, np.array([1., 20., np.inf]))
        p.rewards = p.rewards[::-1].copy()
        b = collect_budget([p], 16, 3, 9, np.array([1., 20., np.inf]))
        np.testing.assert_array_equal(a[..., 1:3], b[..., 1:3])
        np.testing.assert_array_equal(a[0, :, 0, 2], np.ones(6))
        np.testing.assert_array_equal(a[0, :, -1, 2], np.full(6, 16.))


if __name__ == '__main__':
    unittest.main()
