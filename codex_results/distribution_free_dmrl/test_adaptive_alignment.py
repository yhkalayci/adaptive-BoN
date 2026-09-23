"""Equivalence of the public online API and the research replay implementation."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'algorithm'))
from adaptive_alignment import AdaptiveAlignment
from online_cost_optimism import effective_cost
from online_smoothed_spacing import statistics, smooth_numerator
from training_free_spacing import stopping_indices


class OnlineAlignmentTests(unittest.TestCase):
    def test_replay_equivalence(self):
        rng = np.random.default_rng(20260923)
        for _ in range(15):
            rewards = rng.normal(size=80)
            lengths = rng.integers(1, 4000, size=80).astype(float)
            scaled_gain = smooth_numerator(statistics(rewards)['odds'], 'mean')
            cost = effective_cost(lengths, 2)
            for price in [2e-8, 1e-7, 2e-7, 1e-6, 2e-6, 1e-5]:
                with self.subTest(price=price):
                    expected = int(stopping_indices(
                        scaled_gain, cost, [price], 4, 'sequential')[0]) + 1
                    policy = AdaptiveAlignment(price, cap=80)
                    for n, (reward, length) in enumerate(zip(rewards, lengths), 1):
                        decision = policy.observe(reward, length)
                        if n >= 4:
                            self.assertAlmostEqual(decision.estimated_improvement,
                                                   scaled_gain[n-1] / n, places=14)
                            self.assertAlmostEqual(decision.estimated_next_cost,
                                                   price * cost[n-1] / n, places=14)
                        if decision.should_stop:
                            break
                    self.assertEqual(n, expected)
                    self.assertEqual(decision.best_index, int(np.argmax(rewards[:n])))
                    self.assertAlmostEqual(decision.total_cost, price * lengths[:n].sum())

    def test_ties_initial_count_and_stop_guard(self):
        policy = AdaptiveAlignment(1e-6)
        for n in range(4):
            d = policy.observe(2.0, 10)
            self.assertEqual(d.should_stop, n == 3)
        self.assertEqual(d.best_index, 0)
        with self.assertRaises(RuntimeError):
            policy.observe(3, 10)

    def test_cap_returns_best_and_charges_every_response(self):
        policy = AdaptiveAlignment(1e-30, cap=5)
        for n in range(5):
            d = policy.observe(float(n), 10)
        self.assertTrue(d.should_stop)
        self.assertEqual(d.count, 5)
        self.assertEqual(d.best_index, 4)
        self.assertEqual(d.total_length, 50)

    def test_invalid_observation_does_not_mutate_state(self):
        policy = AdaptiveAlignment(1e-6)
        for reward, length in [(float('nan'), 1), (1, 0), (1, 2.5)]:
            with self.assertRaises(ValueError):
                policy.observe(reward, length)
        self.assertEqual(policy.observe(1, 4).count, 1)


if __name__ == '__main__':
    unittest.main()
