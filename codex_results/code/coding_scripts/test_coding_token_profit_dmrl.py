"""Tests for leakage barriers and token accounting in the DMRL study."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coding_token_profit_dmrl import (  # noqa: E402
    PolicyConfig,
    build_trajectory_batch,
    calibrated_rewards,
    exact_fixed_accuracy,
    fixed_metrics,
    selected_correctness,
    stop_counts,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from algorithm.adaptive_coding import AdaptiveCoding, CodingProblem, CodingProfile  # noqa: E402


class CodingTokenProfitTests(unittest.TestCase):
    def setUp(self):
        self.problems = {
            "a": CodingProblem(
                rewards=(0.1, 0.8, 0.3, 0.7, 0.4, 0.9),
                correct=(False, True, False, True, False, True),
                lengths=(10, 20, 30, 40, 50, 60),
            ),
            "b": CodingProblem(
                rewards=(0.2, 0.1, 0.6, 0.5, 0.9, 0.7),
                correct=(False, False, True, False, True, True),
                lengths=(60, 50, 40, 30, 20, 10),
            ),
        }
        self.profile = CodingProfile((0.0, 1.0), (0.0, 1.0), 35.0)
        self.probability = calibrated_rewards(self.problems, self.profile)
        self.permutations = {
            "a": np.asarray([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]]),
            "b": np.asarray([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]]),
        }

    def test_stopping_has_no_correctness_argument_or_dependency(self):
        batch = build_trajectory_batch(
            self.problems, self.probability, self.permutations
        )
        config = PolicyConfig("mean", 1.0, 2.0, None, 0.0)
        first = stop_counts(batch, 1000.0, 5, config)
        flipped = {
            key: CodingProblem(value.rewards, tuple(not x for x in value.correct), value.lengths)
            for key, value in self.problems.items()
        }
        flipped_probability = calibrated_rewards(flipped, self.profile)
        flipped_batch = build_trajectory_batch(
            flipped, flipped_probability, self.permutations
        )
        second = stop_counts(flipped_batch, 1000.0, 5, config)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(batch.cumulative_tokens, flipped_batch.cumulative_tokens)

    def test_correctness_revealed_only_after_prefix_selection(self):
        batch = build_trajectory_batch(
            self.problems, self.probability, self.permutations
        )
        labels = selected_correctness(batch, self.problems, self.permutations)
        metrics = fixed_metrics(batch, labels, divisor=100.0, fixed_n=2)
        np.testing.assert_array_equal(metrics["generations"], 2)
        rows = np.arange(batch.trials)
        expected_tokens = batch.cumulative_tokens[rows, 1]
        np.testing.assert_array_equal(metrics["tokens"], expected_tokens)
        np.testing.assert_allclose(
            metrics["profit"], metrics["correct"] - expected_tokens / 100.0
        )

    def test_exact_fixed_accuracy_matches_enumeration_at_full_n(self):
        accuracy = exact_fixed_accuracy(self.problems)
        expected = []
        for problem_id, problem in self.problems.items():
            best = int(np.argmax(self.probability[problem_id]))
            expected.append(problem.correct[best])
        self.assertAlmostEqual(accuracy[-1], np.mean(expected))
        self.assertAlmostEqual(
            accuracy[0],
            np.mean([np.mean(x.correct) for x in self.problems.values()]),
        )

    def test_early_stop_signal_is_held_until_minimum(self):
        problem = CodingProblem(
            rewards=(0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.2),
            correct=(False,) * 7,
            lengths=(100,) * 7,
        )
        profile = CodingProfile((0.0, 1.0), (0.0, 1.0), 100.0)
        batch = build_trajectory_batch(
            {"a": problem},
            calibrated_rewards({"a": problem}, profile),
            {"a": np.arange(7, dtype=np.int16)[None, :]},
        )
        config = PolicyConfig("current", 1.0, 0.0, None, 1.0)
        batch_stop = int(stop_counts(batch, 10000.0, 6, config)[0])
        policy = AdaptiveCoding(
            profile, price=1 / 10000.0, cap=7, minimum=6,
            smoothing="current", cost_adjustment=0.0,
        )
        online_stop = next(
            decision.count
            for reward in problem.rewards
            if (decision := policy.observe(reward, 100)).should_stop
        )
        self.assertEqual(batch_stop, 6)
        self.assertEqual(online_stop, batch_stop)

    def test_width_three_batch_matches_online(self):
        problem = CodingProblem(
            rewards=(0.1, 0.4, 0.2, 0.8, 0.3, 0.9, 0.5),
            correct=(False,) * 7,
            lengths=(10,) * 7,
        )
        profile = CodingProfile((0.0, 1.0), (0.0, 1.0), 10.0)
        batch = build_trajectory_batch(
            {"a": problem},
            calibrated_rewards({"a": problem}, profile),
            {"a": np.arange(7, dtype=np.int16)[None, :]},
            width=3,
        )
        self.assertAlmostEqual(
            float(batch.residual_current[0, 3]),
            (0.8 + 0.4 + 0.2 - 3 * 0.1) / 3,
        )
        config = PolicyConfig("mean", 1.0, 0.0, None, 0.0, width=3)
        batch_stop = int(stop_counts(batch, 1000.0, 6, config)[0])
        policy = AdaptiveCoding(
            profile, price=1 / 1000.0, cap=7, width=3,
            smoothing="mean", cost_adjustment=0.0,
        )
        online_stop = next(
            decision.count
            for reward in problem.rewards
            if (decision := policy.observe(reward, 10)).should_stop
        )
        self.assertEqual(online_stop, batch_stop)


if __name__ == "__main__":
    unittest.main()
