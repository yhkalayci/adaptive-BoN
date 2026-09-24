"""Safety and exact-formula tests for non-parametric adaptive coding."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from algorithm.adaptive_coding import (  # noqa: E402
    AdaptiveCoding,
    CodingProblem,
    CodingProfile,
    fit_coding_profile,
    load_coding_problems,
    split_problem_ids,
)


class AdaptiveCodingTests(unittest.TestCase):
    @staticmethod
    def fixture():
        rng = np.random.default_rng(20260923)
        problems = {}
        for problem_id, shift in zip(("a", "b", "c", "d"), (-1.0, -0.2, 0.4, 1.1)):
            rewards = rng.normal(shift, 0.9, size=40)
            probability = 1.0 / (1.0 + np.exp(-(rewards - 0.15)))
            correct = rng.random(40) < probability
            lengths = rng.integers(20, 500, size=40)
            problems[problem_id] = CodingProblem(
                tuple(rewards),
                tuple(bool(x) for x in correct),
                tuple(int(x) for x in lengths),
            )
        return problems

    def test_profile_fits_only_isotonic_calibration(self):
        problems = self.fixture()
        profile = fit_coding_profile(problems)
        rewards = np.concatenate([np.asarray(x.rewards) for x in problems.values()])
        correct = np.concatenate([np.asarray(x.correct) for x in problems.values()])
        expected = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        ).fit(rewards, correct)
        np.testing.assert_array_equal(profile.reward_knots, expected.X_thresholds_)
        np.testing.assert_array_equal(profile.probability_knots, expected.y_thresholds_)
        self.assertIsNone(profile.to_dict()["online_distribution"])
        self.assertNotIn("tail", json.dumps(profile.to_dict()).lower())

    def test_top_four_above_fifth_formula_and_cost(self):
        profile = CodingProfile((0.0, 1.0), (0.0, 1.0), mean_length=10.0)
        policy = AdaptiveCoding(
            profile,
            price=0.001,
            cap=20,
            width=4,
            multiplier=1.0,
            smoothing="mean",
            cost_adjustment=2.0,
        )
        for reward, length in zip((0.1, 0.2, 0.8, 0.5), (5, 10, 15, 20)):
            decision = policy.observe(reward, length)
            self.assertIsNone(decision.estimated_improvement)
        decision = policy.observe(0.4, 25)
        residual = np.mean(np.asarray((0.2, 0.4, 0.5, 0.8)) - 0.1)
        mean_length = 15.0
        standard_error = math.sqrt(np.var((5, 10, 15, 20, 25)) / 5)
        expected_cost = 0.001 * mean_length / (
            1.0 + 2.0 * standard_error / mean_length
        )
        self.assertAlmostEqual(decision.residual_scale, residual)
        self.assertAlmostEqual(decision.smoothed_residual_scale, residual)
        self.assertAlmostEqual(decision.estimated_improvement, residual / 5)
        self.assertAlmostEqual(decision.estimated_next_cost, expected_cost)
        self.assertEqual(decision.best_index, 2)
        self.assertEqual(decision.total_cost, 0.075)

    def test_smoothing_modes_are_observation_only(self):
        profile = CodingProfile((0.0, 1.0), (0.0, 1.0), mean_length=10.0)
        rewards = (0.1, 0.2, 0.8, 0.5, 0.4, 0.95, 0.3)
        current = AdaptiveCoding(profile, 1e-8, cap=7, smoothing="current")
        mean = AdaptiveCoding(profile, 1e-8, cap=7, smoothing="mean")
        for reward in rewards:
            current_decision = current.observe(reward, 10)
            mean_decision = mean.observe(reward, 10)
        self.assertAlmostEqual(
            current_decision.smoothed_residual_scale,
            current_decision.residual_scale,
        )
        self.assertNotAlmostEqual(
            mean_decision.smoothed_residual_scale,
            mean_decision.residual_scale,
        )
        self.assertEqual(current_decision.best_index, 5)

    def test_profile_round_trip_and_clipping(self):
        profile = CodingProfile(
            (-1.0, 0.0, 2.0),
            (0.1, 0.2, 0.9),
            mean_length=30.0,
            metadata={"fit_problem_ids": ["a"]},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile.save(path)
            loaded = CodingProfile.load(path)
            self.assertEqual(profile, loaded)
            self.assertTrue(path.read_bytes().endswith(b"\n"))
        self.assertEqual(profile.calibrate(-10), 0.1)
        self.assertEqual(profile.calibrate(10), 0.9)
        self.assertAlmostEqual(profile.calibrate(1), 0.55)

    def test_split_is_deterministic_and_disjoint(self):
        ids = [f"p{index}" for index in range(11)]
        fit_a, test_a = split_problem_ids(ids, seed=4)
        fit_b, test_b = split_problem_ids(reversed(ids), seed=4)
        self.assertEqual((fit_a, test_a), (fit_b, test_b))
        self.assertFalse(set(fit_a) & set(test_a))
        self.assertEqual(set(fit_a) | set(test_a), set(ids))
        self.assertEqual((len(fit_a), len(test_a)), (5, 6))

    def test_loader_validates_and_keeps_only_fit_fields(self):
        samples = [{
            "idx": index,
            "r_score": float(index),
            "correct": index == 2,
            "output_tokens": index + 1,
            "text": "must not survive loading",
        } for index in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps({"id": "x", "samples": samples}) + "\n")
            problems = load_coding_problems(path, expected_samples=3)
        self.assertEqual(problems["x"].rewards, (0.0, 1.0, 2.0))
        self.assertEqual(problems["x"].correct, (False, False, True))
        self.assertEqual(problems["x"].lengths, (1, 2, 3))

    def test_invalid_observation_never_mutates_state(self):
        profile = CodingProfile((0.0, 1.0), (0.1, 0.9), mean_length=20.0)
        policy = AdaptiveCoding(profile, 1e-5)
        for reward, length in ((float("nan"), 1), (1, 0), (1, 2.5)):
            with self.assertRaises(ValueError):
                policy.observe(reward, length)
        self.assertEqual(policy.observe(0.5, 4).count, 1)

    def test_cap_and_raw_reward_breaks_calibration_tie(self):
        profile = CodingProfile((0.0, 1.0), (0.5, 0.5), mean_length=20.0)
        policy = AdaptiveCoding(profile, 1e-30, cap=5)
        for index in range(5):
            decision = policy.observe(index / 4, 10)
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.best_index, 4)
        with self.assertRaises(RuntimeError):
            policy.observe(1.0, 10)


if __name__ == "__main__":
    unittest.main()
