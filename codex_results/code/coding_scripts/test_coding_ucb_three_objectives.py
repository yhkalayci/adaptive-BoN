import itertools
import math
import unittest

import numpy as np

from coding_ucb_three_objectives import (
    REPORT_DIVISORS,
    Calibration,
    DistributionPrior,
    PolicyConfig,
    _ei_curve,
    best_policy_mix,
    best_single_config_policy_mix,
    bounded_calibrated_reward_space_configs,
    calibrated_reward_space_configs,
    exact_fixed_accuracy,
    fit_prior,
    fit_pilot_reward_space_isotonic,
    fit_reward_space_isotonic,
    pandora_stop_from_curve,
    pilot_transform_rewards,
    transform_reward_space,
    uncapped_tail_decay_configs,
)


class CodingUCBThreeObjectivesTests(unittest.TestCase):
    def test_report_divisors_cover_dense_100k_grid(self):
        self.assertEqual(
            REPORT_DIVISORS,
            tuple(float(value) for value in range(100_000, 1_000_001, 100_000)),
        )

    def test_probability_identity_clips_exponential_tail_extrapolation(self):
        calibration = Calibration("identity")
        np.testing.assert_allclose(
            calibration(np.asarray([-0.2, 0.25, 1.4])),
            np.asarray([0.0, 0.25, 1.0]),
        )

    def test_reward_space_isotonic_is_train_fit_and_monotone(self):
        train = {
            "a": (
                np.asarray([-2.0, -1.0, 1.0, 2.0]),
                np.asarray([0, 0, 1, 1]),
                np.ones(4),
            )
        }
        heldout = {
            "b": (
                np.asarray([-3.0, 0.0, 3.0]),
                np.asarray([1, 1, 0]),
                np.asarray([2.0, 3.0, 4.0]),
            )
        }
        calibration = fit_reward_space_isotonic(train)
        transformed = transform_reward_space(heldout, calibration)
        probability, correct, chars = transformed["b"]
        self.assertTrue(np.all(np.diff(probability) >= 0.0))
        self.assertGreaterEqual(float(probability.min()), 0.0)
        self.assertLessEqual(float(probability.max()), 1.0)
        np.testing.assert_array_equal(correct, heldout["b"][1])
        np.testing.assert_array_equal(chars, heldout["b"][2])

    def test_pilot_isotonic_uses_only_first_three_for_context(self):
        problems = {
            "a": (
                np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
                np.asarray([0, 0, 0, 1, 1]),
                np.ones(5),
            )
        }
        permutations = {"a": [np.asarray([0, 1, 2, 3, 4])]}
        calibration = fit_pilot_reward_space_isotonic(
            problems, permutations, beta=-0.5
        )
        first = pilot_transform_rewards(
            problems["a"][0], permutations["a"][0], calibration
        )
        changed_future = problems["a"][0].copy()
        changed_future[3:] += 100.0
        second = pilot_transform_rewards(
            changed_future, permutations["a"][0], calibration
        )
        np.testing.assert_allclose(first[:3], second[:3])
        self.assertTrue(np.all(np.diff(first) >= 0.0))
        self.assertEqual(calibration.context_mean_coef, -0.5)

    def test_calibrated_grid_tests_gaussian_and_exponential_tail(self):
        configs = calibrated_reward_space_configs()
        self.assertEqual(
            {config.family for config in configs},
            {"gaussian", "shifted_exponential"},
        )
        self.assertEqual({config.calibration for config in configs}, {"identity"})
        self.assertEqual(
            {config.method for config in configs},
            {"gaussian_calibrated_probability",
             "shifted_exponential_calibrated_probability"},
        )

    def test_extended_tail_decay_grid_is_uncapped_and_bounded(self):
        configs = uncapped_tail_decay_configs()
        self.assertEqual(len(configs), 324)
        self.assertEqual(
            {config.tail_decay for config in configs},
            {0.0, 0.25, 0.5, 1.0, 2.0, 4.0},
        )
        self.assertEqual({config.calibration for config in configs}, {
            "bounded_identity"
        })
        self.assertTrue(all(config.cap_factor is None for config in configs))

    def test_bounded_probability_models_produce_valid_expected_improvement(self):
        rewards = np.linspace(0.02, 0.75, 30)
        best = np.maximum.accumulate(rewards)
        prior = DistributionPrior(
            0.2, 0.04, {0.5: 0.2, 0.75: 0.3},
            {0.5: 0.1, 0.75: 0.08}, 100.0,
        )
        calibration = Calibration("bounded_identity")
        configs = bounded_calibrated_reward_space_configs()
        for family in ("gaussian", "shifted_exponential"):
            config = next(item for item in configs if item.family == family)
            values = _ei_curve(rewards, best, calibration, prior, config, 0.05)
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all(values >= 0.0))
            self.assertTrue(np.all(values <= 1.0 - best + 1e-12))
        self.assertEqual(
            {config.calibration for config in configs}, {"bounded_identity"}
        )
        self.assertEqual(
            {config.method for config in configs},
            {"gaussian_calibrated_probability",
             "shifted_exponential_calibrated_probability"},
        )

    def test_infinite_reward_prior_is_exactly_global(self):
        first = np.linspace(0.05, 0.75, 30)
        second = np.linspace(0.20, 0.60, 30) ** 2
        shared_best = np.full(30, 0.80)
        prior = DistributionPrior(
            0.3, 0.04, {0.5: 0.25, 0.75: 0.4},
            {0.5: 0.12, 0.75: 0.08}, 100.0,
        )
        calibration = Calibration("bounded_identity")
        for family, quantile in (("gaussian", None),
                                 ("shifted_exponential", 0.5)):
            config = PolicyConfig(
                family, "bounded_identity", 0.8, math.inf, 0.0,
                None, quantile, 1.0,
            )
            one = _ei_curve(first, shared_best, calibration, prior, config, 0.05)
            two = _ei_curve(second, shared_best, calibration, prior, config, 0.05)
            np.testing.assert_allclose(one, two, atol=1e-14, rtol=0.0)

    def test_exact_fixed_curve_matches_exhaustive_enumeration(self):
        rewards = np.asarray([0.0, 1.0, 1.0, 2.0])
        correct = np.asarray([0, 0, 1, 1])
        problems = {"p": (rewards, correct, np.ones(4))}
        exact = exact_fixed_accuracy(problems)
        brute = []
        for n in range(1, 5):
            outcomes = []
            for permutation in itertools.permutations(range(4), n):
                chosen = np.asarray(permutation)
                selected = chosen[int(np.argmax(rewards[chosen]))]
                outcomes.append(correct[selected])
            brute.append(np.mean(outcomes))
        np.testing.assert_allclose(exact, brute, atol=1e-12)

    def test_stop_uses_running_character_mean_and_minimum_three(self):
        ei = np.asarray([1.0, 1.0, 0.03, 0.001, 0.0])
        cumulative_chars = np.asarray([10.0, 30.0, 60.0, 100.0, 150.0])
        config = PolicyConfig("gaussian", "bt", 0.0, 0.0, 0.0, None)
        opened = pandora_stop_from_curve(
            ei, cumulative_chars, 1000.0, 20.0, config, fixed_n=5, min_open=3
        )
        self.assertEqual(opened, 4)
        self.assertEqual(cumulative_chars[opened - 1], 100.0)

    def test_tail_decay_smoothly_stops_persistent_low_value_paths(self):
        ei = np.full(100, 0.02, dtype=np.float64)
        cumulative = np.arange(1, 101, dtype=np.float64) * 100.0
        plain = PolicyConfig(
            "gaussian", "bounded_identity", 0.0, 5.0, 0.0, None
        )
        decayed = PolicyConfig(
            "gaussian", "bounded_identity", 0.0, 5.0, 0.0, None,
            tail_decay=1.0,
        )
        plain_n = pandora_stop_from_curve(
            ei, cumulative, 1e4, 100.0, plain, fixed_n=1
        )
        decayed_n = pandora_stop_from_curve(
            ei, cumulative, 1e4, 100.0, decayed, fixed_n=1
        )
        self.assertGreater(plain_n, decayed_n)
        self.assertGreaterEqual(decayed_n, 3)

    def test_all_distribution_families_produce_finite_nonnegative_ucb_ei(self):
        rng = np.random.default_rng(4)
        rewards = rng.normal(size=30)
        best = np.maximum.accumulate(rewards)
        prior = DistributionPrior(
            0.0, 1.0, {0.25: -0.67, 0.5: 0.0}, {0.25: 1.0, 0.5: 0.8}, 100.0
        )
        calibration = Calibration("bt", intercept=-1.0, slope=0.5)
        for family, quantile in (("gaussian", None), ("gaussian_kde", None),
                                 ("shifted_exponential", 0.5)):
            config = PolicyConfig(family, "bt", 0.8, 5.0, 0.0, None, quantile)
            values = _ei_curve(rewards, best, calibration, prior, config, 0.05)
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all(values >= 0.0))
            self.assertTrue(np.all(values <= 1.0 + 1e-12))

    def test_relative_bt_is_monotone_at_fixed_reference_distribution(self):
        calibration = Calibration("relative_bt", intercept=-2.0, slope=0.7)
        x = np.linspace(-5.0, 5.0, 100)
        y = calibration(x)
        self.assertTrue(np.all(np.diff(y) >= 0.0))
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)

    def test_contextual_relative_bt_exp_tail_is_finite_and_monotone(self):
        rewards = np.linspace(-3.0, 4.0, 30)
        best = np.maximum.accumulate(rewards)
        prior = DistributionPrior(
            0.0, 1.0, {0.75: 0.7}, {0.75: 0.6}, 100.0
        )
        calibration = Calibration(
            "contextual_relative_bt", intercept=-2.0, slope=0.7,
            context_mean_center=0.0, context_mean_scale=2.0,
            context_mean_coef=0.5,
        )
        config = PolicyConfig(
            "shifted_exponential", "contextual_relative_bt",
            0.2, 5.0, 0.0, 2.0, 0.75,
        )
        values = _ei_curve(rewards, best, calibration, prior, config, 0.05)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0 + 1e-12))

    def test_prior_covers_expanded_tail_quantile(self):
        rewards = np.linspace(-2.0, 3.0, 20)
        problems = {"p": (rewards, np.zeros(20), np.ones(20))}
        prior = fit_prior(problems)
        self.assertIn(0.75, prior.exp_locations)
        self.assertIn(0.75, prior.exp_scales)

    def test_quality_match_is_exact_for_nonmonotone_fixed_curve(self):
        quality = np.asarray([0.20, 0.40, 0.30, 0.50])
        chars = np.asarray([100.0, 200.0, 300.0, 400.0])
        low, high, weight, reached = best_policy_mix(quality, chars, 0.35)
        matched_quality = (1.0 - weight) * quality[low] + weight * quality[high]
        self.assertTrue(reached)
        self.assertAlmostEqual(matched_quality, 0.35)
        self.assertAlmostEqual((1.0 - weight) * chars[low] + weight * chars[high], 175.0)

    def test_target_mix_uses_one_ucb_config_and_selects_cheapest_path(self):
        quality = np.asarray([
            [0.20, 0.40],
            [0.10, 0.50],
        ])
        chars = np.asarray([
            [1.0, 10.0],
            [2.0, 4.0],
        ])
        config, low, high, weight, reached = best_single_config_policy_mix(
            quality, chars, config_ids=(0, 1), divisor_ids=(0, 1), target=0.30
        )
        self.assertTrue(reached)
        self.assertEqual(config, 1)
        self.assertEqual((low, high), (0, 1))
        self.assertAlmostEqual(weight, 0.5)
        self.assertAlmostEqual(
            (1.0 - weight) * quality[config, low] + weight * quality[config, high],
            0.30,
        )
        self.assertAlmostEqual(
            (1.0 - weight) * chars[config, low] + weight * chars[config, high],
            3.0,
        )


if __name__ == "__main__":
    unittest.main()
