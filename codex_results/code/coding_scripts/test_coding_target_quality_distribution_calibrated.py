import unittest

import numpy as np

from coding_target_quality_distribution_calibrated import (
    METHOD_FAMILY,
    FROZEN_BOUNDED_CONFIG_BY_TARGET,
    FROZEN_PREDICTED_CONFIG_BY_TARGET,
    attach_equal_quality_fixed_baseline,
    candidate_configs,
    ratio_saving_interval,
    select_matched_fixed_mix,
    select_oracle_fixed_n,
    select_profiled_policy,
    select_single_policy,
)


class DistributionCalibratedTargetTests(unittest.TestCase):
    def test_candidate_grid_has_both_families_and_no_fixed_n_cap(self):
        configs = candidate_configs()
        self.assertEqual(len(configs), 54)
        self.assertEqual(
            {config.family for config in configs},
            {"gaussian", "shifted_exponential"},
        )
        self.assertEqual({config.calibration for config in configs}, {"identity"})
        self.assertTrue(all(config.cap_factor is None for config in configs))

    def test_tail_decay_grid_adds_smooth_uncapped_tail_corrections(self):
        configs = candidate_configs(True, True)
        self.assertEqual(len(configs), 216)
        self.assertEqual(
            {config.tail_decay for config in configs}, {0.0, 0.25, 0.5, 1.0}
        )
        self.assertEqual(
            {config.calibration for config in configs}, {"bounded_identity"}
        )
        self.assertTrue(all(config.cap_factor is None for config in configs))

    def test_simple_global_policy_is_one_uncapped_configuration(self):
        configs = candidate_configs(simple_global_policy=True)
        self.assertEqual(len(configs), 1)
        config = configs[0]
        self.assertEqual(config.family, "shifted_exponential")
        self.assertEqual(config.calibration, "bounded_identity")
        self.assertEqual(config.confidence_scale, 0.8)
        self.assertTrue(np.isinf(config.reward_prior_strength))
        self.assertEqual(config.cost_prior_strength, 0.0)
        self.assertIsNone(config.cap_factor)
        self.assertEqual(config.tail_quantile, 0.75)
        self.assertEqual(config.tail_decay, 1.0)

    def test_profiled_selection_uses_transfer_attainment_and_cost(self):
        configs = candidate_configs(True, True)[:2]
        values = np.zeros((2, 2, 4, 4), dtype=np.float64)
        values[..., 0] = np.asarray([[0.10, 0.45], [0.45, 0.45]])[:, :, None]
        values[..., 1] = np.asarray([[20.0, 30.0], [10.0, 40.0]])[:, :, None]
        profile = {
            "accuracy": np.asarray([[0.31, 0.32], [0.28, 0.34]]),
            "log_chars": np.log(np.asarray([[100.0, 80.0], [5.0, 120.0]])),
        }
        selected = select_profiled_policy(
            values, configs, (1.0, 2.0), 0.30, profile
        )
        # Candidate (1, 0) is cheap but its 0.9-profile + 0.1-train
        # attainment is below target. The selector deploys exactly (0, 1).
        self.assertEqual((selected["config_id"], selected["divisor_id"]), (0, 1))
        self.assertAlmostEqual(selected["selection_accuracy"], 0.333)
        self.assertAlmostEqual(selected["selection_chars"], 80.0)
        self.assertIsNone(selected["cap_factor"])

    def test_profiled_selection_does_not_shift_requested_target(self):
        configs = candidate_configs(True, True)[:1]
        values = np.zeros((1, 2, 2, 4), dtype=np.float64)
        values[..., 0] = 0.30
        values[..., 1] = 10.0
        profile = {
            "accuracy": np.asarray([[0.30, 0.299]]),
            "log_chars": np.log(np.asarray([[20.0, 1.0]])),
        }
        selected = select_profiled_policy(
            values, configs, (1.0, 2.0), 0.30, profile
        )
        self.assertEqual(selected["divisor_id"], 0)
        self.assertAlmostEqual(selected["selection_accuracy"], 0.30)

    def test_selection_uses_cheapest_single_feasible_policy(self):
        configs = candidate_configs()[:2]
        values = np.zeros((2, 2, 4, 4), dtype=np.float64)
        values[0, 0, :, 0] = 0.25
        values[0, 0, :, 1] = 90.0
        values[0, 1, :, 0] = 0.30
        values[0, 1, :, 1] = 120.0
        values[1, 0, :, 0] = 0.30
        values[1, 0, :, 1] = 100.0
        values[1, 1, :, 0] = 0.40
        values[1, 1, :, 1] = 140.0
        selected = select_single_policy(values, configs, (1.0, 2.0), 0.30)
        self.assertTrue(selected["target_reached_train"])
        self.assertEqual((selected["config_id"], selected["divisor_id"]), (1, 0))
        self.assertEqual(selected["train_chars"], 100.0)

    def test_selection_fallback_is_cheapest_maximum_accuracy(self):
        configs = candidate_configs()[:2]
        values = np.zeros((2, 2, 3, 4), dtype=np.float64)
        values[..., 0] = np.asarray([[0.2, 0.3], [0.3, 0.25]])[:, :, None]
        values[..., 1] = np.asarray([[50.0, 90.0], [70.0, 60.0]])[:, :, None]
        selected = select_single_policy(values, configs, (1.0, 2.0), 0.50)
        self.assertFalse(selected["target_reached_train"])
        self.assertEqual((selected["config_id"], selected["divisor_id"]), (1, 0))

    def test_selection_can_use_calibrated_probability_without_binary_outcomes(self):
        configs = candidate_configs()[:1]
        values = np.zeros((1, 2, 4, 4), dtype=np.float64)
        values[0, :, :, 0] = 0.1
        values[0, 0, :, 1] = 50.0
        values[0, 1, :, 1] = 80.0
        values[0, 0, :, 3] = 0.29
        values[0, 1, :, 3] = 0.31
        selected = select_single_policy(
            values, configs, (1.0, 2.0), 0.30,
            selection_metric="calibrated_probability",
        )
        self.assertTrue(selected["target_reached_train"])
        self.assertEqual(selected["divisor_id"], 1)
        self.assertEqual(selected["train_accuracy"], 0.1)
        self.assertEqual(selected["train_predicted_accuracy"], 0.31)

    def test_predicted_metric_has_development_frozen_single_configs(self):
        self.assertEqual(
            set(FROZEN_PREDICTED_CONFIG_BY_TARGET), {0.25, 0.30, 0.35}
        )
        self.assertTrue(all(
            isinstance(config_id, int)
            for config_id in FROZEN_PREDICTED_CONFIG_BY_TARGET.values()
        ))

    def test_selection_can_freeze_one_configuration_without_policy_mixture(self):
        configs = candidate_configs(True)
        values = np.zeros((len(configs), 2, 3, 4), dtype=np.float64)
        values[..., 0] = 0.4
        values[..., 1] = 1000.0
        frozen = FROZEN_BOUNDED_CONFIG_BY_TARGET[0.30]
        values[frozen, 1, :, 1] = 50.0
        selected = select_single_policy(
            values, configs, (1.0, 2.0), 0.30, config_ids=[frozen]
        )
        self.assertEqual(selected["config_id"], frozen)
        self.assertEqual(selected["divisor_id"], 1)

    def test_aggregate_saving_uses_ratio_of_total_characters(self):
        saving, low, high = ratio_saving_interval(
            [50.0, 100.0], [100.0, 200.0], seed=4
        )
        self.assertAlmostEqual(saving, 50.0)
        self.assertLessEqual(low, saving)
        self.assertGreaterEqual(high, saving)

    def test_integer_fixed_n_selector_uses_cheapest_target_reaching_n(self):
        n, accuracy, chars, reached = select_oracle_fixed_n(
            [0.20, 0.31, 0.29, 0.35], [10.0, 20.0, 30.0, 40.0], 0.30
        )
        self.assertTrue(reached)
        self.assertEqual(n, 2)
        self.assertEqual((accuracy, chars), (0.31, 20.0))

    def test_matched_fixed_mix_hits_exact_quality(self):
        low, high, weight, accuracy, chars, reached = select_matched_fixed_mix(
            [0.20, 0.28, 0.36], [10.0, 20.0, 30.0], 0.30
        )
        self.assertTrue(reached)
        self.assertEqual((low, high), (2, 3))
        self.assertAlmostEqual(weight, 0.25)
        self.assertAlmostEqual(accuracy, 0.30)
        self.assertAlmostEqual(chars, 22.5)

    def test_matched_fixed_mix_uses_cheapest_exact_pair(self):
        low, high, weight, accuracy, chars, reached = select_matched_fixed_mix(
            [0.10, 0.20, 0.40], [10.0, 100.0, 110.0], 0.25
        )
        self.assertTrue(reached)
        # Mixing N=1 and N=3 costs less than mixing adjacent N=2 and N=3.
        self.assertEqual((low, high), (1, 3))
        self.assertAlmostEqual(weight, 0.5)
        self.assertAlmostEqual(accuracy, 0.25)
        self.assertAlmostEqual(chars, 60.0)

    def test_aggregate_equal_quality_baseline_matches_adaptive_mean(self):
        rows = [
            {"split": 1, "method": "m", "target_accuracy": 0.3,
             "adaptive_accuracy": 0.25, "adaptive_chars": 15.0},
            {"split": 2, "method": "m", "target_accuracy": 0.3,
             "adaptive_accuracy": 0.35, "adaptive_chars": 25.0},
        ]
        curves = {
            1: (np.asarray([0.20, 0.40]), np.asarray([10.0, 20.0])),
            2: (np.asarray([0.10, 0.50]), np.asarray([12.0, 24.0])),
        }
        attach_equal_quality_fixed_baseline(rows, curves)
        self.assertAlmostEqual(np.mean([
            row["equal_quality_fixed_accuracy"] for row in rows
        ]), 0.30)
        self.assertEqual(rows[0]["equal_quality_fixed_n_low"], 1)
        self.assertEqual(rows[0]["equal_quality_fixed_n_high"], 2)

    def test_reported_methods_include_family_restrictions_and_selection(self):
        self.assertEqual(
            set(METHOD_FAMILY),
            {"gaussian_calibrated_probability",
             "shifted_exponential_calibrated_probability",
             "train_selected_ucb"},
        )


if __name__ == "__main__":
    unittest.main()
