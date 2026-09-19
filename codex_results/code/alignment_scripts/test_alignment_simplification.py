import unittest

import numpy as np

from alignment_pandora_ucb import PriorFit, pandora_stop_many
from alignment_simplification_study import (
    VARIANTS,
    prefix_only_placeholder_prior,
)


class AlignmentSimplificationTest(unittest.TestCase):
    def test_training_free_variants_have_no_active_training_weight(self):
        for variant in VARIANTS:
            if variant.uses_training_scale:
                continue
            self.assertEqual(variant.config.reward_prior_strength, 0.0)
            self.assertEqual(variant.config.cost_prior_strength, 0.0)
            self.assertEqual(variant.config.benchmark_calibration, 0.0)

    def test_placeholder_values_cannot_change_prefix_only_stops(self):
        rng = np.random.default_rng(9)
        rewards = rng.normal(size=60)
        chars = rng.integers(20, 500, size=60).astype(np.float64)
        first = prefix_only_placeholder_prior()
        second = PriorFit(
            raw_sigma=999.0,
            gaussian_quantile_z=-999.0,
            raw_tail_scales={0.5: 777.0},
            exp_tail_ratios={0.5: 555.0},
            raw_tail_quantile_k={0.5: -333.0},
            exp_tail_quantile_k={0.5: -111.0},
            cost_beta=np.full(6, 8.0),
            feature_mean=np.full(5, 4.0),
            feature_scale=np.full(5, 2.0),
        )
        for variant in VARIANTS:
            if variant.uses_training_scale:
                continue
            one = pandora_stop_many(
                rewards, chars, (1e5, 1e6), 1.0, first,
                variant.config, variant.min_open,
            )
            two = pandora_stop_many(
                rewards, chars, (1e5, 1e6), 1e9, second,
                variant.config, variant.min_open,
            )
            self.assertEqual(one, two)


if __name__ == "__main__":
    unittest.main()
