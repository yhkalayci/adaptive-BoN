import unittest

import numpy as np

from build_revised_benchmarks import (
    adjacent_cost_mix,
    adjacent_quality_mix,
    evaluate_fixed_character_budget,
    minimum_cost_quality_mix,
    oracle_budget_choice,
    select_cheapest_target,
    tune_fixed_character_budgets,
)


class OracleBudgetChoiceTest(unittest.TestCase):
    def test_adjacent_cost_mix_matches_interior_target(self):
        curve = np.asarray([10.0, 25.0, 60.0])
        low, high, weight = adjacent_cost_mix(curve, 32.0)
        self.assertEqual((low, high), (1, 2))
        self.assertAlmostEqual(
            (1.0 - weight) * curve[low] + weight * curve[high], 32.0
        )

    def test_adjacent_quality_mix_matches_interior_target(self):
        curve = np.asarray([0.20, 0.27, 0.35])
        low, high, weight = adjacent_quality_mix(curve, 0.30)
        self.assertEqual((low, high), (1, 2))
        self.assertAlmostEqual(
            (1.0 - weight) * curve[low] + weight * curve[high], 0.30
        )

    def test_quality_oracle_finds_nonadjacent_lower_cost_mix(self):
        low, high, weight, reached = minimum_cost_quality_mix(
            np.asarray([0.10, 0.20, 0.40]),
            np.asarray([10.0, 100.0, 110.0]),
            0.25,
        )
        self.assertTrue(reached)
        self.assertEqual((low, high), (0, 2))
        self.assertAlmostEqual(weight, 0.5)

    def test_enumerates_prompt_specific_budget_allocations(self):
        # At R=20, lengths [10, 20] imply allocations [2, 1], which is the
        # unique best allocation in this example.
        utility = np.asarray([[0.1, 0.9, 0.8], [0.7, 0.6, 0.5]])
        chars = np.asarray([[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]])
        budget, value, mean_n, actual_chars = oracle_budget_choice(
            utility, chars, np.asarray([10.0, 20.0])
        )
        self.assertEqual(budget, 20.0)
        self.assertEqual(value, 0.8)
        self.assertEqual(mean_n, 1.5)
        self.assertEqual(actual_chars, 20.0)

    def test_requires_positive_mean_lengths(self):
        with self.assertRaises(ValueError):
            oracle_budget_choice(
                np.ones((1, 2)), np.ones((1, 2)), np.asarray([0.0])
            )

    def test_train_selected_fixed_character_threshold(self):
        accuracy = np.asarray([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        chars = np.asarray([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        selected = tune_fixed_character_budgets(accuracy, chars, (100.0,))
        budget, utility, mean_accuracy, mean_chars = selected[100.0]
        self.assertEqual(budget, 20.0)
        self.assertAlmostEqual(utility, 0.7)
        self.assertEqual(mean_accuracy, 1.0)
        self.assertEqual(mean_chars, 30.0)
        evaluated = evaluate_fixed_character_budget(
            accuracy, chars, budget=10.0, divisor=100.0
        )
        self.assertAlmostEqual(evaluated[0], 0.3)
        self.assertEqual(evaluated[1:], (0.5, 20.0, 2.0))


class TargetSelectionTest(unittest.TestCase):
    def test_selects_one_cheapest_target_reaching_policy(self):
        index, reached = select_cheapest_target(
            np.asarray([0.20, 0.31, 0.35]),
            np.asarray([10.0, 30.0, 50.0]),
            0.30,
        )
        self.assertTrue(reached)
        self.assertEqual(index, 1)

    def test_unattainable_target_uses_cheapest_maximum_quality(self):
        index, reached = select_cheapest_target(
            np.asarray([0.20, 0.29, 0.29]),
            np.asarray([10.0, 50.0, 40.0]),
            0.30,
        )
        self.assertFalse(reached)
        self.assertEqual(index, 2)


if __name__ == "__main__":
    unittest.main()
