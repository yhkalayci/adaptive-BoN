import csv
import tempfile
import unittest
from pathlib import Path

from coding_utility_search import load_profile, policy_grids, selection_score
from profiled_utility_frontier import (
    leave_one_split_out_meta_selection,
    load_profile_candidates,
    select_frozen,
)


class CodingUtilityImprovementTests(unittest.TestCase):
    def test_uncapped_decay_grid_is_exponential_uncapped_and_has_expected_decays(self):
        records, _ = policy_grids(("exp_tail_uncapped_decay",))
        self.assertEqual(len(records), 144)
        self.assertEqual(
            {record["config"].family for record in records},
            {"shifted_exponential"},
        )
        self.assertTrue(all(record["config"].cap_factor is None for record in records))
        self.assertEqual(
            {record["config"].tail_decay for record in records},
            {0.0, 0.25, 0.5, 1.0},
        )

    def test_current_grid_is_exponential_only(self):
        records, _ = policy_grids(("exp_tail_current",))
        self.assertEqual(len(records), 252)
        self.assertEqual(
            {record["config"].family for record in records},
            {"shifted_exponential"},
        )
        self.assertEqual({record["config"].tail_decay for record in records}, {0.0})
        self.assertTrue(any(record["config"].cap_factor is not None for record in records))

    def test_profile_blend_uses_requested_weight_and_risk_penalty(self):
        records, _ = policy_grids(("exp_tail_uncapped_decay",))
        record = records[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "grid", "grid_config_id", "divisor", "test_utility_mean",
                    "test_utility_std", "splits",
                ))
                writer.writeheader()
                writer.writerow({
                    "grid": "exp_tail_uncapped_decay", "grid_config_id": 0,
                    "divisor": 100000.0, "test_utility_mean": 0.30,
                    "test_utility_std": 0.04, "splits": 5,
                })
            profile = load_profile(path)
            score, robust = selection_score(
                0.10, record, 100000.0, profile,
                profile_weight=0.75, risk_penalty=0.5,
            )
        self.assertAlmostEqual(robust, 0.28)
        self.assertAlmostEqual(score, 0.25 * 0.10 + 0.75 * 0.28)

    def test_frontier_shortlist_is_uncapped_exponential_and_genuine_ucb(self):
        profile = (
            Path(__file__).resolve().parents[1]
            / "code" / "coding_scripts" / "coding_target_tail_decay_profile.csv"
        )
        records = load_profile_candidates(profile, (100_000.0, 500_000.0), top_k=3)
        self.assertEqual(len(records), 6)
        self.assertEqual({record["utility_divisor"] for record in records}, {
            100_000.0, 500_000.0,
        })
        self.assertTrue(all(
            record["config"].family == "shifted_exponential"
            and record["config"].confidence_scale > 0.0
            and record["config"].cap_factor is None
            for record in records
        ))

    def test_frontier_validation_selection_applies_risk_penalty(self):
        common = {
            "utility_divisor": 100_000.0,
            "reservation_divisor_id": 0,
            "family": "shifted_exponential",
            "calibration": "bounded_identity",
            "confidence_scale": 0.2,
            "reward_prior_strength": 5.0,
            "cost_prior_strength": 0.0,
            "cap_factor": None,
            "tail_quantile": 0.75,
            "tail_decay": 1.0,
            "profile_accuracy": 0.3,
            "profile_chars_geomean": 1000.0,
            "profile_utility": 0.29,
            "fixed_utility_mean": 0.2,
            "utility_gap_mean": 0.02,
            "relative_utility_gain_pct_mean": 10.0,
            "positive_gap_splits": 4,
            "splits": 5,
        }
        risky = common | {
            "reservation_divisor": 50_000.0, "config_id": 1,
            "profile_rank": 1, "adaptive_utility_mean": 0.24,
            "adaptive_utility_std": 0.05,
        }
        stable = common | {
            "reservation_divisor": 70_000.0, "config_id": 2,
            "profile_rank": 2, "adaptive_utility_mean": 0.23,
            "adaptive_utility_std": 0.01,
        }
        selected = select_frozen(
            [risky, stable], risk_penalty=0.5, profile_weight=0.0
        )
        self.assertEqual(selected[0]["config_id"], 2)

    def test_meta_selection_audit_covers_frozen_weight_grid(self):
        rows = []
        for split in (1, 2):
            for config_id, gap in ((1, 0.02), (2, 0.01)):
                rows.append({
                    "split": split, "utility_divisor": 100_000.0,
                    "config_id": config_id,
                    "reservation_divisor": 50_000.0 + config_id,
                    "profile_utility": 0.2 + 0.01 * config_id,
                    "adaptive_utility": 0.25 + gap,
                    "utility_gap": gap,
                    "relative_utility_gain_pct": 100.0 * gap / 0.25,
                })
        audit = leave_one_split_out_meta_selection(rows)
        self.assertEqual(len(audit), 24)
        chosen = next(
            row for row in audit
            if row["profile_weight"] == 0.5 and row["risk_penalty"] == 0.1
        )
        self.assertEqual(chosen["heldout_cells"], 2)


if __name__ == "__main__":
    unittest.main()
