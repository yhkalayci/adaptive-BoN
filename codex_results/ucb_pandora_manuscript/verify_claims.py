"""Verify every quantitative claim in the simplified-policy manuscript."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CODEX_RESULTS = ROOT.parent
ALIGNMENT = CODEX_RESULTS / "results/alignment/simplification/final_confirmation"
CODING_UTILITY = (
    CODEX_RESULTS / "results/coding/simplification/final_confirmation_reservation"
)
CODING_TARGET = (
    CODEX_RESULTS / "results/coding/simplification/target_final_profile_only"
)
CODING_PROFILE = (
    CODEX_RESULTS / "results/coding/simplification/target_development/"
    "simple_policy_profile.csv"
)
REVISED = ROOT / "artifacts/revised_benchmarks"
ALIGNMENT_VARIANT = "local_exp_open5_conf06"
CODING_VARIANT = "global_exp_q75_conf08_085x"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, tolerance=5e-3):
    assert math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance), (
        actual,
        expected,
    )


def verify_alignment_policy() -> None:
    manifest = json.loads((ALIGNMENT / "METHOD.json").read_text())
    selected = next(
        item for item in manifest["variants"] if item["name"] == ALIGNMENT_VARIANT
    )
    assert selected["uses_training_scale"] is False
    assert selected["min_open"] == 5
    assert selected["config"]["reward_prior_strength"] == 0.0
    assert selected["config"]["cost_prior_strength"] == 0.0
    assert selected["config"]["benchmark_calibration"] == 0.0
    close(selected["config"]["confidence_scale"], 0.6)
    close(selected["config"]["ei_bonus_scale"], 0.002)

    variants = {
        row["variant"]: row for row in rows(ALIGNMENT / "variant_summary.csv")
    }
    simple = variants[ALIGNMENT_VARIANT]
    close(simple["mean_utility_delta_vs_reference"], -0.000398823)
    assert int(simple["positive_utility_cells"]) == 30
    close(simple["mean_target_abs_error_pp"], 2.491235)
    assert int(simple["positive_target_saving_cells"]) == 22

    h2h = [
        row for row in rows(ALIGNMENT / "head_to_head_cells.csv")
        if row["variant"] == ALIGNMENT_VARIANT
    ]
    assert len(h2h) == 30
    assert all(float(row["adaptive_score_ci_low"]) > 0.5 for row in h2h)
    means = [float(row["adaptive_score_mean"]) for row in h2h]
    close(min(means), 0.511618)
    close(max(means), 0.632381)


def verify_alignment_results() -> None:
    summary = rows(REVISED / "alignment_benchmark_summary.csv")
    target = rows(REVISED / "alignment_target_summary.csv")
    target_splits = rows(REVISED / "alignment_target_splits.csv")
    assert len(summary) == 30
    assert len(target) == 35
    assert len(target_splits) == 350

    metrics = (
        "relative_improvement_vs_train_fixed_pct_mean",
        "relative_improvement_vs_oracle_fixed_n_pct_mean",
        "relative_improvement_vs_oracle_budget_pct_mean",
    )
    expected_overall = (4.647599, 3.808493, -2.373127)
    for metric, expected in zip(metrics, expected_overall):
        close(np.mean([float(row[metric]) for row in summary]), expected)
    assert [sum(float(row[m]) > 0 for row in summary) for m in metrics] == [30, 28, 6]

    by_generator = {
        "gemma2_9b": (4.390001, 3.764285, -0.802697),
        "llama3.1_8b": (6.645194, 5.630190, -2.363199),
        "llama3.2_3b": (5.322395, 4.477418, -3.220379),
        "mistral_7b": (2.953835, 2.107280, -1.445032),
        "qwen2.5_7b": (3.926570, 3.063292, -4.034328),
    }
    for generator, expected in by_generator.items():
        group = [row for row in summary if row["generator"] == generator]
        for metric, wanted in zip(metrics, expected):
            close(np.mean([float(row[metric]) for row in group]), wanted)

    assert all(
        row["adaptive_target_reached_train"].lower() == "true"
        and row["fixed_target_reached_train"].lower() == "true"
        for row in target_splits
    )
    adaptive_mae = np.mean([
        abs(float(row["adaptive_test_quality"]) - float(row["target_quality"]))
        * 100.0 for row in target_splits
    ])
    fixed_mae = np.mean([
        abs(float(row["fixed_test_quality"]) - float(row["target_quality"]))
        * 100.0 for row in target_splits
    ])
    close(adaptive_mae, 2.491235)
    close(fixed_mae, 1.086442)
    assert sum(
        float(row["character_saving_vs_train_fixed_pct_mean"]) > 0
        for row in target
    ) == 22
    assert all(
        row["matched_target_reached"].lower() == "true"
        for row in target_splits
    )
    assert max(
        abs(float(row["matched_quality_error_pp"])) for row in target_splits
    ) < 2e-12
    assert sum(
        float(row["character_saving_vs_matched_fixed_pct_mean"]) > 0
        for row in target
    ) == 35
    expected_savings = {
        0.30: -1.991745, 0.35: -5.442784, 0.40: -8.828962,
        0.45: 6.851481, 0.50: 6.896733, 0.55: -0.690454, 0.60: 2.318187,
    }
    for target_value, expected in expected_savings.items():
        group = [row for row in target if float(row["target_quality"]) == target_value]
        close(np.mean([
            float(row["character_saving_vs_train_fixed_pct_mean"]) for row in group
        ]), expected)
    expected_matched_savings = {
        0.30: 11.222873, 0.35: 21.593029, 0.40: 23.263371,
        0.45: 27.166902, 0.50: 34.216853, 0.55: 33.095510,
        0.60: 25.822773,
    }
    for target_value, expected in expected_matched_savings.items():
        group = [row for row in target if float(row["target_quality"]) == target_value]
        close(np.mean([
            float(row["character_saving_vs_matched_fixed_pct_mean"])
            for row in group
        ]), expected)
    close(np.mean([
        float(row["character_saving_vs_matched_fixed_pct_mean"])
        for row in target
    ]), 25.197330)


def verify_coding_policy() -> None:
    manifest = json.loads((CODING_UTILITY / "METHOD.json").read_text())
    assert manifest["fixed_n_guard"] is False
    selected = manifest["variants"][0]
    assert selected["name"] == CODING_VARIANT
    assert selected["family"] == "shifted_exponential"
    close(selected["confidence_scale"], 0.8)
    assert math.isinf(float(selected["reward_prior_strength"]))
    assert selected["cost_prior_strength"] == 0.0
    close(selected["tail_quantile"], 0.75)
    close(selected["tail_decay"], 1.0)
    assert selected["reservation_rule"] == "0.85x"

    split_rows = [
        row for row in rows(CODING_UTILITY / "utility_splits.csv")
        if row["variant"] == CODING_VARIANT
    ]
    assert len(split_rows) == 100
    assert all(
        math.isclose(
            float(row["reservation_divisor"]),
            0.85 * float(row["utility_divisor"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ) for row in split_rows
    )
    variant = next(
        row for row in rows(CODING_UTILITY / "variant_summary.csv")
        if row["variant"] == CODING_VARIANT
    )
    close(variant["mean_utility_delta_vs_reference"], -0.000569418)


def verify_coding_results() -> None:
    summary = rows(REVISED / "coding_benchmark_summary.csv")
    equal = rows(REVISED / "coding_equal_budget_summary.csv")
    assert len(summary) == len(equal) == 10
    metrics = (
        "relative_improvement_vs_train_fixed_pct_mean",
        "relative_improvement_vs_oracle_fixed_n_pct_mean",
        "relative_improvement_vs_train_fixed_budget_pct_mean",
    )
    expected_overall = (2.626350, -0.118616, 3.482824)
    for metric, expected in zip(metrics, expected_overall):
        close(np.mean([float(row[metric]) for row in summary]), expected)
    assert [sum(float(row[m]) > 0 for row in summary) for m in metrics] == [10, 5, 10]
    assert sum(
        float(row["relative_improvement_vs_train_fixed_budget_pct_ci_low"]) > 0
        for row in summary
    ) == 6
    expected_rows = {
        1e5: (7.7316, 2.5230, 4.5210), 2e5: (4.1037, 0.7300, 5.1960),
        3e5: (3.3209, 0.0910, 5.2120), 4e5: (2.6756, 0.2790, 3.3350),
        5e5: (2.4560, 0.0350, 2.8120), 6e5: (1.9785, -0.2830, 2.8930),
        7e5: (1.7859, -0.4700, 2.7790), 8e5: (1.1321, -0.9000, 2.2070),
        9e5: (0.7503, -1.3990, 3.3140), 1e6: (0.3290, -1.7930, 2.5590),
    }
    for row in summary:
        for metric, expected in zip(metrics, expected_rows[float(row["divisor"])]):
            close(row[metric], expected)

    expected_equal = {
        1e5: (1.598829, 0.723867, 2.473791),
        2e5: (1.414462, 0.639157, 2.189766),
        3e5: (1.048408, 0.427883, 1.668933),
        4e5: (0.965467, 0.265419, 1.665515),
        5e5: (0.811154, 0.090475, 1.531833),
        6e5: (0.822815, 0.014041, 1.631589),
    }
    by_divisor = {float(row["divisor"]): row for row in equal}
    for divisor, expected in expected_equal.items():
        row = by_divisor[divisor]
        for field, wanted in zip(
            ("accuracy_gap_pp_mean", "accuracy_gap_pp_ci_low", "accuracy_gap_pp_ci_high"),
            expected,
        ):
            close(row[field], wanted)
    assert all(abs(float(row["cost_mismatch_mean"])) < 1e-9 for row in equal)


def verify_coding_target() -> None:
    split_rows = rows(CODING_TARGET / "target_quality_splits.csv")
    summary = {
        float(row["target_accuracy"]): row
        for row in rows(CODING_TARGET / "target_quality_summary.csv")
    }
    assert len(split_rows) == 30
    assert {float(row["divisor"]) for row in split_rows} == {
        25_000.0, 140_000.0, 2_800_000.0
    }
    expected = {
        0.25: (0.260218, 54.041668, 8.061959, 74.709809,
               34.179497, 29.272386, 39.137837, 0.339563),
        0.30: (0.295288, 74.406230, 27.041069, 86.185504,
               28.795420, 18.864404, 37.004469, 0.120063),
        0.35: (0.346181, 54.168460, 0.829210, 73.576842,
               38.546438, -10.675850, 65.834707, -0.013561),
    }
    fields = (
        "adaptive_accuracy_mean", "aggregate_saving_vs_train_fixed_pct",
        "aggregate_saving_vs_train_fixed_ci_low",
        "aggregate_saving_vs_train_fixed_ci_high",
        "aggregate_saving_vs_oracle_fixed_pct",
        "aggregate_saving_vs_oracle_fixed_ci_low",
        "aggregate_saving_vs_oracle_fixed_ci_high",
        "oracle_accuracy_overshoot_pp_mean",
    )
    for target, wanted in expected.items():
        for field, value in zip(fields, wanted):
            close(summary[target][field], value)

    exact_quality = {
        0.25: (0.260218, -0.0, 8.0, 9.0, 0.157897,
               28.259843, 22.771166, 33.289810),
        0.30: (0.295288, 0.0, 18.0, 19.0, 0.602514,
               22.237559, 12.148499, 31.167684),
        0.35: (0.346181, -0.0, 77.0, 78.0, 0.450623,
               1.126310, -16.924224, 17.149719),
    }
    exact_fields = (
        "equal_quality_fixed_accuracy_mean",
        "equal_quality_accuracy_error_pp_mean",
        "equal_quality_fixed_n_low_mean",
        "equal_quality_fixed_n_high_mean",
        "equal_quality_fixed_high_weight_mean",
        "aggregate_saving_vs_equal_quality_fixed_pct",
        "aggregate_saving_vs_equal_quality_fixed_ci_low",
        "aggregate_saving_vs_equal_quality_fixed_ci_high",
    )
    for target, wanted in exact_quality.items():
        for field, value in zip(exact_fields, wanted):
            close(summary[target][field], value)
        close(
            summary[target]["equal_quality_fixed_accuracy_mean"],
            float(summary[target]["adaptive_accuracy_mean"]),
            tolerance=2e-12,
        )
        close(summary[target]["equal_quality_target_reached_fraction"], 1.0)

    manifest = json.loads((CODING_TARGET / "METHOD.json").read_text())
    assert manifest["simple_global_policy"] is True
    assert manifest["policy_is_mixture"] is False
    assert manifest["cap_factor"] is None
    assert manifest["target_margin"] == 0.0
    assert manifest["development_profile"]["train_accuracy_weight"] == 0.0
    assert manifest["development_profile"]["profile_accuracy_weight"] == 1.0
    profile = rows(CODING_PROFILE)
    assert len(profile) == 19
    assert all(row["family"] == "shifted_exponential" for row in profile)


def verify_profile_overlap() -> None:
    manifest = json.loads((CODING_TARGET / "METHOD.json").read_text())
    seed = int(manifest["seed"])
    development = manifest["development_profile"]["development_split_ids"]
    final = range(manifest["split_start"], manifest["split_start"] + manifest["splits"])

    def test_half(split):
        order = np.random.default_rng(seed + split).permutation(83)
        return set(order[83 // 2:])

    development_union = set().union(*(test_half(split) for split in development))
    assert len(development_union) == 81
    overlap = [len(test_half(split) & development_union) for split in final]
    assert min(overlap) == 40 and max(overlap) == 41


def main() -> None:
    verify_alignment_policy()
    verify_alignment_results()
    verify_coding_policy()
    verify_coding_results()
    verify_coding_target()
    verify_profile_overlap()
    print("All simplified-policy manuscript claims verified against result artifacts.")


if __name__ == "__main__":
    main()
