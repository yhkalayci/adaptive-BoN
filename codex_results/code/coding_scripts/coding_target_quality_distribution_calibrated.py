"""Target-quality coding evaluation in isotonic probability space.

Each outer split fits raw reward -> P(correct) isotonic regression on the
41-problem training half and freezes the map.  Gaussian and conditional
shifted-exponential reward distributions are then fitted after transformation.
For each requested accuracy, train-only cross-fitted trajectories optionally
combine with a frozen development transfer profile to select exactly one
globally uncapped UCB-Pandora configuration and cost divisor. There is no
policy mixture, Fixed-N guard, Fixed-N fallback, target margin, or test-time
selection.

The deployable Fixed-N comparator is the cheapest integer N reaching the
requested target on train.  A second, evaluation-only oracle matches the
adaptive policy's held-out accuracy exactly.  If no single integer N has that
exact accuracy, the oracle randomizes between two fixed counts before seeing
any program; this remains non-adaptive and gives the minimum expected
character cost among all such two-count mixtures.

The primary target-quality plot keeps the deployable comparison---select N on
the outer-training half, freeze it, and evaluate that same N on test---and adds
the exact-aggregate-quality oracle as a separate diagnostic panel.  The plot
therefore distinguishes train-to-test target transfer from efficiency at
matched realized quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from coding_ucb_three_objectives import (
    PolicyConfig,
    crossfit_evaluate,
    evaluate,
    exact_fixed_accuracy,
    fit_config_calibrations,
    fit_prior,
    fit_reward_space_isotonic,
    load_problems,
    make_permutations,
    reward_space_calibration_record,
    split_problems,
    transform_reward_space,
    write_csv,
)


TARGETS = (0.25, 0.30, 0.35)
PROFILE_TRAIN_ACCURACY_WEIGHT = 0.10
TARGET_DIVISORS = (
    25_000.0, 35_000.0, 50_000.0, 70_000.0, 100_000.0,
    140_000.0, 200_000.0, 280_000.0, 400_000.0, 560_000.0,
    700_000.0, 1_000_000.0, 1_400_000.0, 2_000_000.0,
    2_800_000.0, 4_000_000.0, 5_600_000.0, 7_000_000.0,
    10_000_000.0,
)
FROZEN_BOUNDED_CONFIG_BY_TARGET = {0.25: 51, 0.30: 3, 0.35: 17}
FROZEN_PREDICTED_CONFIG_BY_TARGET = {0.25: 50, 0.30: 49, 0.35: 26}
METHOD_FAMILY = {
    "gaussian_calibrated_probability": "gaussian",
    "shifted_exponential_calibrated_probability": "shifted_exponential",
    "train_selected_ucb": None,
}
LABELS = {
    "gaussian_calibrated_probability": "Gaussian on isotonic probability",
    "shifted_exponential_calibrated_probability": "Exp-tail on isotonic probability",
    "train_selected_ucb": "Train-selected UCB Pandora",
}
COLORS = {
    "gaussian_calibrated_probability": "#3A6EA5",
    "shifted_exponential_calibrated_probability": "#C45A00",
    "train_selected_ucb": "#1B7F3A",
}


def parse_floats(value):
    return tuple(float(item) for item in value.split(",") if item.strip())


def candidate_configs(
    bounded_probability_models=False, tail_decay_grid=False,
    simple_global_policy=False,
) -> list[PolicyConfig]:
    """Matched Gaussian/exponential grid with no Fixed-N-derived cap."""
    if simple_global_policy:
        return [PolicyConfig(
            family="shifted_exponential",
            calibration="bounded_identity",
            confidence_scale=0.8,
            reward_prior_strength=math.inf,
            cost_prior_strength=0.0,
            cap_factor=None,
            tail_quantile=0.75,
            tail_decay=1.0,
        )]
    calibration = (
        "bounded_identity" if bounded_probability_models else "identity"
    )
    common = itertools.product(
        (0.0, 0.2, 0.8),
        (5.0, 20.0, 1e6),
        (0.0, 10.0),
    )
    common = tuple(common)
    gaussian = [
        PolicyConfig("gaussian", calibration, cs, rp, cp, None, None)
        for cs, rp, cp in common
    ]
    exponential = [
        PolicyConfig(
            "shifted_exponential", calibration, cs, rp, cp, None, quantile
        )
        for quantile in (0.5, 0.75)
        for cs, rp, cp in common
    ]
    base_configs = gaussian + exponential
    configs = [
        PolicyConfig(
            config.family, config.calibration, config.confidence_scale,
            config.reward_prior_strength, config.cost_prior_strength,
            config.cap_factor, config.tail_quantile, tail_decay,
        )
        for config in base_configs
        for tail_decay in (
            (0.0, 0.25, 0.5, 1.0) if tail_decay_grid else (0.0,)
        )
    ]
    if any(config.cap_factor is not None for config in configs):
        raise AssertionError("target-quality candidates must be globally uncapped")
    return configs


def exact_fixed_curves(problems):
    accuracy = exact_fixed_accuracy(problems)
    mean_chars = float(np.mean(np.concatenate([
        values[2] for values in problems.values()
    ])))
    chars = mean_chars * np.arange(1, len(accuracy) + 1, dtype=np.float64)
    return accuracy, chars


def select_oracle_fixed_n(accuracy_curve, char_curve, target_accuracy):
    """Cheapest integer N attaining target; cheapest maximum if unreachable."""
    accuracy = np.asarray(accuracy_curve, dtype=np.float64)
    chars = np.asarray(char_curve, dtype=np.float64)
    if accuracy.ndim != 1 or chars.shape != accuracy.shape or not len(accuracy):
        raise ValueError("accuracy and character curves must be nonempty 1-D peers")
    if not np.all(np.isfinite(accuracy)) or not np.all(np.isfinite(chars)):
        raise ValueError("accuracy and character curves must be finite")
    eligible = np.flatnonzero(accuracy >= float(target_accuracy) - 1e-12)
    reached = bool(len(eligible))
    if not reached:
        eligible = np.flatnonzero(accuracy >= float(np.max(accuracy)) - 1e-12)
    index = int(min(eligible, key=lambda item: (chars[item], item)))
    return index + 1, float(accuracy[index]), float(chars[index]), reached


def select_matched_fixed_mix(accuracy_curve, char_curve, target_accuracy):
    """Minimum-cost non-adaptive Fixed-N mixture at exact expected quality.

    A deterministic integer N is returned whenever it hits the target.  When
    integer accuracies straddle the target, randomization between two counts
    is mathematically necessary for an exact expected-quality comparison.
    The random choice is made before generation and never depends on observed
    rewards, correctness, or character counts.
    """
    accuracy = np.asarray(accuracy_curve, dtype=np.float64)
    chars = np.asarray(char_curve, dtype=np.float64)
    if accuracy.ndim != 1 or chars.shape != accuracy.shape or not len(accuracy):
        raise ValueError("accuracy and character curves must be nonempty 1-D peers")
    if not np.all(np.isfinite(accuracy)) or not np.all(np.isfinite(chars)):
        raise ValueError("accuracy and character curves must be finite")
    target = float(target_accuracy)
    low_ids = np.flatnonzero(accuracy <= target + 1e-12)
    high_ids = np.flatnonzero(accuracy >= target - 1e-12)
    if not len(low_ids) or not len(high_ids):
        error = np.abs(accuracy - target)
        closest_error = float(np.min(error))
        candidates = np.flatnonzero(error <= closest_error + 1e-12)
        index = int(min(candidates, key=lambda item: (chars[item], int(item))))
        return (
            index + 1, index + 1, 0.0,
            float(accuracy[index]), float(chars[index]), False,
        )

    best = None
    for low in low_ids:
        q_low = float(accuracy[low])
        q_high = accuracy[high_ids]
        denominator = q_high - q_low
        weights = np.divide(
            target - q_low,
            denominator,
            out=np.zeros_like(denominator),
            where=np.abs(denominator) > 1e-15,
        )
        weights = np.clip(weights, 0.0, 1.0)
        mixed_chars = (1.0 - weights) * chars[low] + weights * chars[high_ids]
        for position, high in enumerate(high_ids):
            weight = float(weights[position])
            mixed_accuracy = (
                (1.0 - weight) * q_low + weight * float(accuracy[high])
            )
            if abs(mixed_accuracy - target) > 2e-12:
                continue
            candidate = (
                float(mixed_chars[position]),
                abs(int(high) - int(low)),
                int(low), int(high), weight,
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise AssertionError("bracketing Fixed-N policies failed to match target")
    mixed_chars, _, low, high, weight = best
    mixed_accuracy = (
        (1.0 - weight) * float(accuracy[low])
        + weight * float(accuracy[high])
    )
    return (
        low + 1, high + 1, weight,
        float(mixed_accuracy), float(mixed_chars), True,
    )


def select_single_policy(
    values, configs, divisors, target, family=None, config_ids=None,
    selection_metric="observed_accuracy",
):
    """Cheapest one-policy train candidate attaining target, without margin.

    ``calibrated_probability`` uses the mean selected isotonic probability
    already returned by the evaluator (metric 3).  This avoids reintroducing
    binary-outcome noise after fitting the probability reward transform.
    """
    if selection_metric not in ("observed_accuracy", "calibrated_probability"):
        raise ValueError(f"unknown selection metric: {selection_metric}")
    observed_accuracy = np.asarray(
        values[..., 0].mean(axis=2), dtype=np.float64
    )
    calibrated_probability = np.asarray(
        values[..., 3].mean(axis=2), dtype=np.float64
    )
    selection_accuracy = (
        calibrated_probability if selection_metric == "calibrated_probability"
        else observed_accuracy
    )
    chars = np.asarray(values[..., 1].mean(axis=2), dtype=np.float64)
    candidates = []
    allowed_ids = None if config_ids is None else set(config_ids)
    for config_id, config in enumerate(configs):
        if allowed_ids is not None and config_id not in allowed_ids:
            continue
        if family is not None and config.family != family:
            continue
        for divisor_id, divisor in enumerate(divisors):
            candidates.append({
                "config_id": config_id,
                "divisor_id": divisor_id,
                "divisor": float(divisor),
                **asdict(config),
                "train_accuracy": float(
                    observed_accuracy[config_id, divisor_id]
                ),
                "train_predicted_accuracy": float(
                    calibrated_probability[config_id, divisor_id]
                ),
                "selection_accuracy": float(
                    selection_accuracy[config_id, divisor_id]
                ),
                "train_chars": float(chars[config_id, divisor_id]),
            })
    eligible = [
        row for row in candidates
        if row["selection_accuracy"] >= float(target) - 1e-12
    ]
    if eligible:
        selected = min(eligible, key=lambda row: (
            row["train_chars"], row["selection_accuracy"] - target,
            row["config_id"], row["divisor_id"],
        ))
        return {"target_reached_train": True, "fallback_reason": None, **selected}
    maximum = max(row["selection_accuracy"] for row in candidates)
    plateau = [
        row for row in candidates
        if row["selection_accuracy"] >= maximum - 1e-12
    ]
    selected = min(plateau, key=lambda row: (
        row["train_chars"], row["config_id"], row["divisor_id"],
    ))
    return {
        "target_reached_train": False,
        "fallback_reason": "target_unattainable_cheapest_maximum_accuracy",
        **selected,
    }


def load_development_profile(path, configs, divisors):
    """Load and validate the frozen development transfer profile."""
    path = Path(path)
    divisors = np.asarray(divisors, dtype=np.float64)
    shape = (len(configs), len(divisors))
    accuracy = np.full(shape, np.nan, dtype=np.float64)
    log_chars = np.full(shape, np.nan, dtype=np.float64)
    seen = set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != accuracy.size:
        raise ValueError(
            f"development profile has {len(rows)} rows; expected {accuracy.size}"
        )
    for row in rows:
        config_id = int(row["config_id"])
        divisor_id = int(row["divisor_id"])
        key = (config_id, divisor_id)
        if key in seen:
            raise ValueError(f"duplicate development-profile key {key}")
        if not (0 <= config_id < len(configs)) or not (
            0 <= divisor_id < len(divisors)
        ):
            raise ValueError(f"out-of-range development-profile key {key}")
        seen.add(key)
        config = configs[config_id]
        if row["family"] != config.family or row["calibration"] != config.calibration:
            raise ValueError(f"configuration mismatch at development-profile key {key}")
        for field in (
            "confidence_scale", "reward_prior_strength", "cost_prior_strength",
            "tail_decay",
        ):
            if not math.isclose(
                float(row[field]), float(getattr(config, field)),
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{field} mismatch at development-profile key {key}"
                )
        for field in ("cap_factor", "tail_quantile"):
            expected = getattr(config, field)
            actual = None if row[field] == "" else float(row[field])
            if actual != expected:
                raise ValueError(
                    f"{field} mismatch at development-profile key {key}"
                )
        if not math.isclose(
            float(row["divisor"]), divisors[divisor_id],
            rel_tol=0.0, abs_tol=1e-9,
        ):
            raise ValueError(f"divisor mismatch at development-profile key {key}")
        accuracy[key] = float(row["profile_accuracy"])
        log_chars[key] = float(row["profile_log_chars"])
    if not np.all(np.isfinite(accuracy)) or not np.all(np.isfinite(log_chars)):
        raise ValueError("development profile contains missing or non-finite values")
    if np.any((accuracy < 0.0) | (accuracy > 1.0)):
        raise ValueError("development profile accuracy must lie in [0, 1]")
    return {
        "accuracy": accuracy,
        "log_chars": log_chars,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def select_profiled_policy(
    values, configs, divisors, target, profile, family=None,
    train_accuracy_weight=PROFILE_TRAIN_ACCURACY_WEIGHT,
):
    """Select one policy using a frozen development transfer profile.

    The requested target is not shifted. A small contribution from the
    current split's cross-fitted accuracy adapts the frozen development
    attainment estimate; development geometric-mean characters rank feasible
    policies without selecting on the noisy current split cost.
    """
    if not 0.0 <= train_accuracy_weight <= 1.0:
        raise ValueError("train_accuracy_weight must lie in [0, 1]")
    observed_accuracy = np.asarray(values[..., 0].mean(axis=2), dtype=np.float64)
    calibrated_probability = np.asarray(
        values[..., 3].mean(axis=2), dtype=np.float64
    )
    train_chars = np.asarray(values[..., 1].mean(axis=2), dtype=np.float64)
    profile_accuracy = np.asarray(profile["accuracy"], dtype=np.float64)
    profile_log_chars = np.asarray(profile["log_chars"], dtype=np.float64)
    if profile_accuracy.shape != observed_accuracy.shape or (
        profile_log_chars.shape != observed_accuracy.shape
    ):
        raise ValueError("development profile does not match candidate grid")
    selection_accuracy = (
        train_accuracy_weight * observed_accuracy
        + (1.0 - train_accuracy_weight) * profile_accuracy
    )
    selection_chars = np.exp(profile_log_chars)
    candidates = []
    for config_id, config in enumerate(configs):
        if family is not None and config.family != family:
            continue
        for divisor_id, divisor in enumerate(divisors):
            candidates.append({
                "config_id": config_id,
                "divisor_id": divisor_id,
                "divisor": float(divisor),
                **asdict(config),
                "train_accuracy": float(observed_accuracy[config_id, divisor_id]),
                "train_predicted_accuracy": float(
                    calibrated_probability[config_id, divisor_id]
                ),
                "selection_accuracy": float(
                    selection_accuracy[config_id, divisor_id]
                ),
                "train_chars": float(train_chars[config_id, divisor_id]),
                "profile_accuracy": float(profile_accuracy[config_id, divisor_id]),
                "profile_log_chars": float(
                    profile_log_chars[config_id, divisor_id]
                ),
                "profile_chars": float(selection_chars[config_id, divisor_id]),
                "selection_chars": float(selection_chars[config_id, divisor_id]),
            })
    eligible = [
        row for row in candidates
        if row["selection_accuracy"] >= float(target) - 1e-12
    ]
    if eligible:
        selected = min(eligible, key=lambda row: (
            row["selection_chars"], row["selection_accuracy"] - target,
            row["config_id"], row["divisor_id"],
        ))
        return {"target_reached_train": True, "fallback_reason": None, **selected}
    maximum = max(row["selection_accuracy"] for row in candidates)
    plateau = [
        row for row in candidates
        if row["selection_accuracy"] >= maximum - 1e-12
    ]
    selected = min(plateau, key=lambda row: (
        row["selection_chars"], row["config_id"], row["divisor_id"],
    ))
    return {
        "target_reached_train": False,
        "fallback_reason": "target_unattainable_cheapest_maximum_accuracy",
        **selected,
    }


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    half = float(
        student_t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1) / math.sqrt(len(values))
    )
    return mean, mean - half, mean + half


def ratio_saving_interval(adaptive, baseline, seed):
    adaptive = np.asarray(adaptive, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    saving = 100.0 * (baseline.mean() - adaptive.mean()) / baseline.mean()
    if len(adaptive) <= 1:
        return float(saving), float(saving), float(saving)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(adaptive), size=(50_000, len(adaptive)))
    boot_adaptive = adaptive[indices].mean(axis=1)
    boot_baseline = baseline[indices].mean(axis=1)
    boot = 100.0 * (boot_baseline - boot_adaptive) / boot_baseline
    low, high = np.quantile(boot, (0.025, 0.975))
    return float(saving), float(low), float(high)


def heldout_fixed_curves(problems, args):
    """Reconstruct exact Fixed-N curves for the final held-out halves."""
    curves = {}
    audit_rows = []
    for split in range(args.split_start, args.split_start + args.splits):
        raw_train, raw_test = split_problems(problems, args.seed + split)
        calibration = fit_reward_space_isotonic(raw_train)
        test = transform_reward_space(raw_test, calibration)
        accuracy, chars = exact_fixed_curves(test)
        curves[split] = (accuracy, chars)
        audit_rows.extend({
            "split": split,
            "n": index + 1,
            "accuracy": float(accuracy[index]),
            "characters": float(chars[index]),
        } for index in range(len(accuracy)))
    return curves, audit_rows


def attach_equal_quality_fixed_baseline(rows, fixed_curves):
    """Attach one test-oracle Fixed-N rule at exact aggregate accuracy.

    For each reported method and target, average the ten held-out Fixed-N
    curves, then choose the minimum-cost deterministic N or two-count
    pre-generation randomization whose average accuracy equals the adaptive
    average.  The same oracle choice is evaluated on every split, permitting
    paired uncertainty intervals while guaranteeing exact aggregate quality.
    """
    for method in sorted({row["method"] for row in rows}):
        for target in sorted({
            row["target_accuracy"] for row in rows if row["method"] == method
        }):
            group = [
                row for row in rows
                if row["method"] == method and row["target_accuracy"] == target
            ]
            split_ids = [int(row["split"]) for row in group]
            mean_accuracy_curve = np.mean(
                [fixed_curves[split][0] for split in split_ids], axis=0
            )
            mean_char_curve = np.mean(
                [fixed_curves[split][1] for split in split_ids], axis=0
            )
            adaptive_accuracy = float(np.mean([
                row["adaptive_accuracy"] for row in group
            ]))
            (low_n, high_n, high_weight, matched_accuracy, _,
             reached) = select_matched_fixed_mix(
                mean_accuracy_curve, mean_char_curve, adaptive_accuracy
            )
            for row in group:
                accuracy_curve, char_curve = fixed_curves[int(row["split"])]
                matched_split_accuracy = float(
                    (1.0 - high_weight) * accuracy_curve[low_n - 1]
                    + high_weight * accuracy_curve[high_n - 1]
                )
                matched_split_chars = float(
                    (1.0 - high_weight) * char_curve[low_n - 1]
                    + high_weight * char_curve[high_n - 1]
                )
                row.update({
                    "equal_quality_fixed_n_low": low_n,
                    "equal_quality_fixed_n_high": high_n,
                    "equal_quality_fixed_high_weight": high_weight,
                    "equal_quality_fixed_expected_n": (
                        (1.0 - high_weight) * low_n + high_weight * high_n
                    ),
                    "equal_quality_fixed_accuracy": matched_split_accuracy,
                    "equal_quality_fixed_chars": matched_split_chars,
                    "equal_quality_target_reached": reached,
                    "equal_quality_accuracy_error_pp": 100.0 * (
                        matched_split_accuracy - row["adaptive_accuracy"]
                    ),
                    "saving_vs_equal_quality_fixed_pct": 100.0 * (
                        matched_split_chars - row["adaptive_chars"]
                    ) / matched_split_chars,
                })
            aggregate_matched_accuracy = float(np.mean([
                row["equal_quality_fixed_accuracy"] for row in group
            ]))
            if reached and not math.isclose(
                aggregate_matched_accuracy, adaptive_accuracy,
                rel_tol=0.0, abs_tol=2e-12,
            ):
                raise AssertionError("equal-quality Fixed-N aggregate mismatch")


def run_split(split, problems, configs, divisors, args, development_profile=None):
    raw_train, raw_test = split_problems(problems, args.seed + split)
    reward_calibration = fit_reward_space_isotonic(raw_train)
    calibration_record = reward_space_calibration_record(
        split, raw_train, raw_test, reward_calibration
    )
    train = transform_reward_space(raw_train, reward_calibration)
    test = transform_reward_space(raw_test, reward_calibration)

    train_permutations = make_permutations(
        train, args.train_permutations, args.seed + 101_003 * split + 17
    )
    test_permutations = make_permutations(
        test, args.test_permutations, args.seed + 101_003 * split + 31
    )
    # cap_factor=None makes this legacy evaluator argument irrelevant.
    unused_fixed_ns = {float(divisor): 1 for divisor in divisors}
    train_values = crossfit_evaluate(
        train, train_permutations, configs, divisors, unused_fixed_ns,
        args.delta, args.workers, args.seed + 17_003 * split + 71,
        folds=args.folds,
    )

    selections = {}
    search_rows = []
    train_accuracy = train_values[..., 0].mean(axis=2)
    train_predicted_accuracy = train_values[..., 3].mean(axis=2)
    train_chars = train_values[..., 1].mean(axis=2)
    for target in args.targets:
        for method, family in METHOD_FAMILY.items():
            frozen_ids = None
            if args.frozen_target_configs and method == "train_selected_ucb":
                frozen_map = (
                    FROZEN_PREDICTED_CONFIG_BY_TARGET
                    if args.selection_metric == "calibrated_probability"
                    else FROZEN_BOUNDED_CONFIG_BY_TARGET
                )
                frozen_ids = [frozen_map[target]]
            if development_profile is not None:
                selections[(method, target)] = select_profiled_policy(
                    train_values, configs, divisors, target,
                    development_profile, family,
                    args.profile_train_accuracy_weight,
                )
            else:
                selections[(method, target)] = select_single_policy(
                    train_values, configs, divisors, target, family, frozen_ids,
                    args.selection_metric,
                )
        for config_id, config in enumerate(configs):
            for divisor_id, divisor in enumerate(divisors):
                search_rows.append({
                    "split": split,
                    "target_accuracy": target,
                    "config_id": config_id,
                    "divisor_id": divisor_id,
                    "divisor": divisor,
                    **asdict(config),
                    "train_accuracy": train_accuracy[config_id, divisor_id],
                    "train_predicted_accuracy": (
                        train_predicted_accuracy[config_id, divisor_id]
                    ),
                    "selection_accuracy": (
                        args.profile_train_accuracy_weight
                        * train_accuracy[config_id, divisor_id]
                        + (1.0 - args.profile_train_accuracy_weight)
                        * development_profile["accuracy"][config_id, divisor_id]
                        if development_profile is not None else (
                            train_predicted_accuracy[config_id, divisor_id]
                            if args.selection_metric == "calibrated_probability"
                            else train_accuracy[config_id, divisor_id]
                        )
                    ),
                    "train_chars": train_chars[config_id, divisor_id],
                    "profile_accuracy": (
                        development_profile["accuracy"][config_id, divisor_id]
                        if development_profile is not None else None
                    ),
                    "profile_chars": (
                        np.exp(development_profile["log_chars"][config_id, divisor_id])
                        if development_profile is not None else None
                    ),
                    "eligible": (
                        (
                            args.profile_train_accuracy_weight
                            * train_accuracy[config_id, divisor_id]
                            + (1.0 - args.profile_train_accuracy_weight)
                            * development_profile["accuracy"][config_id, divisor_id]
                            if development_profile is not None else (
                                train_predicted_accuracy[config_id, divisor_id]
                                if args.selection_metric == "calibrated_probability"
                                else train_accuracy[config_id, divisor_id]
                            )
                        ) >= target - 1e-12
                    ),
                })

    selected_config_ids = (
        list(range(len(configs))) if args.diagnostic_all_configs else
        sorted({selected["config_id"] for selected in selections.values()})
    )
    selected_configs = [configs[index] for index in selected_config_ids]
    global_to_local = {
        global_id: local_id
        for local_id, global_id in enumerate(selected_config_ids)
    }
    calibrations = fit_config_calibrations(train, selected_configs)
    prior = fit_prior(train)
    test_values = evaluate(
        test, test_permutations, selected_configs, calibrations, prior,
        divisors, unused_fixed_ns, args.delta, args.workers,
    )

    train_fixed_accuracy, train_fixed_chars = exact_fixed_curves(train)
    test_fixed_accuracy, test_fixed_chars = exact_fixed_curves(test)
    diagnostic_rows = []
    if args.diagnostic_all_configs:
        for config_id, config in enumerate(configs):
            local_id = global_to_local[config_id]
            for divisor_id, divisor in enumerate(divisors):
                values = test_values[local_id, divisor_id]
                diagnostic_rows.append({
                    "split": split,
                    "config_id": config_id,
                    "divisor_id": divisor_id,
                    "divisor": divisor,
                    **asdict(config),
                    "train_accuracy": train_accuracy[config_id, divisor_id],
                    "train_predicted_accuracy": (
                        train_predicted_accuracy[config_id, divisor_id]
                    ),
                    "train_chars": train_chars[config_id, divisor_id],
                    "test_accuracy": float(np.mean(values[:, 0])),
                    "test_predicted_accuracy": float(np.mean(values[:, 3])),
                    "test_chars": float(np.mean(values[:, 1])),
                    "test_opens": float(np.mean(values[:, 2])),
                })
    rows = []
    for target in args.targets:
        (fixed_n, fixed_train_accuracy, fixed_train_chars,
         fixed_reached_train) = select_oracle_fixed_n(
            train_fixed_accuracy, train_fixed_chars, target
        )
        fixed_test_accuracy = float(test_fixed_accuracy[fixed_n - 1])
        fixed_test_chars = float(test_fixed_chars[fixed_n - 1])
        for method in METHOD_FAMILY:
            selected = selections[(method, target)]
            local_id = global_to_local[selected["config_id"]]
            values = test_values[local_id, selected["divisor_id"]]
            adaptive_accuracy = float(np.mean(values[:, 0]))
            adaptive_chars = float(np.mean(values[:, 1]))
            (oracle_n, oracle_accuracy, oracle_chars,
             oracle_reached) = select_oracle_fixed_n(
                test_fixed_accuracy, test_fixed_chars, adaptive_accuracy
            )
            (matched_low_n, matched_high_n, matched_high_weight,
             matched_accuracy, matched_chars,
             matched_reached) = select_matched_fixed_mix(
                test_fixed_accuracy, test_fixed_chars, adaptive_accuracy
            )
            row = {
                "split": split,
                "method": method,
                "target_accuracy": target,
                "target_margin": 0.0,
                "target_reached_train": selected["target_reached_train"],
                "fallback_reason": selected["fallback_reason"],
                "config_id": selected["config_id"],
                "divisor": selected["divisor"],
                "family": selected["family"],
                "calibration": selected["calibration"],
                "confidence_scale": selected["confidence_scale"],
                "reward_prior_strength": selected["reward_prior_strength"],
                "cost_prior_strength": selected["cost_prior_strength"],
                "cap_factor": selected["cap_factor"],
                "tail_quantile": selected["tail_quantile"],
                "tail_decay": selected["tail_decay"],
                "train_accuracy": selected["train_accuracy"],
                "train_predicted_accuracy": selected[
                    "train_predicted_accuracy"
                ],
                "selection_accuracy": selected["selection_accuracy"],
                "train_chars": selected["train_chars"],
                "profile_accuracy": selected.get("profile_accuracy"),
                "profile_chars": selected.get("profile_chars"),
                "selection_chars": selected.get(
                    "selection_chars", selected["train_chars"]
                ),
                "adaptive_accuracy": adaptive_accuracy,
                "adaptive_predicted_accuracy": float(np.mean(values[:, 3])),
                "adaptive_chars": adaptive_chars,
                "adaptive_opens": float(np.mean(values[:, 2])),
                "target_error_pp": 100.0 * (adaptive_accuracy - target),
                "train_fixed_n": fixed_n,
                "train_fixed_target_reached_train": fixed_reached_train,
                "train_fixed_train_accuracy": fixed_train_accuracy,
                "train_fixed_train_chars": fixed_train_chars,
                "train_fixed_test_accuracy": fixed_test_accuracy,
                "train_fixed_test_chars": fixed_test_chars,
                "saving_vs_train_fixed_pct": 100.0 * (
                    fixed_test_chars - adaptive_chars
                ) / fixed_test_chars,
                "oracle_fixed_n": oracle_n,
                "oracle_fixed_accuracy": oracle_accuracy,
                "oracle_fixed_chars": oracle_chars,
                "oracle_target_reached": oracle_reached,
                "oracle_accuracy_overshoot_pp": 100.0 * (
                    oracle_accuracy - adaptive_accuracy
                ),
                "saving_vs_oracle_fixed_pct": 100.0 * (
                    oracle_chars - adaptive_chars
                ) / oracle_chars,
                "matched_fixed_n_low": matched_low_n,
                "matched_fixed_n_high": matched_high_n,
                "matched_fixed_high_weight": matched_high_weight,
                "matched_fixed_expected_n": (
                    (1.0 - matched_high_weight) * matched_low_n
                    + matched_high_weight * matched_high_n
                ),
                "matched_fixed_accuracy": matched_accuracy,
                "matched_fixed_chars": matched_chars,
                "matched_target_reached": matched_reached,
                "matched_accuracy_error_pp": 100.0 * (
                    matched_accuracy - adaptive_accuracy
                ),
                "saving_vs_matched_fixed_pct": 100.0 * (
                    matched_chars - adaptive_chars
                ) / matched_chars,
                "cost_unit": "output_characters",
                "policy_uses_fixed_n": False,
                "policy_is_mixture": False,
            }
            rows.append(row)
            print(
                f"[target-calibrated] split={split} target={target:.2f} "
                f"method={method} family={row['family']} "
                f"divisor={row['divisor']:.0f} "
                f"train-observed={row['train_accuracy']:.3f} "
                f"train-predicted={row['train_predicted_accuracy']:.3f} "
                f"test={adaptive_accuracy:.3f} chars={adaptive_chars:.0f} "
                f"saving-matched={row['saving_vs_matched_fixed_pct']:+.2f}%",
                flush=True,
            )
    return rows, search_rows, calibration_record, diagnostic_rows


def summarize(rows):
    metrics = (
        "train_accuracy", "train_predicted_accuracy", "selection_accuracy",
        "train_chars", "adaptive_accuracy", "adaptive_predicted_accuracy",
        "adaptive_chars", "adaptive_opens", "target_error_pp",
        "train_fixed_n", "train_fixed_train_accuracy",
        "train_fixed_test_accuracy", "train_fixed_test_chars",
        "saving_vs_train_fixed_pct", "oracle_fixed_n",
        "oracle_fixed_accuracy", "oracle_fixed_chars",
        "oracle_accuracy_overshoot_pp", "saving_vs_oracle_fixed_pct",
        "matched_fixed_n_low", "matched_fixed_n_high",
        "matched_fixed_high_weight", "matched_fixed_expected_n",
        "matched_fixed_accuracy", "matched_fixed_chars",
        "matched_accuracy_error_pp", "saving_vs_matched_fixed_pct",
        "equal_quality_fixed_n_low", "equal_quality_fixed_n_high",
        "equal_quality_fixed_high_weight", "equal_quality_fixed_expected_n",
        "equal_quality_fixed_accuracy", "equal_quality_fixed_chars",
        "equal_quality_accuracy_error_pp",
        "saving_vs_equal_quality_fixed_pct",
    )
    output = []
    for method in METHOD_FAMILY:
        for target in sorted({row["target_accuracy"] for row in rows}):
            group = [
                row for row in rows
                if row["method"] == method and row["target_accuracy"] == target
            ]
            record = {"method": method, "target_accuracy": target, "splits": len(group)}
            for metric in metrics:
                mean, low, high = interval([row[metric] for row in group])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_ci_low"] = low
                record[f"{metric}_ci_high"] = high
            record["adaptive_target_hit_fraction"] = float(np.mean([
                row["adaptive_accuracy"] >= target for row in group
            ]))
            record["train_target_reached_fraction"] = float(np.mean([
                row["target_reached_train"] for row in group
            ]))
            record["train_fixed_test_target_hit_fraction"] = float(np.mean([
                row["train_fixed_test_accuracy"] >= target for row in group
            ]))
            record["oracle_target_reached_fraction"] = float(np.mean([
                row["oracle_target_reached"] for row in group
            ]))
            record["matched_target_reached_fraction"] = float(np.mean([
                row["matched_target_reached"] for row in group
            ]))
            record["equal_quality_target_reached_fraction"] = float(np.mean([
                row["equal_quality_target_reached"] for row in group
            ]))
            adaptive = [row["adaptive_chars"] for row in group]
            train_fixed = [row["train_fixed_test_chars"] for row in group]
            oracle = [row["oracle_fixed_chars"] for row in group]
            matched = [row["matched_fixed_chars"] for row in group]
            equal_quality = [
                row["equal_quality_fixed_chars"] for row in group
            ]
            for name, baseline, offset in (
                ("train_fixed", train_fixed, 0),
                ("oracle_fixed", oracle, 1),
                ("matched_fixed", matched, 2),
                ("equal_quality_fixed", equal_quality, 3),
            ):
                saving, low, high = ratio_saving_interval(
                    adaptive, baseline,
                    9_317_011 + int(round(target * 10_000)) + offset,
                )
                record[f"aggregate_saving_vs_{name}_pct"] = saving
                record[f"aggregate_saving_vs_{name}_ci_low"] = low
                record[f"aggregate_saving_vs_{name}_ci_high"] = high
                saved_mean, saved_low, saved_high = interval(
                    np.asarray(baseline) - np.asarray(adaptive)
                )
                record[f"characters_saved_vs_{name}_mean"] = saved_mean
                record[f"characters_saved_vs_{name}_ci_low"] = saved_low
                record[f"characters_saved_vs_{name}_ci_high"] = saved_high
            output.append(record)
    return output


def _band(ax, x, summary, metric, color, marker, label, linestyle="-"):
    mean = np.asarray(
        [row[f"{metric}_mean"] for row in summary], dtype=np.float64
    )
    low = np.asarray(
        [row[f"{metric}_ci_low"] for row in summary], dtype=np.float64
    )
    high = np.asarray(
        [row[f"{metric}_ci_high"] for row in summary], dtype=np.float64
    )
    ax.plot(x, mean, color=color, marker=marker, linestyle=linestyle,
            linewidth=2, label=label)
    ax.fill_between(x, low, high, color=color, alpha=0.12)


def plot_primary(summary, path):
    group = sorted(
        (row for row in summary if row["method"] == "train_selected_ucb"),
        key=lambda row: float(row["target_accuracy"]),
    )
    targets = np.asarray(
        [row["target_accuracy"] for row in group], dtype=np.float64
    )
    fig, axes = plt.subplots(1, 4, figsize=(19.2, 4.6), constrained_layout=True)
    adaptive_color = "#1B7F3A"
    fixed_color = "#555555"
    for axis, prefix, title in (
        (axes[0], "train_fixed", "A. Saving vs train-selected Fixed-$N$"),
        (axes[1], "equal_quality_fixed",
         "B. Saving at exactly matched accuracy"),
    ):
        saving = np.asarray([
            row[f"aggregate_saving_vs_{prefix}_pct"] for row in group
        ], dtype=np.float64)
        saving_low = np.asarray([
            row[f"aggregate_saving_vs_{prefix}_ci_low"] for row in group
        ], dtype=np.float64)
        saving_high = np.asarray([
            row[f"aggregate_saving_vs_{prefix}_ci_high"] for row in group
        ], dtype=np.float64)
        axis.plot(targets, saving, color=adaptive_color, marker="o", linewidth=2)
        axis.fill_between(
            targets, saving_low, saving_high, color=adaptive_color, alpha=0.12,
        )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set(
            xlabel="Requested accuracy",
            ylabel="Adaptive character saving (%)",
            title=title,
        )

    for axis, metric, color, marker, title in (
        (axes[2], "adaptive_accuracy", adaptive_color, "o",
         "C. Adaptive test calibration"),
        (axes[3], "train_fixed_test_accuracy", fixed_color, "s",
         "D. Train-selected Fixed-$N$ calibration"),
    ):
        _band(axis, targets, group, metric, color, marker, title)
        axis.plot(targets, targets, color="black", linestyle="--", linewidth=1)
        axis.set(
            xlabel="Requested accuracy", ylabel="Test accuracy", title=title,
        )
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_families(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for method in METHOD_FAMILY:
        group = sorted(
            (row for row in summary if row["method"] == method),
            key=lambda row: row["target_accuracy"],
        )
        targets = np.asarray([row["target_accuracy"] for row in group])
        saving = np.asarray([
            row["aggregate_saving_vs_equal_quality_fixed_pct"] for row in group
        ])
        low = np.asarray([
            row["aggregate_saving_vs_equal_quality_fixed_ci_low"] for row in group
        ])
        high = np.asarray([
            row["aggregate_saving_vs_equal_quality_fixed_ci_high"] for row in group
        ])
        axes[0].plot(targets, saving, marker="o", linewidth=2,
                     color=COLORS[method], label=LABELS[method])
        axes[0].fill_between(targets, low, high, color=COLORS[method], alpha=0.12)
        _band(axes[1], targets, group, "adaptive_accuracy", COLORS[method],
              "o", LABELS[method])
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Requested accuracy", ylabel="Character saving (%)",
                title="Saving vs exact-quality held-out Fixed-N oracle")
    axes[1].plot(TARGETS, TARGETS, color="black", linestyle="--", linewidth=1,
                 label="Achieved = requested")
    axes[1].set(xlabel="Requested accuracy", ylabel="Held-out achieved accuracy",
                title="Distribution-family target transfer")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(summary, path, args):
    group = sorted(
        (row for row in summary if row["method"] == "train_selected_ucb"),
        key=lambda row: row["target_accuracy"],
    )
    lines = [
        "# Isotonic probability-space target quality", "",
        "| Target | Adaptive accuracy | Adaptive chars | Train-tuned N / test accuracy | Saving vs train N | Exact-quality fixed expected N / accuracy | Saving at equal quality |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group:
        lines.append(
            f"| {row['target_accuracy']:.2f} | {row['adaptive_accuracy_mean']:.4f} "
            f"| {row['adaptive_chars_mean']:.0f} | {row['train_fixed_n_mean']:.1f} / "
            f"{row['train_fixed_test_accuracy_mean']:.4f} | "
            f"{row['aggregate_saving_vs_train_fixed_pct']:+.2f}% | "
            f"{row['equal_quality_fixed_expected_n_mean']:.1f} / "
            f"{row['equal_quality_fixed_accuracy_mean']:.4f} | "
            f"{row['aggregate_saving_vs_equal_quality_fixed_pct']:+.2f}% |"
        )
    lines.extend([
        "", "## Integrity", "",
        "Every adaptive result deploys one configuration and one cost divisor. "
        "There is no policy mixture, target margin, Fixed-N guard, Fixed-N cap, "
        "or Fixed-N fallback. Isotonic reward calibration uses only the "
        "41-problem training half; the corresponding 42-problem test half is "
        "untouched until final evaluation.", "",
        "The first comparator chooses the minimum-character integer N reaching "
        "the requested target on train and freezes it for test. The second is "
        "a descriptive held-out oracle at exactly the adaptive policy's aggregate "
        "accuracy. It uses one common integer N when possible and otherwise "
        "randomizes between two common fixed counts before generation, choosing "
        "the minimum-cost exact-quality mixture on the averaged held-out curve. "
        "Neither comparator affects adaptive stopping.", "",
        f"Splits: {args.split_start}--{args.split_start + args.splits - 1}; "
        f"train/test permutations: {args.train_permutations}/{args.test_permutations}.",
    ])
    if args.development_profile:
        lines.extend([
            "", "The policy selector was frozen from development split IDs "
            f"55--59. It places {100 * args.profile_train_accuracy_weight:.0f}% "
            "weight on current train cross-fit accuracy and the remaining weight "
            "on development held-out accuracy, ranks feasible rules by development "
            "geometric-mean characters, and compares the result to the requested "
            "target with zero offset. Split IDs 95--104 were not used to choose "
            "this selector.",
        ])
    path.write_text("\n".join(lines) + "\n")


def main():
    global METHOD_FAMILY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(
        "algorithm/bestofn_coding/data.jsonl"
    ))
    parser.add_argument("--char-cache", type=Path, default=Path(
        "algorithm/bestofn_coding/practical_algorithm/coding_char_counts_83.npz"
    ))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "codex_results/results/coding/target_quality"
    ))
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--split-start", type=int, default=65)
    parser.add_argument("--train-permutations", type=int, default=16)
    parser.add_argument("--test-permutations", type=int, default=48)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--bounded-probability-models", action="store_true")
    parser.add_argument("--tail-decay-grid", action="store_true")
    parser.add_argument(
        "--simple-global-policy", action="store_true",
        help=("evaluate only the frozen simple policy: one train-fitted global "
              "75-percent exponential tail, confidence 0.8, running-mean "
              "character cost, D_res=D, and 3/n opportunity decay"),
    )
    parser.add_argument("--development-profile", type=Path)
    parser.add_argument(
        "--profile-train-accuracy-weight", type=float,
        default=PROFILE_TRAIN_ACCURACY_WEIGHT,
        help=("weight on current split cross-fit accuracy when a frozen "
              "development profile is supplied; zero gives the simplest "
              "development-only inverse calibration"),
    )
    parser.add_argument("--diagnostic-all-configs", action="store_true")
    parser.add_argument("--frozen-target-configs", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=("observed_accuracy", "calibrated_probability"),
        default="observed_accuracy",
        help=("train-only attainment signal; calibrated_probability uses the "
              "mean selected isotonic P(correct) instead of binary outcomes"),
    )
    parser.add_argument("--targets", type=parse_floats, default=TARGETS)
    parser.add_argument("--divisors", type=parse_floats, default=TARGET_DIVISORS)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    if not args.targets:
        parser.error("--targets must be nonempty")
    if not 0.0 <= args.profile_train_accuracy_weight <= 1.0:
        parser.error("--profile-train-accuracy-weight must lie in [0, 1]")
    if any(not 0.0 <= target <= 1.0 for target in args.targets):
        parser.error("all targets must lie in [0, 1]")
    if args.frozen_target_configs and not args.bounded_probability_models:
        parser.error("--frozen-target-configs requires --bounded-probability-models")
    if args.frozen_target_configs and any(
        target not in FROZEN_BOUNDED_CONFIG_BY_TARGET for target in args.targets
    ):
        parser.error("frozen target configurations exist only for 0.25,0.30,0.35")
    if args.development_profile and not args.simple_global_policy and not (
        args.bounded_probability_models and args.tail_decay_grid
    ):
        parser.error(
            "--development-profile requires --bounded-probability-models "
            "and --tail-decay-grid"
        )
    if args.development_profile and args.frozen_target_configs:
        parser.error(
            "--development-profile and --frozen-target-configs are incompatible"
        )
    if args.simple_global_policy:
        METHOD_FAMILY = {"train_selected_ucb": None}
    configs = candidate_configs(
        args.bounded_probability_models, args.tail_decay_grid,
        args.simple_global_policy,
    )
    development_profile = (
        load_development_profile(args.development_profile, configs, args.divisors)
        if args.development_profile else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    problems = load_problems(args.data, args.char_cache)

    rows, search_rows, calibration_rows, diagnostic_rows = [], [], [], []
    for split in range(args.split_start, args.split_start + args.splits):
        split_rows, split_search, calibration_record, split_diagnostic = run_split(
            split, problems, configs, args.divisors, args, development_profile
        )
        rows.extend(split_rows)
        search_rows.extend(split_search)
        calibration_rows.append(calibration_record)
        diagnostic_rows.extend(split_diagnostic)
        write_csv(args.output_dir / "target_quality_splits.partial.csv", rows)
        write_csv(args.output_dir / "policy_search.partial.csv", search_rows)
        write_csv(
            args.output_dir / "isotonic_calibration_splits.partial.csv",
            calibration_rows,
        )
        if diagnostic_rows:
            write_csv(
                args.output_dir / "diagnostic_all_configs.partial.csv",
                diagnostic_rows,
            )

    fixed_curves, fixed_curve_rows = heldout_fixed_curves(problems, args)
    attach_equal_quality_fixed_baseline(rows, fixed_curves)
    summary = summarize(rows)
    write_csv(args.output_dir / "target_quality_splits.csv", rows)
    write_csv(args.output_dir / "target_quality_summary.csv", summary)
    write_csv(args.output_dir / "heldout_fixed_n_curves.csv", fixed_curve_rows)
    write_csv(args.output_dir / "policy_search.csv", search_rows)
    write_csv(args.output_dir / "isotonic_calibration_splits.csv", calibration_rows)
    if diagnostic_rows:
        write_csv(args.output_dir / "diagnostic_all_configs.csv", diagnostic_rows)
    plot_primary(summary, args.output_dir / "coding_target_quality_calibrated.png")
    plot_families(summary, args.output_dir / "coding_target_quality_families.png")
    write_report(summary, args.output_dir / "REPORT.md", args)
    (args.output_dir / "METHOD.json").write_text(json.dumps({
        "algorithm": "one deterministic uncapped UCB-Pandora target policy",
        "reward_space": "train-only isotonic P(correct | raw reward)",
        "distribution_models": (
            ["one global conditional shifted-exponential upper-quarter tail"]
            if args.simple_global_policy else [
                "complete Gaussian after isotonic transformation",
                "conditional shifted-exponential tail after isotonic transformation",
            ]
        ),
        "target_selection": (
            (
                "minimum frozen development-profile geometric-mean characters "
                "whose development accuracy reaches the unshifted requested target"
                if args.profile_train_accuracy_weight == 0.0 else
                "minimum frozen development-profile geometric-mean characters "
                "among single policies whose weighted current-train/development "
                "profile accuracy reaches the unshifted requested target"
            ) if development_profile is not None else (
                "minimum cross-fitted train characters among single policies with "
                f"mean train {args.selection_metric} >= requested target"
            )
        ),
        "selection_metric": args.selection_metric,
        "target_margin": 0.0,
        "target_lcb": None,
        "policy_uses_fixed_n": False,
        "policy_is_mixture": False,
        "cap_factor": None,
        "terminal_horizon": "absolute 512-generation dataset boundary only",
        "fixed_n_baseline_1": (
            "minimum-character integer N reaching requested accuracy on train; "
            "same N evaluated on test"
        ),
        "fixed_n_baseline_2": (
            "held-out oracle minimum-character non-adaptive rule on the average "
            "final-test Fixed-N curve; it uses one common integer N or a "
            "pre-generation mixture of two common N values whose aggregate "
            "accuracy exactly equals aggregate adaptive held-out accuracy"
        ),
        "fixed_n_baseline_2_splitwise_diagnostic": (
            "minimum-cost pre-generation mixture of at most two fixed N values "
            "matching adaptive accuracy separately on each test split whenever "
            "that accuracy lies in the uniform Fixed-N attainable range"
        ),
        "cost": "exact cumulative generation-output characters",
        "candidate_configs": [asdict(config) for config in configs],
        "simple_global_policy": args.simple_global_policy,
        "divisors": args.divisors,
        "targets": args.targets,
        "splits": args.splits,
        "split_start": args.split_start,
        "train_test": "41/42 problems",
        "train_permutations": args.train_permutations,
        "test_permutations": args.test_permutations,
        "folds": args.folds,
        "delta": args.delta,
        "bounded_probability_models": (
            args.bounded_probability_models or args.simple_global_policy
        ),
        "tail_decay_grid": args.tail_decay_grid,
        "tail_decay_definition": (
            "smooth expected-improvement multiplier "
            "(opens/minimum_opens)^(-tail_decay); no horizon or cap"
        ),
        "development_profile": (
            {
                "path": development_profile["path"],
                "sha256": development_profile["sha256"],
                "train_accuracy_weight": args.profile_train_accuracy_weight,
                "profile_accuracy_weight": 1.0 - args.profile_train_accuracy_weight,
                "current_train_cost_weight": 0.0,
                "target_offset": 0.0,
                "development_split_ids": [55, 56, 57, 58, 59],
                "development_train_permutations": 4,
                "development_test_permutations": 8,
            } if development_profile is not None else None
        ),
        "diagnostic_all_configs": args.diagnostic_all_configs,
        "frozen_target_config_by_target": (
            (
                FROZEN_PREDICTED_CONFIG_BY_TARGET
                if args.selection_metric == "calibrated_probability"
                else FROZEN_BOUNDED_CONFIG_BY_TARGET
            ) if args.frozen_target_configs else None
        ),
        "seed": args.seed,
    }, indent=2) + "\n")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "split_rows": len(rows),
        "summary_rows": len(summary),
        "search_rows": len(search_rows),
        "diagnostic_rows": len(diagnostic_rows),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
