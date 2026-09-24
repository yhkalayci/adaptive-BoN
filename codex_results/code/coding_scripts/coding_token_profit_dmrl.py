"""Leakage-free token-profit study for non-parametric DMRL coding.

Each outer split uses calibration problems to fit an increasing isotonic
reward-to-success map, tune Fixed-N, and tune the top-four DMRL policy.  Both
policies are then frozen and evaluated on unseen problems and identical sample
orders.  Online stopping sees only calibrated rewards and output-token counts;
correctness is revealed after selection to compute profit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from scipy.stats import t as student_t


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from algorithm.adaptive_coding import (  # noqa: E402
    CodingProblem,
    CodingProfile,
    fit_coding_profile,
    load_coding_problems,
    split_problem_ids,
)


DEFAULT_DIVISORS = (
    25_000.0,
    35_000.0,
    50_000.0,
    70_000.0,
    100_000.0,
    140_000.0,
    200_000.0,
    280_000.0,
    400_000.0,
    560_000.0,
    700_000.0,
    1_000_000.0,
)
SMOOTHING_MODES = ("current", "mean", "recent_half")
MULTIPLIERS = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
COST_ADJUSTMENTS = (0.0, 1.0, 2.0, 4.0)
CAP_FACTORS: tuple[float | None, ...] = (0.5, 0.75, 0.875, 1.0)
MINIMUM_FACTORS = (0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0)


def policy_family(width: int) -> str:
    ordinal = {3: "fourth", 4: "fifth", 5: "sixth"}.get(width)
    return (
        f"nonparametric_dmrl_top{width}_above_{ordinal}"
        if ordinal else f"nonparametric_dmrl_top{width}_above_next"
    )


@dataclass(frozen=True)
class PolicyConfig:
    smoothing: str
    multiplier: float
    cost_adjustment: float
    cap_factor: float | None
    minimum_factor: float
    width: int = 4

    def to_dict(self) -> dict[str, object]:
        return {
            "family": policy_family(self.width),
            "width": self.width,
            "minimum": self.width + 1,
            "smoothing": self.smoothing,
            "multiplier": self.multiplier,
            "cost_adjustment": self.cost_adjustment,
            "cap_factor_vs_train_tuned_fixed_n": self.cap_factor,
            "minimum_factor_vs_train_tuned_fixed_n": self.minimum_factor,
        }


@dataclass
class TrajectoryBatch:
    problem_ids: np.ndarray
    permutation_ids: np.ndarray
    best_indices: np.ndarray
    best_probabilities: np.ndarray
    cumulative_tokens: np.ndarray
    mean_tokens: np.ndarray
    token_se_ratio: np.ndarray
    residual_current: np.ndarray
    residual_mean: np.ndarray
    residual_recent_half: np.ndarray

    @property
    def trials(self) -> int:
        return int(self.best_indices.shape[0])

    @property
    def samples(self) -> int:
        return int(self.best_indices.shape[1])

    def residual(self, smoothing: str) -> np.ndarray:
        if smoothing == "current":
            return self.residual_current
        if smoothing == "mean":
            return self.residual_mean
        if smoothing == "recent_half":
            return self.residual_recent_half
        raise ValueError(f"unknown smoothing {smoothing!r}")


def parse_divisors(value: str) -> tuple[float, ...]:
    values = tuple(float(x) for x in value.split(",") if x.strip())
    if not values or any(not math.isfinite(x) or x <= 0.0 for x in values):
        raise argparse.ArgumentTypeError("divisors must be comma-separated positives")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("divisors must be unique")
    return tuple(sorted(values))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        with Path(temporary).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def calibrated_rewards(
    problems: dict[str, CodingProblem], profile: CodingProfile
) -> dict[str, np.ndarray]:
    return {
        problem_id: np.interp(
            np.asarray(problem.rewards, dtype=np.float64),
            profile.reward_knots,
            profile.probability_knots,
            left=profile.probability_knots[0],
            right=profile.probability_knots[-1],
        )
        for problem_id, problem in problems.items()
    }


def exact_fixed_accuracy(problems: dict[str, CodingProblem]) -> np.ndarray:
    """Exact random-order success of selecting the maximum reward among N."""
    sample_count = min(len(x.rewards) for x in problems.values())
    if any(len(x.rewards) != sample_count for x in problems.values()):
        raise ValueError("all problems must have equal sample counts")
    curve = np.zeros(sample_count, dtype=np.float64)
    for problem_id, problem in problems.items():
        # Best-of-N conventionally selects the largest raw reward.  Isotonic
        # calibration is non-strict and must not collapse its ranking plateaus.
        rewards = np.asarray(problem.rewards, dtype=np.float64)
        correct = np.asarray(problem.correct, dtype=np.float64)
        order = np.argsort(rewards, kind="stable")
        sorted_rewards = rewards[order]
        sorted_correct = correct[order]
        starts = np.r_[0, 1 + np.flatnonzero(sorted_rewards[1:] != sorted_rewards[:-1])]
        ends = np.r_[starts[1:], sample_count]
        group_correct = np.add.reduceat(sorted_correct, starts) / (ends - starts)
        for opened in range(1, sample_count + 1):
            inclusion = np.zeros(sample_count + 1, dtype=np.float64)
            inclusion[sample_count] = 1.0
            for available in range(sample_count, opened, -1):
                inclusion[available - 1] = (
                    inclusion[available] * (available - opened) / available
                )
            curve[opened - 1] += float(
                (inclusion[ends] - inclusion[starts]) @ group_correct
            )
    return curve / len(problems)


def select_fixed_n(
    problems: dict[str, CodingProblem],
    divisors: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, float]:
    accuracy = exact_fixed_accuracy(problems)
    mean_tokens = float(np.mean(np.concatenate([
        np.asarray(x.lengths, dtype=np.float64) for x in problems.values()
    ])))
    counts = np.arange(1, len(accuracy) + 1, dtype=np.float64)
    selected = np.asarray([
        int(np.argmax(accuracy - mean_tokens * counts / divisor)) + 1
        for divisor in divisors
    ], dtype=np.int64)
    return selected, accuracy, mean_tokens


def make_permutations(
    problems: dict[str, CodingProblem], count: int, seed: int
) -> dict[str, np.ndarray]:
    output = {}
    for offset, problem_id in enumerate(sorted(problems)):
        rng = np.random.default_rng(seed + 1_000_003 * (offset + 1))
        size = len(problems[problem_id].rewards)
        output[problem_id] = np.asarray(
            [rng.permutation(size) for _ in range(count)], dtype=np.int16
        )
    return output


def build_trajectory_batch(
    problems: dict[str, CodingProblem],
    probability: dict[str, np.ndarray],
    permutations: dict[str, np.ndarray],
    width: int = 4,
) -> TrajectoryBatch:
    """Build label-free prefix statistics used by every policy candidate."""
    if width < 1:
        raise ValueError("width must be positive")
    ids = sorted(problems)
    if not ids:
        raise ValueError("trajectory batch needs at least one problem")
    samples = len(problems[ids[0]].rewards)
    per_problem = permutations[ids[0]].shape[0]
    trials = len(ids) * per_problem
    shape = (trials, samples)
    best_indices = np.empty(shape, dtype=np.int16)
    best_probabilities = np.empty(shape, dtype=np.float32)
    cumulative_tokens = np.empty(shape, dtype=np.int64)
    mean_tokens = np.empty(shape, dtype=np.float32)
    token_se_ratio = np.empty(shape, dtype=np.float32)
    residual_current = np.zeros(shape, dtype=np.float32)
    residual_mean = np.zeros(shape, dtype=np.float32)
    residual_recent_half = np.zeros(shape, dtype=np.float32)
    problem_ids = np.empty(trials, dtype=object)
    permutation_ids = np.empty(trials, dtype=np.int16)

    row = 0
    for problem_id in ids:
        source_probability = probability[problem_id]
        source_rewards = np.asarray(problems[problem_id].rewards, dtype=np.float64)
        source_lengths = np.asarray(problems[problem_id].lengths, dtype=np.int64)
        for permutation_id, permutation in enumerate(permutations[problem_id]):
            ordered_probability = source_probability[permutation]
            ordered_rewards = source_rewards[permutation]
            ordered_lengths = source_lengths[permutation]
            top_values: list[float] = []
            best_probability = -math.inf
            best_reward = -math.inf
            best_index = -1
            length_sum = 0.0
            length_square_sum = 0.0
            residual_sum = 0.0
            residual_prefix: list[float] = []
            residual_cumulative = [0.0]
            for index, (value, reward, length) in enumerate(
                zip(ordered_probability, ordered_rewards, ordered_lengths)
            ):
                value = float(value)
                reward = float(reward)
                if value > best_probability or (
                    value == best_probability and reward > best_reward
                ):
                    best_probability = value
                    best_reward = reward
                    best_index = index
                best_indices[row, index] = best_index
                best_probabilities[row, index] = best_probability
                length_sum += float(length)
                length_square_sum += float(length) ** 2
                count = index + 1
                cumulative_tokens[row, index] = int(length_sum)
                mean = length_sum / count
                variance = max(0.0, length_square_sum / count - mean * mean)
                mean_tokens[row, index] = mean
                token_se_ratio[row, index] = math.sqrt(variance / count) / mean

                if len(top_values) < width + 1:
                    heapq.heappush(top_values, value)
                elif value > top_values[0]:
                    heapq.heapreplace(top_values, value)
                if count >= width + 1:
                    residual = max(
                        0.0,
                        (sum(top_values) - (width + 1) * top_values[0]) / width,
                    )
                    residual_prefix.append(residual)
                    residual_sum += residual
                    residual_cumulative.append(residual_sum)
                    eligible_count = len(residual_prefix)
                    recent_count = max(1, (eligible_count + 1) // 2)
                    recent_sum = (
                        residual_cumulative[-1]
                        - residual_cumulative[eligible_count - recent_count]
                    )
                    residual_current[row, index] = residual
                    residual_mean[row, index] = residual_sum / eligible_count
                    residual_recent_half[row, index] = recent_sum / recent_count
            problem_ids[row] = problem_id
            permutation_ids[row] = permutation_id
            row += 1
    return TrajectoryBatch(
        problem_ids,
        permutation_ids,
        best_indices,
        best_probabilities,
        cumulative_tokens,
        mean_tokens,
        token_se_ratio,
        residual_current,
        residual_mean,
        residual_recent_half,
    )


def selected_correctness(
    batch: TrajectoryBatch,
    problems: dict[str, CodingProblem],
    permutations: dict[str, np.ndarray],
) -> np.ndarray:
    """Reveal labels after prefix selection; never used to construct statistics."""
    output = np.empty_like(batch.best_indices, dtype=np.int8)
    for row, (problem_id, permutation_id) in enumerate(
        zip(batch.problem_ids, batch.permutation_ids)
    ):
        correct = np.asarray(problems[str(problem_id)].correct, dtype=np.int8)
        ordered = correct[permutations[str(problem_id)][int(permutation_id)]]
        output[row] = ordered[batch.best_indices[row]]
    return output


def cap_for_config(config: PolicyConfig, fixed_n: int, sample_count: int) -> int:
    minimum = minimum_for_config(config, fixed_n, sample_count)
    if config.cap_factor is None:
        return sample_count
    return min(
        sample_count,
        max(minimum, config.width + 1, int(round(fixed_n * config.cap_factor))),
    )


def minimum_for_config(
    config: PolicyConfig, fixed_n: int, sample_count: int
) -> int:
    return min(
        sample_count,
        max(config.width + 1, int(round(fixed_n * config.minimum_factor))),
    )


def stop_counts(
    batch: TrajectoryBatch,
    divisor: float,
    fixed_n: int,
    config: PolicyConfig,
) -> np.ndarray:
    """Compute stops from reward/token features only, with no label argument."""
    cap = cap_for_config(config, fixed_n, batch.samples)
    minimum = minimum_for_config(config, fixed_n, batch.samples)
    residual = batch.residual(config.smoothing)[:, config.width:cap]
    opened = np.arange(config.width + 1, cap + 1, dtype=np.float64)[None, :]
    gain = config.multiplier * residual / opened
    next_cost = (
        batch.mean_tokens[:, config.width:cap]
        / (1.0 + config.cost_adjustment * batch.token_se_ratio[:, config.width:cap])
        / divisor
    )
    condition = gain <= next_cost
    any_stop = np.any(condition, axis=1)
    first = np.argmax(condition, axis=1) + config.width + 1
    return np.minimum(np.maximum(np.where(any_stop, first, cap), minimum), cap).astype(np.int16)


def fixed_metrics(
    batch: TrajectoryBatch, labels: np.ndarray, divisor: float, fixed_n: int
) -> dict[str, np.ndarray]:
    rows = np.arange(batch.trials)
    index = fixed_n - 1
    correct = labels[rows, index].astype(np.float64)
    tokens = batch.cumulative_tokens[rows, index].astype(np.float64)
    return {
        "correct": correct,
        "tokens": tokens,
        "generations": np.full(batch.trials, fixed_n, dtype=np.int16),
        "probability": batch.best_probabilities[rows, index].astype(np.float64),
        "profit": correct - tokens / divisor,
    }


def adaptive_metrics(
    batch: TrajectoryBatch,
    labels: np.ndarray,
    divisor: float,
    fixed_n: int,
    config: PolicyConfig,
) -> dict[str, np.ndarray]:
    stop = stop_counts(batch, divisor, fixed_n, config)
    rows = np.arange(batch.trials)
    index = stop.astype(np.int64) - 1
    correct = labels[rows, index].astype(np.float64)
    tokens = batch.cumulative_tokens[rows, index].astype(np.float64)
    return {
        "correct": correct,
        "tokens": tokens,
        "generations": stop,
        "probability": batch.best_probabilities[rows, index].astype(np.float64),
        "profit": correct - tokens / divisor,
    }


def candidate_grid(width: int = 4) -> tuple[PolicyConfig, ...]:
    return tuple(
        PolicyConfig(
            smoothing, multiplier, cost_adjustment, cap_factor, minimum_factor,
            width,
        )
        for smoothing in SMOOTHING_MODES
        for multiplier in MULTIPLIERS
        for cost_adjustment in COST_ADJUSTMENTS
        for cap_factor in CAP_FACTORS
        for minimum_factor in MINIMUM_FACTORS
    )


def tune_policy(
    batch: TrajectoryBatch,
    labels: np.ndarray,
    divisor: float,
    fixed_n: int,
    candidates: tuple[PolicyConfig, ...],
    tuning_target: str = "calibrated",
    tuning_cost_scale: float = 1.0,
    width: int = 4,
) -> tuple[PolicyConfig, dict[str, float]]:
    """Select using only calibration trajectories and their offline labels."""
    if tuning_target not in ("calibrated", "observed"):
        raise ValueError(f"unknown tuning target {tuning_target!r}")
    if not math.isfinite(tuning_cost_scale) or tuning_cost_scale <= 0.0:
        raise ValueError("tuning_cost_scale must be finite and positive")
    del candidates  # the structured loops below avoid recomputing each cap path
    rows = np.arange(batch.trials)
    best_config: PolicyConfig | None = None
    best_metrics: dict[str, np.ndarray] | None = None
    best_estimated_profit = -math.inf
    best_tokens = math.inf
    for smoothing in SMOOTHING_MODES:
        residual = batch.residual(smoothing)[:, width:]
        opened = np.arange(width + 1, batch.samples + 1, dtype=np.float64)[None, :]
        for cost_adjustment in COST_ADJUSTMENTS:
            next_cost = (
                batch.mean_tokens[:, width:]
                / (1.0 + cost_adjustment * batch.token_se_ratio[:, width:])
                / divisor
            )
            for multiplier in MULTIPLIERS:
                condition = multiplier * residual / opened <= next_cost
                any_stop = np.any(condition, axis=1)
                unlimited_stop = np.where(
                    any_stop, np.argmax(condition, axis=1) + width + 1, batch.samples
                ).astype(np.int16)
                for cap_factor in CAP_FACTORS:
                    for minimum_factor in MINIMUM_FACTORS:
                        config = PolicyConfig(
                            smoothing,
                            multiplier,
                            cost_adjustment,
                            cap_factor,
                            minimum_factor,
                            width,
                        )
                        cap = cap_for_config(config, fixed_n, batch.samples)
                        minimum = minimum_for_config(
                            config, fixed_n, batch.samples
                        )
                        stop = np.minimum(
                            np.maximum(unlimited_stop, minimum), cap
                        )
                        index = stop.astype(np.int64) - 1
                        correct = labels[rows, index].astype(np.float64)
                        tokens = batch.cumulative_tokens[rows, index].astype(np.float64)
                        probability = batch.best_probabilities[
                            rows, index
                        ].astype(np.float64)
                        metrics = {
                            "correct": correct,
                            "tokens": tokens,
                            "generations": stop,
                            "probability": probability,
                            "profit": correct - tokens / divisor,
                        }
                        selection_value = (
                            probability if tuning_target == "calibrated" else correct
                        )
                        estimated_profit = float(
                            np.mean(selection_value - tuning_cost_scale * tokens / divisor)
                        )
                        mean_tokens = float(np.mean(tokens))
                        if estimated_profit > best_estimated_profit + 1e-15 or (
                            abs(estimated_profit - best_estimated_profit) <= 1e-15
                            and mean_tokens < best_tokens
                        ):
                            best_config = config
                            best_metrics = metrics
                            best_estimated_profit = estimated_profit
                            best_tokens = mean_tokens
    assert best_config is not None and best_metrics is not None
    return best_config, {
        "calibration_estimated_profit": best_estimated_profit,
        "calibration_actual_profit": float(np.mean(best_metrics["profit"])),
        "calibration_accuracy": float(np.mean(best_metrics["correct"])),
        "calibration_mean_generations": float(np.mean(best_metrics["generations"])),
        "calibration_mean_output_tokens": best_tokens,
    }


def interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean
    half = float(student_t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    half *= float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean - half, mean + half


def run_split(
    split: int,
    problems: dict[str, CodingProblem],
    divisors: tuple[float, ...],
    outer_seed: int,
    train_permutations: int,
    test_permutations: int,
    output: Path,
    candidates: tuple[PolicyConfig, ...],
    tuning_target: str = "calibrated",
    tuning_cost_scale: float = 1.0,
    width: int = 4,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    started = time.monotonic()
    train_ids, test_ids = split_problem_ids(
        problems, seed=outer_seed + split, fit_fraction=0.5
    )
    train = {x: problems[x] for x in train_ids}
    test = {x: problems[x] for x in test_ids}
    profile = fit_coding_profile(
        train,
        metadata={
            "outer_split": split,
            "outer_seed": outer_seed + split,
            "fit_problem_ids": list(train_ids),
            "holdout_problem_ids": list(test_ids),
            "online_distribution_fit": False,
            "policy_family": policy_family(width),
        },
    )
    profile.save(output / "profiles" / f"split_{split:02d}.json")
    all_probability = calibrated_rewards(problems, profile)
    train_probability = {x: all_probability[x] for x in train_ids}
    test_probability = {x: all_probability[x] for x in test_ids}
    fixed_ns, fixed_accuracy_curve, train_mean_tokens = select_fixed_n(
        train, divisors
    )

    train_orders = make_permutations(
        train, train_permutations, outer_seed + 100_000 + split * 10_000
    )
    test_orders = make_permutations(
        test, test_permutations, outer_seed + 900_000 + split * 10_000
    )
    train_batch = build_trajectory_batch(
        train, train_probability, train_orders, width=width
    )
    test_batch = build_trajectory_batch(
        test, test_probability, test_orders, width=width
    )
    train_labels = selected_correctness(train_batch, train, train_orders)
    test_labels = selected_correctness(test_batch, test, test_orders)

    rows: list[dict[str, object]] = []
    policies: list[dict[str, object]] = []
    trial_arrays: dict[str, np.ndarray] = {
        "problem_ids": test_batch.problem_ids.astype(str),
        "permutation_ids": test_batch.permutation_ids,
        "divisors": np.asarray(divisors),
        "fixed_n": fixed_ns,
    }
    for divisor_id, (divisor, fixed_n) in enumerate(zip(divisors, fixed_ns)):
        fixed_n = int(fixed_n)
        config, tuning = tune_policy(
            train_batch, train_labels, divisor, fixed_n, candidates,
            tuning_target=tuning_target,
            tuning_cost_scale=tuning_cost_scale,
            width=width,
        )
        adaptive = adaptive_metrics(test_batch, test_labels, divisor, fixed_n, config)
        fixed = fixed_metrics(test_batch, test_labels, divisor, fixed_n)
        delta = adaptive["profit"] - fixed["profit"]
        row = {
            "split": split,
            "utility_divisor": divisor,
            "price_per_token": 1.0 / divisor,
            "train_problems": len(train),
            "test_problems": len(test),
            "train_permutations": train_permutations,
            "test_permutations": test_permutations,
            "test_trials": test_batch.trials,
            "fixed_n": fixed_n,
            "fixed_train_exact_accuracy": float(fixed_accuracy_curve[fixed_n - 1]),
            "fixed_train_mean_token_estimate": train_mean_tokens * fixed_n,
            "smoothing": config.smoothing,
            "multiplier": config.multiplier,
            "cost_adjustment": config.cost_adjustment,
            "cap_factor": config.cap_factor,
            "minimum_factor": config.minimum_factor,
            "minimum": minimum_for_config(config, fixed_n, test_batch.samples),
            "cap": cap_for_config(config, fixed_n, test_batch.samples),
            **tuning,
            "adaptive_accuracy": float(np.mean(adaptive["correct"])),
            "fixed_accuracy": float(np.mean(fixed["correct"])),
            "adaptive_mean_generations": float(np.mean(adaptive["generations"])),
            "fixed_mean_generations": float(np.mean(fixed["generations"])),
            "adaptive_mean_output_tokens": float(np.mean(adaptive["tokens"])),
            "fixed_mean_output_tokens": float(np.mean(fixed["tokens"])),
            "adaptive_total_generations": int(np.sum(adaptive["generations"])),
            "fixed_total_generations": int(np.sum(fixed["generations"])),
            "adaptive_total_output_tokens": int(np.sum(adaptive["tokens"])),
            "fixed_total_output_tokens": int(np.sum(fixed["tokens"])),
            "adaptive_profit": float(np.mean(adaptive["profit"])),
            "fixed_profit": float(np.mean(fixed["profit"])),
            "profit_delta": float(np.mean(delta)),
            "relative_profit_improvement_percent": (
                100.0 * float(np.mean(delta)) / abs(float(np.mean(fixed["profit"])))
                if float(np.mean(fixed["profit"])) != 0.0
                else math.nan
            ),
        }
        rows.append(row)
        policies.append({
            "split": split,
            "utility_divisor": divisor,
            "fixed_n_tuned_on_calibration": fixed_n,
            "adaptive_policy_tuned_on_calibration": config.to_dict(),
            **tuning,
        })
        prefix = f"d{divisor_id:02d}"
        for name, values in adaptive.items():
            trial_arrays[f"adaptive_{name}_{prefix}"] = values
        for name, values in fixed.items():
            trial_arrays[f"fixed_{name}_{prefix}"] = values
        trial_arrays[f"profit_delta_{prefix}"] = delta

    atomic_npz(output / "trials" / f"split_{split:02d}.npz", **trial_arrays)
    elapsed = time.monotonic() - started
    print(
        f"split {split + 1}: {len(train)} train/{len(test)} test, "
        f"{test_batch.trials} test trials, {elapsed:.1f}s",
        flush=True,
    )
    return rows, policies


def crossfit_problem_deltas(
    output: Path, divisors: tuple[float, ...], splits: int
) -> dict[float, np.ndarray]:
    """Average repeated cross-fitted deltas within each unique problem ID."""
    values: dict[float, dict[str, list[float]]] = {
        divisor: {} for divisor in divisors
    }
    for split in range(splits):
        with np.load(output / "trials" / f"split_{split:02d}.npz") as data:
            ids = data["problem_ids"]
            stored_divisors = data["divisors"]
            for divisor in divisors:
                divisor_id = int(np.flatnonzero(stored_divisors == divisor)[0])
                delta = data[f"profit_delta_d{divisor_id:02d}"]
                for problem_id in np.unique(ids):
                    values[divisor].setdefault(str(problem_id), []).append(
                        float(np.mean(delta[ids == problem_id]))
                    )
    return {
        divisor: np.asarray(
            [np.mean(items) for items in by_problem.values()], dtype=np.float64
        )
        for divisor, by_problem in values.items()
    }


def summarize(
    split_rows: list[dict[str, object]],
    divisors: tuple[float, ...],
    output: Path,
    splits: int,
) -> list[dict[str, object]]:
    summary = []
    problem_deltas = crossfit_problem_deltas(output, divisors, splits)
    for divisor in divisors:
        rows = [x for x in split_rows if float(x["utility_divisor"]) == divisor]
        adaptive_profit = np.asarray([x["adaptive_profit"] for x in rows], dtype=float)
        fixed_profit = np.asarray([x["fixed_profit"] for x in rows], dtype=float)
        delta = adaptive_profit - fixed_profit
        low, high = interval(delta)
        problem_low, problem_high = interval(problem_deltas[divisor])
        fixed_mean = float(np.mean(fixed_profit))
        adaptive_total_generations = int(sum(int(x["adaptive_total_generations"]) for x in rows))
        fixed_total_generations = int(sum(int(x["fixed_total_generations"]) for x in rows))
        adaptive_total_tokens = int(sum(int(x["adaptive_total_output_tokens"]) for x in rows))
        fixed_total_tokens = int(sum(int(x["fixed_total_output_tokens"]) for x in rows))
        trial_count = int(sum(int(x["test_trials"]) for x in rows))
        summary.append({
            "utility_divisor": divisor,
            "price_per_token": 1.0 / divisor,
            "splits": len(rows),
            "evaluation_trials": trial_count,
            "mean_train_tuned_fixed_n": float(np.mean([x["fixed_n"] for x in rows])),
            "adaptive_accuracy": float(np.mean([x["adaptive_accuracy"] for x in rows])),
            "fixed_accuracy": float(np.mean([x["fixed_accuracy"] for x in rows])),
            "adaptive_mean_generations": adaptive_total_generations / trial_count,
            "fixed_mean_generations": fixed_total_generations / trial_count,
            "adaptive_total_generations": adaptive_total_generations,
            "fixed_total_generations": fixed_total_generations,
            "adaptive_mean_output_tokens": adaptive_total_tokens / trial_count,
            "fixed_mean_output_tokens": fixed_total_tokens / trial_count,
            "adaptive_total_output_tokens": adaptive_total_tokens,
            "fixed_total_output_tokens": fixed_total_tokens,
            "adaptive_profit": float(np.mean(adaptive_profit)),
            "fixed_profit": fixed_mean,
            "profit_delta": float(np.mean(delta)),
            "profit_delta_ci95_low": low,
            "profit_delta_ci95_high": high,
            "relative_profit_improvement_percent": (
                100.0 * float(np.mean(delta)) / abs(fixed_mean)
                if fixed_mean != 0.0 else math.nan
            ),
            "positive_splits": int(np.count_nonzero(delta > 0.0)),
            "tied_splits": int(np.count_nonzero(delta == 0.0)),
            "unique_problem_clusters": len(problem_deltas[divisor]),
            "crossfit_problem_profit_delta": float(
                np.mean(problem_deltas[divisor])
            ),
            "crossfit_problem_delta_ci95_low": problem_low,
            "crossfit_problem_delta_ci95_high": problem_high,
            "crossfit_problem_significant_positive_95": problem_low > 0.0,
        })
    return summary


def write_report(output: Path, summary: list[dict[str, object]], args: argparse.Namespace) -> None:
    best = max(summary, key=lambda x: float(x["profit_delta"]))
    lines = [
        "# Non-parametric DMRL coding token-profit results",
        "",
        f"The offline half of each split fits only an increasing isotonic map and tunes both the top-{args.width} DMRL policy and the Fixed-N baseline. The frozen policies run on unseen problems; correctness is revealed only after stopping.",
        f"Adaptive calibration ranking uses {args.tuning_target} utility with a {args.tuning_cost_scale:g}x output-token cost weight; held-out profit uses the actual cost.",
        "",
        "Profit is `selected_correct - cumulative_output_tokens / utility_divisor`.",
        "",
        "| divisor | fixed N | adaptive profit | Fixed-N profit | delta | relative | adaptive gen | fixed gen | adaptive tokens | fixed tokens | positive splits | 95% problem-cluster CI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in summary:
        lines.append(
            f"| {float(row['utility_divisor']):,.0f} | {float(row['mean_train_tuned_fixed_n']):.1f} | "
            f"{float(row['adaptive_profit']):.5f} | {float(row['fixed_profit']):.5f} | "
            f"{float(row['profit_delta']):+.5f} | {float(row['relative_profit_improvement_percent']):+.2f}% | "
            f"{float(row['adaptive_mean_generations']):.1f} | {float(row['fixed_mean_generations']):.1f} | "
            f"{float(row['adaptive_mean_output_tokens']):,.0f} | {float(row['fixed_mean_output_tokens']):,.0f} | "
            f"{int(row['positive_splits'])}/{int(row['splits'])} | "
            f"[{float(row['crossfit_problem_delta_ci95_low']):+.5f}, {float(row['crossfit_problem_delta_ci95_high']):+.5f}] |"
        )
    lines.extend([
        "",
        f"Largest mean gain: **{float(best['profit_delta']):+.5f}** ({float(best['relative_profit_improvement_percent']):+.2f}%) at divisor {float(best['utility_divisor']):,.0f}.",
        "",
        "`adaptive_total_generations`, `fixed_total_generations`, `adaptive_total_output_tokens`, and `fixed_total_output_tokens` are in `summary.csv`. They sum all evaluation trials across overlapping outer splits and therefore are computational totals, not unique production requests.",
        "",
        "## Guardrails",
        "",
        "- No shifted-exponential or other parametric tail is fitted.",
        f"- The online statistic is the mean excess of the top {args.width} calibrated utilities above the next-largest, divided by observations.",
        "- Hyperparameters and the comparator N are selected on calibration problems only.",
        "- Test correctness is used only for post-selection profit.",
        "- All comparisons use identical sample permutations and output-token costs.",
        "- The problem-cluster interval first averages repeated cross-fitted outcomes within each unique problem, then computes a t interval across unique problems; split-level intervals remain in `summary.csv`.",
        "",
        f"Run: {args.splits} outer splits, {args.train_permutations} calibration permutations/problem, {args.test_permutations} test permutations/problem.",
    ])
    path = output / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")


def write_plot(output: Path, summary: list[dict[str, object]]) -> bool:
    """Write a dependency-free SVG so binary plotting wheels cannot block results."""
    x = np.asarray([x["utility_divisor"] for x in summary], dtype=float)
    y = np.asarray([x["relative_profit_improvement_percent"] for x in summary], dtype=float)
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 35, 75
    log_x = np.log(x)
    x_span = max(float(np.ptp(log_x)), 1.0)
    y_low = min(float(np.min(y)), 0.0)
    y_high = max(float(np.max(y)), 0.0)
    y_pad = max((y_high - y_low) * 0.12, 0.1)
    y_low -= y_pad
    y_high += y_pad
    plot_width = width - left - right
    plot_height = height - top - bottom
    px = left + (log_x - np.min(log_x)) / x_span * plot_width
    py = top + (y_high - y) / (y_high - y_low) * plot_height
    zero_y = top + y_high / (y_high - y_low) * plot_height
    points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" stroke="#555"/>',
        f'<polyline points="{points}" fill="none" stroke="#1769aa" stroke-width="3"/>',
    ]
    for divisor, value, a, b in zip(x, y, px, py):
        elements.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="5" fill="#1769aa"/>')
        elements.append(f'<text x="{a:.1f}" y="{height-bottom+24}" text-anchor="middle" font-size="12">{divisor/1000:g}k</text>')
        elements.append(f'<text x="{a:.1f}" y="{b-10:.1f}" text-anchor="middle" font-size="11">{value:+.2f}%</text>')
    elements.extend([
        f'<text x="{width/2:.1f}" y="{height-18}" text-anchor="middle" font-size="15">Utility divisor (tokens per unit correctness value; log spacing)</text>',
        f'<text transform="translate(22 {height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="15">Profit improvement over train-tuned Fixed-N (%)</text>',
        '</svg>',
    ])
    (output / "profit_improvement.svg").write_text("\n".join(elements) + "\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    economic = parser.add_mutually_exclusive_group()
    economic.add_argument("--prices", type=parse_divisors,
                          help="comma-separated positive prices in dollars per output token")
    economic.add_argument("--divisors", type=parse_divisors, default=DEFAULT_DIVISORS,
                          help="legacy reciprocal-price input; prefer --prices")
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--outer-seed", type=int, default=20260923)
    parser.add_argument("--train-permutations", type=int, default=24)
    parser.add_argument("--test-permutations", type=int, default=48)
    parser.add_argument("--expected-samples", type=int, default=512)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument(
        "--tuning-target", choices=("calibrated", "observed"),
        default="calibrated",
        help="calibration-only objective for choosing stopping controls",
    )
    parser.add_argument(
        "--tuning-cost-scale", type=float, default=1.0,
        help="scale output-token cost only when selecting a policy offline",
    )
    args = parser.parse_args()
    if args.prices is not None:
        args.divisors = tuple(1.0 / price for price in args.prices)
    if args.splits < 1 or args.train_permutations < 1 or args.test_permutations < 1:
        parser.error("splits and permutation counts must be positive")
    if args.width < 1 or args.width >= args.expected_samples:
        parser.error("width must be between 1 and expected-samples minus one")
    if not math.isfinite(args.tuning_cost_scale) or args.tuning_cost_scale <= 0.0:
        parser.error("tuning-cost-scale must be finite and positive")

    started = time.monotonic()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    problems = load_coding_problems(args.data, expected_samples=args.expected_samples)
    candidates = candidate_grid(args.width)
    manifest = {
        "schema": "coding_nonparametric_dmrl_profit_study",
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data),
        "data_sha256": sha256(args.data),
        "problems": len(problems),
        "samples": sum(len(x.rewards) for x in problems.values()),
        "length_field": "output_tokens",
        "profit": "selected_correct - cumulative_output_tokens / utility_divisor",
        "adaptive_tuning_objective": (
            "selected_isotonic_probability - tuning_cost_scale * cumulative_output_tokens / utility_divisor"
            if args.tuning_target == "calibrated" else
            "selected_actual_correctness - tuning_cost_scale * cumulative_output_tokens / utility_divisor"
        ),
        "tuning_cost_scale": args.tuning_cost_scale,
        "fixed_n_tuning_objective": "exact_random_order_correctness - expected_output_tokens / utility_divisor",
        "divisors": args.divisors,
        "prices_per_output_token": [1.0 / value for value in args.divisors],
        "splits": args.splits,
        "outer_seed": args.outer_seed,
        "train_permutations": args.train_permutations,
        "test_permutations": args.test_permutations,
        "candidate_count_per_divisor": len(candidates),
        "candidate_grid": {
            "smoothing": SMOOTHING_MODES,
            "multipliers": MULTIPLIERS,
            "cost_adjustments": COST_ADJUSTMENTS,
            "cap_factors": CAP_FACTORS,
            "minimum_factors": MINIMUM_FACTORS,
        },
        "test_label_online_access": False,
        "online_distribution_fit": False,
        "width": args.width,
        "policy_family": policy_family(args.width),
        "response_selection": "maximum raw CodeScaler reward; calibrated probability is used for DMRL utility and stopping",
    }
    atomic_json(args.output / "manifest.json", manifest)

    split_rows: list[dict[str, object]] = []
    policies: list[dict[str, object]] = []
    for split in range(args.splits):
        rows, selected = run_split(
            split,
            problems,
            args.divisors,
            args.outer_seed,
            args.train_permutations,
            args.test_permutations,
            args.output,
            candidates,
            tuning_target=args.tuning_target,
            tuning_cost_scale=args.tuning_cost_scale,
            width=args.width,
        )
        split_rows.extend(rows)
        policies.extend(selected)
        atomic_csv(args.output / "split_metrics.partial.csv", split_rows)
        atomic_json(args.output / "selected_policies.partial.json", policies)

    summary = summarize(split_rows, args.divisors, args.output, args.splits)
    atomic_csv(args.output / "split_metrics.csv", split_rows)
    atomic_csv(args.output / "summary.csv", summary)
    atomic_json(args.output / "selected_policies.json", policies)
    write_report(args.output, summary, args)
    write_plot(args.output, summary)
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({
        "output": str(args.output),
        "elapsed_seconds": manifest["elapsed_seconds"],
        "best": max(summary, key=lambda x: float(x["profit_delta"])),
    }, indent=2))


if __name__ == "__main__":
    main()
