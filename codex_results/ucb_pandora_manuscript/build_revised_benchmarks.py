"""Build all manuscript comparisons for the confirmed simple policies."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
CODEX_RESULTS = HERE.parent
REPO = CODEX_RESULTS.parent
ALIGNMENT_CODE = CODEX_RESULTS / "code/alignment_scripts"
CODING_CODE = CODEX_RESULTS / "code/coding_scripts"
for source in (ALIGNMENT_CODE, CODING_CODE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from alignment_pandora_ucb import _stable_alpha_quantile, load_data  # noqa: E402
from coding_ucb_three_objectives import (  # noqa: E402
    fit_reward_space_isotonic,
    fixed_trials,
    load_problems,
    make_permutations,
    split_problems,
    transform_reward_space,
)


OUTPUT = HERE / "artifacts/revised_benchmarks"
FIGURES = HERE / "figures"
ALIGNMENT_SIMPLE_DIR = (
    CODEX_RESULTS / "results/alignment/simplification/final_confirmation"
)
ALIGNMENT_SIMPLE_VARIANT = "local_exp_open5_conf06"
ALIGNMENT_ROWS = ALIGNMENT_SIMPLE_DIR / "utility_splits.csv"
ALIGNMENT_METHOD = ALIGNMENT_SIMPLE_DIR / "METHOD.json"
ALIGNMENT_HEAD_TO_HEAD = ALIGNMENT_SIMPLE_DIR / "head_to_head_cells.csv"
ALIGNMENT_TARGET_ROWS = ALIGNMENT_SIMPLE_DIR / "target_splits.csv"
CODING_SIMPLE_DIR = (
    CODEX_RESULTS
    / "results/coding/simplification/final_confirmation_reservation"
)
CODING_SIMPLE_VARIANT = "global_exp_q75_conf08_085x"
CODING_ROWS = CODING_SIMPLE_DIR / "utility_splits.csv"
CODING_METHOD = CODING_SIMPLE_DIR / "METHOD.json"
CODING_TARGET = (
    CODEX_RESULTS
    / "results/coding/simplification/target_final_profile_only/"
      "target_quality_summary.csv"
)

ALIGNMENT_GENERATORS = (
    "gemma2_9b",
    "llama3.1_8b",
    "llama3.2_3b",
    "mistral_7b",
    "qwen2.5_7b",
)
ALIGNMENT_LABELS = {
    "gemma2_9b": "Gemma-2-9B",
    "llama3.1_8b": "Llama-3.1-8B",
    "llama3.2_3b": "Llama-3.2-3B",
    "mistral_7b": "Mistral-7B",
    "qwen2.5_7b": "Qwen-2.5-7B",
}
ALIGNMENT_COLORS = {
    "gemma2_9b": "#3A6EA5",
    "llama3.1_8b": "#D1495B",
    "llama3.2_3b": "#7A5195",
    "mistral_7b": "#2E7D32",
    "qwen2.5_7b": "#E17C05",
}
ALIGNMENT_FRONTIER_DIVISORS = (
    5e4, 7.5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7,
)
ALIGNMENT_TARGETS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
CODING_METHOD_NAME = "train_selected_ucb"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_interval(values) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half = float(
        student_t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / math.sqrt(len(values))
    )
    return mean, mean - half, mean + half


def select_cheapest_target(
    qualities: np.ndarray, costs: np.ndarray, target: float
) -> tuple[int, bool]:
    """Select one policy using only selection-set quality and cost.

    Among policies reaching the target, choose the lowest-cost policy.  If the
    target is unattainable, choose maximum quality, breaking ties by cost and
    then by index.  No interpolation or policy randomization is used.
    """
    qualities = np.asarray(qualities, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    if qualities.ndim != 1 or qualities.shape != costs.shape:
        raise ValueError("quality and cost must be matching one-dimensional curves")
    feasible = np.flatnonzero(qualities >= target)
    if len(feasible):
        winner = min(feasible, key=lambda index: (costs[index], int(index)))
        return int(winner), True
    best_quality = float(np.max(qualities))
    best = np.flatnonzero(np.isclose(qualities, best_quality, rtol=0.0, atol=1e-15))
    winner = min(best, key=lambda index: (costs[index], int(index)))
    return int(winner), False


def adjacent_monotone_mix(
    curve: np.ndarray, target: float
) -> tuple[int, int, float]:
    """Mix adjacent points of a monotone curve to match a target exactly."""
    curve = np.maximum.accumulate(np.asarray(curve, dtype=np.float64))
    if curve.ndim != 1 or len(curve) == 0:
        raise ValueError("curve must be nonempty and one-dimensional")
    if target <= curve[0]:
        return 0, 0, 0.0
    if target >= curve[-1]:
        last = len(curve) - 1
        return last, last, 0.0
    high = int(np.searchsorted(curve, target, side="left"))
    low = high - 1
    if curve[high] <= curve[low] + 1e-15:
        return high, high, 0.0
    weight = (target - curve[low]) / (curve[high] - curve[low])
    return low, high, float(np.clip(weight, 0.0, 1.0))


def adjacent_cost_mix(curve: np.ndarray, target: float) -> tuple[int, int, float]:
    """Mix adjacent Fixed-N policies to match expected characters."""
    return adjacent_monotone_mix(curve, target)


def adjacent_quality_mix(
    curve: np.ndarray, target: float
) -> tuple[int, int, float]:
    """Mix adjacent Fixed-N policies to match expected quality."""
    return adjacent_monotone_mix(curve, target)


def minimum_cost_quality_mix(
    qualities: np.ndarray, costs: np.ndarray, target: float
) -> tuple[int, int, float, bool]:
    """Solve the exact-quality oracle over randomized Fixed-N policies.

    This is a linear program with normalization and one quality constraint, so
    an optimum has support on at most two N values.  Enumerating all bracketing
    pairs therefore gives the global minimum expected cost.
    """
    qualities = np.asarray(qualities, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    if qualities.ndim != 1 or costs.shape != qualities.shape or not len(qualities):
        raise ValueError("quality and cost curves must be nonempty 1-D peers")
    if not np.all(np.isfinite(qualities)) or not np.all(np.isfinite(costs)):
        raise ValueError("quality and cost curves must be finite")
    target = float(target)
    low_ids = np.flatnonzero(qualities <= target + 1e-12)
    high_ids = np.flatnonzero(qualities >= target - 1e-12)
    if not len(low_ids) or not len(high_ids):
        closest_error = float(np.min(np.abs(qualities - target)))
        candidates = np.flatnonzero(
            np.abs(qualities - target) <= closest_error + 1e-12
        )
        index = int(min(candidates, key=lambda item: (costs[item], int(item))))
        return index, index, 0.0, False

    best = None
    for low in low_ids:
        q_low = float(qualities[low])
        q_high = qualities[high_ids]
        denominator = q_high - q_low
        weights = np.divide(
            target - q_low,
            denominator,
            out=np.zeros_like(denominator),
            where=np.abs(denominator) > 1e-15,
        )
        weights = np.clip(weights, 0.0, 1.0)
        mixed_costs = (1.0 - weights) * costs[low] + weights * costs[high_ids]
        for position, high in enumerate(high_ids):
            weight = float(weights[position])
            matched_quality = (
                (1.0 - weight) * q_low
                + weight * float(qualities[high])
            )
            if abs(matched_quality - target) > 2e-12:
                continue
            candidate = (
                float(mixed_costs[position]),
                abs(int(high) - int(low)),
                int(low), int(high), weight,
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise AssertionError("bracketing policies failed to match target quality")
    _, _, low, high, weight = best
    return low, high, weight, True


def tune_fixed_character_budgets(
    accuracy: np.ndarray,
    cumulative_chars: np.ndarray,
    divisors: tuple[float, ...],
) -> dict[float, tuple[float, float, float, float]]:
    """Tune one observed-character stopping threshold for each divisor.

    The policy always opens one response and continues while cumulative
    observed characters are at most R.  Its policy changes only when R crosses
    a realized prefix cost, so a sweep over those breakpoints is exact.
    Returns R, utility, accuracy, and actual characters on the tuning trials.
    """
    accuracy = np.asarray(accuracy, dtype=np.float64)
    cumulative_chars = np.asarray(cumulative_chars, dtype=np.float64)
    if accuracy.shape != cumulative_chars.shape or accuracy.ndim != 2:
        raise ValueError("accuracy and character curves must be matching matrices")
    if accuracy.shape[0] == 0 or accuracy.shape[1] == 0:
        raise ValueError("at least one trial and one response are required")
    if np.any(np.diff(cumulative_chars, axis=1) <= 0.0):
        raise ValueError("cumulative character curves must be strictly increasing")

    trial_count, horizon = accuracy.shape
    mean_accuracy = [float(np.mean(accuracy[:, 0]))]
    mean_chars = [float(np.mean(cumulative_chars[:, 0]))]
    budgets = [0.0]
    if horizon > 1:
        events = cumulative_chars[:, :-1].ravel()
        delta_accuracy = np.diff(accuracy, axis=1).ravel()
        delta_chars = np.diff(cumulative_chars, axis=1).ravel()
        order = np.argsort(events, kind="mergesort")
        sorted_events = events[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_events) > 0.0) + 1]
        unique_events = sorted_events[starts]
        grouped_accuracy = np.add.reduceat(delta_accuracy[order], starts)
        grouped_chars = np.add.reduceat(delta_chars[order], starts)
        budgets.extend(float(value) for value in unique_events)
        mean_accuracy.extend(
            mean_accuracy[0] + np.cumsum(grouped_accuracy) / trial_count
        )
        mean_chars.extend(mean_chars[0] + np.cumsum(grouped_chars) / trial_count)

    budgets = np.asarray(budgets, dtype=np.float64)
    mean_accuracy = np.asarray(mean_accuracy, dtype=np.float64)
    mean_chars = np.asarray(mean_chars, dtype=np.float64)
    selected = {}
    for divisor in divisors:
        utility = mean_accuracy - mean_chars / float(divisor)
        winner = int(np.argmax(utility))
        selected[float(divisor)] = (
            float(budgets[winner]),
            float(utility[winner]),
            float(mean_accuracy[winner]),
            float(mean_chars[winner]),
        )
    return selected


def evaluate_fixed_character_budget(
    accuracy: np.ndarray,
    cumulative_chars: np.ndarray,
    budget: float,
    divisor: float,
) -> tuple[float, float, float, float]:
    """Evaluate a frozen threshold; return utility, accuracy, chars, and opens."""
    accuracy = np.asarray(accuracy, dtype=np.float64)
    cumulative_chars = np.asarray(cumulative_chars, dtype=np.float64)
    if accuracy.shape != cumulative_chars.shape or accuracy.ndim != 2:
        raise ValueError("accuracy and character curves must be matching matrices")
    stops = 1 + np.sum(cumulative_chars[:, :-1] <= budget, axis=1)
    row = np.arange(len(stops))
    chosen_accuracy = accuracy[row, stops - 1]
    chosen_chars = cumulative_chars[row, stops - 1]
    return (
        float(np.mean(chosen_accuracy - chosen_chars / float(divisor))),
        float(np.mean(chosen_accuracy)),
        float(np.mean(chosen_chars)),
        float(np.mean(stops)),
    )


def oracle_budget_choice(
    utility_by_problem_n: np.ndarray,
    chars_by_problem_n: np.ndarray,
    oracle_mean_lengths: np.ndarray,
) -> tuple[float, float, float, float]:
    """Choose the best global character budget on the evaluation problems.

    For a budget R and oracle per-problem mean length l_x, the rule opens
    floor(R / l_x) responses, clipped to the finite candidate horizon and with
    at least one response.  Every allocation change occurs at R = n l_x, so
    enumerating those breakpoints is exact for this policy class.
    """
    if utility_by_problem_n.shape != chars_by_problem_n.shape:
        raise ValueError("utility and character curves must have the same shape")
    problems, horizon = utility_by_problem_n.shape
    if oracle_mean_lengths.shape != (problems,):
        raise ValueError("one oracle mean response length is required per problem")
    if np.any(oracle_mean_lengths <= 0.0):
        raise ValueError("oracle mean response lengths must be positive")

    breakpoints = np.unique(
        (
            oracle_mean_lengths[:, None]
            * np.arange(1, horizon + 1, dtype=np.float64)[None, :]
        ).ravel()
    )
    allocations = np.floor(
        breakpoints[:, None] / oracle_mean_lengths[None, :] + 1e-12
    ).astype(np.int64)
    allocations = np.clip(allocations, 1, horizon)
    problem_indices = np.arange(problems)[None, :]
    budget_utilities = utility_by_problem_n[
        problem_indices, allocations - 1
    ].mean(axis=1)
    winner = int(np.argmax(budget_utilities))
    chosen_n = allocations[winner]
    chosen_chars = chars_by_problem_n[
        np.arange(problems), chosen_n - 1
    ]
    return (
        float(breakpoints[winner]),
        float(budget_utilities[winner]),
        float(chosen_n.mean()),
        float(chosen_chars.mean()),
    )


def alignment_test_curves(
    rewards: np.ndarray,
    chars: np.ndarray,
    test_indices: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the exact fixed-policy test permutations of the saved run."""
    rng = np.random.default_rng(seed)
    quality = np.empty(
        (len(test_indices), permutations, rewards.shape[1]), dtype=np.float64
    )
    cumulative_chars = np.empty_like(quality)
    for problem_position, problem_index in enumerate(test_indices):
        problem_index = int(problem_index)
        benchmark = _stable_alpha_quantile(rewards[problem_index], 0.99)
        for permutation_id in range(permutations):
            order = rng.permutation(rewards.shape[1])
            # The retained runner draws an independent head-to-head order here.
            # It does not affect this comparison, but consuming the draw is
            # necessary to reproduce all subsequent fixed-policy orders.
            rng.permutation(rewards.shape[1])
            ordered_rewards = rewards[problem_index, order]
            ordered_chars = chars[problem_index, order]
            quality[problem_position, permutation_id] = expit(
                np.maximum.accumulate(ordered_rewards) - benchmark
            )
            cumulative_chars[problem_position, permutation_id] = np.cumsum(
                ordered_chars, dtype=np.float64
            )
    return quality, cumulative_chars


def build_alignment() -> tuple[list[dict], list[dict]]:
    saved = [
        row for row in read_csv(ALIGNMENT_ROWS)
        if row.get("variant") == ALIGNMENT_SIMPLE_VARIANT
    ]
    method = json.loads(ALIGNMENT_METHOD.read_text())
    seed = int(method.get("seed", 20260802))
    split_start = int(method["split_start"])
    split_count = int(method["splits"])
    permutations = int(method.get(
        "test_permutations_per_prompt", method["test_permutations"]
    ))
    reward_key = method.get("reward_key", "mistral_rm_reward")
    data_dir = REPO / "dataset/alpaca"
    detailed = []

    saved_by_key = {
        (row["generator"], int(row["split"]), float(row["divisor"])): row
        for row in saved
    }
    for generator_id, generator in enumerate(ALIGNMENT_GENERATORS):
        rewards, chars, _ = load_data(
            data_dir / f"{generator}_output.merged_rm.jsonl.gz", reward_key
        )
        oracle_mean_lengths_all = chars.mean(axis=1)
        for split in range(split_start, split_start + split_count):
            split_order = np.random.default_rng(seed + split).permutation(len(rewards))
            test_indices = split_order[len(split_order) // 2 :]
            quality, cumulative_chars = alignment_test_curves(
                rewards,
                chars,
                test_indices,
                permutations,
                seed + 100_000 * generator_id + 1009 * split + 31,
            )
            quality_curve = quality.mean(axis=(0, 1))
            chars_curve = cumulative_chars.mean(axis=(0, 1))
            problem_quality = quality.mean(axis=1)
            problem_chars = cumulative_chars.mean(axis=1)

            divisors = sorted(
                key[2]
                for key in saved_by_key
                if key[0] == generator and key[1] == split
            )
            for divisor in divisors:
                original = saved_by_key[(generator, split, divisor)]
                train_n = int(original["fixed_n"])
                train_utility = float(
                    quality_curve[train_n - 1] - chars_curve[train_n - 1] / divisor
                )
                if not math.isclose(
                    train_utility,
                    float(original["fixed_utility"]),
                    rel_tol=0.0,
                    abs_tol=2e-12,
                ):
                    raise AssertionError(
                        f"alignment replay mismatch for {generator}, split {split}, "
                        f"divisor {divisor}"
                    )

                test_utility_curve = quality_curve - chars_curve / divisor
                oracle_n_index = int(np.argmax(test_utility_curve))
                oracle_n_utility = float(test_utility_curve[oracle_n_index])
                problem_utility = problem_quality - problem_chars / divisor
                budget, budget_utility, budget_mean_n, budget_actual_chars = (
                    oracle_budget_choice(
                        problem_utility,
                        problem_chars,
                        oracle_mean_lengths_all[test_indices],
                    )
                )
                adaptive = float(original["adaptive_utility"])
                detailed.append({
                    "generator": generator,
                    "split": split,
                    "divisor": divisor,
                    "adaptive_utility": adaptive,
                    "train_fixed_n": train_n,
                    "train_fixed_utility": train_utility,
                    "oracle_fixed_n": oracle_n_index + 1,
                    "oracle_fixed_n_utility": oracle_n_utility,
                    "oracle_budget_chars": budget,
                    "oracle_budget_mean_n": budget_mean_n,
                    "oracle_budget_actual_chars": budget_actual_chars,
                    "oracle_budget_utility": budget_utility,
                    "ratio_vs_train_fixed": adaptive / train_utility,
                    "ratio_vs_oracle_fixed_n": adaptive / oracle_n_utility,
                    "ratio_vs_oracle_budget": adaptive / budget_utility,
                    "relative_improvement_vs_train_fixed_pct":
                        100.0 * (adaptive - train_utility) / train_utility,
                    "relative_improvement_vs_oracle_fixed_n_pct":
                        100.0 * (adaptive - oracle_n_utility) / oracle_n_utility,
                    "relative_improvement_vs_oracle_budget_pct":
                        100.0 * (adaptive - budget_utility) / budget_utility,
                    "gap_vs_train_fixed": adaptive - train_utility,
                    "gap_vs_oracle_fixed_n": adaptive - oracle_n_utility,
                    "gap_vs_oracle_budget": adaptive - budget_utility,
                })
        print(f"[alignment benchmark] {generator} complete", flush=True)

    summary = aggregate(
        detailed,
        ("generator", "divisor"),
        (
            "adaptive_utility",
            "train_fixed_n",
            "train_fixed_utility",
            "oracle_fixed_n",
            "oracle_fixed_n_utility",
            "oracle_budget_chars",
            "oracle_budget_mean_n",
            "oracle_budget_actual_chars",
            "oracle_budget_utility",
            "ratio_vs_train_fixed",
            "ratio_vs_oracle_fixed_n",
            "ratio_vs_oracle_budget",
            "relative_improvement_vs_train_fixed_pct",
            "relative_improvement_vs_oracle_fixed_n_pct",
            "relative_improvement_vs_oracle_budget_pct",
            "gap_vs_train_fixed",
            "gap_vs_oracle_fixed_n",
            "gap_vs_oracle_budget",
        ),
    )
    return detailed, summary


def build_alignment_target() -> tuple[list[dict], list[dict]]:
    """Add an exact-quality held-out Fixed-N diagnostic to target results."""
    saved = [
        row for row in read_csv(ALIGNMENT_TARGET_ROWS)
        if row.get("variant") == ALIGNMENT_SIMPLE_VARIANT
    ]
    method = json.loads(ALIGNMENT_METHOD.read_text())
    seed = int(method.get("seed", 20260802))
    permutations = int(method.get(
        "test_permutations_per_prompt", method["test_permutations"]
    ))
    reward_key = method.get("reward_key", "mistral_rm_reward")
    data_dir = REPO / "dataset/alpaca"
    rows_by_key: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in saved:
        rows_by_key.setdefault(
            (row["generator"], int(row["split"])), []
        ).append(row)

    detailed = []
    for generator_id, generator in enumerate(ALIGNMENT_GENERATORS):
        rewards, chars, _ = load_data(
            data_dir / f"{generator}_output.merged_rm.jsonl.gz", reward_key
        )
        for generator_name, split in sorted(
            key for key in rows_by_key if key[0] == generator
        ):
            split_order = np.random.default_rng(seed + split).permutation(len(rewards))
            test_indices = split_order[len(split_order) // 2 :]
            quality, cumulative_chars = alignment_test_curves(
                rewards,
                chars,
                test_indices,
                permutations,
                seed + 100_000 * generator_id + 1009 * split + 31,
            )
            quality_curve = quality.mean(axis=(0, 1))
            chars_curve = cumulative_chars.mean(axis=(0, 1))
            for source in rows_by_key[(generator_name, split)]:
                row = dict(source)
                for key in (
                    "target_quality", "selected_divisor", "adaptive_test_quality",
                    "adaptive_test_chars", "fixed_test_quality", "fixed_test_chars",
                    "adaptive_target_error_pp", "fixed_target_error_pp",
                    "character_saving_vs_train_fixed_pct",
                ):
                    row[key] = float(row[key])
                row["split"] = split
                for key in (
                    "adaptive_target_reached_train", "fixed_target_reached_train"
                ):
                    row[key] = row[key].lower() == "true"

                target = row["adaptive_test_quality"]
                low, high, weight, reached = minimum_cost_quality_mix(
                    quality_curve, chars_curve, target
                )
                matched_quality = float(
                    (1.0 - weight) * quality_curve[low]
                    + weight * quality_curve[high]
                )
                matched_chars = float(
                    (1.0 - weight) * chars_curve[low]
                    + weight * chars_curve[high]
                )
                row.update({
                    "matched_fixed_n_low": low + 1,
                    "matched_fixed_n_high": high + 1,
                    "matched_fixed_high_weight": weight,
                    "matched_fixed_expected_n": (
                        (1.0 - weight) * (low + 1) + weight * (high + 1)
                    ),
                    "matched_fixed_quality": matched_quality,
                    "matched_fixed_chars": matched_chars,
                    "matched_target_reached": reached,
                    "matched_quality_error_pp": 100.0 * (
                        matched_quality - target
                    ),
                    "character_saving_vs_matched_fixed_pct": 100.0 * (
                        matched_chars - row["adaptive_test_chars"]
                    ) / matched_chars,
                })
                detailed.append(row)
        print(f"[alignment target match] {generator} complete", flush=True)
    summary = aggregate(
        detailed,
        ("generator", "target_quality"),
        (
            "selected_divisor",
            "adaptive_test_quality",
            "adaptive_test_chars",
            "fixed_test_quality",
            "fixed_test_chars",
            "adaptive_target_error_pp",
            "fixed_target_error_pp",
            "character_saving_vs_train_fixed_pct",
            "matched_fixed_n_low",
            "matched_fixed_n_high",
            "matched_fixed_high_weight",
            "matched_fixed_expected_n",
            "matched_fixed_quality",
            "matched_fixed_chars",
            "matched_quality_error_pp",
            "character_saving_vs_matched_fixed_pct",
        ),
    )
    return detailed, summary


def build_coding() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    saved = [
        row for row in read_csv(CODING_ROWS)
        if row.get("variant") == CODING_SIMPLE_VARIANT
    ]
    method = json.loads(CODING_METHOD.read_text())
    seed = int(method["seed"])
    permutations = int(method["test_permutations"])
    problems = load_problems(
        REPO / "algorithm/bestofn_coding/data.jsonl",
        REPO
        / "algorithm/bestofn_coding/practical_algorithm/coding_char_counts_83.npz",
    )
    rows_by_split: dict[int, list[dict[str, str]]] = {}
    for row in saved:
        rows_by_split.setdefault(int(row["split"]), []).append(row)

    detailed = []
    equal_budget = []
    for split in sorted(rows_by_split):
        raw_train, raw_test = split_problems(problems, seed + split)
        calibration = fit_reward_space_isotonic(raw_train)
        train = transform_reward_space(raw_train, calibration)
        test = transform_reward_space(raw_test, calibration)
        train_permutations = make_permutations(
            train, permutations, seed + 101_003 * split + 17
        )
        train_budget_accuracy, train_budget_chars, _ = fixed_trials(
            train, train_permutations
        )
        split_divisors = tuple(sorted({
            float(row["utility_divisor"]) for row in rows_by_split[split]
        }))
        selected_budgets = tune_fixed_character_budgets(
            train_budget_accuracy, train_budget_chars, split_divisors
        )
        test_permutations = make_permutations(
            test, permutations, seed + 101_003 * split + 31
        )
        fixed_accuracy, fixed_chars, _ = fixed_trials(test, test_permutations)
        accuracy_curve = fixed_accuracy.mean(axis=0)
        chars_curve = fixed_chars.mean(axis=0)
        opponent_permutations = make_permutations(
            test, permutations, seed + 101_003 * split + 47
        )
        opponent_accuracy, opponent_chars, _ = fixed_trials(
            test, opponent_permutations
        )
        opponent_char_curve = opponent_chars.mean(axis=0)
        for original in rows_by_split[split]:
            divisor = float(original["utility_divisor"])
            train_n = int(original["fixed_n"])
            train_utility = float(
                accuracy_curve[train_n - 1] - chars_curve[train_n - 1] / divisor
            )
            if not math.isclose(
                train_utility,
                float(original["fixed_utility"]),
                rel_tol=0.0,
                abs_tol=2e-12,
            ):
                raise AssertionError(
                    f"coding replay mismatch for split {split}, divisor {divisor}"
                )
            test_utility_curve = accuracy_curve - chars_curve / divisor
            oracle_n_index = int(np.argmax(test_utility_curve))
            oracle_utility = float(test_utility_curve[oracle_n_index])
            budget, budget_train_utility, _, _ = selected_budgets[divisor]
            (
                budget_test_utility,
                budget_test_accuracy,
                budget_test_chars,
                budget_test_opens,
            ) = evaluate_fixed_character_budget(
                fixed_accuracy, fixed_chars, budget, divisor
            )
            adaptive = float(original["adaptive_utility"])
            detailed.append({
                "split": split,
                "divisor": divisor,
                "adaptive_utility": adaptive,
                "train_fixed_n": train_n,
                "train_fixed_utility": train_utility,
                "oracle_fixed_n": oracle_n_index + 1,
                "oracle_fixed_n_utility": oracle_utility,
                "train_fixed_budget_chars": budget,
                "train_fixed_budget_train_utility": budget_train_utility,
                "train_fixed_budget_test_utility": budget_test_utility,
                "train_fixed_budget_test_accuracy": budget_test_accuracy,
                "train_fixed_budget_test_chars": budget_test_chars,
                "train_fixed_budget_test_opens": budget_test_opens,
                "ratio_vs_train_fixed": adaptive / train_utility,
                "ratio_vs_oracle_fixed_n": adaptive / oracle_utility,
                "ratio_vs_train_fixed_budget": adaptive / budget_test_utility,
                "relative_improvement_vs_train_fixed_pct":
                    100.0 * (adaptive - train_utility) / train_utility,
                "relative_improvement_vs_oracle_fixed_n_pct":
                    100.0 * (adaptive - oracle_utility) / oracle_utility,
                "relative_improvement_vs_train_fixed_budget_pct":
                    100.0 * (adaptive - budget_test_utility) / budget_test_utility,
                "gap_vs_train_fixed": adaptive - train_utility,
                "gap_vs_oracle_fixed_n": adaptive - oracle_utility,
                "gap_vs_train_fixed_budget": adaptive - budget_test_utility,
            })
            adaptive_chars = float(original["adaptive_chars"])
            low, high, high_weight = adjacent_cost_mix(
                opponent_char_curve, adaptive_chars
            )
            matched_accuracy = float(np.mean(
                (1.0 - high_weight) * opponent_accuracy[:, low]
                + high_weight * opponent_accuracy[:, high]
            ))
            matched_chars = float(np.mean(
                (1.0 - high_weight) * opponent_chars[:, low]
                + high_weight * opponent_chars[:, high]
            ))
            adaptive_accuracy = float(original["adaptive_accuracy"])
            equal_budget.append({
                "split": split,
                "divisor": divisor,
                "adaptive_accuracy": adaptive_accuracy,
                "fixed_accuracy": matched_accuracy,
                "accuracy_gap_pp": 100.0 * (
                    adaptive_accuracy - matched_accuracy
                ),
                "adaptive_chars": adaptive_chars,
                "fixed_chars": matched_chars,
                "cost_mismatch": adaptive_chars - matched_chars,
                "fixed_n_low": low + 1,
                "fixed_n_high": high + 1,
                "fixed_high_weight": high_weight,
            })
        print(f"[coding benchmark] split {split} complete", flush=True)

    summary = aggregate(
        detailed,
        ("divisor",),
        (
            "adaptive_utility",
            "train_fixed_n",
            "train_fixed_utility",
            "oracle_fixed_n",
            "oracle_fixed_n_utility",
            "train_fixed_budget_chars",
            "train_fixed_budget_train_utility",
            "train_fixed_budget_test_utility",
            "train_fixed_budget_test_accuracy",
            "train_fixed_budget_test_chars",
            "train_fixed_budget_test_opens",
            "ratio_vs_train_fixed",
            "ratio_vs_oracle_fixed_n",
            "ratio_vs_train_fixed_budget",
            "relative_improvement_vs_train_fixed_pct",
            "relative_improvement_vs_oracle_fixed_n_pct",
            "relative_improvement_vs_train_fixed_budget_pct",
            "gap_vs_train_fixed",
            "gap_vs_oracle_fixed_n",
            "gap_vs_train_fixed_budget",
        ),
    )
    equal_budget_summary = aggregate(
        equal_budget,
        ("divisor",),
        (
            "accuracy_gap_pp",
            "adaptive_accuracy",
            "fixed_accuracy",
            "adaptive_chars",
            "fixed_chars",
            "cost_mismatch",
        ),
    )
    return detailed, summary, equal_budget, equal_budget_summary


def aggregate(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]):
    groups = sorted({tuple(row[key] for key in keys) for row in rows})
    output = []
    for group_key in groups:
        subset = [
            row for row in rows
            if tuple(row[key] for key in keys) == group_key
        ]
        result = dict(zip(keys, group_key))
        result["splits"] = len(subset)
        for metric in metrics:
            mean, low, high = mean_interval([row[metric] for row in subset])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        output.append(result)
    return output


def plot_alignment(summary: list[dict]) -> None:
    metrics = (
        ("relative_improvement_vs_train_fixed_pct", "A. Train-selected Fixed-$N$"),
        ("relative_improvement_vs_oracle_fixed_n_pct", "B. Test-oracle Fixed-$N$"),
        ("relative_improvement_vs_oracle_budget_pct", "C. Test-oracle character budget"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        for generator in ALIGNMENT_GENERATORS:
            group = sorted(
                (row for row in summary if row["generator"] == generator),
                key=lambda row: float(row["divisor"]),
            )
            x = np.asarray([float(row["divisor"]) for row in group])
            mean = np.asarray([float(row[f"{metric}_mean"]) for row in group])
            low = np.asarray([float(row[f"{metric}_ci_low"]) for row in group])
            high = np.asarray([float(row[f"{metric}_ci_high"]) for row in group])
            color = ALIGNMENT_COLORS[generator]
            axis.plot(
                x, mean, marker="o", linewidth=2.0, color=color,
                label=ALIGNMENT_LABELS[generator],
            )
            axis.fill_between(x, low, high, color=color, alpha=0.12)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xscale("log")
        axis.set(
            xlabel="Character-cost divisor",
            ylabel="Relative utility improvement (%)",
            title=title,
        )
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Alignment: adaptive stopping against three fixed-policy baselines")
    base = FIGURES / "alignment/alignment_revised_baselines"
    fig.savefig(base.with_suffix(".png"), dpi=220)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def plot_coding(summary: list[dict]) -> None:
    metrics = (
        ("relative_improvement_vs_train_fixed_pct", "A. Train-selected Fixed-$N$"),
        ("relative_improvement_vs_oracle_fixed_n_pct", "B. Test-oracle Fixed-$N$"),
        (
            "relative_improvement_vs_train_fixed_budget_pct",
            "C. Train-selected fixed character budget",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        x = np.asarray([float(row["divisor"]) for row in summary])
        mean = np.asarray([float(row[f"{metric}_mean"]) for row in summary])
        low = np.asarray([float(row[f"{metric}_ci_low"]) for row in summary])
        high = np.asarray([float(row[f"{metric}_ci_high"]) for row in summary])
        axis.plot(x, mean, color="#C45A00", marker="o", linewidth=2.2)
        axis.fill_between(x, low, high, color="#C45A00", alpha=0.16)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set(
            xlabel="Character-cost divisor",
            ylabel="Relative utility improvement (%)",
            title=title,
        )
        axis.grid(alpha=0.25)
    fig.suptitle("Coding: adaptive stopping against three fixed-policy baselines")
    base = FIGURES / "coding/coding_revised_baselines"
    fig.savefig(base.with_suffix(".png"), dpi=220)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def plot_alignment_win_rate() -> None:
    summary = [
        row for row in read_csv(ALIGNMENT_HEAD_TO_HEAD)
        if row.get("variant") == ALIGNMENT_SIMPLE_VARIANT
    ]
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for generator in ALIGNMENT_GENERATORS:
        group = sorted(
            (row for row in summary if row["generator"] == generator),
            key=lambda row: float(row["divisor"]),
        )
        x = np.asarray([float(row["divisor"]) for row in group])
        mean = 100.0 * np.asarray([float(row["adaptive_score_mean"]) for row in group])
        low = 100.0 * np.asarray([float(row["adaptive_score_ci_low"]) for row in group])
        high = 100.0 * np.asarray([float(row["adaptive_score_ci_high"]) for row in group])
        color = ALIGNMENT_COLORS[generator]
        axis.plot(x, mean, marker="o", linewidth=2.0, color=color,
                  label=ALIGNMENT_LABELS[generator])
        axis.fill_between(x, low, high, color=color, alpha=0.12)
    axis.axhline(50.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xscale("log")
    axis.set(
        xlabel="Character-cost divisor",
        ylabel="Adaptive head-to-head win rate (%)",
        title="Alignment: independent responses at matched expected characters",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    base = FIGURES / "alignment/alignment_equal_cost_win_rate"
    fig.savefig(base.with_suffix(".png"), dpi=220)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def plot_alignment_target(summary: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
    for generator in ALIGNMENT_GENERATORS:
        group = sorted(
            (row for row in summary if row["generator"] == generator),
            key=lambda row: float(row["target_quality"]),
        )
        x = np.asarray([float(row["target_quality"]) for row in group])
        color = ALIGNMENT_COLORS[generator]
        axes[0].plot(
            x,
            [float(row["character_saving_vs_matched_fixed_pct_mean"])
             for row in group],
            marker="o", linewidth=2.0, color=color,
            label=ALIGNMENT_LABELS[generator],
        )
        axes[0].fill_between(
            x,
            [float(row["character_saving_vs_matched_fixed_pct_ci_low"])
             for row in group],
            [float(row["character_saving_vs_matched_fixed_pct_ci_high"])
             for row in group],
            color=color, alpha=0.12,
        )
        axes[1].plot(
            x,
            [float(row["adaptive_test_quality_mean"]) for row in group],
            marker="o", linewidth=2.0, color=color,
        )
        axes[1].fill_between(
            x,
            [float(row["adaptive_test_quality_ci_low"]) for row in group],
            [float(row["adaptive_test_quality_ci_high"]) for row in group],
            color=color, alpha=0.12,
        )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set(
        xlabel="Target quality",
        ylabel="Adaptive character saving at equal quality (%)",
        title="A. Character saving at equal quality",
    )
    axes[1].plot(
        ALIGNMENT_TARGETS, ALIGNMENT_TARGETS,
        color="black", linestyle="--", linewidth=1.0,
    )
    axes[1].set(
        xlabel="Requested target quality",
        ylabel="Adaptive held-out quality",
        title="B. Calibration to the requested target",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Alignment target quality")
    base = FIGURES / "alignment/alignment_target_quality_train_selected"
    fig.savefig(base.with_suffix(".png"), dpi=220)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def plot_coding_equal_budget(summary: list[dict]) -> None:
    summary = sorted(summary, key=lambda row: float(row["divisor"]))
    x = np.asarray([float(row["divisor"]) for row in summary])
    mean = np.asarray([float(row["accuracy_gap_pp_mean"]) for row in summary])
    low = np.asarray([float(row["accuracy_gap_pp_ci_low"]) for row in summary])
    high = np.asarray([float(row["accuracy_gap_pp_ci_high"]) for row in summary])
    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    axis.plot(x, mean, color="#C45A00", marker="o", linewidth=2.2)
    axis.fill_between(x, low, high, color="#C45A00", alpha=0.16)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set(
        xlabel="Character-cost divisor",
        ylabel="Adaptive accuracy improvement (percentage points)",
        title="Coding: accuracy at exactly matched expected characters",
    )
    axis.grid(alpha=0.25)
    base = FIGURES / "coding/coding_equal_budget_accuracy_revised"
    fig.savefig(base.with_suffix(".png"), dpi=220)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def plot_coding_target() -> None:
    summary = sorted(
        (row for row in read_csv(CODING_TARGET)
         if row["method"] == CODING_METHOD_NAME),
        key=lambda row: float(row["target_accuracy"]),
    )
    x = np.asarray([float(row["target_accuracy"]) for row in summary])
    fig, axes = plt.subplots(1, 4, figsize=(18.6, 4.2), constrained_layout=True)
    for axis, prefix, title in (
        (axes[0], "train_fixed", "A. Saving vs train-selected Fixed-$N$"),
        (axes[1], "equal_quality_fixed",
         "B. Saving at exactly matched accuracy"),
    ):
        axis.plot(
            x,
            [float(row[f"aggregate_saving_vs_{prefix}_pct"]) for row in summary],
            color="#C45A00", marker="o", linewidth=2.2,
        )
        axis.fill_between(
            x,
            [float(row[f"aggregate_saving_vs_{prefix}_ci_low"])
             for row in summary],
            [float(row[f"aggregate_saving_vs_{prefix}_ci_high"])
             for row in summary],
            color="#C45A00", alpha=0.16,
        )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set(
            xlabel="Training target accuracy",
            ylabel="Adaptive character saving (%)",
            title=title,
        )
    for axis, prefix, title in (
        (axes[2], "adaptive_accuracy", "C. Adaptive test calibration"),
        (axes[3], "train_fixed_test_accuracy",
         "D. Train-selected Fixed-$N$ calibration"),
    ):
        axis.plot(
            x, [float(row[f"{prefix}_mean"]) for row in summary],
            color="#C45A00" if prefix == "adaptive_accuracy" else "#555555",
            marker="o" if prefix == "adaptive_accuracy" else "s", linewidth=2.2,
        )
        axis.fill_between(
            x,
            [float(row[f"{prefix}_ci_low"]) for row in summary],
            [float(row[f"{prefix}_ci_high"]) for row in summary],
            color="#C45A00" if prefix == "adaptive_accuracy" else "#555555",
            alpha=0.14,
        )
        axis.plot(x, x, color="black", linestyle="--", linewidth=1.0)
        axis.set(xlabel="Training target accuracy", ylabel="Test accuracy", title=title)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(
        "Coding target quality: transfer and equal-accuracy oracle diagnostics"
    )
    base = FIGURES / "coding/coding_target_quality_train_selected"
    fig.savefig(base.with_suffix(".png"), dpi=220)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (FIGURES / "alignment").mkdir(parents=True, exist_ok=True)
    (FIGURES / "coding").mkdir(parents=True, exist_ok=True)
    alignment_rows, alignment_summary = build_alignment()
    alignment_target_rows, alignment_target_summary = build_alignment_target()
    coding_rows, coding_summary, coding_equal_rows, coding_equal_summary = (
        build_coding()
    )
    write_csv(OUTPUT / "alignment_benchmark_splits.csv", alignment_rows)
    write_csv(OUTPUT / "alignment_benchmark_summary.csv", alignment_summary)
    write_csv(OUTPUT / "alignment_target_splits.csv", alignment_target_rows)
    write_csv(OUTPUT / "alignment_target_summary.csv", alignment_target_summary)
    write_csv(OUTPUT / "coding_benchmark_splits.csv", coding_rows)
    write_csv(OUTPUT / "coding_benchmark_summary.csv", coding_summary)
    write_csv(OUTPUT / "coding_equal_budget_splits.csv", coding_equal_rows)
    write_csv(OUTPUT / "coding_equal_budget_summary.csv", coding_equal_summary)
    plot_alignment(alignment_summary)
    plot_alignment_win_rate()
    plot_alignment_target(alignment_target_summary)
    plot_coding(coding_summary)
    plot_coding_equal_budget(coding_equal_summary)
    plot_coding_target()
    (OUTPUT / "METHOD.json").write_text(json.dumps({
        "purpose": "streamlined manuscript baseline comparison",
        "adaptive_results": (
            "utility and target means read unchanged from the held-out "
            "confirmations of the selected training-free alignment rule and "
            "single-global-tail coding rule"
        ),
        "implementable_fixed_n": (
            "choose the utility-maximizing integer N on the outer training half; "
            "freeze N and evaluate on the test half"
        ),
        "oracle_fixed_n": (
            "choose the utility-maximizing integer N using the test half itself"
        ),
        "alignment_oracle_character_budget": (
            "use each test prompt's full-pool mean response length l_x; for each "
            "global budget R open clip(floor(R/l_x),1,H) responses; choose R on "
            "the test half from the exact union of allocation breakpoints"
        ),
        "alignment_target_quality": (
            "on the training half choose one adaptive divisor and one integer N; "
            "among policies attaining the target choose minimum mean characters; "
            "freeze both and evaluate on test; additionally report a post-hoc "
            "minimum-character randomization over at most two N values whose "
            "expected test quality exactly equals the adaptive policy's expected "
            "test quality"
        ),
        "alignment_win_rate": (
            "independent fixed-policy responses; adjacent-N randomization exactly "
            "matches the adaptive policy's expected test characters"
        ),
        "coding_equal_budget": (
            "frozen uncapped utility selector compared with independent Fixed-N "
            "at exactly matched expected test characters"
        ),
        "coding_train_selected_fixed_character_budget": (
            "choose one cumulative-character threshold R on the outer training "
            "half for each utility divisor; always open once, continue while "
            "observed cumulative characters are at most R, freeze R, and "
            "evaluate on test with exact realized character cost"
        ),
        "coding_target_quality": (
            "adaptive reservation divisor selected by the frozen development "
            "inverse profile; integer N selected to attain target accuracy on "
            "the outer training half; both frozen before test evaluation; also "
            "report a test-oracle common Fixed-N randomization at exactly the "
            "adaptive aggregate test accuracy"
        ),
        "reported_cost": "exact cumulative characters actually opened",
        "alignment_replay": {
            "source": str(ALIGNMENT_ROWS.relative_to(REPO)),
            "method": str(ALIGNMENT_METHOD.relative_to(REPO)),
        },
        "coding_replay": {
            "source": str(CODING_ROWS.relative_to(REPO)),
            "method": str(CODING_METHOD.relative_to(REPO)),
        },
        "uncertainty": "pointwise 95% Student-t intervals across ten splits",
    }, indent=2) + "\n")
    print(json.dumps({
        "alignment_split_rows": len(alignment_rows),
        "alignment_summary_rows": len(alignment_summary),
        "alignment_target_split_rows": len(alignment_target_rows),
        "alignment_target_summary_rows": len(alignment_target_summary),
        "coding_split_rows": len(coding_rows),
        "coding_summary_rows": len(coding_summary),
        "coding_equal_budget_split_rows": len(coding_equal_rows),
        "coding_equal_budget_summary_rows": len(coding_equal_summary),
        "output": str(OUTPUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
