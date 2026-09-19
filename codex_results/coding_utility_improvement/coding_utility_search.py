"""Isolated search for stronger coding utility policies.

This runner reuses the retained leakage-free coding evaluator and compares the
shifted-exponential part of the original capped/no-decay probability-space grid
with an exponential-only uncapped grid carrying the smooth expected-improvement
decay used by the target-quality experiment. Gaussian policies and cross-family
model selection are deliberately excluded.
All policy selection is based on cross-fitted outer-training trajectories,
optionally blended with a profile frozen from designated development splits.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
RETAINED_CODE = HERE.parent / "code" / "coding_scripts"
if str(RETAINED_CODE) not in sys.path:
    sys.path.insert(0, str(RETAINED_CODE))

from coding_ucb_three_objectives import (  # noqa: E402
    PolicyConfig,
    bounded_calibrated_reward_space_configs,
    crossfit_evaluate,
    evaluate,
    fit_config_calibrations,
    fit_prior,
    fit_reward_space_isotonic,
    fixed_curves,
    fixed_trials,
    load_problems,
    make_permutations,
    split_problems,
    transform_reward_space,
    uncapped_tail_decay_configs,
)


DEFAULT_DIVISORS = tuple(float(value) for value in range(100_000, 1_000_001, 100_000))


def policy_grids(names: tuple[str, ...]):
    """Return deterministic grid records and their combined configuration list."""
    builders = {
        "exp_tail_current": lambda: [
            config for config in bounded_calibrated_reward_space_configs()
            if config.family == "shifted_exponential"
        ],
        "exp_tail_uncapped_decay": lambda: [
            config for config in uncapped_tail_decay_configs(
                tail_decays=(0.0, 0.25, 0.5, 1.0)
            ) if config.family == "shifted_exponential"
        ],
    }
    records = []
    for grid in names:
        if grid not in builders:
            raise ValueError(f"unknown policy grid: {grid}")
        for grid_config_id, config in enumerate(builders[grid]()):
            records.append({
                "grid": grid,
                "grid_config_id": grid_config_id,
                "config": config,
            })
    for config_id, record in enumerate(records):
        record["config_id"] = config_id
    return records, [record["config"] for record in records]


def load_profile(path: Path | None):
    if path is None:
        return None
    profile = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["grid"], int(row["grid_config_id"]), float(row["divisor"]))
            if key in profile:
                raise ValueError(f"duplicate development-profile key: {key}")
            profile[key] = {
                "utility_mean": float(row["test_utility_mean"]),
                "utility_std": float(row["test_utility_std"]),
                "splits": int(row["splits"]),
            }
    return profile


def selection_score(train_utility, record, divisor, profile, profile_weight, risk_penalty):
    if profile is None:
        return float(train_utility), None
    key = (record["grid"], record["grid_config_id"], float(divisor))
    if key not in profile:
        raise ValueError(f"development profile is missing {key}")
    item = profile[key]
    robust_profile = item["utility_mean"] - risk_penalty * item["utility_std"]
    score = (1.0 - profile_weight) * float(train_utility) + profile_weight * robust_profile
    return score, robust_profile


def mean_interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean, 0.0
    std = float(values.std(ddof=1))
    half = float(student_t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
    return mean, mean - half, mean + half, std


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_selected(rows):
    output = []
    groups = sorted({(row["grid"], float(row["divisor"])) for row in rows})
    for grid, divisor in groups:
        subset = [row for row in rows if row["grid"] == grid and float(row["divisor"]) == divisor]
        result = {"grid": grid, "divisor": divisor, "splits": len(subset)}
        for metric in (
            "adaptive_utility", "fixed_utility", "utility_gap",
            "relative_utility_gain_pct", "adaptive_accuracy", "adaptive_chars",
            "adaptive_opens",
        ):
            mean, low, high, std = mean_interval([row[metric] for row in subset])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
            result[f"{metric}_std"] = std
        result["positive_gap_splits"] = sum(float(row["utility_gap"]) > 0.0 for row in subset)
        output.append(result)
    return output


def aggregate_profile(rows):
    output = []
    keys = sorted({
        (row["grid"], int(row["grid_config_id"]), float(row["divisor"]))
        for row in rows
    })
    for grid, grid_config_id, divisor in keys:
        subset = [
            row for row in rows
            if row["grid"] == grid
            and int(row["grid_config_id"]) == grid_config_id
            and float(row["divisor"]) == divisor
        ]
        first = subset[0]
        result = {
            "grid": grid,
            "grid_config_id": grid_config_id,
            "divisor": divisor,
            "splits": len(subset),
        }
        for field in asdict(first["_config"]):
            result[field] = getattr(first["_config"], field)
        for metric in (
            "train_utility", "test_utility", "test_accuracy", "test_chars", "test_opens"
        ):
            mean, low, high, std = mean_interval([row[metric] for row in subset])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
            result[f"{metric}_std"] = std
        output.append(result)
    return output


def run_split(split, problems, records, configs, divisors, args, profile):
    raw_train, raw_test = split_problems(problems, args.seed + split)
    reward_calibration = fit_reward_space_isotonic(raw_train)
    train = transform_reward_space(raw_train, reward_calibration)
    test = transform_reward_space(raw_test, reward_calibration)

    _, _, fixed_ns = fixed_curves(train, divisors)
    train_permutations = make_permutations(
        train, args.train_permutations, args.seed + 101_003 * split + 17
    )
    test_permutations = make_permutations(
        test, args.test_permutations, args.seed + 101_003 * split + 31
    )

    train_values = crossfit_evaluate(
        train, train_permutations, configs, divisors, fixed_ns,
        args.delta, args.workers, args.seed + 17_003 * split + 71,
        folds=args.folds,
    )
    train_accuracy = train_values[..., 0].mean(axis=2)
    train_chars = train_values[..., 1].mean(axis=2)
    train_utility = train_accuracy - train_chars / np.asarray(divisors)[None, :]

    grid_names = tuple(dict.fromkeys(record["grid"] for record in records))
    selections = {}
    tuning_rows = []
    for divisor_id, divisor in enumerate(divisors):
        for grid in grid_names:
            eligible = [record for record in records if record["grid"] == grid]
            scored = []
            for record in eligible:
                config_id = record["config_id"]
                score, robust_profile = selection_score(
                    train_utility[config_id, divisor_id], record, divisor,
                    profile, args.profile_weight, args.profile_risk_penalty,
                )
                scored.append((score, -record["grid_config_id"], record, robust_profile))
            _, _, selected, robust_profile = max(scored, key=lambda item: item[:2])
            selections[(grid, divisor)] = selected
            config_id = selected["config_id"]
            tuning_rows.append({
                "split": split,
                "grid": grid,
                "divisor": divisor,
                "config_id": config_id,
                "grid_config_id": selected["grid_config_id"],
                **asdict(selected["config"]),
                "train_accuracy": float(train_accuracy[config_id, divisor_id]),
                "train_chars": float(train_chars[config_id, divisor_id]),
                "train_utility": float(train_utility[config_id, divisor_id]),
                "profile_robust_utility": robust_profile,
                "selection_score": selection_score(
                    train_utility[config_id, divisor_id], selected, divisor,
                    profile, args.profile_weight, args.profile_risk_penalty,
                )[0],
                "fixed_n": fixed_ns[divisor],
            })

    evaluate_ids = (
        list(range(len(configs))) if args.diagnostic_all_configs else
        sorted({record["config_id"] for record in selections.values()})
    )
    evaluate_configs = [configs[index] for index in evaluate_ids]
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(evaluate_ids)}
    calibrations = fit_config_calibrations(train, evaluate_configs)
    prior = fit_prior(train)
    test_values = evaluate(
        test, test_permutations, evaluate_configs, calibrations, prior,
        divisors, fixed_ns, args.delta, args.workers,
    )
    fixed_accuracy, fixed_chars, _ = fixed_trials(test, test_permutations)

    selected_rows = []
    for divisor_id, divisor in enumerate(divisors):
        fixed_n = fixed_ns[divisor]
        fixed_trials_utility = (
            fixed_accuracy[:, fixed_n - 1] - fixed_chars[:, fixed_n - 1] / divisor
        )
        fixed_utility = float(fixed_trials_utility.mean())
        for grid in grid_names:
            selected = selections[(grid, divisor)]
            config_id = selected["config_id"]
            values = test_values[global_to_local[config_id], divisor_id]
            adaptive_trials = values[:, 0] - values[:, 1] / divisor
            adaptive_utility = float(adaptive_trials.mean())
            selected_rows.append({
                "split": split,
                "grid": grid,
                "divisor": divisor,
                "config_id": config_id,
                "grid_config_id": selected["grid_config_id"],
                **asdict(selected["config"]),
                "adaptive_utility": adaptive_utility,
                "fixed_utility": fixed_utility,
                "utility_gap": adaptive_utility - fixed_utility,
                "relative_utility_gain_pct": (
                    100.0 * (adaptive_utility - fixed_utility) / fixed_utility
                ),
                "adaptive_accuracy": float(values[:, 0].mean()),
                "adaptive_chars": float(values[:, 1].mean()),
                "adaptive_opens": float(values[:, 2].mean()),
                "fixed_n": fixed_n,
            })

    diagnostic_rows = []
    if args.diagnostic_all_configs:
        for record in records:
            config_id = record["config_id"]
            local_id = global_to_local[config_id]
            config = record["config"]
            for divisor_id, divisor in enumerate(divisors):
                values = test_values[local_id, divisor_id]
                test_utility = values[:, 0] - values[:, 1] / divisor
                diagnostic_rows.append({
                    "split": split,
                    "grid": record["grid"],
                    "grid_config_id": record["grid_config_id"],
                    "divisor": divisor,
                    "_config": config,
                    "train_utility": float(train_utility[config_id, divisor_id]),
                    "test_utility": float(test_utility.mean()),
                    "test_accuracy": float(values[:, 0].mean()),
                    "test_chars": float(values[:, 1].mean()),
                    "test_opens": float(values[:, 2].mean()),
                })
    return selected_rows, tuning_rows, diagnostic_rows


def selection_frequency_rows(tuning_rows):
    counts = Counter(
        (row["grid"], float(row["divisor"]), int(row["grid_config_id"]))
        for row in tuning_rows
    )
    output = []
    for (grid, divisor, grid_config_id), count in sorted(counts.items()):
        row = next(
            item for item in tuning_rows
            if item["grid"] == grid
            and float(item["divisor"]) == divisor
            and int(item["grid_config_id"]) == grid_config_id
        )
        output.append({
            "grid": grid,
            "divisor": divisor,
            "grid_config_id": grid_config_id,
            "selected_splits": count,
            **{field: row[field] for field in asdict(PolicyConfig(
                "gaussian", "bounded_identity", 0.0, 0.0, 0.0, None
            ))},
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path("algorithm/bestofn_coding/data.jsonl"))
    parser.add_argument("--char-cache", type=Path, default=Path(
        "algorithm/bestofn_coding/practical_algorithm/coding_char_counts_83.npz"
    ))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--grids", nargs="+",
        choices=("exp_tail_current", "exp_tail_uncapped_decay"),
        default=("exp_tail_current", "exp_tail_uncapped_decay"),
    )
    parser.add_argument("--divisors", type=float, nargs="+", default=DEFAULT_DIVISORS)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--split-start", type=int, default=60)
    parser.add_argument("--train-permutations", type=int, default=4)
    parser.add_argument("--test-permutations", type=int, default=4)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--development-profile", type=Path)
    parser.add_argument("--profile-weight", type=float, default=0.9)
    parser.add_argument("--profile-risk-penalty", type=float, default=0.0)
    parser.add_argument("--diagnostic-all-configs", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.profile_weight <= 1.0:
        parser.error("--profile-weight must lie in [0, 1]")
    args.grids = tuple(args.grids)
    args.divisors = tuple(float(value) for value in args.divisors)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, configs = policy_grids(args.grids)
    profile = load_profile(args.development_profile)
    problems = load_problems(args.data, args.char_cache)
    selected_rows, tuning_rows, diagnostic_rows = [], [], []
    for position, split in enumerate(range(args.split_start, args.split_start + args.splits), 1):
        selected, tuning, diagnostic = run_split(
            split, problems, records, configs, args.divisors, args, profile
        )
        selected_rows.extend(selected)
        tuning_rows.extend(tuning)
        diagnostic_rows.extend(diagnostic)
        write_csv(args.output_dir / "selected_splits.partial.csv", selected_rows)
        print(f"[utility-improvement] split {position}/{args.splits} (id={split}) complete",
              flush=True)

    summary = aggregate_selected(selected_rows)
    write_csv(args.output_dir / "selected_splits.csv", selected_rows)
    write_csv(args.output_dir / "utility_summary.csv", summary)
    write_csv(args.output_dir / "tuning_selections.csv", tuning_rows)
    write_csv(args.output_dir / "selection_frequencies.csv", selection_frequency_rows(tuning_rows))
    if diagnostic_rows:
        serializable_diagnostics = [
            {key: value for key, value in row.items() if key != "_config"}
            | asdict(row["_config"])
            for row in diagnostic_rows
        ]
        profile_rows = aggregate_profile(diagnostic_rows)
        write_csv(args.output_dir / "all_configs_splits.csv", serializable_diagnostics)
        write_csv(args.output_dir / "development_profile.csv", profile_rows)

    method = {
        "purpose": "isolated coding utility improvement search",
        "retained_code": str(RETAINED_CODE / "coding_ucb_three_objectives.py"),
        "grids": args.grids,
        "grid_sizes": dict(Counter(record["grid"] for record in records)),
        "reward_space": "outer-train-only isotonic P(correct | raw reward)",
        "distribution": "conditional shifted-exponential upper tail only",
        "distribution_support": "shifted exponential truncated to [0, 1]",
        "selection": (
            "cross-fitted train utility" if profile is None else
            "blend of cross-fitted train utility and frozen development-profile utility"
        ),
        "profile": str(args.development_profile) if args.development_profile else None,
        "profile_weight": args.profile_weight,
        "profile_risk_penalty_standard_deviations": args.profile_risk_penalty,
        "splits": args.splits,
        "split_start": args.split_start,
        "train_permutations": args.train_permutations,
        "test_permutations": args.test_permutations,
        "folds": args.folds,
        "workers": args.workers,
        "divisors": args.divisors,
        "diagnostic_all_configs": args.diagnostic_all_configs,
    }
    with (args.output_dir / "METHOD.json").open("w") as handle:
        json.dump(method, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
