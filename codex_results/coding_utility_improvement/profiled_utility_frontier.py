"""Select a utility-optimal UCB-Pandora policy from a frozen frontier.

The retained utility runner evaluates the Pandora reservation rule only at the
reporting divisor.  That unnecessarily couples the economic reporting price
to a possibly miscalibrated stopping threshold.  This runner keeps them
separate:

* ``utility_divisor`` is the actual character price used for evaluation;
* ``reservation_divisor`` is a train/development-selected calibration of the
  Pandora continue/stop comparison.

Every candidate remains an uncapped, shifted-exponential UCB-Pandora policy.
The candidate shortlist is obtained from the frozen target-quality development
profile (splits 55--59).  A disjoint validation stage can select one candidate
per utility divisor, after which a final stage evaluates only the frozen
selection on untouched outer splits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
RETAINED_CODE = HERE.parent / "code" / "coding_scripts"
if str(RETAINED_CODE) not in sys.path:
    sys.path.insert(0, str(RETAINED_CODE))

from coding_target_quality_distribution_calibrated import (  # noqa: E402
    candidate_configs as target_candidate_configs,
)
from coding_ucb_three_objectives import (  # noqa: E402
    Calibration,
    _ei_curve,
    _prefix_success,
    fit_prior,
    fit_reward_space_isotonic,
    fixed_curves,
    fixed_trials,
    load_problems,
    make_permutations,
    pandora_stop_from_curve,
    split_problems,
    transform_reward_space,
)


DEFAULT_UTILITY_DIVISORS = tuple(
    float(value) for value in range(100_000, 1_000_001, 100_000)
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_profile_candidates(profile_path, utility_divisors, top_k):
    """Shortlist profile policies without consulting validation/final data."""
    configs = target_candidate_configs(True, True)
    with profile_path.open(newline="") as handle:
        profile_rows = list(csv.DictReader(handle))
    expected = len(configs) * len({float(row["divisor"]) for row in profile_rows})
    if len(profile_rows) != expected:
        raise ValueError(
            f"profile has {len(profile_rows)} rows; expected a complete {expected}-row grid"
        )

    candidates = []
    for utility_divisor in utility_divisors:
        eligible = [
            row for row in profile_rows
            if row["family"] == "shifted_exponential"
            and float(row["confidence_scale"]) > 0.0
            and row["calibration"] == "bounded_identity"
            and row["cap_factor"] == ""
        ]
        for rank, row in enumerate(sorted(
            eligible,
            key=lambda item: (
                float(item["profile_accuracy"])
                - float(item["profile_chars_geomean"]) / utility_divisor,
                -int(item["config_id"]),
                -int(item["divisor_id"]),
            ),
            reverse=True,
        )[:top_k], start=1):
            config_id = int(row["config_id"])
            config = configs[config_id]
            if config.confidence_scale <= 0.0:
                raise AssertionError("shortlist must contain genuine UCB policies")
            if config.family != "shifted_exponential" or config.cap_factor is not None:
                raise AssertionError("shortlist must contain uncapped exponential-tail policies")
            candidates.append({
                "utility_divisor": float(utility_divisor),
                "reservation_divisor": float(row["divisor"]),
                "reservation_divisor_id": int(row["divisor_id"]),
                "config_id": config_id,
                "config": config,
                "profile_rank": rank,
                "profile_accuracy": float(row["profile_accuracy"]),
                "profile_chars_geomean": float(row["profile_chars_geomean"]),
                "profile_utility": (
                    float(row["profile_accuracy"])
                    - float(row["profile_chars_geomean"]) / utility_divisor
                ),
            })
    return candidates


def load_frozen_candidates(path):
    configs = target_candidate_configs(True, True)
    candidates = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            config_id = int(row["config_id"])
            config = configs[config_id]
            if config.confidence_scale <= 0.0:
                raise ValueError("frozen selection contains a non-UCB policy")
            candidates.append({
                "utility_divisor": float(row["utility_divisor"]),
                "reservation_divisor": float(row["reservation_divisor"]),
                "reservation_divisor_id": int(row["reservation_divisor_id"]),
                "config_id": config_id,
                "config": config,
                "profile_rank": int(row["profile_rank"]),
                "profile_accuracy": float(row["profile_accuracy"]),
                "profile_chars_geomean": float(row["profile_chars_geomean"]),
                "profile_utility": float(row["profile_utility"]),
            })
    expected = sorted(DEFAULT_UTILITY_DIVISORS)
    actual = sorted(item["utility_divisor"] for item in candidates)
    if actual != expected:
        raise ValueError(
            "frozen selection must contain exactly one row for each default utility divisor"
        )
    return candidates


def _evaluate_pair_problem(task):
    (problem_id, rewards, correct, chars, permutations, candidates, prior, delta) = task
    result = np.empty((len(candidates), len(permutations), 4), dtype=np.float64)
    calibration = Calibration("bounded_identity")
    core_keys = [(
        item["config"].family,
        item["config"].confidence_scale,
        item["config"].reward_prior_strength,
        item["config"].tail_quantile,
    ) for item in candidates]
    for permutation_id, permutation in enumerate(permutations):
        ordered_rewards = rewards[permutation]
        ordered_correct = correct[permutation]
        ordered_chars = chars[permutation]
        cumulative_chars = np.cumsum(ordered_chars, dtype=np.float64)
        best_rewards, success = _prefix_success(ordered_rewards, ordered_correct)
        curves = {}
        for item, key in zip(candidates, core_keys):
            if key not in curves:
                curves[key] = _ei_curve(
                    ordered_rewards, best_rewards, calibration, prior,
                    item["config"], delta,
                )
        for candidate_id, (item, key) in enumerate(zip(candidates, core_keys)):
            opened = pandora_stop_from_curve(
                curves[key], cumulative_chars, item["reservation_divisor"],
                prior.mean_chars, item["config"], fixed_n=1, min_open=3,
            )
            result[candidate_id, permutation_id] = (
                success[opened - 1], cumulative_chars[opened - 1], opened,
                best_rewards[opened - 1],
            )
    return problem_id, result


def evaluate_pairs(problems, permutations, candidates, prior, delta, workers):
    tasks = [(
        problem_id, *problems[problem_id], permutations[problem_id],
        candidates, prior, delta,
    ) for problem_id in sorted(problems)]
    if workers == 1:
        pieces = [_evaluate_pair_problem(task)[1] for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pieces = [
                values for _, values in pool.map(
                    _evaluate_pair_problem, tasks, chunksize=1
                )
            ]
    return np.concatenate(pieces, axis=1)


def run_split(split, problems, candidates, args):
    raw_train, raw_test = split_problems(problems, args.seed + split)
    reward_calibration = fit_reward_space_isotonic(raw_train)
    train = transform_reward_space(raw_train, reward_calibration)
    test = transform_reward_space(raw_test, reward_calibration)
    utility_divisors = tuple(sorted({item["utility_divisor"] for item in candidates}))
    _, _, fixed_ns = fixed_curves(train, utility_divisors)
    test_permutations = make_permutations(
        test, args.test_permutations, args.seed + 101_003 * split + 31
    )
    prior = fit_prior(train)
    values = evaluate_pairs(
        test, test_permutations, candidates, prior, args.delta, args.workers
    )
    fixed_accuracy, fixed_chars, _ = fixed_trials(test, test_permutations)

    fixed = {}
    for divisor in utility_divisors:
        n = fixed_ns[divisor]
        trial_utility = fixed_accuracy[:, n - 1] - fixed_chars[:, n - 1] / divisor
        fixed[divisor] = {
            "utility": float(trial_utility.mean()),
            "accuracy": float(fixed_accuracy[:, n - 1].mean()),
            "chars": float(fixed_chars[:, n - 1].mean()),
            "n": n,
        }

    rows = []
    for candidate_id, item in enumerate(candidates):
        divisor = item["utility_divisor"]
        candidate_values = values[candidate_id]
        trial_utility = candidate_values[:, 0] - candidate_values[:, 1] / divisor
        adaptive_utility = float(trial_utility.mean())
        baseline = fixed[divisor]
        rows.append({
            "split": split,
            "utility_divisor": divisor,
            "reservation_divisor": item["reservation_divisor"],
            "reservation_divisor_id": item["reservation_divisor_id"],
            "config_id": item["config_id"],
            **asdict(item["config"]),
            "profile_rank": item["profile_rank"],
            "profile_accuracy": item["profile_accuracy"],
            "profile_chars_geomean": item["profile_chars_geomean"],
            "profile_utility": item["profile_utility"],
            "adaptive_utility": adaptive_utility,
            "fixed_utility": baseline["utility"],
            "utility_gap": adaptive_utility - baseline["utility"],
            "relative_utility_gain_pct": (
                100.0 * (adaptive_utility - baseline["utility"])
                / baseline["utility"]
            ),
            "adaptive_accuracy": float(candidate_values[:, 0].mean()),
            "adaptive_chars": float(candidate_values[:, 1].mean()),
            "adaptive_opens": float(candidate_values[:, 2].mean()),
            "fixed_accuracy": baseline["accuracy"],
            "fixed_chars": baseline["chars"],
            "fixed_n": baseline["n"],
        })
    return rows


def aggregate_candidates(rows):
    keys = sorted({
        (float(row["utility_divisor"]), int(row["config_id"]),
         float(row["reservation_divisor"]))
        for row in rows
    })
    output = []
    for utility_divisor, config_id, reservation_divisor in keys:
        subset = [
            row for row in rows
            if float(row["utility_divisor"]) == utility_divisor
            and int(row["config_id"]) == config_id
            and float(row["reservation_divisor"]) == reservation_divisor
        ]
        first = subset[0]
        result = {
            key: first[key] for key in (
                "utility_divisor", "reservation_divisor",
                "reservation_divisor_id", "config_id", "family", "calibration",
                "confidence_scale", "reward_prior_strength", "cost_prior_strength",
                "cap_factor", "tail_quantile", "tail_decay", "profile_rank",
                "profile_accuracy", "profile_chars_geomean", "profile_utility",
            )
        }
        result["splits"] = len(subset)
        for metric in (
            "adaptive_utility", "fixed_utility", "utility_gap",
            "relative_utility_gain_pct", "adaptive_accuracy", "adaptive_chars",
            "adaptive_opens", "fixed_accuracy", "fixed_chars", "fixed_n",
        ):
            mean, low, high, std = mean_interval([row[metric] for row in subset])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
            result[f"{metric}_std"] = std
        result["positive_gap_splits"] = sum(
            float(row["utility_gap"]) > 0.0 for row in subset
        )
        output.append(result)
    return output


def select_frozen(summary, risk_penalty, profile_weight=0.5):
    if not 0.0 <= profile_weight <= 1.0:
        raise ValueError("profile_weight must lie in [0, 1]")
    selected = []
    for divisor in sorted({float(row["utility_divisor"]) for row in summary}):
        eligible = [row for row in summary if float(row["utility_divisor"]) == divisor]
        winner = max(eligible, key=lambda row: (
            (1.0 - profile_weight) * float(row["adaptive_utility_mean"])
            + profile_weight * float(row["profile_utility"])
            - risk_penalty * float(row["adaptive_utility_std"]),
            -int(row["profile_rank"]),
            -int(row["config_id"]),
        ))
        selected.append({
            key: winner[key] for key in (
                "utility_divisor", "reservation_divisor",
                "reservation_divisor_id", "config_id", "family", "calibration",
                "confidence_scale", "reward_prior_strength", "cost_prior_strength",
                "cap_factor", "tail_quantile", "tail_decay", "profile_rank",
                "profile_accuracy", "profile_chars_geomean", "profile_utility",
                "adaptive_utility_mean", "adaptive_utility_std", "utility_gap_mean",
                "relative_utility_gain_pct_mean", "positive_gap_splits", "splits",
            )
        })
    return selected


def leave_one_split_out_meta_selection(rows):
    """Audit profile/validation selector weights without using final splits."""
    output = []
    profile_weights = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    risk_penalties = (0.0, 0.1, 0.25, 0.5)
    divisors = sorted({float(row["utility_divisor"]) for row in rows})
    split_ids = sorted({int(row["split"]) for row in rows})
    for profile_weight in profile_weights:
        for risk_penalty in risk_penalties:
            heldout_gaps = []
            heldout_relative_gains = []
            for divisor in divisors:
                subset = [
                    row for row in rows
                    if float(row["utility_divisor"]) == divisor
                ]
                keys = sorted({
                    (int(row["config_id"]), float(row["reservation_divisor"]))
                    for row in subset
                })
                table = {
                    (int(row["split"]), int(row["config_id"]),
                     float(row["reservation_divisor"])): row
                    for row in subset
                }
                for heldout_split in split_ids:
                    scored = []
                    for key in keys:
                        fit_utility = np.asarray([
                            float(table[(split, *key)]["adaptive_utility"])
                            for split in split_ids if split != heldout_split
                        ])
                        profile_utility = float(
                            table[(heldout_split, *key)]["profile_utility"]
                        )
                        score = (
                            (1.0 - profile_weight) * float(fit_utility.mean())
                            + profile_weight * profile_utility
                            - risk_penalty * (
                                float(fit_utility.std(ddof=1))
                                if len(fit_utility) > 1 else 0.0
                            )
                        )
                        scored.append((score, key))
                    selected_key = max(scored)[1]
                    heldout = table[(heldout_split, *selected_key)]
                    heldout_gaps.append(float(heldout["utility_gap"]))
                    heldout_relative_gains.append(
                        float(heldout["relative_utility_gain_pct"])
                    )
            output.append({
                "profile_weight": profile_weight,
                "validation_weight": 1.0 - profile_weight,
                "risk_penalty": risk_penalty,
                "heldout_cells": len(heldout_gaps),
                "mean_heldout_utility_gap": float(np.mean(heldout_gaps)),
                "mean_heldout_relative_gain_pct": float(
                    np.mean(heldout_relative_gains)
                ),
                "positive_heldout_cells": int(np.sum(np.asarray(heldout_gaps) > 0.0)),
            })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path("algorithm/bestofn_coding/data.jsonl"))
    parser.add_argument("--char-cache", type=Path, default=Path(
        "algorithm/bestofn_coding/practical_algorithm/coding_char_counts_83.npz"
    ))
    parser.add_argument("--profile", type=Path, default=Path(
        "codex_results/code/coding_scripts/coding_target_tail_decay_profile.csv"
    ))
    parser.add_argument("--frozen-selection", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--utility-divisors", type=float, nargs="+",
                        default=DEFAULT_UTILITY_DIVISORS)
    parser.add_argument("--profile-top-k", type=int, default=20)
    parser.add_argument("--selection-profile-weight", type=float, default=0.5)
    parser.add_argument("--selection-risk-penalty", type=float, default=0.1)
    parser.add_argument("--reuse-candidate-splits", type=Path)
    parser.add_argument("--split-start", type=int, default=65)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--test-permutations", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.utility_divisors = tuple(float(value) for value in args.utility_divisors)
    if not 0.0 <= args.selection_profile_weight <= 1.0:
        parser.error("--selection-profile-weight must lie in [0, 1]")
    if args.frozen_selection:
        if args.utility_divisors != DEFAULT_UTILITY_DIVISORS:
            parser.error("a frozen selection uses the default utility-divisor grid")
        candidates = load_frozen_candidates(args.frozen_selection)
        stage = "final_frozen_evaluation"
    else:
        candidates = load_profile_candidates(
            args.profile, args.utility_divisors, args.profile_top_k
        )
        stage = "disjoint_validation_selection"

    if args.reuse_candidate_splits:
        if args.frozen_selection:
            parser.error("--reuse-candidate-splits cannot be combined with --frozen-selection")
        with args.reuse_candidate_splits.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        stage = "selection_from_saved_disjoint_validation"
    else:
        problems = load_problems(args.data, args.char_cache)
        rows = []
        for position, split in enumerate(
            range(args.split_start, args.split_start + args.splits), 1
        ):
            rows.extend(run_split(split, problems, candidates, args))
            write_csv(args.output_dir / "candidate_splits.partial.csv", rows)
            print(
                f"[profiled-utility] split {position}/{args.splits} (id={split}) complete",
                flush=True,
            )

    summary = aggregate_candidates(rows)
    write_csv(args.output_dir / "candidate_splits.csv", rows)
    write_csv(args.output_dir / "candidate_summary.csv", summary)
    if args.frozen_selection:
        selected = summary
        write_csv(args.output_dir / "utility_summary.csv", summary)
    else:
        meta_selection_rows = leave_one_split_out_meta_selection(rows)
        write_csv(args.output_dir / "meta_selection_loo.csv", meta_selection_rows)
        selected = select_frozen(
            summary, args.selection_risk_penalty, args.selection_profile_weight
        )
        write_csv(args.output_dir / "frozen_selection.csv", selected)

    method = {
        "purpose": "maximize coding utility along the frozen UCB-Pandora frontier",
        "stage": stage,
        "algorithm": "uncapped shifted-exponential UCB-Pandora reservation stopping",
        "reward_space": "outer-train-only isotonic P(correct | raw reward)",
        "actual_cost": "exact cumulative output characters / utility_divisor",
        "reservation_calibration": (
            "Pandora expected improvement is compared with estimated next characters "
            "/ reservation_divisor"
        ),
        "blueprint_equivalent_gain": (
            "A_n = (reservation_divisor / utility_divisor) * "
            "(n / minimum_openings)^(-tail_decay), B_n = 0; hence the same rule "
            "compares A_n * UCB_expected_improvement with next_characters / "
            "utility_divisor"
        ),
        "profile": str(args.profile),
        "profile_sha256": _sha256(args.profile),
        "profile_development_splits": [55, 56, 57, 58, 59],
        "profile_top_k_per_utility_divisor": (
            None if args.frozen_selection else args.profile_top_k
        ),
        "frozen_selection": (
            str(args.frozen_selection) if args.frozen_selection else None
        ),
        "frozen_selection_sha256": (
            _sha256(args.frozen_selection) if args.frozen_selection else None
        ),
        "selection_risk_penalty_standard_deviations": args.selection_risk_penalty,
        "selection_profile_weight": args.selection_profile_weight,
        "selection_validation_weight": 1.0 - args.selection_profile_weight,
        "selection_weights_chosen_by_leave_one_split_out_validation": True,
        "meta_selection_audit": (
            str(args.frozen_selection.parent / "meta_selection_loo.csv")
            if args.frozen_selection else
            str(args.output_dir / "meta_selection_loo.csv")
        ),
        "reuse_candidate_splits": (
            str(args.reuse_candidate_splits) if args.reuse_candidate_splits else None
        ),
        "genuine_ucb_constraint": "confidence_scale > 0",
        "distribution_constraint": "shifted_exponential only, truncated to [0,1]",
        "cap_constraint": "no Fixed-N cap; only the absolute 512-box dataset horizon",
        "splits": args.splits,
        "split_start": args.split_start,
        "test_permutations": args.test_permutations,
        "workers": args.workers,
        "seed": args.seed,
        "delta": args.delta,
        "utility_divisors": args.utility_divisors,
        "candidate_count": len(candidates),
    }
    with (args.output_dir / "METHOD.json").open("w") as handle:
        json.dump(method, handle, indent=2)
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
