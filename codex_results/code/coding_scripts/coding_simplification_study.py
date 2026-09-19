"""Development/final evaluation for conceptually simpler coding policies.

The retained policy uses a different tail configuration and reservation
divisor for several economic divisors.  This study asks whether one frozen
training distribution and one common stopping formula can replace that table.
Every variant uses train-only isotonic reward calibration, exact realized
character cost, and no Fixed-N guard.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from coding_ucb_three_objectives import (
    Calibration,
    PolicyConfig,
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


UTILITY_DIVISORS = tuple(float(value) for value in range(100_000, 1_000_001, 100_000))


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    family: str
    confidence_scale: float
    reward_prior_strength: float
    cost_prior_strength: float
    tail_quantile: float | None
    tail_decay: float
    reservation_rule: str

    def config(self) -> PolicyConfig:
        return PolicyConfig(
            family=self.family,
            calibration="bounded_identity",
            confidence_scale=self.confidence_scale,
            reward_prior_strength=self.reward_prior_strength,
            cost_prior_strength=self.cost_prior_strength,
            cap_factor=None,
            tail_quantile=self.tail_quantile,
            tail_decay=self.tail_decay,
        )

    def reservation_divisor(self, utility_divisor: float) -> float:
        if self.reservation_rule == "equal":
            return utility_divisor
        if self.reservation_rule == "0.7x":
            return 0.7 * utility_divisor
        if self.reservation_rule == "0.85x":
            return 0.85 * utility_divisor
        if self.reservation_rule == "1.15x":
            return 1.15 * utility_divisor
        if self.reservation_rule == "1.4x":
            return 1.4 * utility_divisor
        if self.reservation_rule == "retained_piecewise":
            if utility_divisor <= 200_000:
                return 70_000.0
            if utility_divisor <= 600_000:
                return 400_000.0
            return 1_000_000.0
        raise ValueError(f"unknown reservation rule: {self.reservation_rule}")


VARIANTS = (
    Variant(
        "global_exp_piecewise",
        "one global exponential-tail configuration with retained reservation bins",
        "shifted_exponential", 0.8, math.inf, 10.0, 0.5, 1.0,
        "retained_piecewise",
    ),
    Variant(
        "global_exp_equal",
        "one global exponential-tail configuration and D_res = D",
        "shifted_exponential", 0.8, math.inf, 10.0, 0.5, 1.0, "equal",
    ),
    Variant(
        "global_exp_equal_running_cost",
        "global exponential tail, D_res = D, and prefix-mean character cost",
        "shifted_exponential", 0.8, math.inf, 0.0, 0.5, 1.0, "equal",
    ),
    Variant(
        "global_exp_07x_running_cost",
        "global exponential tail and one 0.7 reservation multiplier",
        "shifted_exponential", 0.8, math.inf, 0.0, 0.5, 1.0, "0.7x",
    ),
    Variant(
        "global_exp_085x_running_cost",
        "global exponential tail and one 0.85 reservation multiplier",
        "shifted_exponential", 0.8, math.inf, 0.0, 0.5, 1.0, "0.85x",
    ),
    Variant(
        "global_exp_115x_running_cost",
        "global exponential tail and one 1.15 reservation multiplier",
        "shifted_exponential", 0.8, math.inf, 0.0, 0.5, 1.0, "1.15x",
    ),
    Variant(
        "global_exp_14x_running_cost",
        "global exponential tail and one 1.4 reservation multiplier",
        "shifted_exponential", 0.8, math.inf, 0.0, 0.5, 1.0, "1.4x",
    ),
    Variant(
        "global_exp_equal_conf02",
        "global exponential tail with weaker confidence inflation",
        "shifted_exponential", 0.2, math.inf, 0.0, 0.5, 1.0, "equal",
    ),
    Variant(
        "global_exp_equal_conf04",
        "global exponential tail with confidence scale 0.4",
        "shifted_exponential", 0.4, math.inf, 0.0, 0.5, 1.0, "equal",
    ),
    Variant(
        "global_exp_equal_conf06",
        "global exponential tail with confidence scale 0.6",
        "shifted_exponential", 0.6, math.inf, 0.0, 0.5, 1.0, "equal",
    ),
    Variant(
        "global_exp_equal_conf10",
        "global exponential tail with confidence scale 1.0",
        "shifted_exponential", 1.0, math.inf, 0.0, 0.5, 1.0, "equal",
    ),
    Variant(
        "global_exp_equal_no_ucb",
        "global exponential-tail plug-in rule without confidence inflation",
        "shifted_exponential", 0.0, math.inf, 0.0, 0.5, 1.0, "equal",
    ),
    Variant(
        "global_exp_equal_q75",
        "global 75-percent exponential tail with D_res = D",
        "shifted_exponential", 0.2, math.inf, 0.0, 0.75, 1.0, "equal",
    ),
    Variant(
        "global_exp_equal_q75_conf08",
        "global 75-percent exponential tail with confidence scale 0.8",
        "shifted_exponential", 0.8, math.inf, 0.0, 0.75, 1.0, "equal",
    ),
    Variant(
        "global_exp_q75_conf08_07x",
        "global 75-percent exponential tail with D_res = 0.7 D",
        "shifted_exponential", 0.8, math.inf, 0.0, 0.75, 1.0, "0.7x",
    ),
    Variant(
        "global_exp_q75_conf08_085x",
        "global 75-percent exponential tail with D_res = 0.85 D",
        "shifted_exponential", 0.8, math.inf, 0.0, 0.75, 1.0, "0.85x",
    ),
    Variant(
        "global_exp_q75_conf08_115x",
        "global 75-percent exponential tail with D_res = 1.15 D",
        "shifted_exponential", 0.8, math.inf, 0.0, 0.75, 1.0, "1.15x",
    ),
    Variant(
        "global_exp_q75_conf08_14x",
        "global 75-percent exponential tail with D_res = 1.4 D",
        "shifted_exponential", 0.8, math.inf, 0.0, 0.75, 1.0, "1.4x",
    ),
    Variant(
        "global_gaussian_equal",
        "one global truncated Gaussian rule with D_res = D",
        "gaussian", 0.8, math.inf, 0.0, None, 1.0, "equal",
    ),
    Variant(
        "local_exp_equal",
        "train-calibrated but otherwise prefix-local exponential-tail rule",
        "shifted_exponential", 0.8, 0.0, 0.0, 0.5, 1.0, "equal",
    ),
    Variant(
        "global_exp_equal_decay05",
        "global exponential tail with square-root opportunity decay",
        "shifted_exponential", 0.8, math.inf, 0.0, 0.5, 0.5, "equal",
    ),
    Variant(
        "global_exp_equal_no_decay",
        "global exponential tail without opportunity decay",
        "shifted_exponential", 0.8, math.inf, 0.0, 0.5, 0.0, "equal",
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def load_reference_candidates(path: Path):
    output = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            output.append({
                "variant": "reference_profiled",
                "utility_divisor": float(row["utility_divisor"]),
                "reservation_divisor": float(row["reservation_divisor"]),
                "config": PolicyConfig(
                    family=row["family"],
                    calibration=row["calibration"],
                    confidence_scale=float(row["confidence_scale"]),
                    reward_prior_strength=float(row["reward_prior_strength"]),
                    cost_prior_strength=float(row["cost_prior_strength"]),
                    cap_factor=None,
                    tail_quantile=float(row["tail_quantile"]),
                    tail_decay=float(row["tail_decay"]),
                ),
            })
    if sorted(item["utility_divisor"] for item in output) != list(UTILITY_DIVISORS):
        raise ValueError("reference selection must contain one row per utility divisor")
    return output


def build_candidates(reference_path: Path, variant_names: tuple[str, ...]):
    candidates = load_reference_candidates(reference_path)
    for name in variant_names:
        variant = VARIANT_BY_NAME[name]
        for divisor in UTILITY_DIVISORS:
            candidates.append({
                "variant": name,
                "utility_divisor": divisor,
                "reservation_divisor": variant.reservation_divisor(divisor),
                "config": variant.config(),
            })
    return candidates


def _evaluate_problem(task):
    problem_id, rewards, correct, chars, permutations, candidates, prior, delta = task
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
                success[opened - 1],
                cumulative_chars[opened - 1],
                opened,
                best_rewards[opened - 1],
            )
    return problem_id, result


def evaluate_candidates(problems, permutations, candidates, prior, delta, workers):
    tasks = [(
        problem_id, *problems[problem_id], permutations[problem_id],
        candidates, prior, delta,
    ) for problem_id in sorted(problems)]
    if workers == 1:
        pieces = [_evaluate_problem(task)[1] for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pieces = [
                values for _, values in pool.map(
                    _evaluate_problem, tasks, chunksize=1
                )
            ]
    return np.concatenate(pieces, axis=1)


def run_split(split, problems, candidates, args):
    raw_train, raw_test = split_problems(problems, args.seed + split)
    isotonic = fit_reward_space_isotonic(raw_train)
    train = transform_reward_space(raw_train, isotonic)
    test = transform_reward_space(raw_test, isotonic)
    _, _, fixed_ns = fixed_curves(train, UTILITY_DIVISORS)
    permutations = make_permutations(
        test, args.test_permutations, args.seed + 101_003 * split + 31
    )
    prior = fit_prior(train)
    values = evaluate_candidates(
        test, permutations, candidates, prior, args.delta, args.workers
    )
    fixed_accuracy, fixed_chars, _ = fixed_trials(test, permutations)

    fixed = {}
    for divisor in UTILITY_DIVISORS:
        n = fixed_ns[divisor]
        trial_utility = fixed_accuracy[:, n - 1] - fixed_chars[:, n - 1] / divisor
        fixed[divisor] = {
            "n": n,
            "utility": float(np.mean(trial_utility)),
            "accuracy": float(np.mean(fixed_accuracy[:, n - 1])),
            "chars": float(np.mean(fixed_chars[:, n - 1])),
        }

    reference_utility = {}
    for candidate_id, item in enumerate(candidates):
        if item["variant"] != "reference_profiled":
            continue
        divisor = item["utility_divisor"]
        candidate_values = values[candidate_id]
        reference_utility[divisor] = (
            candidate_values[:, 0] - candidate_values[:, 1] / divisor
        )

    rows = []
    for candidate_id, item in enumerate(candidates):
        divisor = item["utility_divisor"]
        config = item["config"]
        candidate_values = values[candidate_id]
        trial_utility = candidate_values[:, 0] - candidate_values[:, 1] / divisor
        adaptive_utility = float(np.mean(trial_utility))
        baseline = fixed[divisor]
        rows.append({
            "variant": item["variant"],
            "split": split,
            "utility_divisor": divisor,
            "reservation_divisor": item["reservation_divisor"],
            **asdict(config),
            "adaptive_utility": adaptive_utility,
            "fixed_utility": baseline["utility"],
            "utility_gap": adaptive_utility - baseline["utility"],
            "relative_utility_gain_pct": (
                100.0 * (adaptive_utility - baseline["utility"])
                / baseline["utility"]
            ),
            "utility_delta_vs_reference": float(np.mean(
                trial_utility - reference_utility[divisor]
            )),
            "adaptive_accuracy": float(np.mean(candidate_values[:, 0])),
            "adaptive_chars": float(np.mean(candidate_values[:, 1])),
            "adaptive_opens": float(np.mean(candidate_values[:, 2])),
            "fixed_accuracy": baseline["accuracy"],
            "fixed_chars": baseline["chars"],
            "fixed_n": baseline["n"],
        })
    return rows


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean, 0.0
    std = float(np.std(values, ddof=1))
    half = float(student_t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
    return mean, mean - half, mean + half, std


def aggregate(rows):
    output = []
    keys = sorted({(row["variant"], float(row["utility_divisor"])) for row in rows})
    for variant, divisor in keys:
        group = [
            row for row in rows
            if row["variant"] == variant
            and float(row["utility_divisor"]) == divisor
        ]
        first = group[0]
        item = {
            "variant": variant,
            "utility_divisor": divisor,
            "reservation_divisor": first["reservation_divisor"],
            "splits": len(group),
        }
        for metric in (
            "adaptive_utility", "fixed_utility", "utility_gap",
            "relative_utility_gain_pct", "utility_delta_vs_reference",
            "adaptive_accuracy", "adaptive_chars", "adaptive_opens",
            "fixed_accuracy", "fixed_chars", "fixed_n",
        ):
            mean, low, high, std = interval([row[metric] for row in group])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci_low"] = low
            item[f"{metric}_ci_high"] = high
            item[f"{metric}_std"] = std
        item["positive_gap_splits"] = sum(row["utility_gap"] > 0 for row in group)
        output.append(item)
    return output


def summarize_variants(summary):
    output = []
    for variant in sorted({row["variant"] for row in summary}):
        group = [row for row in summary if row["variant"] == variant]
        description = (
            "retained per-divisor profiled policy"
            if variant == "reference_profiled"
            else VARIANT_BY_NAME[variant].description
        )
        output.append({
            "variant": variant,
            "description": description,
            "mean_relative_utility_gain_pct": float(np.mean([
                row["relative_utility_gain_pct_mean"] for row in group
            ])),
            "mean_utility_delta_vs_reference": float(np.mean([
                row["utility_delta_vs_reference_mean"] for row in group
            ])),
            "positive_divisor_means": sum(
                row["utility_gap_mean"] > 0 for row in group
            ),
            "divisors": len(group),
            "mean_accuracy": float(np.mean([
                row["adaptive_accuracy_mean"] for row in group
            ])),
            "mean_chars": float(np.mean([
                row["adaptive_chars_mean"] for row in group
            ])),
        })
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary, path):
    labels = [row["variant"].replace("_", "\n") for row in summary]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    axes[0].bar(x, [row["mean_relative_utility_gain_pct"] for row in summary],
                color="#C45A00")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Relative utility improvement over Fixed-$N$ (%)")
    axes[1].bar(x, [row["mean_utility_delta_vs_reference"] for row in summary],
                color="#3A6EA5")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Absolute utility difference from retained policy")
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Coding utility simplification study")
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path("algorithm/bestofn_coding/data.jsonl"))
    parser.add_argument("--char-cache", type=Path, default=Path(
        "algorithm/bestofn_coding/practical_algorithm/coding_char_counts_83.npz"
    ))
    parser.add_argument("--reference-selection", type=Path, default=Path(
        "codex_results/coding_utility_improvement/results/"
        "frontier_validation/frozen_selection.csv"
    ))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANT_BY_NAME),
                        default=tuple(VARIANT_BY_NAME))
    parser.add_argument("--split-start", type=int, default=60)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--test-permutations", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(args.reference_selection, tuple(args.variants))
    problems = load_problems(args.data, args.char_cache)
    rows = []
    for position, split in enumerate(
        range(args.split_start, args.split_start + args.splits), start=1
    ):
        rows.extend(run_split(split, problems, candidates, args))
        write_csv(args.output_dir / "utility_splits.partial.csv", rows)
        print(
            f"[coding simplify] split {position}/{args.splits} (id={split}) complete",
            flush=True,
        )
    summary = aggregate(rows)
    variant_summary = summarize_variants(summary)
    write_csv(args.output_dir / "utility_splits.csv", rows)
    write_csv(args.output_dir / "utility_summary.csv", summary)
    write_csv(args.output_dir / "variant_summary.csv", variant_summary)
    plot_summary(variant_summary, args.output_dir / "simplification_summary")
    method = {
        "purpose": "coding utility simplification without final-split selection",
        "split_start": args.split_start,
        "splits": args.splits,
        "test_permutations": args.test_permutations,
        "seed": args.seed,
        "reference_selection": str(args.reference_selection),
        "variants": [asdict(VARIANT_BY_NAME[name]) for name in args.variants],
        "global_prior_semantics": (
            "reward_prior_strength=Infinity uses exactly the train-fitted global "
            "distribution; no local reward-tail estimate is formed"
        ),
        "shared_streams": "identical split and test permutations for every variant",
        "cost": "exact cumulative characters actually opened",
        "fixed_n_guard": False,
    }
    (args.output_dir / "METHOD.json").write_text(json.dumps(method, indent=2) + "\n")
    print(json.dumps(variant_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
