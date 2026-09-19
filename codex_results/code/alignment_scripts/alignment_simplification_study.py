"""Controlled simplification study for the retained alignment policy.

Each variant is evaluated on the same prompt splits and response permutations.
The reference reproduces the retained training-shrunk exponential-tail rule.
All other variants are genuinely training-free at decision time: their
``PriorFit`` object is a numerical placeholder multiplied by zero in every
stopping equation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t as student_t

from alignment_pandora_all_generators import (
    FRONTIER_DIVISORS,
    GENERATORS,
    REPORT_DIVISORS,
    TARGETS,
    h2h_score,
    make_trials,
    mixed_index,
    mixed_mean,
)
from alignment_pandora_ucb import (
    PandoraConfig,
    PriorFit,
    fit_priors,
    load_data,
    prompt_features,
)


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    config: PandoraConfig
    min_open: int
    uses_training_scale: bool


def _config(
    model="exp_tail_exponentiated",
    confidence=0.4,
    reward_prior=0.0,
    bonus=0.002,
) -> PandoraConfig:
    return PandoraConfig(
        model=model,
        confidence_scale=confidence,
        reward_prior_strength=reward_prior,
        cost_prior_strength=0.0,
        threshold_quantile=0.5,
        benchmark_calibration=0.0,
        ei_bonus_scale=bonus,
    )


VARIANTS = (
    Variant(
        "reference_train_scale",
        "retained exponentiated-tail UCB with five training pseudo-observations",
        _config(reward_prior=5.0),
        3,
        True,
    ),
    Variant(
        "local_exp_open3",
        "remove the training tail-scale prior and otherwise change nothing",
        _config(),
        3,
        False,
    ),
    Variant(
        "local_exp_open5",
        "training-free local tail with five initial responses",
        _config(),
        5,
        False,
    ),
    Variant(
        "local_exp_open4",
        "training-free local tail with four initial responses",
        _config(),
        4,
        False,
    ),
    Variant(
        "local_exp_open6",
        "training-free local tail with six initial responses",
        _config(),
        6,
        False,
    ),
    Variant(
        "local_exp_open8",
        "training-free local tail with eight initial responses",
        _config(),
        8,
        False,
    ),
    Variant(
        "local_exp_open5_no_bonus",
        "remove the training prior and additive EI bonus",
        _config(bonus=0.0),
        5,
        False,
    ),
    Variant(
        "local_exp_open5_plugin",
        "plain local shifted-exponential plug-in rule without UCB inflation",
        _config(confidence=0.0, bonus=0.0),
        5,
        False,
    ),
    Variant(
        "local_exp_open5_bonus_only",
        "local tail with only the additive anytime optimism bonus",
        _config(confidence=0.0, bonus=0.002),
        5,
        False,
    ),
    Variant(
        "local_exp_open5_bonus_only_004",
        "local tail with a larger additive anytime optimism bonus",
        _config(confidence=0.0, bonus=0.004),
        5,
        False,
    ),
    Variant(
        "local_exp_open5_conf02",
        "local tail with weaker scale confidence inflation",
        _config(confidence=0.2, bonus=0.002),
        5,
        False,
    ),
    Variant(
        "local_exp_open5_conf06",
        "local tail with stronger scale confidence inflation",
        _config(confidence=0.6, bonus=0.002),
        5,
        False,
    ),
    Variant(
        "local_raw_tail_open5",
        "fit the shifted-exponential tail directly in raw-reward space",
        _config(model="exp_tail_raw"),
        5,
        False,
    ),
    Variant(
        "local_halfnormal_open5_conf02",
        "fit a shifted half-normal upper tail directly to raw rewards",
        _config(model="halfnormal_tail_raw", confidence=0.2),
        5,
        False,
    ),
    Variant(
        "local_halfnormal_open5_conf04",
        "fit a shifted half-normal upper tail directly to raw rewards",
        _config(model="halfnormal_tail_raw", confidence=0.4),
        5,
        False,
    ),
    Variant(
        "local_halfnormal_open5_conf06",
        "fit a shifted half-normal upper tail directly to raw rewards",
        _config(model="halfnormal_tail_raw", confidence=0.6),
        5,
        False,
    ),
    Variant(
        "local_halfnormal_open5_conf08",
        "fit a shifted half-normal upper tail directly to raw rewards",
        _config(model="halfnormal_tail_raw", confidence=0.8),
        5,
        False,
    ),
    Variant(
        "local_rayleigh_open5_conf00",
        "fit a shifted Rayleigh upper tail directly to raw rewards",
        _config(model="rayleigh_tail_raw", confidence=0.0),
        5,
        False,
    ),
    Variant(
        "local_rayleigh_open5_conf02",
        "fit a shifted Rayleigh upper tail directly to raw rewards",
        _config(model="rayleigh_tail_raw", confidence=0.2),
        5,
        False,
    ),
    Variant(
        "local_rayleigh_open5_conf04",
        "fit a shifted Rayleigh upper tail directly to raw rewards",
        _config(model="rayleigh_tail_raw", confidence=0.4),
        5,
        False,
    ),
    Variant(
        "local_rayleigh_open5_conf06",
        "fit a shifted Rayleigh upper tail directly to raw rewards",
        _config(model="rayleigh_tail_raw", confidence=0.6),
        5,
        False,
    ),
    Variant(
        "local_rayleigh_open5_conf08",
        "fit a shifted Rayleigh upper tail directly to raw rewards",
        _config(model="rayleigh_tail_raw", confidence=0.8),
        5,
        False,
    ),
    Variant(
        "local_rayleigh_open5_conf10",
        "fit a shifted Rayleigh upper tail directly to raw rewards",
        _config(model="rayleigh_tail_raw", confidence=1.0),
        5,
        False,
    ),
    Variant(
        "local_rayleigh_open5_conf12",
        "fit a shifted Rayleigh upper tail directly to raw rewards",
        _config(model="rayleigh_tail_raw", confidence=1.2),
        5,
        False,
    ),
    Variant(
        "local_rayleigh_open5_conf14",
        "fit a shifted Rayleigh upper tail directly to raw rewards",
        _config(model="rayleigh_tail_raw", confidence=1.4),
        5,
        False,
    ),
    Variant(
        "local_rayleigh_open5_conf16",
        "fit a shifted Rayleigh upper tail directly to raw rewards",
        _config(model="rayleigh_tail_raw", confidence=1.6),
        5,
        False,
    ),
    Variant(
        "local_rayleigh_open5_conf18",
        "fit a shifted Rayleigh upper tail directly to raw rewards",
        _config(model="rayleigh_tail_raw", confidence=1.8),
        5,
        False,
    ),
    Variant(
        "local_rayleigh_open5_conf20",
        "fit a shifted Rayleigh upper tail directly to raw rewards",
        _config(model="rayleigh_tail_raw", confidence=2.0),
        5,
        False,
    ),
    Variant(
        "local_gaussian_open5",
        "training-free Gaussian UCB fit in raw-reward space",
        _config(model="gaussian_raw"),
        5,
        False,
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def prefix_only_placeholder_prior() -> PriorFit:
    """Return a finite placeholder whose contribution is exactly zero."""
    theoretical_k = -math.log((1.0 - 0.99) / 0.5)
    return PriorFit(
        raw_sigma=1.0,
        gaussian_quantile_z=float(norm.ppf(0.99)),
        raw_tail_scales={0.5: 1.0},
        exp_tail_ratios={0.5: 1.0},
        raw_tail_quantile_k={0.5: theoretical_k},
        exp_tail_quantile_k={0.5: theoretical_k},
        cost_beta=np.zeros(6, dtype=np.float64),
        feature_mean=np.zeros(5, dtype=np.float64),
        feature_scale=np.ones(5, dtype=np.float64),
    )


def select_cheapest_target(quality, chars, target) -> tuple[int, bool]:
    quality = np.asarray(quality, dtype=np.float64)
    chars = np.asarray(chars, dtype=np.float64)
    feasible = np.flatnonzero(quality >= target)
    if len(feasible):
        winner = min(feasible, key=lambda index: (chars[index], int(index)))
        return int(winner), True
    best_quality = float(np.max(quality))
    best = np.flatnonzero(np.isclose(quality, best_quality, atol=1e-15, rtol=0.0))
    winner = min(best, key=lambda index: (chars[index], int(index)))
    return int(winner), False


def evaluate_generator(generator: str, args_dict: dict, variant_names: tuple[str, ...]):
    args = argparse.Namespace(**args_dict)
    path = args.data_dir / f"{generator}_output.merged_rm.jsonl.gz"
    rewards, chars, prompts = load_data(path, args.reward_key)
    features = prompt_features(prompts)
    all_divisors = tuple(sorted(set(REPORT_DIVISORS) | set(FRONTIER_DIVISORS)))
    placeholder = prefix_only_placeholder_prior()
    generator_id = GENERATORS.index(generator)
    utility_rows, h2h_rows, target_rows = [], [], []

    for split in range(args.split_start, args.split_start + args.splits):
        order = np.random.default_rng(args.seed + split).permutation(len(rewards))
        midpoint = len(order) // 2
        train_idx, test_idx = order[:midpoint], order[midpoint:]
        fitted_prior = fit_priors(rewards, chars, features, train_idx, (0.5,))
        train_seed = args.seed + 100_000 * generator_id + 1009 * split + 17
        test_seed = args.seed + 100_000 * generator_id + 1009 * split + 31

        reference = VARIANT_BY_NAME["reference_train_scale"]
        reference_train = make_trials(
            rewards, chars, train_idx, args.train_permutations, train_seed,
            features, fitted_prior, reference.config, all_divisors,
            reference.min_open,
        )
        reference_test = make_trials(
            rewards, chars, test_idx, args.test_permutations, test_seed,
            features, fitted_prior, reference.config, all_divisors,
            reference.min_open,
        )
        train_fixed_quality = reference_train.quality_prefix.mean(axis=0)
        train_fixed_chars = reference_train.chars_prefix.mean(axis=0)
        test_opponent_chars = reference_test.opponent_chars_prefix.mean(axis=0)
        reference_utility = {
            divisor: (
                reference_test.adaptive_quality[divisor]
                - reference_test.adaptive_chars[divisor] / divisor
            )
            for divisor in REPORT_DIVISORS
        }

        fixed_target_selection = {
            target: select_cheapest_target(
                train_fixed_quality, train_fixed_chars, target
            )
            for target in TARGETS
        }

        for variant_name in variant_names:
            variant = VARIANT_BY_NAME[variant_name]
            if variant.name == reference.name:
                train, test = reference_train, reference_test
            else:
                policy_prior = fitted_prior if variant.uses_training_scale else placeholder
                train = make_trials(
                    rewards, chars, train_idx, args.train_permutations,
                    train_seed, features, policy_prior, variant.config,
                    all_divisors, variant.min_open,
                )
                test = make_trials(
                    rewards, chars, test_idx, args.test_permutations,
                    test_seed, features, policy_prior, variant.config,
                    all_divisors, variant.min_open,
                )

            for divisor in REPORT_DIVISORS:
                train_fixed_utility = train_fixed_quality - train_fixed_chars / divisor
                fixed_index = int(np.argmax(train_fixed_utility))
                fixed_trial_utility = (
                    reference_test.quality_prefix[:, fixed_index]
                    - reference_test.chars_prefix[:, fixed_index] / divisor
                )
                adaptive_trial_utility = (
                    test.adaptive_quality[divisor]
                    - test.adaptive_chars[divisor] / divisor
                )
                fixed_mean = float(np.mean(fixed_trial_utility))
                adaptive_mean = float(np.mean(adaptive_trial_utility))
                utility_rows.append({
                    "variant": variant.name,
                    "generator": generator,
                    "split": split,
                    "divisor": divisor,
                    "fixed_n": fixed_index + 1,
                    "adaptive_utility": adaptive_mean,
                    "fixed_utility": fixed_mean,
                    "utility_gap": adaptive_mean - fixed_mean,
                    "relative_utility_gain_pct": (
                        100.0 * (adaptive_mean - fixed_mean) / fixed_mean
                    ),
                    "utility_delta_vs_reference": float(np.mean(
                        adaptive_trial_utility - reference_utility[divisor]
                    )),
                    "adaptive_quality": float(np.mean(test.adaptive_quality[divisor])),
                    "adaptive_chars": float(np.mean(test.adaptive_chars[divisor])),
                    "adaptive_opens": float(np.mean(test.adaptive_opens[divisor])),
                })

                adaptive_chars = float(np.mean(test.adaptive_chars[divisor]))
                cost_mix = mixed_index(test_opponent_chars, adaptive_chars)
                h2h_rows.append({
                    "variant": variant.name,
                    "generator": generator,
                    "split": split,
                    "divisor": divisor,
                    "adaptive_score": h2h_score(
                        test.adaptive_best[divisor],
                        reference_test.opponent_best_prefix,
                        cost_mix,
                    ),
                    "adaptive_chars": adaptive_chars,
                    "fixed_chars": mixed_mean(
                        reference_test.opponent_chars_prefix, cost_mix
                    ),
                })

            train_adaptive_quality = np.asarray([
                np.mean(train.adaptive_quality[divisor])
                for divisor in FRONTIER_DIVISORS
            ])
            train_adaptive_chars = np.asarray([
                np.mean(train.adaptive_chars[divisor])
                for divisor in FRONTIER_DIVISORS
            ])
            for target in TARGETS:
                adaptive_index, adaptive_reached = select_cheapest_target(
                    train_adaptive_quality, train_adaptive_chars, target
                )
                fixed_index, fixed_reached = fixed_target_selection[target]
                selected_divisor = FRONTIER_DIVISORS[adaptive_index]
                adaptive_quality = float(np.mean(
                    test.adaptive_quality[selected_divisor]
                ))
                adaptive_chars = float(np.mean(
                    test.adaptive_chars[selected_divisor]
                ))
                fixed_quality = float(np.mean(
                    reference_test.quality_prefix[:, fixed_index]
                ))
                fixed_chars = float(np.mean(
                    reference_test.chars_prefix[:, fixed_index]
                ))
                target_rows.append({
                    "variant": variant.name,
                    "generator": generator,
                    "split": split,
                    "target_quality": target,
                    "selected_divisor": selected_divisor,
                    "adaptive_target_reached_train": adaptive_reached,
                    "fixed_target_reached_train": fixed_reached,
                    "adaptive_test_quality": adaptive_quality,
                    "fixed_test_quality": fixed_quality,
                    "adaptive_target_error_pp": 100.0 * (adaptive_quality - target),
                    "fixed_target_error_pp": 100.0 * (fixed_quality - target),
                    "adaptive_test_chars": adaptive_chars,
                    "fixed_test_chars": fixed_chars,
                    "character_saving_vs_train_fixed_pct": (
                        100.0 * (fixed_chars - adaptive_chars) / fixed_chars
                    ),
                })

        print(
            f"[alignment simplify:{generator}] "
            f"split {split - args.split_start + 1}/{args.splits}",
            flush=True,
        )
    return utility_rows, h2h_rows, target_rows


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    half = float(
        student_t.ppf(0.975, len(values) - 1)
        * np.std(values, ddof=1) / math.sqrt(len(values))
    )
    return mean, mean - half, mean + half


def cell_summary(rows, x_key, metrics):
    output = []
    keys = sorted({
        (row["variant"], row["generator"], float(row[x_key])) for row in rows
    })
    for variant, generator, x_value in keys:
        group = [
            row for row in rows
            if row["variant"] == variant
            and row["generator"] == generator
            and float(row[x_key]) == x_value
        ]
        item = {
            "variant": variant,
            "generator": generator,
            x_key: x_value,
            "splits": len(group),
        }
        for metric in metrics:
            mean, low, high = interval([row[metric] for row in group])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci_low"] = low
            item[f"{metric}_ci_high"] = high
        output.append(item)
    return output


def variant_summary(utility, utility_cells, h2h_cells, target, target_cells):
    output = []
    for variant in (VARIANT_BY_NAME[name] for name in sorted({r["variant"] for r in utility})):
        utility_group = [r for r in utility if r["variant"] == variant.name]
        utility_cell_group = [r for r in utility_cells if r["variant"] == variant.name]
        h2h_cell_group = [r for r in h2h_cells if r["variant"] == variant.name]
        target_group = [r for r in target if r["variant"] == variant.name]
        target_cell_group = [r for r in target_cells if r["variant"] == variant.name]
        output.append({
            "variant": variant.name,
            "description": variant.description,
            "uses_training_scale": variant.uses_training_scale,
            "model": variant.config.model,
            "confidence_scale": variant.config.confidence_scale,
            "ei_bonus_scale": variant.config.ei_bonus_scale,
            "min_open": variant.min_open,
            "mean_relative_utility_gain_pct": float(np.mean([
                row["relative_utility_gain_pct_mean"] for row in utility_cell_group
            ])),
            "mean_utility_delta_vs_reference": float(np.mean([
                row["utility_delta_vs_reference"] for row in utility_group
            ])),
            "positive_utility_cells": sum(
                row["utility_gap_mean"] > 0.0 for row in utility_cell_group
            ),
            "utility_cells": len(utility_cell_group),
            "mean_h2h_score": float(np.mean([
                row["adaptive_score_mean"] for row in h2h_cell_group
            ])),
            "minimum_h2h_cell_mean": float(np.min([
                row["adaptive_score_mean"] for row in h2h_cell_group
            ])),
            "h2h_cells_above_half": sum(
                row["adaptive_score_mean"] > 0.5 for row in h2h_cell_group
            ),
            "mean_target_abs_error_pp": float(np.mean([
                abs(row["adaptive_target_error_pp"]) for row in target_group
            ])),
            "mean_target_character_saving_pct": float(np.mean([
                row["character_saving_vs_train_fixed_pct_mean"]
                for row in target_cell_group
            ])),
            "positive_target_saving_cells": sum(
                row["character_saving_vs_train_fixed_pct_mean"] > 0.0
                for row in target_cell_group
            ),
            "target_cells": len(target_cell_group),
        })
    return output


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary, output):
    labels = [row["variant"].replace("_", "\n") for row in summary]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    specs = (
        ("mean_relative_utility_gain_pct", "Utility improvement over Fixed-$N$ (%)", 0.0),
        ("mean_h2h_score", "Matched-cost head-to-head score", 0.5),
        ("mean_target_abs_error_pp", "Target mean absolute error (points)", None),
    )
    for axis, (metric, ylabel, baseline) in zip(axes, specs):
        axis.bar(x, [row[metric] for row in summary], color="#3A6EA5")
        if baseline is not None:
            axis.axhline(baseline, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(x, labels, rotation=0, fontsize=7)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Alignment simplification study")
    fig.savefig(output.with_suffix(".png"), dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/alpaca"))
    parser.add_argument("--reward-key", default="mistral_rm_reward")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANT_BY_NAME),
                        default=tuple(VARIANT_BY_NAME))
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--split-start", type=int, default=10)
    parser.add_argument("--train-permutations", type=int, default=1)
    parser.add_argument("--test-permutations", type=int, default=2)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict.pop("variants")
    args_dict.pop("workers")
    args_dict.pop("output_dir")
    variant_names = tuple(args.variants)

    utility, h2h, target = [], [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(GENERATORS))) as pool:
        futures = {
            pool.submit(evaluate_generator, generator, args_dict, variant_names): generator
            for generator in GENERATORS
        }
        for future in as_completed(futures):
            generator = futures[future]
            generator_utility, generator_h2h, generator_target = future.result()
            utility.extend(generator_utility)
            h2h.extend(generator_h2h)
            target.extend(generator_target)
            print(f"[alignment simplify] completed {generator}", flush=True)

    utility_cells = cell_summary(
        utility, "divisor",
        ("relative_utility_gain_pct", "utility_gap", "utility_delta_vs_reference",
         "adaptive_utility", "adaptive_quality", "adaptive_chars", "adaptive_opens"),
    )
    h2h_cells = cell_summary(
        h2h, "divisor", ("adaptive_score", "adaptive_chars", "fixed_chars")
    )
    target_cells = cell_summary(
        target, "target_quality",
        ("adaptive_test_quality", "fixed_test_quality", "adaptive_target_error_pp",
         "fixed_target_error_pp", "adaptive_test_chars", "fixed_test_chars",
         "character_saving_vs_train_fixed_pct"),
    )
    summary = variant_summary(utility, utility_cells, h2h_cells, target, target_cells)
    for filename, rows in (
        ("utility_splits.csv", utility),
        ("utility_cells.csv", utility_cells),
        ("head_to_head_splits.csv", h2h),
        ("head_to_head_cells.csv", h2h_cells),
        ("target_splits.csv", target),
        ("target_cells.csv", target_cells),
        ("variant_summary.csv", summary),
    ):
        write_csv(args.output_dir / filename, rows)
    plot_summary(summary, args.output_dir / "simplification_summary")
    method = {
        "purpose": "stepwise alignment-policy simplification without final-split selection",
        "split_start": args.split_start,
        "splits": args.splits,
        "train_permutations": args.train_permutations,
        "test_permutations": args.test_permutations,
        "seed": args.seed,
        "variants": [
            {
                **asdict(VARIANT_BY_NAME[name]),
                "config": asdict(VARIANT_BY_NAME[name].config),
            }
            for name in variant_names
        ],
        "shared_streams": "identical prompt splits and response permutations",
        "training_free_contract": (
            "all non-reference variants have zero reward/cost prior weights and "
            "zero benchmark-calibration weight; the placeholder prior cannot "
            "affect a stopping decision"
        ),
    }
    (args.output_dir / "METHOD.json").write_text(json.dumps(method, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
