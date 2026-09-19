"""All-generator evaluation of the frozen UCB-Pandora alignment policy.

Produces three held-out objectives with 95% confidence bands across prompt
splits: utility against train-tuned fixed N, independent head-to-head reward at
exactly matched expected character computation, and target-quality character
savings.  Every adaptive decision is the reservation rule implemented by
``pandora_stop_many``; no contextual horizon policy is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.stats import t as student_t

from alignment_pandora_ucb import (
    PandoraConfig,
    _stable_alpha_quantile,
    fit_priors,
    load_data,
    pandora_stop_many,
    predict_mean_chars,
    prompt_features,
)


GENERATORS = (
    "gemma2_9b",
    "llama3.1_8b",
    "llama3.2_3b",
    "mistral_7b",
    "qwen2.5_7b",
)
LABELS = {
    "gemma2_9b": "Gemma-2-9B",
    "llama3.1_8b": "Llama-3.1-8B",
    "llama3.2_3b": "Llama-3.2-3B",
    "mistral_7b": "Mistral-7B",
    "qwen2.5_7b": "Qwen-2.5-7B",
}
COLORS = {
    "gemma2_9b": "#3A6EA5",
    "llama3.1_8b": "#D1495B",
    "llama3.2_3b": "#7A5195",
    "mistral_7b": "#2E7D32",
    "qwen2.5_7b": "#E17C05",
}
REPORT_DIVISORS = (1e5, 5e5, 1e6, 5e6, 1e7, 5e7)
FRONTIER_DIVISORS = (5e4, 7.5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7)
TARGETS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


@dataclass
class TrialSet:
    quality_prefix: np.ndarray
    chars_prefix: np.ndarray
    best_prefix: np.ndarray
    opponent_best_prefix: np.ndarray
    opponent_chars_prefix: np.ndarray
    adaptive_best: dict[float, np.ndarray]
    adaptive_quality: dict[float, np.ndarray]
    adaptive_chars: dict[float, np.ndarray]
    adaptive_opens: dict[float, np.ndarray]


def make_trials(rewards: np.ndarray, chars: np.ndarray, indices: np.ndarray,
                permutations: int, seed: int, features: np.ndarray, prior,
                cfg: PandoraConfig, divisors: tuple[float, ...],
                min_open_count: int = 3) -> TrialSet:
    rng = np.random.default_rng(seed)
    predicted_chars = predict_mean_chars(prior, features)
    quality_prefix, chars_prefix, best_prefix = [], [], []
    opponent_best, opponent_chars = [], []
    adaptive_best = {d: [] for d in divisors}
    adaptive_quality = {d: [] for d in divisors}
    adaptive_chars = {d: [] for d in divisors}
    adaptive_opens = {d: [] for d in divisors}
    for prompt_idx in indices:
        prompt_idx = int(prompt_idx)
        benchmark = _stable_alpha_quantile(rewards[prompt_idx], 0.99)
        for _ in range(permutations):
            order = rng.permutation(rewards.shape[1])
            opponent_order = rng.permutation(rewards.shape[1])
            rr, cc = rewards[prompt_idx, order], chars[prompt_idx, order]
            fixed_best = np.maximum.accumulate(rr)
            fixed_chars = np.cumsum(cc)
            quality_prefix.append(expit(fixed_best - benchmark))
            chars_prefix.append(fixed_chars)
            best_prefix.append(fixed_best)
            opponent_best.append(np.maximum.accumulate(rewards[prompt_idx, opponent_order]))
            opponent_chars.append(np.cumsum(chars[prompt_idx, opponent_order]))
            stops = pandora_stop_many(
                rr, cc, divisors, float(predicted_chars[prompt_idx]), prior, cfg,
                min_open_count,
            )
            for divisor in divisors:
                opened, best, total_chars = stops[divisor]
                adaptive_best[divisor].append(best)
                adaptive_quality[divisor].append(float(expit(best - benchmark)))
                adaptive_chars[divisor].append(total_chars)
                adaptive_opens[divisor].append(opened)
    convert = lambda mapping: {d: np.asarray(values, dtype=np.float64) for d, values in mapping.items()}
    return TrialSet(
        np.asarray(quality_prefix), np.asarray(chars_prefix), np.asarray(best_prefix),
        np.asarray(opponent_best), np.asarray(opponent_chars), convert(adaptive_best),
        convert(adaptive_quality), convert(adaptive_chars), convert(adaptive_opens),
    )


def mixed_index(curve: np.ndarray, target: float) -> tuple[int, int, float]:
    curve = np.maximum.accumulate(np.asarray(curve, dtype=np.float64))
    if target <= curve[0]:
        return 0, 0, 0.0
    if target >= curve[-1]:
        return len(curve) - 1, len(curve) - 1, 0.0
    hi = int(np.searchsorted(curve, target, side="left"))
    lo = hi - 1
    if curve[hi] <= curve[lo] + 1e-15:
        return hi, hi, 0.0
    weight = (target - curve[lo]) / (curve[hi] - curve[lo])
    return lo, hi, float(np.clip(weight, 0.0, 1.0))


def mixed_mean(values: np.ndarray, mix: tuple[int, int, float]) -> float:
    lo, hi, weight = mix
    return float(np.mean((1.0 - weight) * values[:, lo] + weight * values[:, hi]))


def h2h_score(adaptive_best: np.ndarray, fixed_best: np.ndarray,
              mix: tuple[int, int, float]) -> float:
    lo, hi, weight = mix

    def score(values: np.ndarray) -> np.ndarray:
        return np.where(adaptive_best > values, 1.0,
                        np.where(adaptive_best == values, 0.5, 0.0))

    return float(np.mean((1.0 - weight) * score(fixed_best[:, lo]) +
                         weight * score(fixed_best[:, hi])))


def evaluate_generator(generator: str, args_dict: dict) -> tuple[list[dict], list[dict], list[dict]]:
    args = argparse.Namespace(**args_dict)
    path = args.data_dir / f"{generator}_output.merged_rm.jsonl.gz"
    rewards, chars, prompts = load_data(path, args.reward_key)
    features = prompt_features(prompts)
    all_divisors = tuple(sorted(set(REPORT_DIVISORS) | set(FRONTIER_DIVISORS)))
    cfg = PandoraConfig(
        "exp_tail_exponentiated", confidence_scale=0.4,
        reward_prior_strength=5.0, cost_prior_strength=0.0,
        threshold_quantile=0.5, benchmark_calibration=0.0,
        ei_bonus_scale=0.002,
    )
    utility_rows, h2h_rows, target_rows = [], [], []
    generator_id = GENERATORS.index(generator)
    for split in range(args.split_start, args.split_start + args.splits):
        split_rng = np.random.default_rng(args.seed + split)
        order = split_rng.permutation(len(rewards))
        train_idx, test_idx = order[:len(order) // 2], order[len(order) // 2:]
        prior = fit_priors(rewards, chars, features, train_idx, (0.5,))
        train = make_trials(
            rewards, chars, train_idx, args.train_permutations,
            args.seed + 100_000 * generator_id + 1009 * split + 17,
            features, prior, cfg, all_divisors,
        )
        test = make_trials(
            rewards, chars, test_idx, args.test_permutations,
            args.seed + 100_000 * generator_id + 1009 * split + 31,
            features, prior, cfg, all_divisors,
        )
        train_quality_curve = train.quality_prefix.mean(axis=0)
        train_char_curve = train.chars_prefix.mean(axis=0)
        test_quality_curve = test.quality_prefix.mean(axis=0)
        test_opponent_char_curve = test.opponent_chars_prefix.mean(axis=0)

        for divisor in REPORT_DIVISORS:
            train_utility_curve = train_quality_curve - train_char_curve / divisor
            fixed_n = int(np.argmax(train_utility_curve))
            fixed_utility = test.quality_prefix[:, fixed_n] - test.chars_prefix[:, fixed_n] / divisor
            adaptive_utility = (test.adaptive_quality[divisor] -
                                test.adaptive_chars[divisor] / divisor)
            fixed_mean = float(np.mean(fixed_utility))
            adaptive_mean = float(np.mean(adaptive_utility))
            utility_rows.append({
                "generator": generator, "split": split, "divisor": divisor,
                "fixed_n": fixed_n + 1,
                "fixed_utility": fixed_mean, "adaptive_utility": adaptive_mean,
                "utility_gap": adaptive_mean - fixed_mean,
                "relative_utility_gain_pct": 100.0 * (adaptive_mean - fixed_mean) / fixed_mean,
                "adaptive_quality": float(np.mean(test.adaptive_quality[divisor])),
                "adaptive_chars": float(np.mean(test.adaptive_chars[divisor])),
                "adaptive_opens": float(np.mean(test.adaptive_opens[divisor])),
            })

            adaptive_chars_mean = float(np.mean(test.adaptive_chars[divisor]))
            cost_mix = mixed_index(test_opponent_char_curve, adaptive_chars_mean)
            matched_chars = mixed_mean(test.opponent_chars_prefix, cost_mix)
            h2h_rows.append({
                "generator": generator, "split": split, "divisor": divisor,
                "adaptive_score": h2h_score(
                    test.adaptive_best[divisor], test.opponent_best_prefix, cost_mix,
                ),
                "adaptive_chars": adaptive_chars_mean, "fixed_chars": matched_chars,
                "cost_mismatch": adaptive_chars_mean - matched_chars,
                "fixed_n_low": cost_mix[0] + 1, "fixed_n_high": cost_mix[1] + 1,
                "fixed_high_weight": cost_mix[2],
            })

        frontier_quality = np.asarray([
            float(np.mean(train.adaptive_quality[d])) for d in FRONTIER_DIVISORS
        ])
        for target in TARGETS:
            adaptive_mix = mixed_index(frontier_quality, target)
            adaptive_lo, adaptive_hi, adaptive_weight = adaptive_mix
            divisor_low = FRONTIER_DIVISORS[adaptive_lo]
            divisor_high = FRONTIER_DIVISORS[adaptive_hi]
            reached = bool(frontier_quality[0] <= target <= np.maximum.accumulate(frontier_quality)[-1])
            fixed_mix = mixed_index(train_quality_curve, target)
            fixed_quality = mixed_mean(test.quality_prefix, fixed_mix)
            fixed_chars = mixed_mean(test.chars_prefix, fixed_mix)
            adaptive_quality = float(np.mean(
                (1.0 - adaptive_weight) * test.adaptive_quality[divisor_low] +
                adaptive_weight * test.adaptive_quality[divisor_high]
            ))
            adaptive_chars = float(np.mean(
                (1.0 - adaptive_weight) * test.adaptive_chars[divisor_low] +
                adaptive_weight * test.adaptive_chars[divisor_high]
            ))
            heldout_mix = mixed_index(test_quality_curve, adaptive_quality)
            matched_fixed_chars = mixed_mean(test.chars_prefix, heldout_mix)
            target_rows.append({
                "generator": generator, "split": split, "target_quality": target,
                "adaptive_divisor_low": divisor_low, "adaptive_divisor_high": divisor_high,
                "adaptive_high_weight": adaptive_weight, "target_reached_train": reached,
                "fixed_quality": fixed_quality, "adaptive_quality": adaptive_quality,
                "quality_gap": adaptive_quality - fixed_quality,
                "fixed_chars": fixed_chars, "adaptive_chars": adaptive_chars,
                "direct_char_saving_pct": 100.0 * (fixed_chars - adaptive_chars) / fixed_chars,
                "matched_fixed_chars": matched_fixed_chars,
                "matched_char_saving_pct": 100.0 * (matched_fixed_chars - adaptive_chars) /
                                            matched_fixed_chars,
            })
        print(f"[{generator}] split {split - args.split_start + 1}/{args.splits}", flush=True)
    return utility_rows, h2h_rows, target_rows


def summarize(rows: list[dict], x_key: str, metrics: tuple[str, ...]) -> list[dict]:
    output = []
    keys = sorted({(r["generator"], float(r[x_key])) for r in rows})
    for generator, x_value in keys:
        group = [r for r in rows if r["generator"] == generator and float(r[x_key]) == x_value]
        out = {"generator": generator, x_key: x_value, "splits": len(group)}
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in group])
            mean = float(values.mean())
            half = (float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) /
                          math.sqrt(len(values))) if len(values) > 1 else 0.0)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_ci_low"] = mean - half
            out[f"{metric}_ci_high"] = mean + half
        output.append(out)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def band(ax, x, mean, low, high, generator, label=None):
    color = COLORS[generator]
    ax.plot(x, mean, marker="o", linewidth=2, color=color, label=label or LABELS[generator])
    ax.fill_between(x, low, high, color=color, alpha=0.13)


def plot_metric(summary: list[dict], x_key: str, metric: str, path: Path,
                ylabel: str, title: str, baseline: float, log_x: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for generator in GENERATORS:
        group = sorted((r for r in summary if r["generator"] == generator), key=lambda r: r[x_key])
        x = np.asarray([r[x_key] for r in group])
        band(ax, x,
             np.asarray([r[f"{metric}_mean"] for r in group]),
             np.asarray([r[f"{metric}_ci_low"] for r in group]),
             np.asarray([r[f"{metric}_ci_high"] for r in group]), generator)
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1.1)
    if log_x:
        ax.set_xscale("log")
    ax.set(xlabel=("Character-cost divisor" if x_key == "divisor" else "Target quality"),
           ylabel=ylabel, title=title)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_target(summary: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), constrained_layout=True)
    for generator in GENERATORS:
        group = sorted((r for r in summary if r["generator"] == generator),
                       key=lambda r: r["target_quality"])
        x = np.asarray([r["target_quality"] for r in group])
        band(axes[0], x,
             np.asarray([r["matched_char_saving_pct_mean"] for r in group]),
             np.asarray([r["matched_char_saving_pct_ci_low"] for r in group]),
             np.asarray([r["matched_char_saving_pct_ci_high"] for r in group]), generator)
        band(axes[1], x,
             np.asarray([r["adaptive_quality_mean"] for r in group]),
             np.asarray([r["adaptive_quality_ci_low"] for r in group]),
             np.asarray([r["adaptive_quality_ci_high"] for r in group]), generator)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set(xlabel="Target quality", ylabel="Character saving over quality-matched fixed N (%)",
                title="Target-quality computation saving")
    axes[1].plot(TARGETS, TARGETS, color="black", linestyle="--", linewidth=1.0,
                 label="Achieved = target")
    axes[1].set(xlabel="Target quality", ylabel="Held-out adaptive quality",
                title="Target tracking")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_overview(utility_summary: list[dict], h2h_summary: list[dict],
                  target_summary: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.7), constrained_layout=True)
    specs = (
        (axes[0], utility_summary, "divisor", "relative_utility_gain_pct", 0.0,
         "Utility gain (%)", "A. Utility", True),
        (axes[1], h2h_summary, "divisor", "adaptive_score", 0.5,
         "Adaptive head-to-head score", "B. Equal-cost head-to-head", True),
        (axes[2], target_summary, "target_quality", "matched_char_saving_pct", 0.0,
         "Character saving (%)", "C. Target quality", False),
    )
    for ax, summary, x_key, metric, baseline, ylabel, title, log_x in specs:
        for generator in GENERATORS:
            group = sorted((r for r in summary if r["generator"] == generator), key=lambda r: r[x_key])
            x = np.asarray([r[x_key] for r in group])
            band(ax, x,
                 np.asarray([r[f"{metric}_mean"] for r in group]),
                 np.asarray([r[f"{metric}_ci_low"] for r in group]),
                 np.asarray([r[f"{metric}_ci_high"] for r in group]), generator)
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1.0)
        if log_x:
            ax.set_xscale("log")
        ax.set(xlabel=("Character-cost divisor" if x_key == "divisor" else "Target quality"),
               ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Frozen UCB Pandora across all alignment generators", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("dataset/alpaca"))
    p.add_argument("--reward-key", default="mistral_rm_reward")
    p.add_argument("--output-dir", type=Path,
                   default=Path("codex_results/pandora_all_generators"))
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--split-start", type=int, default=30)
    p.add_argument("--train-permutations", type=int, default=6)
    p.add_argument("--test-permutations", type=int, default=12)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260802)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args).copy()
    utility_rows, h2h_rows, target_rows = [], [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(GENERATORS))) as pool:
        futures = {pool.submit(evaluate_generator, generator, args_dict): generator
                   for generator in GENERATORS}
        for future in as_completed(futures):
            generator = futures[future]
            utility, h2h, target = future.result()
            utility_rows.extend(utility)
            h2h_rows.extend(h2h)
            target_rows.extend(target)
            print(f"completed {generator}", flush=True)

    utility_summary = summarize(
        utility_rows, "divisor",
        ("relative_utility_gain_pct", "utility_gap", "adaptive_utility", "fixed_utility",
         "adaptive_quality", "adaptive_chars", "adaptive_opens"),
    )
    h2h_summary = summarize(
        h2h_rows, "divisor", ("adaptive_score", "adaptive_chars", "fixed_chars", "cost_mismatch"),
    )
    target_summary = summarize(
        target_rows, "target_quality",
        ("matched_char_saving_pct", "direct_char_saving_pct", "adaptive_quality",
         "fixed_quality", "adaptive_chars", "fixed_chars", "quality_gap"),
    )
    for name, rows in (("utility_splits.csv", utility_rows), ("utility_summary.csv", utility_summary),
                       ("head_to_head_splits.csv", h2h_rows), ("head_to_head_summary.csv", h2h_summary),
                       ("target_quality_splits.csv", target_rows),
                       ("target_quality_summary.csv", target_summary)):
        write_csv(args.output_dir / name, rows)
    plot_metric(utility_summary, "divisor", "relative_utility_gain_pct",
                args.output_dir / "utility_gap_all_generators.png",
                "Relative utility gain over fixed N (%)",
                "UCB Pandora utility advantage; 95% CI across splits", 0.0, True)
    plot_metric(h2h_summary, "divisor", "adaptive_score",
                args.output_dir / "head_to_head_equal_chars_all_generators.png",
                "Adaptive head-to-head score",
                "Independent outputs at exactly matched expected characters", 0.5, True)
    plot_target(target_summary, args.output_dir / "target_quality_all_generators.png")
    plot_overview(utility_summary, h2h_summary, target_summary,
                  args.output_dir / "all_generators_three_objectives.png")
    method = {
        "algorithm": "UCB Pandora reservation stopping only",
        "generators": GENERATORS, "reward_key": args.reward_key,
        "configuration": {
            "model": "shifted exponential tail on exponentiated reward",
            "confidence_scale": 0.4, "ei_bonus_scale": 0.002,
            "reward_prior_strength": 5.0, "threshold_quantile": 0.5,
            "min_open_count": 3,
        },
        "splits": args.splits, "split_start": args.split_start,
        "train_permutations_per_prompt": args.train_permutations,
        "test_permutations_per_prompt": args.test_permutations,
        "train_test": "50/50 prompts",
        "head_to_head": "independent draws; adjacent fixed-N mixture exactly matches held-out expected characters; ties=0.5",
        "target_quality": "training-selected mixture of adjacent UCB-Pandora divisors and adjacent fixed N; primary saving exactly quality-matched descriptively on heldout",
        "cost": "exact cumulative output characters",
    }
    with open(args.output_dir / "METHOD.json", "w") as handle:
        json.dump(method, handle, indent=2)
    print(json.dumps({"utility": utility_summary, "head_to_head": h2h_summary,
                      "target_quality": target_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
