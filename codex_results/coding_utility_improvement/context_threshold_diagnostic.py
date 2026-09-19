"""Development-only diagnostic for contextual Pandora reservation signals.

This script is intentionally not the final algorithm.  It measures whether a
non-anticipating score threshold, optionally shifted by the first three
rewards, allocates character budget better than Fixed-N.  A positive result
justifies putting the same context into the UCB tail/value model; a negative
result prevents an expensive blind grid expansion.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RETAINED_CODE = HERE.parent / "code" / "coding_scripts"
if str(RETAINED_CODE) not in sys.path:
    sys.path.insert(0, str(RETAINED_CODE))

from coding_ucb_three_objectives import (  # noqa: E402
    fixed_curves,
    fixed_trials,
    load_problems,
    make_permutations,
    split_problems,
)


DEFAULT_DIVISORS = tuple(float(value) for value in range(100_000, 1_000_001, 100_000))


def trajectories(problems, permutations):
    rows = []
    for problem_id in sorted(problems):
        rewards, correct, chars = problems[problem_id]
        for permutation in permutations[problem_id]:
            rows.append((rewards[permutation], correct[permutation], chars[permutation]))
    return rows


def evaluate_thresholds(rows, center, beta, cap, thresholds, min_open=3):
    """Return threshold x metric arrays for one contextual threshold family."""
    successes = np.empty((len(rows), len(thresholds)), dtype=np.float64)
    costs = np.empty_like(successes)
    opens = np.empty_like(successes)
    for row_id, (rewards, correct, chars) in enumerate(rows):
        shifted = rewards + beta * (float(np.mean(rewards[:min_open])) - center)
        previous_best = np.concatenate(([-np.inf], np.maximum.accumulate(shifted)[:-1]))
        running_best_index = np.maximum.accumulate(
            np.where(shifted > previous_best, np.arange(len(shifted)), 0)
        )
        running_best = shifted[running_best_index]
        cumulative_chars = np.cumsum(chars, dtype=np.float64)
        # running_best is monotone, so search all reservation thresholds
        # without materializing a 512 x n_threshold Boolean matrix.
        first = np.searchsorted(running_best, thresholds, side="left")
        first = np.maximum(first, min_open - 1)
        first = np.minimum(first, len(rewards) - 1)
        if cap is not None:
            first = np.minimum(first, cap - 1)
        successes[row_id] = correct[running_best_index[first]]
        costs[row_id] = cumulative_chars[first]
        opens[row_id] = first + 1
    return successes.mean(axis=0), costs.mean(axis=0), opens.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path("algorithm/bestofn_coding/data.jsonl"))
    parser.add_argument("--char-cache", type=Path, default=Path(
        "algorithm/bestofn_coding/practical_algorithm/coding_char_counts_83.npz"
    ))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-start", type=int, default=60)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--train-permutations", type=int, default=8)
    parser.add_argument("--test-permutations", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--divisors", type=float, nargs="+", default=DEFAULT_DIVISORS)
    args = parser.parse_args()

    problems = load_problems(args.data, args.char_cache)
    betas = (-0.5, 0.0, 0.25, 0.5, 1.0, 2.0)
    cap_factors = (0.75, 1.0, 1.25, 1.5, 2.0, None)
    output = []
    for split in range(args.split_start, args.split_start + args.splits):
        train, test = split_problems(problems, args.seed + split)
        _, _, fixed_ns = fixed_curves(train, args.divisors)
        train_perm = make_permutations(
            train, args.train_permutations, args.seed + 101_003 * split + 17
        )
        test_perm = make_permutations(
            test, args.test_permutations, args.seed + 101_003 * split + 31
        )
        train_rows = trajectories(train, train_perm)
        test_rows = trajectories(test, test_perm)
        center = float(np.mean(np.concatenate([values[0] for values in train.values()])))
        raw_train = np.concatenate([row[0] for row in train_rows])
        thresholds = np.unique(np.quantile(raw_train, np.linspace(0.25, 0.999, 121)))
        fixed_acc, fixed_chars, _ = fixed_trials(test, test_perm)

        train_curves = {}
        test_curves = {}
        for beta in betas:
            for factor in cap_factors:
                for divisor in args.divisors:
                    cap = None if factor is None else max(
                        3, min(512, int(np.ceil(factor * fixed_ns[divisor])))
                    )
                    key = (beta, factor, divisor)
                    train_curves[key] = evaluate_thresholds(
                        train_rows, center, beta, cap, thresholds
                    )
                    test_curves[key] = evaluate_thresholds(
                        test_rows, center, beta, cap, thresholds
                    )

        for divisor in args.divisors:
            n_fixed = fixed_ns[divisor]
            fixed_utility = float(np.mean(
                fixed_acc[:, n_fixed - 1] - fixed_chars[:, n_fixed - 1] / divisor
            ))
            for beta in betas:
                for factor in cap_factors:
                    key = (beta, factor, divisor)
                    train_acc, train_chars, _ = train_curves[key]
                    utility = train_acc - train_chars / divisor
                    selected = int(np.argmax(utility))
                    test_acc, test_chars, test_opens = test_curves[key]
                    adaptive = float(test_acc[selected] - test_chars[selected] / divisor)
                    output.append({
                        "split": split,
                        "divisor": divisor,
                        "beta": beta,
                        "cap_factor": factor,
                        "threshold": float(thresholds[selected]),
                        "train_utility": float(utility[selected]),
                        "adaptive_utility": adaptive,
                        "fixed_utility": fixed_utility,
                        "utility_gap": adaptive - fixed_utility,
                        "adaptive_accuracy": float(test_acc[selected]),
                        "adaptive_chars": float(test_chars[selected]),
                        "adaptive_opens": float(test_opens[selected]),
                        "fixed_n": n_fixed,
                    })
        print(f"[context-threshold] split {split} complete", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
