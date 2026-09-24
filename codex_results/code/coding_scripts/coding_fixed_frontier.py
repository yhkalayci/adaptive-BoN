"""Evaluate a dense, train-tuned Fixed-N frontier on unseen coding problems.

The split and permutation seeds match ``coding_token_profit_dmrl.py``. This
script evaluates only Fixed-N, so adding frontier points does not retune or
rerun an adaptive policy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from coding_token_profit_dmrl import make_permutations, select_fixed_n
from algorithm.adaptive_coding import load_coding_problems, split_problem_ids


def parse_divisors(value: str) -> tuple[float, ...]:
    divisors = tuple(float(item) for item in value.split(","))
    if not divisors or any(not np.isfinite(item) or item <= 0 for item in divisors):
        raise argparse.ArgumentTypeError("divisors must be positive and finite")
    return divisors


def evaluate_split(problems, divisors, split, seed, test_permutations):
    train_ids, test_ids = split_problem_ids(problems, seed=seed + split)
    train = {key: problems[key] for key in train_ids}
    test = {key: problems[key] for key in test_ids}
    fixed_ns, _, _ = select_fixed_n(train, divisors)
    orders = make_permutations(
        test, test_permutations, seed + 900_000 + split * 10_000
    )
    correctness = np.zeros(len(divisors), dtype=np.float64)
    tokens = np.zeros(len(divisors), dtype=np.float64)
    for problem_id in sorted(test):
        problem = test[problem_id]
        permutation = orders[problem_id]
        rewards = np.asarray(problem.rewards)[permutation]
        correct = np.asarray(problem.correct, dtype=np.float64)[permutation]
        lengths = np.asarray(problem.lengths, dtype=np.float64)[permutation]
        cumulative_tokens = np.cumsum(lengths, axis=1)
        best_index = np.zeros(len(permutation), dtype=np.int64)
        best_reward = np.full(len(permutation), -np.inf)
        selected = np.empty_like(permutation, dtype=np.int64)
        for count in range(rewards.shape[1]):
            improved = rewards[:, count] > best_reward
            best_index[improved] = count
            best_reward[improved] = rewards[improved, count]
            selected[:, count] = best_index
        rows = np.arange(len(permutation))
        for index, n in enumerate(fixed_ns):
            n = int(n)
            correctness[index] += float(np.sum(correct[rows, selected[:, n - 1]]))
            tokens[index] += float(np.sum(cumulative_tokens[:, n - 1]))
    trials = len(test) * test_permutations
    return [
        {
            "split": split,
            "utility_divisor": divisor,
            "fixed_n": int(n),
            "trials": trials,
            "fixed_accuracy": value / trials,
            "fixed_mean_output_tokens": token_count / trials,
        }
        for divisor, n, value, token_count in zip(
            divisors, fixed_ns, correctness, tokens
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--divisors", type=parse_divisors, required=True)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--outer-seed", type=int, default=20260923)
    parser.add_argument("--test-permutations", type=int, default=48)
    args = parser.parse_args()
    if args.splits < 1 or args.test_permutations < 1:
        parser.error("splits and test-permutations must be positive")
    problems = load_coding_problems(args.data)
    rows = [
        row
        for split in range(args.splits)
        for row in evaluate_split(
            problems, args.divisors, split, args.outer_seed,
            args.test_permutations,
        )
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for divisor in args.divisors:
        group = [row for row in rows if row["utility_divisor"] == divisor]
        print(
            f"{divisor:g}: N={np.mean([row['fixed_n'] for row in group]):.1f}, "
            f"accuracy={np.mean([row['fixed_accuracy'] for row in group]):.6f}, "
            f"tokens={np.mean([row['fixed_mean_output_tokens'] for row in group]):.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
