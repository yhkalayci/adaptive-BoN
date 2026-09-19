"""Build an uncapped calibrated utility profile on development splits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from coding_ucb_three_objectives import (
    REPORT_DIVISORS,
    evaluate,
    fit_config_calibrations,
    fit_pilot_prior,
    fit_pilot_reward_space_isotonic,
    fit_prior,
    fit_reward_space_isotonic,
    load_problems,
    make_permutations,
    split_problems,
    transform_reward_space,
    uncapped_tail_decay_configs,
    write_csv,
)


def evaluate_split(split, problems, configs, args):
    raw_train, raw_test = split_problems(problems, args.seed + split)
    train_permutations = make_permutations(
        raw_train, args.train_permutations,
        args.seed + 101_003 * split + 17,
    )
    if args.pilot_context_beta is None:
        reward_calibration = fit_reward_space_isotonic(raw_train)
        train = transform_reward_space(raw_train, reward_calibration)
        test = transform_reward_space(raw_test, reward_calibration)
        reward_transform = None
        prior = fit_prior(train)
    else:
        train, test = raw_train, raw_test
        reward_transform = fit_pilot_reward_space_isotonic(
            train, train_permutations, args.pilot_context_beta
        )
        prior = fit_pilot_prior(
            train, train_permutations, reward_transform
        )
    permutations = make_permutations(
        test, args.test_permutations, args.seed + 101_003 * split + 31
    )
    calibrations = fit_config_calibrations(train, configs)
    unused_fixed_ns = {float(divisor): 1 for divisor in REPORT_DIVISORS}
    values = evaluate(
        test, permutations, configs, calibrations, prior, REPORT_DIVISORS,
        unused_fixed_ns, args.delta, args.workers, reward_transform,
    )
    return values[..., 0].mean(axis=2), values[..., 1].mean(axis=2)


def build_profile(output_dir, splits, configs):
    accuracy, chars = [], []
    for split in splits:
        with np.load(output_dir / f"utility_tail_decay_{split}.npz") as data:
            if not np.array_equal(data["divisors"], REPORT_DIVISORS):
                raise ValueError(f"divisor mismatch for split {split}")
            accuracy.append(np.asarray(data["test_accuracy"], dtype=np.float64))
            chars.append(np.asarray(data["test_chars"], dtype=np.float64))
    accuracy = np.asarray(accuracy)
    chars = np.asarray(chars)
    expected = (len(splits), len(configs), len(REPORT_DIVISORS))
    if accuracy.shape != expected or chars.shape != expected:
        raise ValueError(f"development array shape mismatch: {accuracy.shape}")
    rows = []
    for config_id, config in enumerate(configs):
        for divisor_id, divisor in enumerate(REPORT_DIVISORS):
            split_utility = (
                accuracy[:, config_id, divisor_id]
                - chars[:, config_id, divisor_id] / divisor
            )
            rows.append({
                "config_id": config_id,
                "divisor_id": divisor_id,
                "divisor": divisor,
                **asdict(config),
                "development_splits": len(splits),
                "profile_accuracy": float(np.mean(
                    accuracy[:, config_id, divisor_id]
                )),
                "profile_chars": float(np.mean(chars[:, config_id, divisor_id])),
                "profile_utility": float(np.mean(split_utility)),
                "profile_utility_se": float(
                    np.std(split_utility, ddof=1) / np.sqrt(len(splits))
                ),
            })
    write_csv(output_dir / "coding_utility_tail_decay_profile.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(
        "algorithm/bestofn_coding/data.jsonl"
    ))
    parser.add_argument("--char-cache", type=Path, default=Path(
        "algorithm/bestofn_coding/practical_algorithm/coding_char_counts_83.npz"
    ))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-start", type=int, default=55)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--test-permutations", type=int, default=24)
    parser.add_argument("--train-permutations", type=int, default=4)
    parser.add_argument("--pilot-context-beta", type=float)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    problems = load_problems(args.data, args.char_cache)
    configs = uncapped_tail_decay_configs()
    splits = tuple(range(args.split_start, args.split_start + args.splits))
    for split in splits:
        path = args.output_dir / f"utility_tail_decay_{split}.npz"
        if path.exists():
            print(f"reusing {path}", flush=True)
            continue
        accuracy, chars = evaluate_split(split, problems, configs, args)
        np.savez_compressed(
            path, divisors=np.asarray(REPORT_DIVISORS),
            test_accuracy=accuracy, test_chars=chars,
        )
        print(f"wrote {path}", flush=True)
    build_profile(args.output_dir, splits, configs)
    print(
        f"wrote profile for {len(configs)} uncapped policies x "
        f"{len(REPORT_DIVISORS)} divisors",
        flush=True,
    )


if __name__ == "__main__":
    main()
