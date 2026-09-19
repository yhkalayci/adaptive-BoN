"""Build the frozen target-policy profile from development diagnostics."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from coding_target_quality_distribution_calibrated import candidate_configs
from coding_ucb_three_objectives import write_csv


DEVELOPMENT_SPLITS = tuple(range(55, 60))


def build_profile(diagnostic_dir: Path):
    configs = candidate_configs(True, True)
    pieces = []
    divisors = None
    for split in DEVELOPMENT_SPLITS:
        path = diagnostic_dir / f"decay_diagnostic_{split}.npz"
        with np.load(path, allow_pickle=False) as data:
            current_divisors = np.asarray(data["divisors"], dtype=np.float64)
            test_accuracy = np.asarray(data["test_accuracy"], dtype=np.float64)
            test_chars = np.asarray(data["test_chars"], dtype=np.float64)
        expected = (len(configs), len(current_divisors))
        if test_accuracy.shape != expected or test_chars.shape != expected:
            raise ValueError(f"unexpected diagnostic shape in {path}")
        if divisors is None:
            divisors = current_divisors
        elif not np.array_equal(divisors, current_divisors):
            raise ValueError(f"divisor mismatch in {path}")
        if not np.all(np.isfinite(test_accuracy)) or not np.all(test_chars > 0):
            raise ValueError(f"non-finite development result in {path}")
        pieces.append((test_accuracy, test_chars))

    accuracy = np.mean([item[0] for item in pieces], axis=0)
    log_chars = np.mean([np.log(item[1]) for item in pieces], axis=0)
    rows = []
    for config_id, config in enumerate(configs):
        for divisor_id, divisor in enumerate(divisors):
            rows.append({
                "config_id": config_id,
                "divisor_id": divisor_id,
                "divisor": float(divisor),
                **asdict(config),
                "development_splits": len(DEVELOPMENT_SPLITS),
                "profile_accuracy": float(accuracy[config_id, divisor_id]),
                "profile_log_chars": float(log_chars[config_id, divisor_id]),
                "profile_chars_geomean": float(
                    np.exp(log_chars[config_id, divisor_id])
                ),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_profile(args.diagnostic_dir)
    write_csv(args.output, rows)
    print(f"wrote {len(rows)} profile rows to {args.output}")


if __name__ == "__main__":
    main()
