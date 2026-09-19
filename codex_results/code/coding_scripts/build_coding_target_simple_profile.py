"""Freeze the one-policy coding target profile from development results."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from coding_target_quality_distribution_calibrated import candidate_configs
from coding_ucb_three_objectives import write_csv


DEVELOPMENT_SPLITS = tuple(range(55, 60))


def build_profile(diagnostic_csv: Path) -> list[dict]:
    with diagnostic_csv.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    config = candidate_configs(simple_global_policy=True)[0]
    rows = [
        row for row in source
        if int(row["split"]) in DEVELOPMENT_SPLITS and int(row["config_id"]) == 0
    ]
    if {int(row["split"]) for row in rows} != set(DEVELOPMENT_SPLITS):
        raise ValueError("diagnostics must contain every development split 55--59")
    divisors = sorted({float(row["divisor"]) for row in rows})
    expected = len(DEVELOPMENT_SPLITS) * len(divisors)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} split/divisor rows, found {len(rows)}")

    output = []
    for divisor_id, divisor in enumerate(divisors):
        group = [row for row in rows if float(row["divisor"]) == divisor]
        accuracy = np.asarray([float(row["test_accuracy"]) for row in group])
        chars = np.asarray([float(row["test_chars"]) for row in group])
        if not np.all(np.isfinite(accuracy)) or not np.all(chars > 0.0):
            raise ValueError(f"non-finite development result for divisor {divisor}")
        output.append({
            "config_id": 0,
            "divisor_id": divisor_id,
            "divisor": divisor,
            **asdict(config),
            "development_splits": len(DEVELOPMENT_SPLITS),
            "profile_accuracy": float(np.mean(accuracy)),
            "profile_log_chars": float(np.mean(np.log(chars))),
            "profile_chars_geomean": float(np.exp(np.mean(np.log(chars)))),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_profile(args.diagnostic_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    print(f"wrote {len(rows)} simple-policy profile rows to {args.output}")


if __name__ == "__main__":
    main()
