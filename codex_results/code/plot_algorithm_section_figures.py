"""Create the coding figure used by ``algorithm_section.tex``.

The retained coding plots compare several distribution families.  The paper
section intentionally discusses only ``shifted_exponential_calibrated_probability``,
so this script redraws only that family's four relevant held-out summaries.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHOD = "shifted_exponential_calibrated_probability"
COLOR = "#C45A00"


def read_method(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["method"] == METHOD]


def values(rows, field):
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def line_with_band(ax, x, mean, low, high):
    ax.plot(x, mean, color=COLOR, marker="o", linewidth=2.2)
    ax.fill_between(x, low, high, color=COLOR, alpha=0.16)


def main() -> None:
    utility = sorted(
        read_method(ROOT / "results/coding/utility_gap/utility_summary.csv"),
        key=lambda row: float(row["divisor"]),
    )
    budget = sorted(
        read_method(ROOT / "results/coding/utility_gap/equal_budget_summary.csv"),
        key=lambda row: float(row["divisor"]),
    )
    target = sorted(
        read_method(ROOT / "results/coding/target_quality/target_quality_summary.csv"),
        key=lambda row: float(row["target_accuracy"]),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), constrained_layout=True)

    x = values(utility, "divisor")
    line_with_band(
        axes[0, 0], x,
        values(utility, "relative_utility_gain_pct_mean"),
        values(utility, "relative_utility_gain_pct_ci_low"),
        values(utility, "relative_utility_gain_pct_ci_high"),
    )
    axes[0, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set(
        xlabel="Character-cost divisor",
        ylabel="Utility gain over train-tuned Fixed-N (%)",
        title="A. Utility objective",
    )

    x = values(budget, "divisor")
    line_with_band(
        axes[0, 1], x,
        values(budget, "accuracy_gap_pp_mean"),
        values(budget, "accuracy_gap_pp_ci_low"),
        values(budget, "accuracy_gap_pp_ci_high"),
    )
    axes[0, 1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set(
        xlabel="Character-cost divisor",
        ylabel="Accuracy gain at equal expected characters (pp)",
        title="B. Equal-budget accuracy",
    )

    x = values(target, "target_accuracy")
    line_with_band(
        axes[1, 0], x,
        values(target, "aggregate_saving_vs_oracle_fixed_pct"),
        values(target, "aggregate_saving_vs_oracle_fixed_ci_low"),
        values(target, "aggregate_saving_vs_oracle_fixed_ci_high"),
    )
    axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set(
        xlabel="Requested accuracy",
        ylabel="Character saving vs matched held-out Fixed-N (%)",
        title="C. Target-quality cost",
    )

    line_with_band(
        axes[1, 1], x,
        values(target, "adaptive_accuracy_mean"),
        values(target, "adaptive_accuracy_ci_low"),
        values(target, "adaptive_accuracy_ci_high"),
    )
    axes[1, 1].plot(x, x, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].set(
        xlabel="Requested accuracy",
        ylabel="Held-out achieved accuracy",
        title="D. Target tracking",
    )

    for ax in axes.flat:
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Coding: shifted-exponential Pandora on isotonic correctness probability",
        fontsize=14,
    )
    output = ROOT / "results/algorithm_section"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "coding_shifted_exponential_overview.png", dpi=200)
    fig.savefig(output / "coding_shifted_exponential_overview.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
