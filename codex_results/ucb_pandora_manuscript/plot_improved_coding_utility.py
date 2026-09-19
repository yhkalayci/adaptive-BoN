"""Plot the frozen profiled coding utility-frontier result used in the paper."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/coding_utility_improved/utility_summary.csv"
OUTPUT = ROOT / "figures/coding/coding_utility_improvement_frontier"
COLOR = "#C45A00"
METHOD = "shifted_exponential_calibrated_probability"


def values(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def main() -> None:
    with SOURCE.open(newline="") as handle:
        rows = sorted(
            csv.DictReader(handle), key=lambda row: float(row["utility_divisor"])
        )

    x = values(rows, "utility_divisor")
    gain = values(rows, "relative_utility_gain_pct_mean")
    gap = values(rows, "utility_gap_mean")
    gap_low = values(rows, "utility_gap_ci_low")
    gap_high = values(rows, "utility_gap_ci_high")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), constrained_layout=True)

    axes[0].plot(x, gain, color=COLOR, marker="o", linewidth=2.2)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set(
        xlabel="Utility divisor $D$ (characters per utility unit)",
        ylabel="Mean relative utility gain over Fixed-$N$ (%)",
        title="A. Relative utility gain",
    )

    axes[1].plot(x, gap, color=COLOR, marker="o", linewidth=2.2)
    axes[1].fill_between(x, gap_low, gap_high, color=COLOR, alpha=0.18)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set(
        xlabel="Utility divisor $D$ (characters per utility unit)",
        ylabel="Additive utility gap",
        title="B. Additive gap with pointwise 95% intervals",
    )

    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.25)

    fig.suptitle("Coding: frozen profiled UCB\N{EN DASH}Pandora utility frontier", fontsize=14)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=220)
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    plt.close(fig)

    with (ROOT / "artifacts/coding_utility/equal_budget_summary.csv").open(
        newline=""
    ) as handle:
        budget = sorted(
            (row for row in csv.DictReader(handle) if row["method"] == METHOD),
            key=lambda row: float(row["divisor"]),
        )
    x = values(budget, "divisor")
    mean = values(budget, "accuracy_gap_pp_mean")
    low = values(budget, "accuracy_gap_pp_ci_low")
    high = values(budget, "accuracy_gap_pp_ci_high")
    fig, axis = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    axis.plot(x, mean, color=COLOR, marker="o", linewidth=2.2)
    axis.fill_between(x, low, high, color=COLOR, alpha=0.18)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xscale("log")
    axis.set(
        xlabel="Character-cost divisor",
        ylabel="Accuracy gain at equal expected characters (pp)",
        title="Coding equal-budget accuracy: retained exp-tail selector",
    )
    axis.grid(alpha=0.25)
    budget_output = ROOT / "figures/coding/coding_equal_budget_exp_tail"
    fig.savefig(budget_output.with_suffix(".png"), dpi=220)
    fig.savefig(budget_output.with_suffix(".pdf"))
    plt.close(fig)

    with (ROOT / "artifacts/coding_target/target_quality_summary.csv").open(
        newline=""
    ) as handle:
        target = sorted(
            (row for row in csv.DictReader(handle) if row["method"] == METHOD),
            key=lambda row: float(row["target_accuracy"]),
        )
    x = values(target, "target_accuracy")
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), constrained_layout=True)
    adaptive_color = "#C45A00"
    fixed_color = "#555555"

    for metric, color, marker, label in (
        ("adaptive_chars", adaptive_color, "o", "Adaptive UCB\N{EN DASH}Pandora"),
        ("train_fixed_test_chars", fixed_color, "s", "Held-out-selected Fixed-$N$"),
    ):
        axes[0].plot(
            x, values(target, f"{metric}_mean"), color=color,
            marker=marker, linewidth=2.2, label=label,
        )
        axes[0].fill_between(
            x, values(target, f"{metric}_ci_low"),
            values(target, f"{metric}_ci_high"), color=color, alpha=0.12,
        )
    axes[0].set(
        xlabel="Target accuracy on selection set",
        ylabel="Test output characters",
        title="A. Test character cost",
    )

    axes[1].plot(
        x, values(target, "aggregate_saving_vs_train_fixed_pct"),
        color=adaptive_color, marker="o", linewidth=2.2,
    )
    axes[1].fill_between(
        x, values(target, "aggregate_saving_vs_train_fixed_ci_low"),
        values(target, "aggregate_saving_vs_train_fixed_ci_high"),
        color=adaptive_color, alpha=0.12,
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set(
        xlabel="Target accuracy on selection set",
        ylabel="Adaptive character saving (%)",
        title="B. Saving vs held-out-selected Fixed-$N$",
    )

    for metric, color, marker, label in (
        ("adaptive_accuracy", adaptive_color, "o", "Adaptive UCB\N{EN DASH}Pandora"),
        ("train_fixed_test_accuracy", fixed_color, "s", "Held-out-selected Fixed-$N$"),
    ):
        axes[2].plot(
            x, values(target, f"{metric}_mean"), color=color,
            marker=marker, linewidth=2.2, label=label,
        )
        axes[2].fill_between(
            x, values(target, f"{metric}_ci_low"),
            values(target, f"{metric}_ci_high"), color=color, alpha=0.12,
        )
    axes[2].plot(
        x, x, color="black", linestyle="--", linewidth=1.0,
        label="Test accuracy = selection target",
    )
    axes[2].set(
        xlabel="Target accuracy on selection set",
        ylabel="Test accuracy",
        title="C. Test target transfer",
    )

    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Coding target quality: adaptive vs held-out-selected Fixed-$N$",
        fontsize=14,
    )
    target_output = ROOT / "figures/coding/coding_target_quality_exp_tail"
    fig.savefig(target_output.with_suffix(".png"), dpi=220)
    fig.savefig(target_output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
