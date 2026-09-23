"""Plot held-out fixed-N profit curves and adaptive-policy operating points."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(result_dir):
    with (result_dir / "summary.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["partition"] == "test"]
    with (result_dir / "comparisons.csv").open() as handle:
        comparisons = list(csv.DictReader(handle))
    prices = sorted({float(row["price"]) for row in rows})
    fig, axes = plt.subplots((len(prices) + 2) // 3, 3, figsize=(12, 6), squeeze=False)
    styles = {"dmrl_doubling": ("Doubling checkpoints", "o", "#D55E00"),
              "dmrl_sequential": ("Every response", "s", "#0072B2")}
    for ax, price in zip(axes.flat, prices):
        selected = [row for row in rows if float(row["price"]) == price]
        fixed = sorted((row for row in selected if row["method"].startswith("fixed_")),
                       key=lambda row: int(row["method"].split("_")[1]))
        ax.plot([float(row["samples"]) for row in fixed],
                [float(row["profit"]) for row in fixed], color="0.4", label="Fixed N")
        chosen = next(row["train_selected_fixed"] for row in comparisons
                      if float(row["price"]) == price)
        baseline = next(row for row in fixed if row["method"] == chosen)
        ax.scatter(float(baseline["samples"]), float(baseline["profit"]), marker="*",
                   s=100, color="black", label="Training-selected N", zorder=4)
        for method, (label, marker, color) in styles.items():
            row = next(row for row in selected if row["method"] == method)
            ax.scatter(float(row["samples"]), float(row["profit"]), marker=marker,
                       color=color, label=label, zorder=4)
        ax.axhline(0, color="0.8", linewidth=.7)
        ax.set(xscale="log", xlabel="Mean responses generated", ylabel="Mean test profit ($)",
               title=f"Price = {price:g} per recorded length unit")
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(prices):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(f"{result_dir.parent.name}: cached alignment replay ({result_dir.name})")
    fig.tight_layout(rect=(0, .06, 1, .94))
    output = result_dir / "profit_vs_fixed_n.png"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    for result_dir in args.results:
        plot(result_dir)
