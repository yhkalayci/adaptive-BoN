"""Build coding profit and matched-quality token-saving curves from saved runs.

This is a post-processing script.  It does not refit a profile, retune a
policy, or replay an evaluation.  Duplicate divisors across the saved sweeps
must agree.  Matched-quality Fixed-N token use is linearly interpolated on the
cost-efficient frontier of the aggregate Fixed-N quality/token operating
points produced by the same train-tuned experiments.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE.parent / "adaptive_coding_codescaler_qwen25_3b"
INPUT_DIRECTORIES = (
    "dmrl_top4_rawselect_capped_10splits_48perm",
    "dmrl_top4_dense_interval_10splits_48perm",
    "dmrl_top4_interval_boundaries_10splits_48perm",
)
OUTPUT_CSV = RESULT_ROOT / "cost_regime_curves.csv"
PROFIT_SVG = RESULT_ROOT / "profit_improvement_by_cost_regime.svg"
SAVING_SVG = RESULT_ROOT / "matched_quality_token_saving_by_cost_regime.svg"
REPORT_MIN_DIVISOR = 500_000.0
REPORT_MAX_DIVISOR = 1_000_000.0
REPORT_CSV = RESULT_ROOT / "cost_regime_curves_500k_1m.csv"
REPORT_PROFIT_SVG = RESULT_ROOT / "profit_improvement_500k_1m.svg"
REPORT_SAVING_SVG = RESULT_ROOT / "matched_quality_token_saving_500k_1m.svg"
REPORT_GENERATIONS_SVG = RESULT_ROOT / "generations_500k_1m.svg"


@dataclass(frozen=True)
class OperatingPoint:
    divisor: float
    adaptive_accuracy: float
    fixed_accuracy: float
    adaptive_tokens: float
    fixed_tokens: float
    adaptive_generations: float
    fixed_generations: float
    profit_delta: float
    relative_profit_improvement: float


def load_points(paths: Iterable[Path]) -> list[OperatingPoint]:
    """Load and merge cost sweeps, requiring duplicate divisors to agree."""
    points: dict[float, OperatingPoint] = {}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                point = OperatingPoint(
                    divisor=float(row["utility_divisor"]),
                    adaptive_accuracy=float(row["adaptive_accuracy"]),
                    fixed_accuracy=float(row["fixed_accuracy"]),
                    adaptive_tokens=float(row["adaptive_mean_output_tokens"]),
                    fixed_tokens=float(row["fixed_mean_output_tokens"]),
                    adaptive_generations=float(row["adaptive_mean_generations"]),
                    fixed_generations=float(row["fixed_mean_generations"]),
                    profit_delta=float(row["profit_delta"]),
                    relative_profit_improvement=float(
                        row["relative_profit_improvement_percent"]
                    ),
                )
                previous = points.get(point.divisor)
                if previous is not None:
                    left = np.asarray(list(previous.__dict__.values()), dtype=float)
                    right = np.asarray(list(point.__dict__.values()), dtype=float)
                    if not np.allclose(left, right, rtol=1e-10, atol=1e-12):
                        raise ValueError(
                            f"inconsistent duplicate divisor {point.divisor:g}"
                        )
                points[point.divisor] = point
    if not points:
        raise ValueError("no saved operating points found")
    return [points[key] for key in sorted(points)]


def fixed_quality_token_frontier(
    points: Iterable[OperatingPoint],
) -> tuple[np.ndarray, np.ndarray]:
    """Return nondominated Fixed-N quality/token points in increasing cost."""
    candidates = sorted((point.fixed_tokens, point.fixed_accuracy) for point in points)
    qualities: list[float] = []
    tokens: list[float] = []
    best_quality = -np.inf
    for token_count, quality in candidates:
        if quality > best_quality + 1e-15:
            qualities.append(quality)
            tokens.append(token_count)
            best_quality = quality
    if len(qualities) < 2:
        raise ValueError("Fixed-N frontier needs at least two quality levels")
    return np.asarray(qualities), np.asarray(tokens)


def curve_rows(points: list[OperatingPoint]) -> list[dict[str, float]]:
    fixed_quality, fixed_tokens = fixed_quality_token_frontier(points)
    rows: list[dict[str, float]] = []
    for point in points:
        if point.adaptive_accuracy > fixed_quality[-1]:
            raise ValueError(
                f"adaptive quality at divisor {point.divisor:g} lies outside "
                "the observed Fixed-N frontier"
            )
        matched_tokens = float(
            np.interp(point.adaptive_accuracy, fixed_quality, fixed_tokens)
        )
        saving = matched_tokens - point.adaptive_tokens
        rows.append(
            {
                "utility_divisor": point.divisor,
                "price_per_output_token": 1.0 / point.divisor,
                "adaptive_accuracy": point.adaptive_accuracy,
                "adaptive_mean_output_tokens": point.adaptive_tokens,
                "matched_quality_fixed_n_mean_output_tokens": matched_tokens,
                "matched_quality_token_saving": saving,
                "matched_quality_token_saving_percent": 100.0
                * saving
                / matched_tokens,
                "adaptive_mean_generations": point.adaptive_generations,
                "train_tuned_fixed_n_mean_generations": point.fixed_generations,
                "generation_reduction_percent": 100.0
                * (point.fixed_generations - point.adaptive_generations)
                / point.fixed_generations,
                "profit_delta": point.profit_delta,
                "relative_profit_improvement_percent": (
                    point.relative_profit_improvement
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_line_plot(
    path: Path,
    rows: list[dict[str, float]],
    value_key: str,
    title: str,
    y_label: str,
    color: str,
    note: str,
) -> None:
    width, height = 1000, 600
    left, right, top, bottom = 105, 35, 68, 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    x = np.asarray([row["utility_divisor"] for row in rows], dtype=float)
    y = np.asarray([row[value_key] for row in rows], dtype=float)
    log_x = np.log10(x)
    x_low, x_high = float(log_x.min()), float(log_x.max())
    y_low, y_high = min(float(y.min()), 0.0), max(float(y.max()), 0.0)
    y_pad = max(0.08 * (y_high - y_low), 0.08)
    y_low -= y_pad
    y_high += y_pad

    def px(value: float) -> float:
        return left + (np.log10(value) - x_low) / (x_high - x_low) * plot_width

    def py(value: float) -> float:
        return top + (y_high - value) / (y_high - y_low) * plot_height

    if x[0] >= 500_000 and x[-1] <= 1_000_000:
        x_ticks = [500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000]
    else:
        x_ticks = [25_000, 50_000, 100_000, 200_000, 400_000, 600_000,
                   800_000, 1_000_000, 1_200_000]
    x_ticks = [value for value in x_ticks if x[0] <= value <= x[-1]]
    y_ticks = np.linspace(y_low + y_pad, y_high - y_pad, 6)
    points = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x, y))
    best = int(np.argmax(y))

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#17202a}'
        '.title{font-size:25px;font-weight:700}.axis{font-size:15px}'
        '.tick{font-size:12px;fill:#52616b}.note{font-size:12px;fill:#52616b}'
        '.peak{font-size:13px;font-weight:700}</style>',
        f'<text x="{width / 2:.1f}" y="36" text-anchor="middle" class="title">{title}</text>',
    ]
    for tick in y_ticks:
        y_pos = py(float(tick))
        elements.extend(
            [
                f'<line x1="{left}" y1="{y_pos:.1f}" x2="{width-right}" y2="{y_pos:.1f}" stroke="#e5e9ed"/>',
                f'<text x="{left-12}" y="{y_pos+4:.1f}" text-anchor="end" class="tick">{tick:+.1f}%</text>',
            ]
        )
    zero_y = py(0.0)
    elements.append(
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" stroke="#59636e" stroke-width="1.5"/>'
    )
    for tick in x_ticks:
        x_pos = px(float(tick))
        label = f"{tick / 1_000:g}k" if tick < 1_000_000 else f"{tick / 1_000_000:g}M"
        elements.extend(
            [
                f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{height-bottom}" stroke="#f0f2f4"/>',
                f'<text x="{x_pos:.1f}" y="{height-bottom+25}" text-anchor="middle" class="tick">{label}</text>',
            ]
        )
    elements.extend(
        [
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>',
            *[
                f'<circle cx="{px(float(a)):.1f}" cy="{py(float(b)):.1f}" r="3.5" fill="{color}"/>'
                for a, b in zip(x, y)
            ],
            f'<circle cx="{px(float(x[best])):.1f}" cy="{py(float(y[best])):.1f}" r="6" fill="white" stroke="{color}" stroke-width="3"/>',
            f'<text x="{px(float(x[best])):.1f}" y="{py(float(y[best]))-13:.1f}" text-anchor="middle" class="peak">{y[best]:+.2f}% at {x[best]/1000:g}k</text>',
            f'<text x="{width/2:.1f}" y="{height-49}" text-anchor="middle" class="axis">Utility divisor D (cost per output token = 1/D; log spacing)</text>',
            f'<text transform="translate(25 {top+plot_height/2:.1f}) rotate(-90)" text-anchor="middle" class="axis">{y_label}</text>',
            f'<text x="{width/2:.1f}" y="{height-17}" text-anchor="middle" class="note">{note}</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(elements) + "\n")


def svg_generation_plot(path: Path, rows: list[dict[str, float]]) -> None:
    width, height = 1000, 600
    left, right, top, bottom = 105, 35, 68, 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    x = np.asarray([row["utility_divisor"] for row in rows], dtype=float)
    adaptive = np.asarray([row["adaptive_mean_generations"] for row in rows])
    fixed = np.asarray(
        [row["train_tuned_fixed_n_mean_generations"] for row in rows]
    )
    x_low, x_high = float(np.log10(x).min()), float(np.log10(x).max())
    y_low = 0.0
    y_high = float(max(adaptive.max(), fixed.max()) * 1.10)

    def px(value: float) -> float:
        return left + (np.log10(value) - x_low) / (x_high - x_low) * plot_width

    def py(value: float) -> float:
        return top + (y_high - value) / (y_high - y_low) * plot_height

    x_ticks = [500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000]
    y_ticks = np.linspace(0.0, y_high, 7)
    adaptive_points = " ".join(
        f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x, adaptive)
    )
    fixed_points = " ".join(
        f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x, fixed)
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#17202a}'
        '.title{font-size:25px;font-weight:700}.axis{font-size:15px}'
        '.tick{font-size:12px;fill:#52616b}.note{font-size:12px;fill:#52616b}'
        '.legend{font-size:14px;font-weight:700}</style>',
        f'<text x="{width/2:.1f}" y="36" text-anchor="middle" class="title">Coding generations across utility divisors 500k–1M</text>',
    ]
    for tick in y_ticks:
        y_pos = py(float(tick))
        elements.extend([
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{width-right}" y2="{y_pos:.1f}" stroke="#e5e9ed"/>',
            f'<text x="{left-12}" y="{y_pos+4:.1f}" text-anchor="end" class="tick">{tick:.0f}</text>',
        ])
    for tick in x_ticks:
        x_pos = px(float(tick))
        label = f"{tick/1000:g}k" if tick < 1_000_000 else "1M"
        elements.extend([
            f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{height-bottom}" stroke="#f0f2f4"/>',
            f'<text x="{x_pos:.1f}" y="{height-bottom+25}" text-anchor="middle" class="tick">{label}</text>',
        ])
    elements.extend([
        f'<polyline points="{fixed_points}" fill="none" stroke="#9a6700" stroke-width="3" stroke-linejoin="round"/>',
        f'<polyline points="{adaptive_points}" fill="none" stroke="#1769aa" stroke-width="3" stroke-linejoin="round"/>',
        *[
            f'<circle cx="{px(float(a)):.1f}" cy="{py(float(b)):.1f}" r="3.5" fill="#9a6700"/>'
            for a, b in zip(x, fixed)
        ],
        *[
            f'<circle cx="{px(float(a)):.1f}" cy="{py(float(b)):.1f}" r="3.5" fill="#1769aa"/>'
            for a, b in zip(x, adaptive)
        ],
        '<line x1="690" y1="82" x2="725" y2="82" stroke="#1769aa" stroke-width="3"/>',
        '<text x="735" y="87" class="legend">Adaptive DMRL</text>',
        '<line x1="690" y1="108" x2="725" y2="108" stroke="#9a6700" stroke-width="3"/>',
        '<text x="735" y="113" class="legend">Train-tuned Fixed-N</text>',
        f'<text x="{width/2:.1f}" y="{height-49}" text-anchor="middle" class="axis">Utility divisor D (cost per output token = 1/D; log spacing)</text>',
        f'<text transform="translate(25 {top+plot_height/2:.1f}) rotate(-90)" text-anchor="middle" class="axis">Mean generations per evaluation request</text>',
        f'<text x="{width/2:.1f}" y="{height-17}" text-anchor="middle" class="note">Means aggregate 10 outer splits and 48 held-out permutations per problem.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(elements) + "\n")


def main() -> None:
    inputs = [RESULT_ROOT / directory / "summary.csv" for directory in INPUT_DIRECTORIES]
    points = load_points(inputs)
    rows = curve_rows(points)
    write_csv(OUTPUT_CSV, rows)
    svg_line_plot(
        PROFIT_SVG,
        rows,
        "relative_profit_improvement_percent",
        "Coding profit improvement across cost regimes",
        "Profit improvement over train-tuned Fixed-N (%)",
        "#1769aa",
        "Profit = selected correctness - output tokens / D; 10 outer splits and 48 held-out permutations/problem.",
    )
    svg_line_plot(
        SAVING_SVG,
        rows,
        "matched_quality_token_saving_percent",
        "Coding token saving at matched quality across cost regimes",
        "Output-token saving versus matched-quality Fixed-N (%)",
        "#117a65",
        "Fixed-N tokens are interpolated on the efficient aggregate frontier; targets below its first point use that same-or-better endpoint.",
    )
    report_rows = [
        row for row in rows
        if REPORT_MIN_DIVISOR <= row["utility_divisor"] <= REPORT_MAX_DIVISOR
    ]
    write_csv(REPORT_CSV, report_rows)
    svg_line_plot(
        REPORT_PROFIT_SVG,
        report_rows,
        "relative_profit_improvement_percent",
        "Coding profit improvement: utility divisors 500k–1M",
        "Profit improvement over train-tuned Fixed-N (%)",
        "#1769aa",
        "Profit = selected correctness - output tokens / D; 10 outer splits and 48 held-out permutations/problem.",
    )
    svg_line_plot(
        REPORT_SAVING_SVG,
        report_rows,
        "matched_quality_token_saving_percent",
        "Coding matched-quality token saving: utility divisors 500k–1M",
        "Output-token saving versus matched-quality Fixed-N (%)",
        "#117a65",
        "Fixed-N tokens are interpolated on the efficient aggregate quality/token frontier from all evaluated regimes.",
    )
    svg_generation_plot(REPORT_GENERATIONS_SVG, report_rows)
    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {PROFIT_SVG}")
    print(f"wrote {SAVING_SVG}")
    print(f"wrote {REPORT_CSV}")
    print(f"wrote {REPORT_PROFIT_SVG}")
    print(f"wrote {REPORT_SAVING_SVG}")
    print(f"wrote {REPORT_GENERATIONS_SVG}")


if __name__ == "__main__":
    main()
