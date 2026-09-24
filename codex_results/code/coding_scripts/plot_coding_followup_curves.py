"""Plot paired coding DMRL policies in the original cost-regime figure style.

The two adaptive runs share splits and permutations. Matched-quality Fixed-N
tokens are interpolated from a separately evaluated, dense frontier on those
same held-out splits. Only evaluated adaptive divisors are plotted.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


RESULT_ROOT = Path(__file__).resolve().parent.parent / "adaptive_coding_codescaler_qwen25_3b"


def read_summary(path: Path) -> dict[float, dict[str, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty summary: {path}")
    points: dict[float, dict[str, float]] = {}
    for row in rows:
        divisor = float(row["utility_divisor"])
        if divisor in points:
            raise ValueError(f"duplicate divisor {divisor:g} in {path}")
        points[divisor] = {
            key: float(row[key])
            for key in (
                "adaptive_accuracy",
                "adaptive_mean_output_tokens",
                "fixed_accuracy",
                "fixed_mean_output_tokens",
                "fixed_profit",
                "relative_profit_improvement_percent",
            )
        }
    return points


def read_fixed_frontier(path: Path) -> tuple[np.ndarray, np.ndarray, dict[float, tuple[float, float]]]:
    by_divisor: dict[float, list[tuple[float, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_divisor[float(row["utility_divisor"])].append(
                (float(row["fixed_accuracy"]), float(row["fixed_mean_output_tokens"]))
            )
    if not by_divisor:
        raise ValueError(f"empty Fixed-N frontier: {path}")
    means = {
        divisor: (
            float(np.mean([value[0] for value in values])),
            float(np.mean([value[1] for value in values])),
        )
        for divisor, values in by_divisor.items()
    }
    ordered = sorted((tokens, accuracy) for accuracy, tokens in means.values())
    efficient: list[tuple[float, float]] = []
    best_accuracy = -np.inf
    for tokens, accuracy in ordered:
        if accuracy > best_accuracy + 1e-15:
            efficient.append((accuracy, tokens))
            best_accuracy = accuracy
    if len(efficient) < 2:
        raise ValueError("Fixed-N frontier needs at least two quality levels")
    return (
        np.asarray([point[0] for point in efficient]),
        np.asarray([point[1] for point in efficient]),
        means,
    )


def build_rows(
    original: dict[float, dict[str, float]],
    candidate: dict[float, dict[str, float]],
    fixed_quality: np.ndarray,
    fixed_tokens: np.ndarray,
    fixed_means: dict[float, tuple[float, float]],
) -> list[dict[str, float]]:
    if set(original) != set(candidate):
        raise ValueError("adaptive runs must evaluate the same divisors")
    rows: list[dict[str, float]] = []
    for divisor in sorted(original):
        old, new = original[divisor], candidate[divisor]
        if divisor not in fixed_means:
            raise ValueError(f"frontier lacks evaluated divisor {divisor:g}")
        expected_accuracy, expected_tokens = fixed_means[divisor]
        for label, policy in (("original", old), ("candidate", new)):
            if not np.isclose(policy["fixed_accuracy"], expected_accuracy, rtol=0, atol=1e-12):
                raise ValueError(f"{label} Fixed-N accuracy differs at {divisor:g}")
            if not np.isclose(policy["fixed_mean_output_tokens"], expected_tokens, rtol=0, atol=1e-8):
                raise ValueError(f"{label} Fixed-N tokens differ at {divisor:g}")
            if not fixed_quality[0] <= policy["adaptive_accuracy"] <= fixed_quality[-1]:
                raise ValueError(f"{label} accuracy is outside Fixed-N frontier at {divisor:g}")
        if not np.isclose(old["fixed_profit"], new["fixed_profit"], rtol=0, atol=1e-12):
            raise ValueError(f"Fixed-N profit differs between runs at {divisor:g}")
        row = {"utility_divisor": divisor}
        for prefix, policy in (("original", old), ("candidate", new)):
            matched_tokens = float(np.interp(
                policy["adaptive_accuracy"], fixed_quality, fixed_tokens
            ))
            used_tokens = policy["adaptive_mean_output_tokens"]
            row[f"{prefix}_accuracy"] = policy["adaptive_accuracy"]
            row[f"{prefix}_output_tokens"] = used_tokens
            row[f"{prefix}_matched_fixed_tokens"] = matched_tokens
            row[f"{prefix}_matched_quality_saving_percent"] = (
                100.0 * (matched_tokens - used_tokens) / matched_tokens
            )
            row[f"{prefix}_profit_improvement_percent"] = (
                policy["relative_profit_improvement_percent"]
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_svg(
    path: Path,
    rows: list[dict[str, float]],
    *,
    metric: str,
    title: str,
    y_label: str,
    color: str,
    pale_color: str,
    note: str,
) -> None:
    width, height = 1000, 600
    left, right, top, bottom = 105, 35, 90, 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    x = np.asarray([row["utility_divisor"] for row in rows], dtype=float)
    old = np.asarray([row[f"original_{metric}"] for row in rows], dtype=float)
    new = np.asarray([row[f"candidate_{metric}"] for row in rows], dtype=float)
    if len(x) < 2:
        raise ValueError("a curve needs at least two evaluated divisors")
    log_x = np.log10(x)
    x_low, x_high = float(log_x.min()), float(log_x.max())
    y_low = min(0.0, float(old.min()), float(new.min()))
    y_high = max(0.0, float(old.max()), float(new.max()))
    padding = max(0.08 * (y_high - y_low), 0.08)
    y_low -= padding
    y_high += padding

    def px(value: float) -> float:
        return left + (np.log10(value) - x_low) / (x_high - x_low) * plot_width

    def py(value: float) -> float:
        return top + (y_high - value) / (y_high - y_low) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#17202a}'
        '.title{font-size:24px;font-weight:700}.axis{font-size:15px}'
        '.tick{font-size:12px;fill:#52616b}.note{font-size:12px;fill:#52616b}'
        '.legend{font-size:14px;font-weight:700}</style>',
        f'<title>{escape(title)}</title>',
        f'<text x="{width/2}" y="36" text-anchor="middle" class="title">{escape(title)}</text>',
        '<text x="500" y="59" text-anchor="middle" class="note">5 tested divisors · 10 paired outer splits · 48 held-out permutations per problem · seed 20261023</text>',
    ]
    for tick in np.linspace(y_low + padding, y_high - padding, 6):
        y_pos = py(float(tick))
        elements.extend([
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{width-right}" y2="{y_pos:.1f}" stroke="#e5e9ed"/>',
            f'<text x="{left-12}" y="{y_pos+4:.1f}" text-anchor="end" class="tick">{tick:+.1f}%</text>',
        ])
    zero_y = py(0.0)
    elements.append(
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" stroke="#59636e" stroke-width="1.5"/>'
    )
    for divisor in x:
        x_pos = px(float(divisor))
        label = f"{divisor/1000:g}k" if divisor < 1_000_000 else "1M"
        elements.extend([
            f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{height-bottom}" stroke="#f0f2f4"/>',
            f'<text x="{x_pos:.1f}" y="{height-bottom+25}" text-anchor="middle" class="tick">{label}</text>',
        ])
    for values, stroke, dashed, radius in (
        (old, pale_color, True, 3.5),
        (new, color, False, 4.5),
    ):
        points = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x, values))
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="3" stroke-linejoin="round"{dash}/>'
        )
        for divisor, value in zip(x, values):
            fill = "white" if dashed else stroke
            elements.append(
                f'<circle cx="{px(float(divisor)):.1f}" cy="{py(float(value)):.1f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            )
    elements.extend([
        f'<line x1="700" y1="105" x2="735" y2="105" stroke="{pale_color}" stroke-width="3" stroke-dasharray="7 5"/>',
        '<text x="745" y="110" class="legend">Original DMRL</text>',
        f'<line x1="700" y1="131" x2="735" y2="131" stroke="{color}" stroke-width="3"/>',
        '<text x="745" y="136" class="legend">Cost scale 1.25</text>',
        f'<text x="{width/2}" y="{height-49}" text-anchor="middle" class="axis">Utility divisor D (cost per output token = 1/D; log spacing)</text>',
        f'<text transform="translate(25 {top+plot_height/2:.1f}) rotate(-90)" text-anchor="middle" class="axis">{escape(y_label)}</text>',
        f'<text x="{width/2}" y="{height-17}" text-anchor="middle" class="note">{escape(note)}</text>',
        '</svg>',
    ])
    path.write_text("\n".join(elements) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original", type=Path,
        default=RESULT_ROOT / "dmrl_fresh_baseline_seed20261023/summary.csv",
    )
    parser.add_argument(
        "--candidate", type=Path,
        default=RESULT_ROOT / "dmrl_fresh_cost_scale_1p25_seed20261023/summary.csv",
    )
    parser.add_argument(
        "--fixed-frontier", type=Path,
        default=RESULT_ROOT / "dmrl_fresh_fixed_dense_seed20261023.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_quality, fixed_tokens, fixed_means = read_fixed_frontier(args.fixed_frontier)
    rows = build_rows(
        read_summary(args.original), read_summary(args.candidate),
        fixed_quality, fixed_tokens, fixed_means,
    )
    output_csv = args.output_dir / "followup_curves_seed20261023.csv"
    savings_svg = args.output_dir / "matched_quality_token_saving_followup_seed20261023.svg"
    profit_svg = args.output_dir / "profit_improvement_followup_seed20261023.svg"
    write_csv(output_csv, rows)
    draw_svg(
        savings_svg, rows,
        metric="matched_quality_saving_percent",
        title="Coding token saving at matched quality: fresh paired splits",
        y_label="Output-token saving versus matched-quality Fixed-N (%)",
        color="#117a65", pale_color="#8ab9ac",
        note="Fixed-N tokens interpolated on the dense efficient quality/token frontier from the same held-out splits.",
    )
    draw_svg(
        profit_svg, rows,
        metric="profit_improvement_percent",
        title="Coding profit improvement: fresh paired splits",
        y_label="Profit improvement over train-tuned Fixed-N (%)",
        color="#1769aa", pale_color="#91b4d6",
        note="Profit = selected correctness - output tokens / D; both policies use identical held-out orders.",
    )
    print(output_csv)
    print(savings_svg)
    print(profit_svg)


if __name__ == "__main__":
    main()
