"""Plot per-problem mean response-character distributions for three tasks."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_SPECS = (
    (
        "alpaca",
        "Alpaca",
        Path("dataset/alpaca/llama3.1_8b_output.merged_rm.jsonl.gz"),
    ),
    (
        "rlhf",
        "HH-RLHF",
        Path("dataset/hh_rlhf/llama3.1_8b_output.merged_rm.jsonl.gz"),
    ),
)
CODING_CACHE = Path(
    "algorithm/bestofn_coding/practical_algorithm/coding_char_counts_83.npz"
)
COLORS = {
    "alpaca": "#3A6EA5",
    "rlhf": "#C45A00",
    "coding": "#1B7F3A",
}


def load_alignment_means(path: Path):
    """Return problem IDs, mean response characters, and response counts."""
    problem_ids, means, counts = [], [], []
    seen = set()
    with gzip.open(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            problem_id = str(record.get("JSON_idx", line_number - 1))
            if problem_id in seen:
                raise ValueError(f"duplicate problem ID {problem_id} in {path}")
            seen.add(problem_id)
            responses = record.get("generations", [])
            if not responses:
                raise ValueError(f"problem {problem_id} has no responses in {path}")
            lengths = np.asarray(
                [len(str(response["text"])) for response in responses],
                dtype=np.float64,
            )
            problem_ids.append(problem_id)
            means.append(float(np.mean(lengths)))
            counts.append(len(lengths))
    if not problem_ids:
        raise ValueError(f"no problems found in {path}")
    return problem_ids, np.asarray(means), np.asarray(counts, dtype=np.int64)


def load_coding_means(path: Path):
    """Load the exact 83-problem coding character-count cache."""
    with np.load(path, allow_pickle=False) as data:
        problem_ids = [str(value) for value in data["ids"].tolist()]
        characters = np.asarray(data["chars"], dtype=np.float64)
    if characters.ndim != 2 or characters.shape[0] != len(problem_ids):
        raise ValueError("coding cache IDs and character matrix do not match")
    if np.any(~np.isfinite(characters)) or np.any(characters < 0.0):
        raise ValueError("coding character counts must be finite and nonnegative")
    counts = np.full(len(problem_ids), characters.shape[1], dtype=np.int64)
    return problem_ids, characters.mean(axis=1), counts


def write_csv(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_row(task, label, values, counts):
    return {
        "task": task,
        "label": label,
        "problems": len(values),
        "responses_per_problem_min": int(np.min(counts)),
        "responses_per_problem_max": int(np.max(counts)),
        "mean_of_problem_means": float(np.mean(values)),
        "std_of_problem_means": float(np.std(values, ddof=1)),
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }


def plot_distributions(data, output: Path):
    fig, axes = plt.subplots(
        1, 3, figsize=(15.6, 4.8), sharey=True, constrained_layout=True
    )
    for ax, (task, label, values, counts) in zip(axes, data):
        color = COLORS[task]
        # Separate x ranges preserve the shape of each task distribution;
        # identical bin counts and a shared y axis retain comparability.
        edges = np.linspace(float(values.min()), float(values.max()), 16)
        ax.hist(values, bins=edges, color=color, alpha=0.78,
                edgecolor="white", linewidth=0.8)
        mean = float(np.mean(values))
        median = float(np.median(values))
        ax.axvline(mean, color="#222222", linewidth=1.8,
                   label=f"Mean: {mean:,.0f}")
        ax.axvline(median, color="#222222", linewidth=1.5, linestyle="--",
                   label=f"Median: {median:,.0f}")
        ax.set_title(
            f"{label}\n{len(values)} problems × "
            f"{int(np.min(counts)):,} responses"
        )
        ax.set_xlabel("Mean response characters per problem")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Number of problems")
    fig.suptitle(
        "Distribution of per-problem average response length",
        fontsize=15,
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("codex_results/results/distribution_check"),
    )
    parser.add_argument("--coding-cache", type=Path, default=CODING_CACHE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = []
    detail_rows = []
    summary_rows = []
    for task, label, path in TASK_SPECS:
        problem_ids, means, counts = load_alignment_means(path)
        data.append((task, f"{label} (Llama-3.1-8B)", means, counts))
        summary_rows.append(summary_row(task, label, means, counts))
        detail_rows.extend({
            "task": task,
            "problem_id": problem_id,
            "responses": int(count),
            "mean_response_characters": float(mean),
            "generator": "llama3.1_8b",
        } for problem_id, mean, count in zip(problem_ids, means, counts))

    problem_ids, means, counts = load_coding_means(args.coding_cache)
    data.append(("coding", "Coding", means, counts))
    summary_rows.append(summary_row("coding", "Coding", means, counts))
    detail_rows.extend({
        "task": "coding",
        "problem_id": problem_id,
        "responses": int(count),
        "mean_response_characters": float(mean),
        "generator": "coding_experiment_cache",
    } for problem_id, mean, count in zip(problem_ids, means, counts))

    write_csv(args.output_dir / "per_problem_mean_response_characters.csv", detail_rows)
    write_csv(args.output_dir / "response_character_summary.csv", summary_rows)
    plot_distributions(
        data, args.output_dir / "average_response_character_histograms.png"
    )
    (args.output_dir / "METHOD.json").write_text(json.dumps({
        "statistic": (
            "for each problem, arithmetic mean of Python len(response text) "
            "over all available responses"
        ),
        "layout": "1x3 histograms with 15 task-specific equal-width bins",
        "alignment_generator": "llama3.1_8b",
        "alpaca_source": str(TASK_SPECS[0][2]),
        "rlhf_source": str(TASK_SPECS[1][2]),
        "coding_source": str(args.coding_cache),
        "coding_cohort": "83 solvable problems used by the coding experiments",
    }, indent=2) + "\n")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "problems": {task: len(values) for task, _, values, _ in data},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
