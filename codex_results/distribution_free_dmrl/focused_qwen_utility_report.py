"""Qwen-only alignment tables and all-generator relative-utility figures.

Uses the completed token-count replay arrays. No responses are regenerated.
Every fixed N is selected on training prompts for its generator, split, reward
model, and price. Bootstrap weights are shared across overlapping splits and
generators, and intervals condition on the frozen fixed-N selections.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_refinement import interpolated_cost


MODELS = ("gemma3_4b", "granite42_8b", "ministral3_8b", "qwen35_9b")
REWARDS = ("fsfairx_llama3_rm", "rm_mistral_7b")
SELECTED_MODEL = "qwen35_9b"
SEEDS = (71, 72, 73, 74, 75)
LABELS = {"gemma3_4b": "Gemma 3 4B", "granite42_8b": "Granite 4.2 8B",
          "ministral3_8b": "Ministral 3 8B", "qwen35_9b": "Qwen 3.5 9B"}
REWARD_LABELS = {"fsfairx_llama3_rm": "FSFairX Llama 3 RM",
                 "rm_mistral_7b": "RM Mistral 7B"}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
BOOTSTRAP_SEED = 20260925


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_pair(adaptive, fixed, prices, weights, selected):
    """Return point/boot utility percentages and selected-model split metrics."""
    boot = np.zeros((len(weights), len(prices)))
    point = np.zeros((len(SEEDS), len(prices)))
    selected_rows = []
    for si, seed in enumerate(SEEDS):
        train, test = np.array_split(np.random.default_rng(seed).permutation(len(fixed)), 2)
        training_fixed = fixed[train].mean(axis=0)
        fixed_indices = np.argmax(
            training_fixed[:, 0, None] - training_fixed[:, 1, None] * prices[None, :] / 1e6,
            axis=0,
        )
        aq = adaptive[test, :, 0]
        at = adaptive[test, :, 1]
        fq = fixed[test][:, fixed_indices, 0]
        ft = fixed[test][:, fixed_indices, 1]
        aq_mean, fq_mean = aq.mean(axis=0), fq.mean(axis=0)
        point[si] = 100 * (aq_mean - fq_mean) / fq_mean
        w = weights[:, test].copy()
        w /= w.sum(axis=1, keepdims=True)
        boot += 100 * ((w @ aq) - (w @ fq)) / (w @ fq)
        if selected:
            for pi, price_per_million in enumerate(prices):
                price = price_per_million / 1e6
                q_a, q_f = float(aq_mean[pi]), float(fq_mean[pi])
                t_a, t_f = float(at[:, pi].mean()), float(ft[:, pi].mean())
                profit_a, profit_f = q_a - price * t_a, q_f - price * t_f
                matched_tokens = float(interpolated_cost(
                    fixed[test, :, 0].mean(axis=0),
                    fixed[test, :, 1].mean(axis=0), np.array([q_a]))[0])
                selected_rows.append(dict(split_seed=seed, price_per_million=float(price_per_million),
                                          fixed_n=int(fixed_indices[pi]) + 1,
                                          fixed_utility=q_f, adaptive_utility=q_a,
                                          relative_utility_improvement_percent=float(point[si, pi]),
                                          fixed_output_tokens=t_f, adaptive_output_tokens=t_a,
                                          fixed_cost_dollars=price * t_f,
                                          adaptive_cost_dollars=price * t_a,
                                          fixed_profit=profit_f, adaptive_profit=profit_a,
                                          profit_gain=profit_a - profit_f,
                                          direct_cost_saving_percent=100 * (1 - t_a / t_f),
                                          matched_utility_cost_saving_percent=100 * (1 - t_a / matched_tokens)))
    boot /= len(SEEDS)
    lower, upper = np.quantile(boot, [.025, .975], axis=0)
    return point.mean(axis=0), lower, upper, selected_rows


def make_tables(split_rows, prices):
    tables = {}
    for reward in REWARDS:
        tables[reward] = []
        for price in prices:
            group = [r for r in split_rows if r["reward_model"] == reward
                     and r["price_per_million"] == float(price)]
            assert len(group) == len(SEEDS)
            row = dict(reward_model=reward, generator=SELECTED_MODEL,
                       price_per_million=float(price))
            for key in ("fixed_n", "fixed_utility", "adaptive_utility",
                        "relative_utility_improvement_percent", "fixed_output_tokens",
                        "adaptive_output_tokens", "fixed_cost_dollars", "adaptive_cost_dollars",
                        "fixed_profit", "adaptive_profit", "profit_gain",
                        "direct_cost_saving_percent", "matched_utility_cost_saving_percent"):
                row[key] = float(np.mean([r[key] for r in group]))
            tables[reward].append(row)
    return tables


def draw_figures(intervals, prices, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "savefig.dpi": 220})
    for reward in REWARDS:
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        for mi, model in enumerate(MODELS):
            group = sorted((r for r in intervals if r["reward_model"] == reward
                            and r["model"] == model), key=lambda r: r["price_per_million"])
            x = np.array([r["price_per_million"] for r in group])
            y = np.array([r["relative_utility_improvement_percent"] for r in group])
            lo = np.array([r["ci_low"] for r in group])
            hi = np.array([r["ci_high"] for r in group])
            ax.plot(x, y, marker="o", linewidth=2, color=COLORS[mi], label=LABELS[model])
            ax.fill_between(x, lo, hi, color=COLORS[mi], alpha=.14, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(prices, [f"{p:g}" for p in prices])
        ax.axhline(0, color="#555555", linewidth=.9)
        ax.grid(axis="y", color="#dce2e8", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Generation price ($ per million output tokens)")
        ax.set_ylabel("Adaptive utility change vs fixed N (%)")
        ax.set_title(REWARD_LABELS[reward], fontsize=15, pad=12)
        ax.legend(ncol=2, frameon=False, title="Generator", loc="best")
        fig.subplots_adjust(left=.11, right=.97, top=.90, bottom=.18)
        fig.text(.11, .04, "Bands: pointwise 95% paired prompt bootstrap; five splits, eight replay orders per prompt.",
                 fontsize=8.5, color="#444444")
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{reward}_relative_utility.{extension}", facecolor="white")
        plt.close(fig)


def write_report(tables, output):
    lines = ["# Focused token-count alignment comparison", "",
             "Generator in both tables: **Qwen 3.5 9B**. Reward models: FSFairX and RM-Mistral. "
             "Prices: $0.1–$10 per million recorded output tokens.", "",
             "Qwen was chosen after inspecting the earlier four-generator results: it was the "
             "most frequent training-selected generator across these reward models and prices "
             "and had higher average adaptive profit than Ministral. This choice is exploratory.", "",
             "Fixed N is selected on training prompts separately for each split, reward model, and price. "
             "Entries average five held-out split results. Utility is sigmoid(selected reward minus "
             "the prompt's full-pool reward q99). Cost is generation output tokens times the price; "
             "reward scoring cost is excluded. Profit is utility minus generation cost. "
             "Prices are illustrative and utility is a reward-model proxy.", "",
             "The figures keep all four generators and plot the mean splitwise relative utility "
             "change, 100 × (adaptive utility − fixed-N utility) / fixed-N utility. Bands are "
             "pointwise paired prompt-bootstrap intervals, conditional on selected fixed Ns and "
             "the exploratory generator choice.", ""]
    for reward in REWARDS:
        lines += [f"## {REWARD_LABELS[reward]}", "",
                  "| $/M tokens | Fixed N | Fixed utility | Adaptive utility | Relative utility | Fixed cost ($) | Adaptive cost ($) | Fixed profit | Adaptive profit | Profit gain ($) | Direct cost saving | Matched-utility saving |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in tables[reward]:
            lines.append("| " + " | ".join((
                f"{r['price_per_million']:g}", f"{r['fixed_n']:.1f}",
                f"{r['fixed_utility']:.4f}", f"{r['adaptive_utility']:.4f}",
                f"{r['relative_utility_improvement_percent']:+.2f}%",
                f"{r['fixed_cost_dollars']:.5f}", f"{r['adaptive_cost_dollars']:.5f}",
                f"{r['fixed_profit']:.4f}", f"{r['adaptive_profit']:.4f}",
                f"{r['profit_gain']:+.5f}", f"{r['direct_cost_saving_percent']:+.1f}%",
                f"{r['matched_utility_cost_saving_percent']:+.1f}%")) + " |")
        lines += ["", f"![{REWARD_LABELS[reward]} relative utility]({reward}_relative_utility.png)", ""]
    lines += ["Direct cost saving compares the paid token cost at each method's attained utility. "
              "Matched-utility saving uses a retrospective fixed-N mixture at adaptive attained "
              "utility and is a diagnostic rather than a deployable policy.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output directory")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    source = json.loads((args.study / "METHOD.json").read_text())
    all_prices = np.array(source["prices_per_million_output_tokens"], dtype=float)
    indices = np.flatnonzero((all_prices >= .1) & (all_prices <= 10))
    prices = all_prices[indices]
    np.testing.assert_array_equal(prices, [.1, .2, .5, 1., 2., 5., 10.])
    arrays = {}
    ids = None
    hashes = {}
    for reward in REWARDS:
        for model in MODELS:
            path = args.study / f"{model}__{reward}.npz"
            with np.load(path) as data:
                if ids is None:
                    ids = data["ids"]
                else:
                    np.testing.assert_array_equal(ids, data["ids"])
                arrays[reward, model] = (data["adaptive"][:, indices, :], data["fixed"])
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(ids) != 100:
        raise ValueError("Expected 100 shared prompts")
    args.output.mkdir(parents=True)
    weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        100, np.full(100, .01), size=args.bootstrap).astype(float)
    intervals, split_rows = [], []
    for reward in REWARDS:
        for model in MODELS:
            mean, low, high, rows = evaluate_pair(*arrays[reward, model], prices, weights,
                                                   selected=model == SELECTED_MODEL)
            for pi, price in enumerate(prices):
                intervals.append(dict(reward_model=reward, model=model,
                                      price_per_million=float(price),
                                      relative_utility_improvement_percent=float(mean[pi]),
                                      ci_low=float(low[pi]), ci_high=float(high[pi]),
                                      confidence=.95, bootstrap_repetitions=args.bootstrap))
            for row in rows:
                row.update(reward_model=reward, generator=model)
            split_rows.extend(rows)
    tables = make_tables(split_rows, prices)
    write_csv(args.output / "relative_utility_intervals.csv", intervals)
    write_csv(args.output / "qwen_split_results.csv", split_rows)
    for reward in REWARDS:
        write_csv(args.output / f"table_{reward}.csv", tables[reward])
    draw_figures(intervals, prices, args.output)
    write_report(tables, args.output)
    method = dict(source_study=str(args.study.resolve()), source_cache_sha256=hashes,
                  generator_models_in_figures=MODELS, generator_in_tables=SELECTED_MODEL,
                  reward_models=REWARDS, prices_per_million_output_tokens=prices.tolist(),
                  split_seeds=SEEDS, bootstrap_seed=BOOTSTRAP_SEED,
                  bootstrap_repetitions=args.bootstrap,
                  generator_choice="Exploratory selection after inspecting earlier four-generator tables; Qwen had higher average adaptive profit than Ministral over these reward models/prices",
                  fixed_n="selected by training profit separately per split/model/reward/price",
                  relative_utility="mean over splits of 100*(mean adaptive utility - mean fixed-N utility)/mean fixed-N utility",
                  intervals="pointwise paired prompt bootstrap, shared weights across splits and generators; conditional on fixed N and generator choice",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "METHOD.json").write_text(json.dumps(method, indent=2) + "\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
