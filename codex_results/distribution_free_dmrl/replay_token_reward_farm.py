"""Replay the frozen adaptive alignment policy on the reward farm's token data.

All stopping statistics use the observed prefix. The full-pool reward 99th
percentile is used only to evaluate the selected response, as in the prior
alignment replay. Fixed N is chosen on training prompts for each split/price.
"""
from __future__ import annotations

import argparse
from bisect import insort
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_refinement import interpolated_cost


MODELS = ("gemma3_4b", "granite42_8b", "ministral3_8b", "qwen35_9b")
REWARDS = ("armorm_llama3_8b", "fsfairx_llama3_rm", "rm_mistral_7b", "skywork_llama31_8b")
PRICES_PER_MILLION = np.array([0.02, 0.1, 0.2, 1.0, 2.0, 10.0])
SPLIT_SEEDS = (71, 72, 73, 74, 75)
REPLAY_SEED = 20260923
PERMUTATIONS = 8
CAP = 960


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def load_model(root: Path, model: str):
    paths = sorted((root / model / "alpaca").glob("*.json"))
    if len(paths) != 100 or [p.stem for p in paths] != [f"{i:05d}" for i in range(100)]:
        raise ValueError(f"Expected 100 numbered records for {model}, found {len(paths)}")
    rewards = np.empty((100, CAP, len(REWARDS)), dtype=np.float64)
    lengths = np.empty((100, CAP), dtype=np.float64)
    prompt_ids = []
    signatures = set()
    for i, path in enumerate(paths):
        row = json.loads(path.read_text())
        samples = row["generations"]
        if len(samples) != CAP or [s["idx"] for s in samples] != list(range(CAP)):
            raise ValueError(f"Missing or reordered samples: {path}")
        if row.get("reward_suite") is None:
            raise ValueError(f"Incomplete reward suite: {path}")
        prompt_ids.append(str(row["JSON_idx"]))
        signatures.add(row["reward_config_signature"])
        lengths[i] = [s["output_tokens"] for s in samples]
        rewards[i] = [[s["rewards"][key] for key in REWARDS] for s in samples]
    if len(set(prompt_ids)) != 100 or len(signatures) != 1:
        raise ValueError(f"Duplicate prompts or mixed reward configurations: {model}")
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(lengths)):
        raise ValueError(f"Nonfinite reward or token length: {model}")
    if np.any(lengths <= 0) or np.any(lengths != np.floor(lengths)):
        raise ValueError(f"Invalid output-token count: {model}")
    return prompt_ids, rewards, lengths, signatures.pop()


def prefix_thresholds(rewards: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Break-even price per output token for each eligible prefix.

    This is the mean_costse2 policy in AdaptiveAlignment, algebraically
    rearranged so one replay serves every prespecified price.
    """
    ordered = []
    cumulative_gain = 0.0
    length_sum = 0.0
    length_sq_sum = 0.0
    threshold = np.full(len(rewards), np.inf)
    for j, (reward, length) in enumerate(zip(rewards, lengths)):
        insort(ordered, float(reward))
        n = j + 1
        length_sum += float(length)
        length_sq_sum += float(length) ** 2
        if n < 4:
            continue
        k = n // 2
        top = np.asarray(ordered[-k-1:])
        z = np.exp(np.clip(top - ordered[-1], -745, 0))
        cutoff = z[0]
        excess = float(np.mean(z[1:] - cutoff))
        reference = cutoff + excess * (1 + math.log((k / n) / 0.01))
        scaled_gain = n / (n + 1) * reference * excess / (
            (1 + reference) * (1 + reference + excess))
        cumulative_gain += scaled_gain
        smoothed_gain = cumulative_gain / (n - 3)
        mean_length = length_sum / n
        variance = max((length_sq_sum - length_sum**2 / n) / (n - 1), 0.0)
        standard_error = math.sqrt(variance / n)
        estimated_length_sum = length_sum / (1 + 2 * standard_error / mean_length)
        threshold[j] = smoothed_gain / estimated_length_sum
    return np.minimum.accumulate(threshold)


def replay(rewards, lengths, seed=REPLAY_SEED, repetitions=PERMUTATIONS):
    prices = PRICES_PER_MILLION / 1e6
    n_prompts = len(rewards)
    adaptive = np.zeros((n_prompts, len(prices), 3))
    fixed = np.zeros((n_prompts, CAP, 3))
    rng = np.random.default_rng(seed)
    for i in range(n_prompts):
        reference = float(np.sort(rewards[i])[int(.99 * CAP)])
        for _ in range(repetitions):
            order = rng.permutation(CAP)
            r, length = rewards[i, order], lengths[i, order]
            cumulative_length = np.cumsum(length)
            quality = sigmoid(np.maximum.accumulate(r) - reference)
            threshold = prefix_thresholds(r, length)
            stopped = np.array([np.flatnonzero(threshold <= p)[0]
                                if np.any(threshold <= p) else CAP - 1 for p in prices])
            adaptive[i] += np.stack((quality[stopped], cumulative_length[stopped], stopped + 1), axis=1)
            fixed[i] += np.stack((quality, cumulative_length, np.arange(1, CAP + 1)), axis=1)
    return adaptive / repetitions, fixed / repetitions


def summarize(model, reward_key, adaptive, fixed):
    rows = []
    for seed in SPLIT_SEEDS:
        train, test = np.array_split(np.random.default_rng(seed).permutation(len(fixed)), 2)
        training_fixed = fixed[train].mean(axis=0)
        test_fixed = fixed[test].mean(axis=0)
        test_adaptive = adaptive[test].mean(axis=0)
        for pi, price_per_million in enumerate(PRICES_PER_MILLION):
            price = price_per_million / 1e6
            best_n = int(np.argmax(training_fixed[:, 0] - price * training_fixed[:, 1]))
            baseline = test_fixed[best_n, 0] - price * test_fixed[best_n, 1]
            quality, tokens, samples = test_adaptive[pi]
            profit = quality - price * tokens
            matched_tokens = float(interpolated_cost(
                test_fixed[:, 0], test_fixed[:, 1], np.array([quality]))[0])
            rows.append(dict(model=model, reward_model=reward_key, split_seed=seed,
                             price_per_million=float(price_per_million),
                             profit=float(profit), fixed_profit=float(baseline),
                             profit_improvement_percent=float(100 * (profit - baseline) / baseline),
                             quality=float(quality), output_tokens=float(tokens),
                             samples=float(samples), fixed_n=best_n + 1,
                             fixed_quality=float(test_fixed[best_n, 0]),
                             fixed_output_tokens=float(test_fixed[best_n, 1]),
                             matched_fixed_tokens=matched_tokens,
                             matched_cost_saving_percent=float(100 * (1 - tokens / matched_tokens))))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = {}
    for row in rows:
        key = row["reward_model"], row["model"], row["price_per_million"]
        groups.setdefault(key, []).append(row)
    result = []
    for (reward, model, price), group in groups.items():
        out = dict(reward_model=reward, model=model, price_per_million=price)
        for key in ("profit", "fixed_profit", "profit_improvement_percent", "quality",
                    "output_tokens", "samples", "fixed_n", "fixed_quality",
                    "fixed_output_tokens", "matched_fixed_tokens", "matched_cost_saving_percent"):
            out[key] = float(np.mean([row[key] for row in group]))
        result.append(out)
    return result


def make_plots(rows, output):
    labels = {"gemma3_4b": "Gemma 3 4B", "granite42_8b": "Granite 4.2 8B",
              "ministral3_8b": "Ministral 3 8B", "qwen35_9b": "Qwen 3.5 9B"}
    titles = {"armorm_llama3_8b": "ArmoRM Llama 3 8B",
              "fsfairx_llama3_rm": "FSFairX Llama 3 RM",
              "rm_mistral_7b": "RM Mistral 7B",
              "skywork_llama31_8b": "Skywork Llama 3.1 8B"}
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "savefig.dpi": 220})
    for reward in REWARDS:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), sharex=True)
        for mi, model in enumerate(MODELS):
            group = sorted((r for r in rows if r["reward_model"] == reward and r["model"] == model),
                           key=lambda r: r["price_per_million"])
            x = [r["price_per_million"] for r in group]
            axes[0].plot(x, [r["profit_improvement_percent"] for r in group],
                         marker="o", linewidth=1.9, color=colors[mi], label=labels[model])
            axes[1].plot(x, [r["matched_cost_saving_percent"] for r in group],
                         marker="o", linewidth=1.9, color=colors[mi], label=labels[model])
        for ax, ylabel in zip(axes, ("Profit improvement over train-selected fixed N (%)",
                                     "Token saving at matched quality (%)")):
            ax.set_xscale("log")
            ax.set_xticks(PRICES_PER_MILLION, [f"{p:g}" for p in PRICES_PER_MILLION])
            ax.set_xlabel("Generation price ($ per million output tokens)")
            ax.set_ylabel(ylabel)
            ax.axhline(0, color="#666666", linewidth=0.8)
            ax.grid(axis="y", color="#DFE4EA", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(titles[reward] + " · 100 Alpaca prompts per generator", fontsize=14)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=False)
        fig.subplots_adjust(top=0.78, bottom=0.18, left=0.08, right=0.98, wspace=0.27)
        fig.text(0.5, 0.035, "Five 50/50 prompt splits; eight response orders. Matched-quality saving uses a retrospective fixed-N mixture.",
                 ha="center", fontsize=8.5, color="#555555")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{reward}_profit_and_saving.{suffix}", facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(
        "/scratch1/kalayci/alignment_generation_farm/reward_run/data/rewarded_records"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite output")
    args.output.mkdir(parents=True)
    all_rows = []
    signatures = set()
    prompt_ids = None
    for model in MODELS:
        ids, score_array, lengths, signature = load_model(args.input_root, model)
        if prompt_ids is None:
            prompt_ids = ids
        elif ids != prompt_ids:
            raise ValueError("Prompt order differs across generators")
        signatures.add(signature)
        for ri, reward in enumerate(REWARDS):
            print("Replaying", model, reward, flush=True)
            adaptive, fixed = replay(score_array[:, :, ri], lengths)
            np.savez_compressed(args.output / f"{model}__{reward}.npz",
                                adaptive=adaptive, fixed=fixed, ids=ids)
            all_rows.extend(summarize(model, reward, adaptive, fixed))
    if len(signatures) != 1:
        raise ValueError("Mixed reward configuration signatures across models")
    write_csv(args.output / "split_results.csv", all_rows)
    summary = aggregate(all_rows)
    write_csv(args.output / "summary.csv", summary)
    make_plots(summary, args.output)
    method = dict(source=str(args.input_root), reward_config_signature=signatures.pop(),
                  generator_models=MODELS, reward_models=REWARDS,
                  prices_per_million_output_tokens=PRICES_PER_MILLION.tolist(),
                  price_interpretation="Illustrative generation prices; output tokens only; reward scoring costs excluded",
                  split_seeds=SPLIT_SEEDS, replay_seed=REPLAY_SEED,
                  permutations=PERMUTATIONS, cap=CAP,
                  policy="AdaptiveAlignment mean_costse2; minimum=4, reference_quantile=.99, cost_adjustment=2",
                  evaluation_quality="sigmoid(selected raw reward - full-pool empirical q99 reward)",
                  baseline="fixed N selected by mean training profit independently for each split and price",
                  matched_quality="retrospective test fixed-N mixture at adaptive attained quality",
                  caveat="Cached finite-pool replay; shared prompts and orderings; no statistical intervals or deployment guarantee",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "METHOD.json").write_text(json.dumps(method, indent=2) + "\n")
    print("Wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
