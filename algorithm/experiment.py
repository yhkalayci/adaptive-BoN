import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from data_io import read_data, write_results, print_cost_progress, print_fixed_n_progress
from experiment_config import LLMS, RMS, DISTRIBUTIONS, TRANSFORMATIONS
from pandora import (
    build_permutations,
    compute_global_opt,
    compute_adaptive_results,
    compute_beacon_results,
    compute_fixed_n_results,
)


def _build_prompt_entry(prompt_index, costs, pandora_results, fixed_n_results):
    pandora_entries = []
    fixed_n_entries = []
    for cost_index, cost in enumerate(costs):
        pandora_entries.append(
            {
                "cost": cost,
                "median_utility": float(pandora_results[cost_index]["median_utility"]),
            }
        )
        fixed_n_entries.append(
            {
                "cost": cost,
                "results": [
                    {"n": item["n"], "mean_utility": float(item["mean_utility"])}
                    for item in fixed_n_results[cost_index]["results"]
                ],
            }
        )

    return {
        "prompt_index": prompt_index,
        "pandora": pandora_entries,
        "fixed_n": fixed_n_entries,
    }


def _compute_win_rate(adaptive_score, fixed_best):
    if adaptive_score > fixed_best:
        return 1.0
    if adaptive_score == fixed_best:
        return 0.5
    return 0.0


def _process_prompt(args):
    (
        prompt_index,
        prompt,
        seed,
        costs,
        delta,
        alpha,
        distribution,
        transformation,
        batch_size,
        rm_name,
        epoch,
    ) = args

    rewards = np.array([x[f"{rm_name}_reward"] for x in prompt["generations"]])
    rng = np.random.default_rng(int(seed))
    permutations = build_permutations(rewards, epoch, rng)
    global_opt = compute_global_opt(rewards, alpha)

    pandora_results = compute_adaptive_results(
        permutations,
        costs,
        delta,
        alpha,
        distribution=distribution,
        transformation=transformation,
        batch_size=batch_size,
    )
    fixed_n_results, fixed_n_best = compute_fixed_n_results(permutations, costs, global_opt["value"])

    per_prompt_entry = _build_prompt_entry(prompt_index, costs, pandora_results, fixed_n_results)

    ratios = []
    for cost_index in range(len(costs)):
        pandora_mean = pandora_results[cost_index]["mean_utility"]
        fixed_best_mean = fixed_n_best[cost_index]["mean_utility"]
        ratio = np.nan
        if fixed_best_mean != 0:
            ratio = pandora_mean / fixed_best_mean
        ratios.append(ratio)

    prompt_cost_entries = []
    result3_win_rates = []
    result3_avg_samples = []
    for cost_index, cost in enumerate(costs):
        outs = pandora_results[cost_index]["outs"]
        open_counts = [out["open_count"] for out in outs]
        mean_open = float(np.mean(open_counts)) if open_counts else 0.0
        target_n = int(np.ceil(mean_open))
        if target_n < 1:
            target_n = 1

        wins = []
        for perm, out in zip(permutations, outs):
            max_n = min(target_n, perm.shape[0])
            fixed_best = float(np.max(perm[:max_n]))
            wins.append(_compute_win_rate(out["score"], fixed_best))

        win_rate = float(np.mean(wins)) if wins else 0.0
        prompt_cost_entries.append(
            {
                "cost": cost,
                "win_rate": win_rate,
                "average_sample_count": mean_open,
                "ceil_mean_open": target_n,
            }
        )
        result3_win_rates.append(win_rate)
        result3_avg_samples.append(mean_open)

    result3_entry = {"prompt_index": prompt_index, "costs": prompt_cost_entries}

    return {
        "prompt_index": prompt_index,
        "per_prompt_entry": per_prompt_entry,
        "ratios": ratios,
        "result3_entry": result3_entry,
        "result3_win_rates": result3_win_rates,
        "result3_avg_samples": result3_avg_samples,
    }


def run(args):
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    rng = np.random.default_rng(int(args.seed))

    result_tag = (
        f"{args.rm_name}_{args.llm_name}_{args.distribution}_"
        f"{args.transformation}_bs{args.batch_size}_a{args.alpha}"
    )
    result_1_file = f"{args.output_folder}/result_1_{result_tag}.json"
    result_2_file = f"{args.output_folder}/result_2_{result_tag}.json"
    result_3_file = f"{args.output_folder}/result_3_{result_tag}.json"

    data = read_data(args.input_folder, args.llm_name)
    costs = [
        0.02,
        0.01,
        0.008,
        0.006,
        0.004,
        0.002,
        0.001,
        0.0008,
        0.0006,
        0.0004,
        0.0002,
        0.0001,
    ]

    print(f"Running experiment for {args.llm_name} across {len(data)} prompts")
    print(f"  Distribution: {args.distribution}")
    print(f"  Transformation: {args.transformation}")
    print(f"  Batch size: {args.batch_size}")
    print("  Workers: 8")

    prompt_seeds = rng.integers(0, np.iinfo(np.int64).max, size=len(data), dtype=np.int64)
    worker_args = [
        (
            prompt_index,
            prompt,
            int(prompt_seeds[prompt_index]),
            costs,
            args.delta,
            args.alpha,
            args.distribution,
            args.transformation,
            args.batch_size,
            args.rm_name,
            args.epoch,
        )
        for prompt_index, prompt in enumerate(data)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_process_prompt, args) for args in worker_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Prompts"):
            results.append(future.result())

    per_prompt_entries = [result["per_prompt_entry"] for result in results]
    ratio_by_cost = {cost: [] for cost in costs}
    result3_per_prompt = [result["result3_entry"] for result in results]
    result3_win_rates = {cost: [] for cost in costs}
    result3_avg_samples = {cost: [] for cost in costs}

    for result in results:
        for cost_index, cost in enumerate(costs):
            ratio_by_cost[cost].append(result["ratios"][cost_index])
            result3_win_rates[cost].append(result["result3_win_rates"][cost_index])
            result3_avg_samples[cost].append(result["result3_avg_samples"][cost_index])

    result_1_payload = {
        "llm_name": args.llm_name,
        "rm_name": args.rm_name,
        "distribution": args.distribution,
        "transformation": args.transformation,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "alpha": args.alpha,
        "delta": args.delta,
        "costs": costs,
        "per_prompt": per_prompt_entries,
    }
    write_results(result_1_file, result_1_payload)

    ratio_summary = []
    for cost in costs:
        ratios = np.array(ratio_by_cost[cost], dtype=float)
        ratio_summary.append(
            {
                "cost": cost,
                "median_ratio": float(np.nanmedian(ratios)),
                "p25_ratio": float(np.nanpercentile(ratios, 25)),
                "p75_ratio": float(np.nanpercentile(ratios, 75)),
                "prompt_count": int(np.sum(~np.isnan(ratios))),
            }
        )

    result_2_payload = {
        "llm_name": args.llm_name,
        "rm_name": args.rm_name,
        "distribution": args.distribution,
        "transformation": args.transformation,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "alpha": args.alpha,
        "delta": args.delta,
        "costs": costs,
        "ratios": ratio_summary,
    }
    write_results(result_2_file, result_2_payload)

    result3_aggregate = []
    for cost in costs:
        win_rates = np.array(result3_win_rates[cost], dtype=float)
        avg_samples = np.array(result3_avg_samples[cost], dtype=float)
        result3_aggregate.append(
            {
                "cost": cost,
                "win_rate_median": float(np.nanmedian(win_rates)),
                "win_rate_p25": float(np.nanpercentile(win_rates, 25)),
                "win_rate_p75": float(np.nanpercentile(win_rates, 75)),
                "avg_sample_count_median": float(np.nanmedian(avg_samples)),
                "avg_sample_count_p25": float(np.nanpercentile(avg_samples, 25)),
                "avg_sample_count_p75": float(np.nanpercentile(avg_samples, 75)),
            }
        )

    result_3_payload = {
        "llm_name": args.llm_name,
        "rm_name": args.rm_name,
        "distribution": args.distribution,
        "transformation": args.transformation,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "alpha": args.alpha,
        "delta": args.delta,
        "costs": costs,
        "per_prompt": result3_per_prompt,
        "aggregate": result3_aggregate,
    }
    write_results(result_3_file, result_3_payload)




def main():
    parser = argparse.ArgumentParser(description="Run LLM experiment for a single prompt.")
    parser.add_argument(
        "--llm_name",
        type=str,
        required=True,
        choices=LLMS,
        help="Short name of the LLM used for generation.",
    )
    parser.add_argument(
        "--rm_name",
        type=str,
        required=True,
        choices=RMS,
        help="Short name of the RM used for generation.",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="./",
        help="Path to the input folder.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./res",
        help="Path to the output folder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=46,
        help="Seed for random number generators.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
        help="Number of permutations to evaluate.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="Quantile parameter for the benchmark.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Confidence level for Pandora's Box.",
    )
    parser.add_argument(
        "--beacon_grid_size",
        type=int,
        default=200,
        help="Grid size for BEACON h-table precompute.",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="shifted_exponential",
        choices=DISTRIBUTIONS,
        help="Distribution family for modeling rewards: "
             "'shifted_exponential' (tail fitting of e^rewards) or "
             "'lognormal' (full fitting of e^rewards).",
    )
    parser.add_argument(
        "--transformation",
        type=str,
        default="bradley_terry",
        choices=TRANSFORMATIONS,
        help="Transformation method: "
             "'bradley_terry' (v/(v+M) with benchmark M) or "
             "'cdf' (CDF of fitted distribution).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of boxes to open per step in Pandora's Box.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
