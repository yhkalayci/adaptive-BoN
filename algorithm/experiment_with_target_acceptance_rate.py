import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tqdm

from data_io import read_data, write_results
from experiment_config import LLMS, RMS, DISTRIBUTIONS, TRANSFORMATIONS
from pandora import build_permutations, pandoras_box_target_wr
from stats_utils import acceptance_rate, alpha_quantile


def _parse_target_wrs(value):
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return [float(item) for item in parts]


def _non_adaptive_win_rates(permutations, benchmark_val, alpha):
    n_total = permutations[0].shape[0]
    win_rates = []
    for n in range(1, n_total + 1):
        wrs = []
        scores = []
        for perm in permutations:
            sampled_quantile = float(alpha_quantile(perm[:n], alpha))
            scores.append(sampled_quantile)
            wrs.append(acceptance_rate(sampled_quantile, benchmark_val))
        win_rates.append(
            {
                "mean": float(np.mean(wrs)),
                "median": float(np.median(wrs)),
                "sample_count": n,
                "scores": scores,
                "max_score": float(benchmark_val),
            }
        )
    return win_rates


def single_prompt_analysis(
    prompt,
    epoch,
    target_wrs,
    delta,
    distribution,
    transformation,
    alpha,
    batch_size,
    rm_name,
    rng,
    min_open_count,
):
    rewards = np.array([x[f"{rm_name}_reward"] for x in prompt["generations"]])
    permutation_count = min(epoch, 100)
    # Use a fixed permutation set for all target acceptance rates.
    permutations = build_permutations(rewards, permutation_count, rng)

    benchmark_val = float(alpha_quantile(rewards, alpha))
    win_rates = _non_adaptive_win_rates(permutations, benchmark_val, alpha)

    pb_out = []
    for wr in target_wrs:
        outs = []
        achieved_rates = []
        for perm in permutations:
            out = pandoras_box_target_wr(
                perm,
                wr,
                delta,
                min_open_count=min_open_count,
                alpha=alpha,
                distribution=distribution,
                transformation=transformation,
                batch_size=batch_size,
            )
            outs.append(out)
            open_count = out["open_count"]
            if open_count > 0:
                sampled_quantile = float(np.max(perm[:open_count]))
                achieved_rates.append(acceptance_rate(sampled_quantile, benchmark_val))
            else:
                achieved_rates.append(np.nan)

        pb_out.append(
            {
                "target_wr": wr,
                "mean": float(np.nanmean(achieved_rates)),
                "median": float(np.nanmedian(achieved_rates)),
                "index_average": int(np.mean([x["max_until"]["index"] + 1 for x in outs])),
                "sample_count": int(np.mean([x["open_count"] for x in outs])),
                "outs": outs,
            }
        )

    return win_rates, pb_out


def _process_prompt(args):
    (
        prompt_index,
        prompt,
        seed,
        epoch,
        target_wrs,
        delta,
        distribution,
        transformation,
        alpha,
        batch_size,
        rm_name,
        min_open_count,
    ) = args

    prompt_rng = np.random.default_rng(int(seed))
    _, adaptive = single_prompt_analysis(
        prompt,
        epoch=epoch,
        target_wrs=target_wrs,
        delta=delta,
        distribution=distribution,
        transformation=transformation,
        alpha=alpha,
        batch_size=batch_size,
        rm_name=rm_name,
        rng=prompt_rng,
        min_open_count=min_open_count,
    )

    per_prompt_entry = {
        "task_id": prompt_index,
        "results": [
            {
                "target_acceptance_rate": float(item["target_wr"]),
                "achieved_acceptance_rate_mean": float(item["mean"]),
                "sample_mean": float(item["sample_count"]),
            }
            for item in adaptive
        ],
    }

    return {"prompt_index": prompt_index, "per_prompt_entry": per_prompt_entry}


def run(args):
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    rng = np.random.default_rng(int(args.seed))

    output_file = (
        f"{args.output_folder}/target_wr_{args.llm_name}_{args.rm_name}_"
        f"{args.distribution}_{args.transformation}_bs{args.batch_size}_a{args.alpha}_{args.seed}.json"
    )

    data = read_data(args.input_folder, args.llm_name)
    target_wrs = _parse_target_wrs(args.target_wrs)
    prompt_count = min(args.prompt_count, len(data))

    prompt_seeds = rng.integers(0, np.iinfo(np.int64).max, size=prompt_count, dtype=np.int64)

    worker_args = [
        (
            i,
            data[i],
            int(prompt_seeds[i]),
            args.epoch,
            target_wrs,
            args.delta,
            args.distribution,
            args.transformation,
            args.alpha,
            args.batch_size,
            args.rm_name,
            args.min_open_count,
        )
        for i in range(prompt_count)
    ]

    per_prompt = [None] * prompt_count
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_process_prompt, worker_arg) for worker_arg in worker_args]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Prompts"):
            result = future.result()
            per_prompt[result["prompt_index"]] = result["per_prompt_entry"]

    aggregate = []
    for wr_index, wr in enumerate(target_wrs):
        achieved = np.array(
            [prompt["results"][wr_index]["achieved_acceptance_rate_mean"] for prompt in per_prompt],
            dtype=float,
        )
        sample_means = np.array(
            [prompt["results"][wr_index]["sample_mean"] for prompt in per_prompt],
            dtype=float,
        )
        aggregate.append(
            {
                "target_acceptance_rate": float(wr),
                "achieved_acceptance_rate_median": float(np.nanmedian(achieved)),
                "achieved_acceptance_rate_p25": float(np.nanpercentile(achieved, 25)),
                "achieved_acceptance_rate_p75": float(np.nanpercentile(achieved, 75)),
                "sample_mean_median": float(np.nanmedian(sample_means)),
                "sample_mean_p25": float(np.nanpercentile(sample_means, 25)),
                "sample_mean_p75": float(np.nanpercentile(sample_means, 75)),
                "prompt_count": int(np.sum(~np.isnan(achieved))),
            }
        )

    payload = {
        "llm_name": args.llm_name,
        "rm_name": args.rm_name,
        "distribution": args.distribution,
        "transformation": args.transformation,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "alpha": args.alpha,
        "delta": args.delta,
        "min_open_count": args.min_open_count,
        "target_acceptance_rates": target_wrs,
        "per_prompt": per_prompt,
        "aggregate": aggregate,
    }
    write_results(output_file, payload)


def main():
    parser = argparse.ArgumentParser(
        description="Run Pandora's Box with target acceptance rate."
    )
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
        default=10,
        help="Number of permutations to evaluate.",
    )
    parser.add_argument(
        "--prompt_count",
        type=int,
        default=100,
        help="Number of prompts to evaluate.",
    )
    parser.add_argument(
        "--target_wrs",
        type=str,
        default="0.3,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5",
        help="Comma-separated target win-rate values.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Confidence level for Pandora's Box.",
    )
    parser.add_argument(
        "--min_open_count",
        type=int,
        default=10,
        help="Minimum number of boxes to open before stopping.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="Quantile parameter for the benchmark.",
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
