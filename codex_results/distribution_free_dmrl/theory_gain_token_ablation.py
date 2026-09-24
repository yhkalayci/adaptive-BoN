"""Replay the smoothed top-two DMRL gain with practical utility and token cost.

This is a plug-in experiment, not a theorem-backed implementation: the reward
reference changes with the observed prefix and response token costs vary.
"""
from __future__ import annotations

import argparse
from bisect import insort
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pandora_token_alignment_report import (
    BOOTSTRAP_SEED, FOCUS_REWARDS, MODELS, PRICES_PER_MILLION,
    SPLIT_SEEDS, evaluate_pair, focused_tables, write_csv,
)
from replay_token_reward_farm import (
    CAP, PERMUTATIONS, REPLAY_SEED, REWARDS, load_model, sigmoid,
)


def scalar_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(max(x, -745.0))
    return ex / (1.0 + ex)


def theory_stops(rewards: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """First crossing of 4*mean_{j=4..n}(top-two excess_j)/n.

    Observed utility at each prefix is sigmoid(reward - prefix reward q99).
    Next-call cost uses the public practical policy's beta=2 estimate.
    """
    prices = PRICES_PER_MILLION / 1e6
    stops = np.zeros(len(prices), dtype=int)
    ordered = []
    length_sum = length_sq_sum = excess_sum = 0.0
    valid = 0
    for j, (reward, length) in enumerate(zip(rewards, lengths)):
        insort(ordered, float(reward))
        n = j + 1
        length_sum += float(length)
        length_sq_sum += float(length) ** 2
        if n < 4:
            continue
        reference = ordered[min(int(.99 * n), n - 1)]
        u1 = scalar_sigmoid(ordered[-1] - reference)
        u2 = scalar_sigmoid(ordered[-2] - reference)
        u3 = scalar_sigmoid(ordered[-3] - reference)
        if u3 > 0:
            excess_sum += ((u1 - u3) + (u2 - u3)) / 2
            valid += 1
        if valid:
            gain = 4 * excess_sum / (n * valid)
            mean_length = length_sum / n
            variance = max((length_sq_sum - length_sum**2 / n) / (n - 1), 0.0)
            standard_error = math.sqrt(variance / n)
            effective_length = mean_length / (1 + 2 * standard_error / mean_length)
            for pi, price in enumerate(prices):
                if stops[pi] == 0 and gain <= price * effective_length:
                    stops[pi] = n
        if np.all(stops):
            break
    stops[stops == 0] = len(rewards)
    return stops


def replay_pair(rewards: np.ndarray, lengths: np.ndarray, fixed: np.ndarray):
    adaptive = np.zeros((len(rewards), len(PRICES_PER_MILLION), 3))
    rng = np.random.default_rng(REPLAY_SEED)
    first_fixed = np.zeros((CAP, 3))
    for i in range(len(rewards)):
        full_reference = float(np.sort(rewards[i])[int(.99 * CAP)])
        for _ in range(PERMUTATIONS):
            order = rng.permutation(CAP)
            r, length = rewards[i, order], lengths[i, order]
            best_quality = sigmoid(np.maximum.accumulate(r) - full_reference)
            paid_tokens = np.cumsum(length)
            if i == 0:
                first_fixed += np.stack((best_quality, paid_tokens,
                                         np.arange(1, CAP + 1)), axis=1)
            stops = theory_stops(r, length) - 1
            adaptive[i] += np.stack((best_quality[stops], paid_tokens[stops],
                                     stops + 1), axis=1)
        if (i + 1) % 25 == 0:
            print("  prompts", i + 1, "/", len(rewards), flush=True)
    np.testing.assert_allclose(first_fixed / PERMUTATIONS, fixed[0], rtol=0, atol=1e-11)
    return adaptive / PERMUTATIONS


def compare_policies(model, reward, theory, incumbent, fixed):
    rows = []
    for seed in SPLIT_SEEDS:
        train, test = np.array_split(np.random.default_rng(seed).permutation(len(fixed)), 2)
        for pi, price in enumerate(PRICES_PER_MILLION):
            p = price / 1e6
            n = int(np.argmax(fixed[train, :, 0].mean(axis=0) -
                              p * fixed[train, :, 1].mean(axis=0)))
            baseline = np.mean(fixed[test, n, 0] - p * fixed[test, n, 1])
            theory_profit = np.mean(theory[test, pi, 0] - p * theory[test, pi, 1])
            incumbent_profit = np.mean(incumbent[test, pi, 0] - p * incumbent[test, pi, 1])
            rows.append(dict(generator=model, reward_model=reward, split_seed=seed,
                             price_per_million=float(price), fixed_n=n + 1,
                             fixed_net_utility=float(baseline),
                             theory_net_utility=float(theory_profit),
                             incumbent_net_utility=float(incumbent_profit),
                             theory_gain_percent=float(100 * (theory_profit - baseline) / baseline),
                             incumbent_gain_percent=float(100 * (incumbent_profit - baseline) / baseline),
                             theory_minus_incumbent=float(theory_profit - incumbent_profit),
                             theory_calls=float(theory[test, pi, 2].mean()),
                             incumbent_calls=float(incumbent[test, pi, 2].mean())))
    return rows


def write_report(output, split_rows, comparison):
    lines = [
        "# Smoothed top-two DMRL gain on token-counted alignment data", "",
        "This exploratory plug-in restores the pasted theoretical gain formula: "
        "at each n>=4, use the top two utilities above the third, average those "
        "excesses over prefixes 4..n, and multiply by 4/n. It retains the "
        "practical beta=2 estimated next-token cost and uses "
        "sigmoid(reward - observed-prefix q99) as the utility transform. "
        "The reference moves as the prefix grows, token costs vary, and the "
        "replay has a cap of 960; the theorem therefore does not directly apply.", "",
        "Both policies use the same recorded responses, eight response orders, "
        "five 50/50 prompt splits, and one fixed N selected on training prompts "
        "for each generator/reward/price. Reported gains average splitwise "
        "relative net-utility improvements.", "",
        "| Reward model | Generator | $/M tokens | Fixed N | Theory-gain samples | Current samples | Theory-gain net utility | Current net utility | Theory-gain vs fixed | Current vs fixed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    names = {"gemma3_4b": "Gemma 3 4B", "granite42_8b": "Granite 4.2 8B",
             "ministral3_8b": "Ministral 3 8B", "qwen35_9b": "Qwen 3.5 9B"}
    for reward in FOCUS_REWARDS:
        for model in MODELS:
            for price in PRICES_PER_MILLION:
                group = [r for r in comparison if r["reward_model"] == reward and
                         r["generator"] == model and r["price_per_million"] == float(price)]
                mean = lambda key: float(np.mean([r[key] for r in group]))
                lines.append("| " + " | ".join((
                    reward, names[model], f"{price:g}", f"{mean('fixed_n'):.1f}",
                    f"{mean('theory_calls'):.2f}", f"{mean('incumbent_calls'):.2f}",
                    f"{mean('theory_net_utility'):.4f}",
                    f"{mean('incumbent_net_utility'):.4f}",
                    f"{mean('theory_gain_percent'):+.2f}%",
                    f"{mean('incumbent_gain_percent'):+.2f}%")) + " |")
    lines += ["", "The earlier near-direct historical ablation did not include "
              "the pasted formula's time average, so it is not this experiment.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(
        "/scratch1/kalayci/alignment_generation_farm/reward_run/data/rewarded_records"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite output")
    args.output.mkdir(parents=True)
    weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        100, np.full(100, .01), size=args.bootstrap).astype(float)
    split_rows, intervals, comparison, hashes = [], [], [], {}
    ids_reference = None
    for model in MODELS:
        ids, rewards, lengths, signature = load_model(args.input_root, model)
        if ids_reference is None:
            ids_reference = ids
        else:
            np.testing.assert_array_equal(ids_reference, ids)
        for reward in FOCUS_REWARDS:
            source = args.study / f"{model}__{reward}.npz"
            with np.load(source) as data:
                np.testing.assert_array_equal(ids, data["ids"])
                fixed = data["fixed"]
                incumbent = data["adaptive"][:, :len(PRICES_PER_MILLION), :]
            hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
            print("Replaying smoothed top-two", model, reward, flush=True)
            theory = replay_pair(rewards[:, :, REWARDS.index(reward)], lengths, fixed)
            np.savez_compressed(args.output / source.name, theory=theory, incumbent=incumbent,
                                fixed=fixed, ids=ids)
            rows, bands = evaluate_pair(reward, model, theory, fixed, weights)
            split_rows.extend(rows)
            intervals.extend(bands)
            comparison.extend(compare_policies(model, reward, theory, incumbent, fixed))
    write_csv(args.output / "theory_gain_split_results.csv", split_rows)
    write_csv(args.output / "theory_gain_intervals.csv", intervals)
    write_csv(args.output / "paired_policy_comparison.csv", comparison)
    tables = focused_tables(split_rows)
    for reward in FOCUS_REWARDS:
        write_csv(args.output / f"table_{reward}.csv", tables[reward])
    write_report(args.output, split_rows, comparison)
    method = dict(input_root=str(args.input_root.resolve()), study=str(args.study.resolve()),
                  source_cache_sha256=hashes, generator_models=MODELS,
                  reward_models=FOCUS_REWARDS, prices_per_million=PRICES_PER_MILLION.tolist(),
                  split_seeds=SPLIT_SEEDS, permutations=PERMUTATIONS, replay_seed=REPLAY_SEED,
                  utility_transform="sigmoid(reward - empirical q99 of observed prefix)",
                  gain="4/n times the mean top-two utility excess over the third-largest, for prefixes j>=4 with third-largest>0",
                  estimated_next_cost="price*mean observed output tokens/(1+2*SE/mean)",
                  benchmark="train-selected fixed N, frozen on test; incumbent is mean_costse2",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "METHOD.json").write_text(json.dumps(method, indent=2) + "\n")
    print("Wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
