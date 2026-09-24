"""Compare n and n+1 denominators for the smoothed top-two gain."""
from __future__ import annotations

from bisect import bisect_left, insort
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pandora_token_alignment_report import (
    FOCUS_REWARDS, MODELS, PRICES_PER_MILLION, SPLIT_SEEDS, write_csv,
)
from replay_token_reward_farm import (
    CAP, PERMUTATIONS, REPLAY_SEED, REWARDS, load_model, sigmoid,
)
from theory_gain_token_ablation import scalar_sigmoid


POLICIES = (
    "four_over_n", "four_over_n_plus_one", "one_over_n",
    "one_over_n_plus_one", "record_then_smooth",
)


def stops_many(rewards, lengths):
    """First stop for each gain formula, with the same beta=2 cost estimate."""
    prices = tuple(float(price) for price in PRICES_PER_MILLION / 1e6)
    stops = np.zeros((len(POLICIES), len(prices)), dtype=int)
    remaining = np.full(len(POLICIES), len(prices), dtype=int)
    ordered = []
    excess_sum = weighted_excess_sum = length_sum = length_sq_sum = 0.0
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
            m = ((u1 - u3) + (u2 - u3)) / 2
            excess_sum += m
            weighted_excess_sum += n * m / (n + 1)
            valid += 1
        if not valid:
            continue
        mean_excess = excess_sum / valid
        gains = (
            4 * mean_excess / n,
            4 * mean_excess / (n + 1),
            mean_excess / n,
            mean_excess / (n + 1),
            weighted_excess_sum / (valid * n),
        )
        mean_length = length_sum / n
        variance = max((length_sq_sum - length_sum**2 / n) / (n - 1), 0.0)
        standard_error = math.sqrt(variance / n)
        effective_length = mean_length / (1 + 2 * standard_error / mean_length)
        for ci, gain in enumerate(gains):
            if remaining[ci] == 0:
                continue
            first_price = bisect_left(prices, gain / effective_length)
            if first_price < remaining[ci]:
                stops[ci, first_price:remaining[ci]] = n
                remaining[ci] = first_price
        if np.all(remaining == 0):
            break
    for ci in range(len(POLICIES)):
        if remaining[ci]:
            stops[ci, :remaining[ci]] = len(rewards)
    return stops


def replay_pair(rewards, lengths, fixed):
    adaptive = np.zeros((len(POLICIES), len(rewards), len(PRICES_PER_MILLION), 3))
    rng = np.random.default_rng(REPLAY_SEED)
    first_fixed = np.zeros((CAP, 3))
    for i in range(len(rewards)):
        full_reference = float(np.sort(rewards[i])[int(.99 * CAP)])
        for _ in range(PERMUTATIONS):
            order = rng.permutation(CAP)
            r, length = rewards[i, order], lengths[i, order]
            quality = sigmoid(np.maximum.accumulate(r) - full_reference)
            token_sum = np.cumsum(length)
            if i == 0:
                first_fixed += np.stack((quality, token_sum,
                                         np.arange(1, CAP + 1)), axis=1)
            ix = stops_many(r, length) - 1
            adaptive[:, i] += np.stack((quality[ix], token_sum[ix], ix + 1), axis=-1)
        if (i + 1) % 25 == 0:
            print("  prompts", i + 1, "/", len(rewards), flush=True)
    np.testing.assert_allclose(first_fixed / PERMUTATIONS, fixed[0], rtol=0, atol=1e-11)
    return adaptive / PERMUTATIONS


def evaluate(model, reward, adaptive, current, fixed):
    rows = []
    for seed in SPLIT_SEEDS:
        train, test = np.array_split(np.random.default_rng(seed).permutation(len(fixed)), 2)
        train_curve = fixed[train].mean(axis=0)
        for pi, price in enumerate(PRICES_PER_MILLION):
            p = price / 1e6
            n = int(np.argmax(train_curve[:, 0] - p * train_curve[:, 1]))
            baseline = float(np.mean(fixed[test, n, 0] - p * fixed[test, n, 1]))
            for ci, policy in enumerate(POLICIES):
                q, tokens, calls = adaptive[ci, test, pi].mean(axis=0)
                profit = float(q - p * tokens)
                rows.append(dict(generator=model, reward_model=reward,
                                 split_seed=seed, price_per_million=float(price),
                                 policy=policy, fixed_n=n + 1,
                                 fixed_net_utility=baseline,
                                 adaptive_quality=float(q),
                                 adaptive_output_tokens=float(tokens),
                                 adaptive_calls=float(calls),
                                 adaptive_net_utility=profit,
                                 relative_gain_percent=100 * (profit - baseline) / baseline))
            q, tokens, calls = current[test, pi].mean(axis=0)
            profit = float(q - p * tokens)
            rows.append(dict(generator=model, reward_model=reward,
                             split_seed=seed, price_per_million=float(price),
                             policy="current_mean_costse2", fixed_n=n + 1,
                             fixed_net_utility=baseline, adaptive_quality=float(q),
                             adaptive_output_tokens=float(tokens),
                             adaptive_calls=float(calls), adaptive_net_utility=profit,
                             relative_gain_percent=100 * (profit - baseline) / baseline))
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        key = row["generator"], row["reward_model"], row["price_per_million"], row["policy"]
        groups.setdefault(key, []).append(row)
    result = []
    for (model, reward, price, policy), group in groups.items():
        out = dict(generator=model, reward_model=reward,
                   price_per_million=price, policy=policy)
        for key in ("fixed_n", "fixed_net_utility", "adaptive_quality",
                    "adaptive_output_tokens", "adaptive_calls",
                    "adaptive_net_utility", "relative_gain_percent"):
            out[key] = float(np.mean([r[key] for r in group]))
        result.append(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(
        "/scratch1/kalayci/alignment_generation_farm/reward_run/data/rewarded_records"))
    parser.add_argument("--current-study", type=Path, required=True)
    parser.add_argument("--top2-study", type=Path, required=True)
    parser.add_argument("--optimism-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite output")
    args.output.mkdir(parents=True)
    hashes, rows = {}, []
    ids_reference = None
    for model in MODELS:
        ids, rewards, lengths, _ = load_model(args.input_root, model)
        if ids_reference is None:
            ids_reference = ids
        else:
            np.testing.assert_array_equal(ids_reference, ids)
        for reward in FOCUS_REWARDS:
            name = f"{model}__{reward}.npz"
            current_path = args.current_study / name
            top2_path = args.top2_study / name
            audit_path = args.optimism_audit / name
            with np.load(current_path) as data:
                np.testing.assert_array_equal(ids, data["ids"])
                fixed = data["fixed"]
                current = data["adaptive"][:, :len(PRICES_PER_MILLION), :]
            with np.load(top2_path) as data:
                top2 = data["theory"]
            with np.load(audit_path) as data:
                factor_one = data["adaptive"][5]
            hashes[name] = dict(current=hashlib.sha256(current_path.read_bytes()).hexdigest(),
                                top2=hashlib.sha256(top2_path.read_bytes()).hexdigest(),
                                audit=hashlib.sha256(audit_path.read_bytes()).hexdigest())
            print("Replaying n+1 denominators", model, reward, flush=True)
            adaptive = replay_pair(rewards[:, :, REWARDS.index(reward)], lengths, fixed)
            np.testing.assert_allclose(adaptive[0], top2, rtol=0, atol=1e-11)
            np.testing.assert_allclose(adaptive[2], factor_one, rtol=0, atol=1e-11)
            rows.extend(evaluate(model, reward, adaptive, current, fixed))
            np.savez_compressed(args.output / name, adaptive=adaptive, ids=ids)
    write_csv(args.output / "split_results.csv", rows)
    write_csv(args.output / "summary.csv", summarize(rows))
    method = dict(input_root=str(args.input_root.resolve()),
                  current_study=str(args.current_study.resolve()),
                  top2_study=str(args.top2_study.resolve()),
                  optimism_audit=str(args.optimism_audit.resolve()),
                  source_sha256=hashes, policies=POLICIES,
                  prices_per_million=PRICES_PER_MILLION.tolist(),
                  split_seeds=SPLIT_SEEDS, replay_seed=REPLAY_SEED,
                  permutations=PERMUTATIONS, cap=CAP,
                  utility_transform="sigmoid(reward - observed-prefix q99)",
                  estimated_next_cost="price*mean output tokens/(1+2*SE/mean)",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "METHOD.json").write_text(json.dumps(method, indent=2) + "\n")
    print("Wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
