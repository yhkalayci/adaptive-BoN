"""Isolate gain scaling and cost optimism on the token alignment replay."""
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


CONFIGS = (
    ("current_beta0", "current", 1.0, 0.0),
    ("current_beta1", "current", 1.0, 1.0),
    ("current_beta2", "current", 1.0, 2.0),
    ("top2_factor4_beta0", "top2", 4.0, 0.0),
    ("top2_factor4_beta2", "top2", 4.0, 2.0),
    ("top2_factor1_beta2", "top2", 1.0, 2.0),
    ("top2_factor0p25_beta2", "top2", .25, 2.0),
)


def stops_many(rewards, lengths):
    prices = tuple(float(x) for x in PRICES_PER_MILLION / 1e6)
    stops = np.zeros((len(CONFIGS), len(prices)), dtype=int)
    discounts = np.zeros_like(stops, dtype=float)
    remaining = np.full(len(CONFIGS), len(prices), dtype=int)
    ordered = []
    length_sum = length_sq_sum = current_sum = top2_sum = 0.0
    valid = 0
    last_discount = {}
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
        benchmark = cutoff + excess * (1 + math.log((k / n) / .01))
        current_sum += n / (n + 1) * benchmark * excess / (
            (1 + benchmark) * (1 + benchmark + excess))
        current_gain = current_sum / (n - 3) / n

        reference = ordered[min(int(.99 * n), n - 1)]
        u1 = scalar_sigmoid(ordered[-1] - reference)
        u2 = scalar_sigmoid(ordered[-2] - reference)
        u3 = scalar_sigmoid(ordered[-3] - reference)
        if u3 > 0:
            top2_sum += ((u1 - u3) + (u2 - u3)) / 2
            valid += 1
        top2_gain = top2_sum / (n * valid) if valid else math.inf

        mean_length = length_sum / n
        variance = max((length_sq_sum - length_sum**2 / n) / (n - 1), 0.0)
        standard_error = math.sqrt(variance / n)
        for ci, (_, kind, factor, beta) in enumerate(CONFIGS):
            if remaining[ci] == 0:
                continue
            discount = 1 / (1 + beta * standard_error / mean_length)
            last_discount[ci] = discount
            gain = (current_gain if kind == "current" else top2_gain) * factor
            threshold = gain / (mean_length * discount)
            start = bisect_left(prices, threshold)
            if start < remaining[ci]:
                stops[ci, start:remaining[ci]] = n
                discounts[ci, start:remaining[ci]] = discount
                remaining[ci] = start
        if np.all(remaining == 0):
            break
    for ci in range(len(CONFIGS)):
        if remaining[ci]:
            stops[ci, :remaining[ci]] = len(rewards)
            discounts[ci, :remaining[ci]] = last_discount[ci]
    return stops, discounts


def replay_pair(rewards, lengths, fixed):
    adaptive = np.zeros((len(CONFIGS), len(rewards), len(PRICES_PER_MILLION), 3))
    discount = np.zeros((len(CONFIGS), len(rewards), len(PRICES_PER_MILLION)))
    rng = np.random.default_rng(REPLAY_SEED)
    first_fixed = np.zeros((CAP, 3))
    for i in range(len(rewards)):
        reference = float(np.sort(rewards[i])[int(.99 * CAP)])
        for _ in range(PERMUTATIONS):
            order = rng.permutation(CAP)
            r, length = rewards[i, order], lengths[i, order]
            quality = sigmoid(np.maximum.accumulate(r) - reference)
            cumulative_length = np.cumsum(length)
            if i == 0:
                first_fixed += np.stack((quality, cumulative_length,
                                         np.arange(1, CAP + 1)), axis=1)
            stops, disc = stops_many(r, length)
            ix = stops - 1
            adaptive[:, i] += np.stack((quality[ix], cumulative_length[ix], stops), axis=-1)
            discount[:, i] += disc
        if (i + 1) % 25 == 0:
            print("  prompts", i + 1, "/", len(rewards), flush=True)
    np.testing.assert_allclose(first_fixed / PERMUTATIONS, fixed[0], rtol=0, atol=1e-11)
    return adaptive / PERMUTATIONS, discount / PERMUTATIONS


def evaluate(model, reward, adaptive, discount, fixed):
    rows = []
    for seed in SPLIT_SEEDS:
        train, test = np.array_split(np.random.default_rng(seed).permutation(len(fixed)), 2)
        train_fixed = fixed[train].mean(axis=0)
        for pi, price in enumerate(PRICES_PER_MILLION):
            p = price / 1e6
            n = int(np.argmax(train_fixed[:, 0] - p * train_fixed[:, 1]))
            fixed_profit = float(np.mean(fixed[test, n, 0] - p * fixed[test, n, 1]))
            for ci, (name, _, factor, beta) in enumerate(CONFIGS):
                q, tokens, calls = adaptive[ci, test, pi].mean(axis=0)
                profit = float(q - p * tokens)
                rows.append(dict(generator=model, reward_model=reward,
                                 split_seed=seed, price_per_million=float(price),
                                 policy=name, gain_factor=factor, cost_beta=beta,
                                 fixed_n=n + 1, fixed_net_utility=fixed_profit,
                                 adaptive_quality=float(q),
                                 adaptive_output_tokens=float(tokens),
                                 adaptive_calls=float(calls),
                                 adaptive_net_utility=profit,
                                 relative_gain_percent=100 * (profit - fixed_profit) / fixed_profit,
                                 mean_cost_discount=float(discount[ci, test, pi].mean())))
    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row["generator"], row["reward_model"], row["price_per_million"], row["policy"])
        grouped.setdefault(key, []).append(row)
    result = []
    for (model, reward, price, policy), group in grouped.items():
        out = dict(generator=model, reward_model=reward, price_per_million=price,
                   policy=policy)
        for key in ("fixed_n", "fixed_net_utility", "adaptive_quality",
                    "adaptive_output_tokens", "adaptive_calls", "adaptive_net_utility",
                    "relative_gain_percent", "mean_cost_discount"):
            out[key] = float(np.mean([r[key] for r in group]))
        result.append(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(
        "/scratch1/kalayci/alignment_generation_farm/reward_run/data/rewarded_records"))
    parser.add_argument("--current-study", type=Path, required=True)
    parser.add_argument("--top2-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite output")
    args.output.mkdir(parents=True)
    hashes = {}
    rows = []
    ids_reference = None
    for model in MODELS:
        ids, rewards, lengths, _ = load_model(args.input_root, model)
        if ids_reference is None:
            ids_reference = ids
        else:
            np.testing.assert_array_equal(ids_reference, ids)
        for reward in FOCUS_REWARDS:
            filename = f"{model}__{reward}.npz"
            current_path = args.current_study / filename
            top2_path = args.top2_study / filename
            with np.load(current_path) as data:
                np.testing.assert_array_equal(ids, data["ids"])
                fixed = data["fixed"]
                current = data["adaptive"][:, :len(PRICES_PER_MILLION), :]
            with np.load(top2_path) as data:
                np.testing.assert_array_equal(ids, data["ids"])
                top2 = data["theory"]
            hashes[filename] = dict(current=hashlib.sha256(current_path.read_bytes()).hexdigest(),
                                    top2=hashlib.sha256(top2_path.read_bytes()).hexdigest())
            print("Auditing optimism", model, reward, flush=True)
            adaptive, discount = replay_pair(rewards[:, :, REWARDS.index(reward)], lengths, fixed)
            np.testing.assert_allclose(adaptive[2], current, rtol=0, atol=1e-11)
            np.testing.assert_allclose(adaptive[4], top2, rtol=0, atol=1e-11)
            rows.extend(evaluate(model, reward, adaptive, discount, fixed))
            np.savez_compressed(args.output / filename, adaptive=adaptive,
                                discount=discount, ids=ids)
    summary = summarize(rows)
    write_csv(args.output / "split_results.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    method = dict(input_root=str(args.input_root.resolve()),
                  current_study=str(args.current_study.resolve()),
                  top2_study=str(args.top2_study.resolve()), source_sha256=hashes,
                  configs=CONFIGS, prices_per_million=PRICES_PER_MILLION.tolist(),
                  split_seeds=SPLIT_SEEDS, replay_seed=REPLAY_SEED,
                  permutations=PERMUTATIONS, cap=CAP,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "METHOD.json").write_text(json.dumps(method, indent=2) + "\n")
    print("Wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
