"""Exploratory price scan; full-history top-three, minimum 10..512, cap 512.

Reuses the original calibration/test splits and orderings. Selects multiplier,
cost adjustment and every integer minimum on calibration predicted profit only.
Fixed-N is selected by exact calibration correctness profit as in the paper.
No manuscript or previous result is changed. This reused cohort is not an
independent validation, and bootstrap intervals do not adjust for selection.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

import coding_theory_bridge as bridge
from algorithm.adaptive_coding import AdaptiveCoding

old = bridge.old


def gain_base(batch):
    n = np.arange(1, batch.samples + 1)
    total = np.cumsum(np.where(n >= 4, batch.residual_current, 0), axis=1, dtype=float)
    return total / np.maximum(n - 3, 1) / n


def first_crossing(gain, cost, alpha, price):
    crossing = alpha * gain <= price * cost
    crossing[:, :3] = False
    return np.where(crossing.any(axis=1), crossing.argmax(axis=1) + 1, gain.shape[1])


def minimum_objectives(batch, raw_stop, price, lower=10):
    """All min-count choices, exactly equivalent to max(raw_stop, minimum)."""
    minimums = np.arange(lower, batch.samples + 1)
    indices = np.maximum(raw_stop[:, None], minimums) - 1
    rows = np.arange(batch.trials)[:, None]
    token_means = batch.cumulative_tokens[rows, indices].mean(axis=0)
    probability_means = batch.best_probabilities[rows, indices].mean(axis=0)
    return probability_means - price * token_means, token_means


def tune(batch, gain, costs, price):
    best_score, best_tokens, best_rule = -np.inf, np.inf, None
    for beta in old.COST_ADJUSTMENTS:
        for alpha in old.MULTIPLIERS:
            stop = first_crossing(gain, costs[beta], alpha, price)
            scores, tokens = minimum_objectives(batch, stop, price)
            peak = scores.max()
            tied = np.flatnonzero(scores >= peak - 1e-15)
            j = tied[np.argmin(tokens[tied])]
            if scores[j] > best_score + 1e-15 or (
                abs(scores[j] - best_score) <= 1e-15 and tokens[j] < best_tokens
            ):
                best_score, best_tokens = float(scores[j]), float(tokens[j])
                best_rule = bridge.Rule(width=2, alpha=alpha, beta=beta,
                    minimum=int(j + 10), cap=512, smoothing="mean", guard=False, latch=True)
    return best_rule, best_score


def summarize(rows, records, output):
    ids = sorted({r["problem_id"] for r in records})
    lookup = {key: j for j, key in enumerate(ids)}
    weights = np.random.default_rng(20260924).multinomial(
        len(ids), np.full(len(ids), 1 / len(ids)), size=1000)
    summaries = []
    for price in sorted({r["price"] for r in rows}):
        part = [r for r in rows if r["price"] == price]
        means = {key: float(np.mean([r[key] for r in part])) for key in (
            "accuracy", "tokens", "calls", "profit", "fixed_accuracy", "fixed_tokens",
            "fixed_calls", "fixed_profit", "cap_rate", "minimum")}
        mask = np.zeros((len(part), len(ids)))
        profit, fixed = np.zeros_like(mask), np.zeros_like(mask)
        for r in records:
            if r["price"] != price:
                continue
            s, j = r["split"], lookup[r["problem_id"]]
            mask[s, j] = 1
            profit[s, j], fixed[s, j] = r["profit"], r["fixed_profit"]
        den = weights @ mask.T
        pa = (weights @ profit.T / den).mean(axis=1)
        pn = (weights @ fixed.T / den).mean(axis=1)
        assert np.all(pn > 0), "Relative profit undefined with nonpositive baseline"
        lo, hi = np.percentile(100 * (pa / pn - 1), [2.5, 97.5])
        summaries.append(dict(price=price, price_per_million=price * 1e6, **means,
            minimum_low=min(r["minimum"] for r in part), minimum_high=max(r["minimum"] for r in part),
            relative_profit_percent=100 * (means["profit"] / means["fixed_profit"] - 1),
            token_change_percent=100 * (means["tokens"] / means["fixed_tokens"] - 1),
            call_change_percent=100 * (means["calls"] / means["fixed_calls"] - 1),
            profit_ci_low=float(lo), profit_ci_high=float(hi)))
    old.atomic_csv(output / "summary.csv", summaries)
    for r in summaries:
        print(f"${r['price_per_million']:.2f}: N {r['fixed_calls']:.2f} -> {r['calls']:.2f}, "
              f"accuracy {100*r['fixed_accuracy']:.2f}% -> {100*r['accuracy']:.2f}%, "
              f"profit {r['relative_profit_percent']:+.2f}% "
              f"[{r['profit_ci_low']:+.2f}, {r['profit_ci_high']:+.2f}]", flush=True)


def run(args):
    output, source = args.output, args.reference
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "manifest.json").read_text())
    data = Path(manifest["data_path"])
    assert old.sha256(data) == manifest["data_sha256"]
    prices = np.arange(4, 41, dtype=float) / 4 / 1e6
    old.atomic_json(output / "manifest.json", dict(scope=__doc__, source=str(source.resolve()),
        data_path=str(data), data_sha256=manifest["data_sha256"],
        script_sha256=old.sha256(Path(__file__)), seed=manifest["outer_seed"],
        splits=args.splits, calibration_permutations=24, test_permutations=48,
        prices_per_token=prices.tolist(), minimum_grid=list(range(10, 513)), cap=512,
        multipliers=old.MULTIPLIERS, cost_adjustments=old.COST_ADJUSTMENTS,
        statistic_start=4, width=2, smoothing="mean", guard=False, latch=True,
        adaptive_tuning="calibration isotonic probability minus actual token cost",
        baseline_tuning="exact calibration correctness minus expected token cost; N=1..512",
        bootstrap_replicates=1000, bootstrap_seed=20260924,
        bootstrap_unit="shared problem-identity weights across splits; conditional, not selection-adjusted"))
    problems = old.load_coding_problems(data)
    rows, records, policies, replay_checks = [], [], [], 0
    total_start = time.monotonic()
    for split in range(args.splits):
        start = time.monotonic()
        train_ids, test_ids = old.split_problem_ids(problems, manifest["outer_seed"] + split)
        profile = old.CodingProfile.load(source / "profiles" / f"split_{split:02d}.json")
        assert set(profile.metadata["fit_problem_ids"]) == set(train_ids)
        assert set(profile.metadata["holdout_problem_ids"]) == set(test_ids)
        probabilities = old.calibrated_rewards(problems, profile)
        parts = []
        for ids, count, offset in ((train_ids, 24, 100000), (test_ids, 48, 900000)):
            subset = {key: problems[key] for key in ids}
            orders = old.make_permutations(subset, count, manifest["outer_seed"] + offset + 10000 * split)
            batch = old.build_trajectory_batch(subset, probabilities, orders, width=2)
            labels = old.selected_correctness(batch, subset, orders)
            parts.append((batch, labels, orders))
        (train, _, _), (test, labels, test_orders) = parts
        fixed_ns, _, _ = old.select_fixed_n({key: problems[key] for key in train_ids}, tuple(1 / prices))
        gain = gain_base(train)
        costs = {b: train.mean_tokens / (1 + b * train.token_se_ratio) for b in old.COST_ADJUSTMENTS}
        print(f"split {split}: trajectories ready ({time.monotonic()-start:.1f}s)", flush=True)
        for price, fixed_n in zip(prices, fixed_ns):
            price, fixed_n = float(price), int(fixed_n)
            rule, score = tune(train, gain, costs, price)
            count = bridge.stops(test, None, price, rule)
            adaptive = bridge.take(test, labels, count, price)
            fixed = bridge.take(test, labels, np.full(test.trials, fixed_n), price)
            # Check the public online implementation on an actual held-out stream.
            key = str(test.problem_ids[0])
            policy = AdaptiveCoding.from_paper_settings(profile, price, multiplier=rule.alpha,
                cost_adjustment=rule.beta, minimum=rule.minimum, cap=rule.cap)
            for index in test_orders[key][0]:
                result = policy.observe(problems[key].rewards[index], problems[key].lengths[index])
                if result.should_stop:
                    break
            assert result.count == count[0]
            assert result.total_length == adaptive["tokens"][0]
            assert result.best_index == test.best_indices[0, count[0]-1]
            replay_checks += 1
            rows.append(dict(split=split, price=price, minimum=rule.minimum, cap=rule.cap,
                accuracy=float(adaptive["correct"].mean()), tokens=float(adaptive["tokens"].mean()),
                calls=float(count.mean()), profit=float(adaptive["profit"].mean()),
                fixed_accuracy=float(fixed["correct"].mean()), fixed_tokens=float(fixed["tokens"].mean()),
                fixed_calls=fixed_n, fixed_profit=float(fixed["profit"].mean()),
                cap_rate=float(np.mean(count == 512))))
            policies.append(dict(split=split, price=price, rule=asdict(rule), calibration_profit=score))
            for j, identity in enumerate(sorted(test_ids)):
                sl = slice(48*j, 48*(j+1))
                records.append(dict(split=split, price=price, problem_id=identity,
                    profit=float(adaptive["profit"][sl].mean()), fixed_profit=float(fixed["profit"][sl].mean()),
                    accuracy=float(adaptive["correct"][sl].mean()), tokens=float(adaptive["tokens"][sl].mean()),
                    calls=float(count[sl].mean()), fixed_calls=fixed_n,
                    fixed_accuracy=float(fixed["correct"][sl].mean()), fixed_tokens=float(fixed["tokens"][sl].mean())))
        old.atomic_csv(output / "split_metrics.csv", rows)
        old.atomic_csv(output / "problem_metrics.csv", records)
        old.atomic_json(output / "selected_policies.json", policies)
        print(f"split {split}: finished in {time.monotonic()-start:.1f}s", flush=True)
    summarize(rows, records, output)
    old.atomic_json(output / "validation.json", dict(public_api_replay_checks=replay_checks,
        completed_splits=args.splits, elapsed_seconds=time.monotonic()-total_start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=10)
    run(parser.parse_args())
