"""Diagnose expensive-price losses without changing production or manuscript.

Exact finite-cache quality curves expose score-calibration selection effects.
Paired replays decompose large-minimum budgets from subsequent stopping.
Counterfactual settings are selected on calibration problems, never test labels.
This remains exploratory evaluation on a reused cohort.
"""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

import coding_top3_price_scan as scan

old, bridge = scan.old, scan.bridge
PRICES = np.asarray((1, 1.25, 1.5, 2, 3, 5, 8, 10)) / 1e6
METHODS = ("current_mean", "fixed_at_minimum", "minimum10_frozen_alpha",
           "observed_outcome_tuned", "minimum10_16_observed_tuned", "minimum10_16_proxy_tuned",
           "fixed_proxy_tuned", "fixed_observed_mc_tuned", "baseline_cap_control")


def max_cdf(size):
    cdf = np.zeros((size, size + 1))
    cdf[:, -1] = 1
    draws = np.arange(1, size + 1)
    for available in range(size, 0, -1):
        cdf[:, available - 1] = cdf[:, available] * np.maximum(available - draws, 0) / available
    return cdf


def exact_curves(problems, probabilities):
    size = len(next(iter(problems.values())).rewards)
    cdf = max_cdf(size)
    correctness, predicted = [], []
    for key, problem in problems.items():
        order = np.argsort(problem.rewards, kind="stable")
        rewards = np.asarray(problem.rewards)[order]
        starts = np.r_[0, 1 + np.flatnonzero(rewards[1:] != rewards[:-1])]
        ends = np.r_[starts[1:], size]
        weights = cdf[:, ends] - cdf[:, starts]
        correct = np.add.reduceat(np.asarray(problem.correct)[order].astype(float), starts) / (ends - starts)
        estimate = np.add.reduceat(probabilities[key][order], starts) / (ends - starts)
        correctness.append(weights @ correct)
        predicted.append(weights @ estimate)
    return np.mean(correctness, axis=0), np.mean(predicted, axis=0)


def tune_targets(batch, labels, gain, price):
    """One pass yields full-range observed and small-start observed/proxy choices."""
    minimums = np.arange(10, 513)
    row = np.arange(batch.trials)[:, None]
    best = {name: (-np.inf, np.inf, None) for name in (
        "observed_outcome_tuned", "minimum10_16_observed_tuned", "minimum10_16_proxy_tuned")}
    for alpha in old.MULTIPLIERS:
        raw = scan.first_crossing(gain, batch.mean_tokens, alpha, price)
        index = np.maximum(raw[:, None], minimums) - 1
        tokens = batch.cumulative_tokens[row, index].mean(axis=0)
        observed = labels[row, index].mean(axis=0) - price * tokens
        proxy = batch.best_probabilities[row, index].mean(axis=0) - price * tokens
        for name in best:
            scores = proxy if name.endswith("proxy_tuned") else observed
            end = 7 if name.startswith("minimum10_16") else len(scores)
            peak = scores[:end].max()
            ties = np.flatnonzero(scores[:end] >= peak - 1e-15)
            j = ties[np.argmin(tokens[ties])]
            previous, previous_tokens, _ = best[name]
            if scores[j] > previous + 1e-15 or (abs(scores[j]-previous) <= 1e-15 and tokens[j] < previous_tokens):
                rule = bridge.Rule(width=2, alpha=alpha, beta=0, minimum=int(minimums[j]), cap=512,
                                   smoothing="mean", guard=False, latch=True)
                best[name] = (float(scores[j]), float(tokens[j]), rule)
    return best


def run(args):
    output, source, reference = args.output, args.source, args.reference
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((source / "manifest.json").read_text())
    saved = json.loads((reference / "selected_policies.json").read_text())
    reference_rows = list(csv.DictReader((reference / "mean_retuned" / "split_metrics.csv").open()))
    data = Path(meta["data_path"])
    assert old.sha256(data) == meta["data_sha256"]
    old.atomic_json(output / "manifest.json", dict(scope=__doc__, data_sha256=meta["data_sha256"],
        source=str(source.resolve()), reference=str(reference.resolve()), script_sha256=old.sha256(Path(__file__)),
        methods=METHODS, prices=PRICES.tolist(), splits=args.splits, train_permutations=24,
        test_permutations=48, cost="unadjusted prefix mean; actual total token expenditure charged",
        main_tuned_fix="selected calibration correctness minus actual token cost; no test labels",
        count_grid=list(range(10, 513)), small_count_grid=list(range(10, 17)), multipliers=old.MULTIPLIERS,
        uncertainty="1000 paired identity bootstraps; conditional, pointwise, not selection-adjusted"))
    problems = old.load_coding_problems(data)
    rows, records = {name: [] for name in METHODS}, {name: [] for name in METHODS}
    policies, budget, curves, stopping = [], [], [], []
    start_all = time.monotonic()
    for split in range(args.splits):
        tic = time.monotonic()
        train_ids, test_ids = old.split_problem_ids(problems, meta["outer_seed"] + split)
        profile = old.CodingProfile.load(source / "profiles" / f"split_{split:02d}.json")
        assert set(profile.metadata["fit_problem_ids"]) == set(train_ids)
        assert set(profile.metadata["holdout_problem_ids"]) == set(test_ids)
        probability = old.calibrated_rewards(problems, profile)
        parts, exact = [], []
        for group, ids, count, offset in (("calibration", train_ids, 24, 100000), ("test", test_ids, 48, 900000)):
            subset = {key: problems[key] for key in ids}
            orders = old.make_permutations(subset, count, meta["outer_seed"] + offset + 10000 * split)
            batch = old.build_trajectory_batch(subset, probability, orders, width=2)
            labels = old.selected_correctness(batch, subset, orders)
            qc, qp = exact_curves(subset, probability)
            mean_length = float(np.mean([length for key in ids for length in problems[key].lengths]))
            exact.append((qc, qp, mean_length))
            parts.append((batch, labels))
            for n in range(1, 513):
                curves.append(dict(split=split, group=group, n=n, correctness=float(qc[n-1]),
                                   predicted=float(qp[n-1]), expected_tokens=n*mean_length))
            for n in (4, 10, 16, 32, 64, 128, 256):
                stopping.append(dict(split=split, group=group, price=0, statistic="instantaneous_zero_excess",
                                     n=n, value=float(np.mean(batch.residual_current[:, n-1] == 0))))
        (train, train_y), (test, test_y) = parts
        train_gain, test_gain = scan.gain_base(train), scan.gain_base(test)
        cal_q, cal_p, cal_length = exact[0]
        for price in PRICES:
            price = float(price)
            ref = next(r for r in reference_rows if int(r["split"]) == split and abs(float(r["price"])-price)<1e-15)
            original = bridge.Rule(**next(r for r in saved if r["split"] == split and
                abs(r["price"]-price)<1e-15 and r["method"] == "mean_retuned")["rule"])
            fixed_n = int(float(ref["fixed_calls"]))
            assert fixed_n == int(np.argmax(cal_q - price*cal_length*np.arange(1,513)))+1
            fixed = bridge.take(test, test_y, np.full(test.trials, fixed_n), price)
            np.testing.assert_allclose(fixed["profit"].mean(), float(ref["fixed_profit"]), atol=1e-12)
            current_counts = bridge.stops(test, None, price, original)
            current = bridge.take(test, test_y, current_counts, price)
            np.testing.assert_allclose(current["profit"].mean(), float(ref["profit"]), atol=1e-12)
            raw = scan.first_crossing(test_gain, test.mean_tokens, original.alpha, price)
            for n in (4, 10, 16, 32):
                stopping.append(dict(split=split, group="test", price=price, statistic="first_crossing_by",
                                     n=n, value=float(np.mean(raw<=n))))
            stopping.append(dict(split=split, group="test", price=price, statistic="stops_at_minimum",
                                 n=original.minimum, value=float(np.mean(current_counts==original.minimum))))
            zero4 = test.residual_current[:, 3] == 0
            stopping.append(dict(split=split, group="test", price=price, statistic="crossing_at4_zero_excess",
                n=4, value=float(np.mean((raw==4) & zero4))))
            minimum = bridge.take(test, test_y, np.full(test.trials, original.minimum), price)
            n, m = fixed_n-1, original.minimum-1
            budget.append(dict(split=split, price=price, fixed_n=fixed_n, minimum=original.minimum,
                calibration_predicted_quality_gain=float(cal_p[m]-cal_p[n]),
                calibration_actual_quality_gain=float(cal_q[m]-cal_q[n]),
                calibration_extra_cost=price*cal_length*(m-n),
                calibration_predicted_profit_gain=float(cal_p[m]-cal_p[n]-price*cal_length*(m-n)),
                calibration_actual_profit_gain=float(cal_q[m]-cal_q[n]-price*cal_length*(m-n)),
                test_quality_gain=float(np.mean(current["correct"]-fixed["correct"])),
                test_extra_cost=float(price*np.mean(current["tokens"]-fixed["tokens"])),
                test_total_profit_gain=float(np.mean(current["profit"]-fixed["profit"])),
                test_minimum_budget_profit_gain=float(np.mean(minimum["profit"]-fixed["profit"])),
                test_stopping_profit_gain=float(np.mean(current["profit"]-minimum["profit"]))))
            tuned = tune_targets(train, train_y, train_gain, price)
            fixed_proxy = int(np.argmax(cal_p-price*cal_length*np.arange(1,513)))+1
            fixed_mc = int(np.argmax(train_y.mean(axis=0)-price*train.cumulative_tokens.mean(axis=0)))+1
            variants = {
                "current_mean": original,
                "fixed_at_minimum": replace(original, kind="fixed"),
                "minimum10_frozen_alpha": replace(original, minimum=10),
                "fixed_proxy_tuned": replace(original, kind="fixed", minimum=fixed_proxy),
                "fixed_observed_mc_tuned": replace(original, kind="fixed", minimum=fixed_mc),
                "baseline_cap_control": replace(original, minimum=min(original.minimum,fixed_n), cap=fixed_n),
                **{name: value[2] for name, value in tuned.items()},
            }
            for name, rule in variants.items():
                count = bridge.stops(test, None, price, rule)
                result = bridge.take(test, test_y, count, price)
                rows[name].append(dict(split=split, price=price, minimum=rule.minimum, cap=rule.cap,
                    accuracy=float(result["correct"].mean()), tokens=float(result["tokens"].mean()),
                    calls=float(count.mean()), profit=float(result["profit"].mean()),
                    fixed_accuracy=float(fixed["correct"].mean()), fixed_tokens=float(fixed["tokens"].mean()),
                    fixed_calls=fixed_n, fixed_profit=float(fixed["profit"].mean()),
                    cap_rate=float(np.mean(count==rule.cap)), minimum_rate=float(np.mean(count==rule.minimum))))
                policies.append(dict(split=split, price=price, method=name, rule=asdict(rule),
                                     calibration_score=tuned[name][0] if name in tuned else None))
                for j, identity in enumerate(sorted(test_ids)):
                    sl = slice(48*j,48*(j+1))
                    records[name].append(dict(split=split, price=price, problem_id=identity,
                        profit=float(result["profit"][sl].mean()), fixed_profit=float(fixed["profit"][sl].mean()),
                        accuracy=float(result["correct"][sl].mean()), tokens=float(result["tokens"][sl].mean()),
                        calls=float(count[sl].mean()), fixed_calls=fixed_n,
                        fixed_accuracy=float(fixed["correct"][sl].mean()), fixed_tokens=float(fixed["tokens"][sl].mean())))
        for name in METHODS:
            folder=output/name
            folder.mkdir(exist_ok=True)
            old.atomic_csv(folder/"split_metrics.csv",rows[name])
            old.atomic_csv(folder/"problem_metrics.csv",records[name])
        old.atomic_csv(output/"budget_decomposition.csv",budget)
        old.atomic_csv(output/"exact_quality_curves.csv",curves)
        old.atomic_csv(output/"stopping_diagnostics.csv",stopping)
        old.atomic_json(output/"selected_policies.json",policies)
        print(f"split {split}: diagnostics completed in {time.monotonic()-tic:.1f}s",flush=True)
    for name in METHODS:
        print(name,flush=True)
        scan.summarize(rows[name],records[name],output/name)
    old.atomic_json(output/"validation.json",dict(completed_splits=args.splits,
        reference_adaptive_and_fixed_reproduced=True, exact_fixed_n_choices_reproduced=True,
        elapsed_seconds=time.monotonic()-start_all))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--splits",type=int,default=10)
    run(parser.parse_args())
