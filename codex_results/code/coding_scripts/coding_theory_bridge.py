"""Exploratory coding ablations; reuse frozen calibration maps and paired streams.

Never change the manuscript or prior results. Candidate selection uses only
calibration problems; test outcomes are descriptive because this cohort has
already supported development. All policies charge actual output tokens.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import coding_token_profit_dmrl as old


@dataclass(frozen=True)
class Rule:
    width: int = 2
    alpha: float = 4.0
    smoothing: str = "current"
    beta: float = 0.0
    minimum: int = 4
    cap: int = 512
    guard: bool = True
    latch: bool = False
    kind: str = "excess"
    budget: float = 0.0


def stops(batch, positive_counts, price, rule):
    """Only prefix utilities and lengths enter this decision; never labels."""
    cap = min(rule.cap, batch.samples)
    start = max(4, rule.width + 1)
    if rule.kind == "length":
        eligible = batch.cumulative_tokens[:, :cap] >= rule.budget
    elif rule.kind == "fixed":
        return np.full(batch.trials, min(rule.minimum, cap), dtype=int)
    else:
        residual = batch.residual_current[:, :cap].astype(float)
        if rule.smoothing != "current":
            sums = np.cumsum(np.where(np.arange(cap) >= start-1, residual, 0), axis=1)
            counts = np.maximum(np.arange(1, cap+1) - start + 1, 1)
            if rule.smoothing == "mean":
                residual = sums / counts
            elif rule.smoothing == "recent_half":
                recent = (counts + 1) // 2
                before = np.arange(cap) - recent
                previous = np.where(before[None, :] >= 0, sums[:, np.maximum(before, 0)], 0)
                residual = (sums - previous) / recent
            else:
                raise ValueError(rule.smoothing)
        cost = price * batch.mean_tokens[:, :cap] / (1 + rule.beta * batch.token_se_ratio[:, :cap])
        eligible = rule.alpha * residual / np.arange(1, cap+1) <= cost
        if rule.guard:
            eligible &= positive_counts[:, :cap] > rule.width
    earliest = start if rule.latch else max(start, rule.minimum)
    eligible[:, :earliest-1] = False
    first = np.where(eligible.any(axis=1), eligible.argmax(axis=1) + 1, cap)
    if rule.latch:
        first = np.maximum(first, rule.minimum)
    return np.minimum(first, cap)


def candidates():
    """Frozen before inspecting this study's outcomes. No baseline-derived caps."""
    families = {"near_direct": [Rule()], "near_direct_no_guard": [Rule(guard=False)]}
    families["top2_multiplier"] = [Rule(alpha=a) for a in old.MULTIPLIERS]
    families["top2_smoothed"] = [Rule(alpha=a, smoothing=s) for a in old.MULTIPLIERS for s in old.SMOOTHING_MODES]
    families["top2_smoothed_cost"] = [Rule(alpha=a, smoothing=s, beta=b) for a in old.MULTIPLIERS for s in old.SMOOTHING_MODES for b in old.COST_ADJUSTMENTS]
    families["top2_burnin"] = [Rule(alpha=a, smoothing=s, minimum=m) for a in old.MULTIPLIERS for s in old.SMOOTHING_MODES for m in (4, 8, 16, 32)]
    families["top4_mean_cost"] = [Rule(width=4, minimum=5, guard=False, alpha=a, smoothing=s) for a in old.MULTIPLIERS for s in old.SMOOTHING_MODES]
    families["top4_full_cost"] = [Rule(width=4, minimum=5, guard=False, alpha=a, smoothing=s, beta=b) for a in old.MULTIPLIERS for s in old.SMOOTHING_MODES for b in old.COST_ADJUSTMENTS]
    return families


def batch_pair(problems, probability, orders):
    batches = {w: old.build_trajectory_batch(problems, probability, orders, width=w) for w in (2, 4)}
    positive = np.vstack([np.cumsum(probability[key][order] > 0, axis=1) for key, order in sorted(orders.items())])
    labels = old.selected_correctness(batches[2], problems, orders)
    return batches, positive, labels


def take(batch, labels, count, price):
    idx = (np.arange(batch.trials), count-1)
    tokens = batch.cumulative_tokens[idx]
    return dict(correct=labels[idx], tokens=tokens, calls=count,
                predicted=batch.best_probabilities[idx], profit=labels[idx]-price*tokens)


def run(args, reference_variants=None):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    source = args.reference
    manifest = json.loads((source / "manifest.json").read_text())
    data = Path(manifest["data_path"])
    digest = hashlib.file_digest(data.open("rb"), "sha256").hexdigest()
    assert digest == manifest["data_sha256"]
    problems = old.load_coding_problems(data)
    prices = manifest["prices_per_output_token"]
    selected = json.loads((source / "selected_policies.json").read_text())
    grid = candidates()
    meta = dict(scope="exploratory reused-cohort comparison, not independent confirmation",
                source=str(source.resolve()), data_sha256=digest, prices=prices,
                splits=args.splits, family_grid={k: [asdict(r) for r in v] for k,v in grid.items()},
                tuning_targets=["calibrated", "correctness"], cap=512,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    old.atomic_json(output / "manifest.json", meta)
    rows, policies, records = [], [], []
    for split in range(args.splits):
        tic = time.monotonic()
        train_ids, test_ids = old.split_problem_ids(problems, manifest["outer_seed"] + split)
        profile = old.CodingProfile.load(source / "profiles" / f"split_{split:02d}.json")
        assert set(profile.metadata["fit_problem_ids"]) == set(train_ids)
        assert set(profile.metadata["holdout_problem_ids"]) == set(test_ids)
        probabilities = old.calibrated_rewards(problems, profile)
        parts = []
        for ids, permutations, offset in ((train_ids, 24, 100000), (test_ids, 48, 900000)):
            subset = {key: problems[key] for key in ids}
            order = old.make_permutations(subset, permutations, manifest["outer_seed"]+offset+10000*split)
            parts.append(batch_pair(subset, probabilities, order))
        (train, pos_train, y_train), (test, pos_test, y_test) = parts
        print(f"split {split}: trajectories built in {time.monotonic()-tic:.1f}s", flush=True)
        for price in prices:
            ref = next(r for r in selected if r["split"] == split and np.isclose(1/r["utility_divisor"], price, rtol=1e-10, atol=0))
            fixed_n = ref["fixed_n_tuned_on_calibration"]
            config = ref["adaptive_policy_tuned_on_calibration"]
            cfg = old.PolicyConfig(config["smoothing"], config["multiplier"], config["cost_adjustment"], config["cap_factor_vs_train_tuned_fixed_n"], config["minimum_factor_vs_train_tuned_fixed_n"], 4)
            minimum = old.minimum_for_config(cfg, fixed_n, 512)
            cap = old.cap_for_config(cfg, fixed_n, 512)
            original = Rule(4, cfg.multiplier, cfg.smoothing, cfg.cost_adjustment, minimum, cap, False, True)
            original_stop = old.stop_counts(test[4], 1/price, fixed_n, cfg)
            # Existing smoothing is float32: roundoff may affect equality only.
            np.testing.assert_array_equal(stops(test[4], pos_test, price, original), original_stop)
            fixed = take(test[2], y_test, np.full(test[2].trials, fixed_n), price)

            def record(name, rule, selection=None):
                batch = test[rule.width]
                count = stops(batch, pos_test, price, rule)
                metrics = take(batch, y_test, count, price)
                rows.append(dict(split=split, price=price, method=name, fixed_n=fixed_n,
                                 accuracy=float(metrics["correct"].mean()), tokens=float(metrics["tokens"].mean()),
                                 calls=float(count.mean()), profit=float(metrics["profit"].mean()),
                                 fixed_accuracy=float(fixed["correct"].mean()), fixed_tokens=float(fixed["tokens"].mean()),
                                 fixed_profit=float(fixed["profit"].mean()), cap_rate=float(np.mean(count==rule.cap))))
                policies.append(dict(split=split, price=price, method=name, rule=asdict(rule), calibration=selection))
                # Keep paired prompt-level outcomes; orderings are not independent identities.
                for j, identity in enumerate(sorted(test_ids)):
                    sl = slice(j*48, (j+1)*48)
                    records.append(dict(split=split, price=price, method=name, problem_id=identity,
                                        profit=float(metrics["profit"][sl].mean()), fixed_profit=float(fixed["profit"][sl].mean()),
                                        correct=float(metrics["correct"][sl].mean()), tokens=float(metrics["tokens"][sl].mean()),
                                        fixed_correct=float(fixed["correct"][sl].mean()), fixed_tokens=float(fixed["tokens"][sl].mean())))

            record("original", original)
            record("original_fresh_check", Rule(**{**asdict(original), "latch":False}))
            if reference_variants is not None:
                for name, rule in reference_variants(original).items():
                    record(name, rule)
            for family, rules in grid.items():
                winners = {target: (-np.inf, np.inf, None) for target in ("calibrated", "correctness")}
                for rule in rules:
                    batch = train[rule.width]
                    count = stops(batch, pos_train, price, rule)
                    metrics = take(batch, y_train, count, price)
                    mean_tokens = float(metrics["tokens"].mean())
                    for target in winners:
                        outcome = metrics["predicted"] if target=="calibrated" else metrics["correct"]
                        score = float(np.mean(outcome-price*metrics["tokens"]))
                        previous = winners[target]
                        if score>previous[0]+1e-15 or (abs(score-previous[0])<=1e-15 and mean_tokens<previous[1]):
                            winners[target] = (score, mean_tokens, rule)
                for target, (score, _, rule) in winners.items():
                    if len(rules)==1 and target=="correctness":
                        continue
                    name = family if len(rules)==1 else family+"_"+target
                    record(name, rule, score)
            # Simple response-length budgets, selected on the same calibration streams.
            mean_length = np.mean([v for key in train_ids for v in problems[key].lengths])
            controls = [Rule(kind="length", budget=float(mean_length*n)) for n in (4,8,16,32,64,96,128,192,256,384,512)]
            for target in ("calibrated", "correctness"):
                values=[]
                for rule in controls:
                    result=take(train[2], y_train, stops(train[2],pos_train,price,rule), price)
                    outcome=result["predicted"] if target=="calibrated" else result["correct"]
                    values.append(float(np.mean(outcome-price*result["tokens"])))
                best=int(np.argmax(values))
                record("length_budget_"+target,controls[best],values[best])
        old.atomic_csv(output / "split_metrics.csv", rows)
        old.atomic_csv(output / "problem_metrics.csv", records)
        old.atomic_json(output / "selected_policies.json", policies)
        print(f"split {split} complete in {time.monotonic()-tic:.1f}s", flush=True)
    summary=[]
    for method in sorted({r["method"] for r in rows}):
        for price in prices:
            subset=[r for r in rows if r["method"]==method and r["price"]==price]
            means={key:float(np.mean([r[key] for r in subset])) for key in ("accuracy","tokens","calls","profit","fixed_accuracy","fixed_tokens","fixed_profit","cap_rate")}
            summary.append(dict(method=method, price=price, **means,
                                relative_profit_percent=100*(means["profit"]/means["fixed_profit"]-1),
                                token_change_percent=100*(means["tokens"]/means["fixed_tokens"]-1)))
    old.atomic_csv(output / "summary.csv",summary)
    for method in sorted({r["method"] for r in summary}):
        print(method, [round(r["relative_profit_percent"],2) for r in summary if r["method"]==method],flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--splits",type=int,default=10)
    run(parser.parse_args())
