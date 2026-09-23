"""Distribution-free top-three stopping on cached candidate pools only.

No utility distribution is fitted. Coding alone fits a held-out monotone
reward-to-success calibration. Alignment uses an empirical prefix reference.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Pool:
    id: str
    rewards: np.ndarray
    lengths: np.ndarray
    correct: np.ndarray | None = None


def sigmoid(x):
    x = np.clip(x, -700, 700)
    return 1.0 / (1.0 + np.exp(-x))


def q99(values):
    """Same order-statistic convention as the saved alignment runner."""
    return float(np.sort(values)[min(int(.99 * len(values)), len(values) - 1)])


def load_alignment(path, reward_key, length_key):
    opener = gzip.open if path.suffix == '.gz' else open
    pools = []
    with opener(path, 'rt') as handle:
        for i, line in enumerate(handle):
            row = json.loads(line)
            samples = row['generations']
            lengths = [len(s['text']) if length_key == 'text_chars'
                       else float(s[length_key]) for s in samples]
            pools.append(Pool(str(row.get('JSON_idx', i)),
                              np.array([s[reward_key] for s in samples], float),
                              np.array(lengths, float)))
    return pools


def load_coding(path, cache_path, length_key):
    """Native data.jsonl + NPZ {ids, indices, chars|tokens} adapter."""
    scored = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            samples = sorted(row['samples'], key=lambda s: int(s['idx']))
            if any(s['correct'] for s in samples):
                scored[str(row['id'])] = samples
    pools = []
    with np.load(cache_path, allow_pickle=False) as cache:
        for id_, indices, lengths in zip(cache['ids'], cache['indices'], cache[length_key]):
            key = str(id_)
            samples = scored[key]
            if list(map(int, indices)) != [int(s['idx']) for s in samples]:
                raise ValueError(f'Cache index mismatch: {key}')
            pools.append(Pool(key, np.array([s['r_score'] for s in samples], float),
                              np.asarray(lengths, float),
                              np.array([s['correct'] for s in samples], float)))
    if set(p.id for p in pools) != set(scored):
        raise ValueError('Length cache does not cover the solvable coding cohort')
    return pools


def validate(pools):
    if len({p.id for p in pools}) != len(pools):
        raise ValueError('Duplicate prompt/problem IDs')
    for p in pools:
        if not len(p.rewards) or p.rewards.shape != p.lengths.shape:
            raise ValueError(f'Invalid pool lengths for {p.id}')
        if not np.all(np.isfinite(p.rewards)) or not np.all(np.isfinite(p.lengths)):
            raise ValueError('Nonfinite values')
        if np.any(p.lengths < 0):
            raise ValueError('Negative generation lengths')
        if p.correct is not None and (p.correct.shape != p.rewards.shape or
                                      not np.all(np.isin(p.correct, [0, 1]))):
            raise ValueError('Invalid correctness labels')


def stop_count(rewards, lengths, price, transform, variant='doubling', fixed_cost=None):
    """Only a prefix is passed to transform; no labels/reference enter policy.

    doubling: exact a=1 checkpoints and positivity guard, with a finite cap.
    sequential: identical statistic and guard, practical every-sample checks.
    fixed_cost=None uses price * observed mean length, a heuristic extension.
    """
    if variant not in ('doubling', 'sequential'):
        raise ValueError('Unknown variant')
    cap = len(rewards)
    if cap < 4:
        return cap
    checkpoints = (range(4, cap + 1) if variant == 'sequential'
                   else (2 ** j for j in range(2, cap.bit_length())))
    for n in checkpoints:
        utility = np.asarray(transform(rewards[:n]), float)
        if utility.shape != (n,) or np.any(~np.isfinite(utility)) or np.any(utility < 0):
            raise ValueError('Transform must produce finite nonnegative prefix utilities')
        low, middle, high = np.sort(utility)[-3:]
        gain = 2.0 * (middle + high - 2.0 * low) / n
        cost = fixed_cost if fixed_cost is not None else price * float(np.mean(lengths[:n]))
        if low > 0 and gain <= cost:
            return n
    return cap


def replay(pool, orders, prices, counts, transform, alignment):
    """Return prompt-level averages, retaining pairing across all policies."""
    methods = [f'fixed_{n}' for n in counts] + ['dmrl_doubling', 'dmrl_sequential']
    # Quality, monetary cost, samples, profit. Full-pool q99 is evaluation ONLY.
    output = {price: {m: [] for m in methods} for price in prices}
    reference = q99(pool.rewards) if alignment else None
    for order in orders:
        rewards, lengths = pool.rewards[order], pool.lengths[order]
        cumulative = np.cumsum(lengths)
        best = np.maximum.accumulate(rewards)
        best_index = np.empty(len(order), int)
        incumbent = 0
        for j in range(len(order)):
            if rewards[j] > rewards[incumbent]:
                incumbent = j
            best_index[j] = incumbent
        quality = (sigmoid(best - reference) if alignment
                   else pool.correct[order[best_index]])
        for price in prices:
            stops = {f'fixed_{n}': n for n in counts}
            for variant in ('doubling', 'sequential'):
                stops[f'dmrl_{variant}'] = stop_count(rewards, lengths, price, transform, variant)
            for method, n in stops.items():
                q, cost = float(quality[n - 1]), price * float(cumulative[n - 1])
                output[price][method].append([q, cost, n, q - cost])
    return {p: {m: np.mean(v, axis=0) for m, v in methods_.items()}
            for p, methods_ in output.items()}


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(pools, args):
    validate(pools)
    if len(pools) < 4:
        raise ValueError('At least four prompts required for a held-out comparison')
    cap = min(args.cap, min(len(p.rewards) for p in pools))
    counts = list(range(1, cap + 1))
    retained = {f'fixed_{2 ** j}' for j in range(cap.bit_length())} | {f'fixed_{cap}', 'dmrl_doubling', 'dmrl_sequential'}
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(len(pools))
    train_idx = set(map(int, shuffled[:len(pools) // 2]))
    if args.task == 'coding':
        from sklearn.isotonic import IsotonicRegression
        calibration = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(
            np.concatenate([pools[i].rewards for i in train_idx]),
            np.concatenate([pools[i].correct for i in train_idx]))
        transform = calibration.predict
    else:
        calibration = None
        # Unknown per-prompt benchmark estimated from observed rewards only.
        # This recomputes all prefix utilities and is NOT the fixed-utility theorem.
        transform = lambda rewards: sigmoid(rewards - q99(rewards))
    rows, prompt_results = [], []
    for i, pool in enumerate(pools):
        orders = [rng.permutation(len(pool.rewards))[:cap] for _ in range(args.permutations)]
        result = replay(pool, orders, args.prices, counts, transform, args.task == 'alignment')
        prompt_results.append(result)
        for price, by_method in result.items():
            for method, values in by_method.items():
                if method not in retained:
                    continue
                rows.append(dict(id=pool.id, partition='train' if i in train_idx else 'test',
                                 price=price, method=method, quality=values[0], cost=values[1],
                                 samples=values[2], profit=values[3]))
    summaries, comparisons = [], []
    metrics = ('quality', 'cost', 'samples', 'profit')
    for price in args.prices:
        methods = list(prompt_results[0][price])
        means = {}
        for partition in ('train', 'test'):
            for method in methods:
                values = np.array([v[price][method] for i, v in enumerate(prompt_results)
                                   if (i in train_idx) == (partition == 'train')])
                means[partition, method] = values
                summaries.append(dict(partition=partition, price=price, method=method,
                                      prompts=len(values), **dict(zip(metrics, values.mean(axis=0)))))
        chosen = max((f'fixed_{n}' for n in counts),
                     key=lambda m: means['train', m][:, 3].mean())
        oracle = max((f'fixed_{n}' for n in counts),
                     key=lambda m: means['test', m][:, 3].mean())
        if chosen not in retained:
            for i, result in enumerate(prompt_results):
                values = result[price][chosen]
                rows.append(dict(id=pools[i].id, partition='train' if i in train_idx else 'test',
                                 price=price, method=chosen, quality=values[0], cost=values[1],
                                 samples=values[2], profit=values[3]))
        for method in ('dmrl_doubling', 'dmrl_sequential'):
            values, baseline = means['test', method], means['test', chosen]
            delta = values[:, 3] - baseline[:, 3]
            bootstrap = rng.choice(delta, (args.bootstrap, len(delta)), replace=True).mean(axis=1)
            lo, hi = np.quantile(bootstrap, [.025, .975])
            bprofit = baseline[:, 3].mean()
            comparisons.append(dict(price=price, method=method, train_selected_fixed=chosen,
                                    test_oracle_fixed=oracle, mean_profit=values[:, 3].mean(),
                                    baseline_profit=bprofit, delta_profit=delta.mean(),
                                    delta_ci_low=lo, delta_ci_high=hi,
                                    relative_profit_percent=100 * delta.mean() / bprofit if bprofit > 0 else '',
                                    quality=values[:, 0].mean(), baseline_quality=baseline[:, 0].mean(),
                                    cost=values[:, 1].mean(), baseline_cost=baseline[:, 1].mean(),
                                    samples=values[:, 2].mean(),
                                    delta_vs_test_oracle=values[:, 3].mean()-means['test', oracle][:, 3].mean()))
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'prompt_metrics.csv', rows)
    write_csv(args.output / 'summary.csv', summaries)
    write_csv(args.output / 'comparisons.csv', comparisons)
    metadata = dict(task=args.task, data=str(args.data), seed=args.seed, cap=cap,
                    reward_key=args.reward_key, length_key=args.length_key,
                    length_cache=str(args.length_cache) if args.length_cache else None,
                    data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    length_unit=args.length_unit, prices=args.prices, fixed_counts=counts,
                    train_ids=[p.id for i, p in enumerate(pools) if i in train_idx],
                    test_ids=[p.id for i, p in enumerate(pools) if i not in train_idx],
                    permutations=args.permutations, bootstrap=args.bootstrap,
                    inference='paired prompt-cluster bootstrap conditional on training split; pointwise, not multiplicity corrected',
                    candidate_sampling='without replacement; same orders across all policies',
                    cost_estimator='price times observed prefix mean length; all opened lengths charged',
                    alignment_reference='prefix empirical q99 for decisions; full-pool q99 for evaluation only',
                    theorem_applies=False,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if calibration is not None:
        metadata['calibration'] = dict(x=calibration.X_thresholds_.tolist(), y=calibration.y_thresholds_.tolist())
    (args.output / 'METHOD.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['coding', 'alignment'], required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing outputs are never overwritten')
    parser.add_argument('--reward-key', default='mistral_rm_reward')
    parser.add_argument('--length-key', default='text_chars', help='Alignment: text_chars or recorded length field. Coding: chars or tokens NPZ key')
    parser.add_argument('--length-unit', choices=['characters', 'tokens'], required=True)
    parser.add_argument('--length-cache', type=Path, help='Coding NPZ with ids, indices, length-key')
    parser.add_argument('--prices', type=float, nargs='+', default=[2e-8, 1e-7, 2e-7, 1e-6, 2e-6, 1e-5])
    parser.add_argument('--cap', type=int, default=512)
    parser.add_argument('--permutations', type=int, default=8)
    parser.add_argument('--seed', type=int, default=30)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists; choose a new path')
    if args.cap < 1 or args.permutations < 1 or args.bootstrap < 1 or any(p <= 0 for p in args.prices):
        parser.error('Cap, permutations, bootstrap and prices must be positive')
    if args.length_key in ('text_chars', 'chars') and args.length_unit != 'characters':
        parser.error('Character lengths cannot be labeled tokens')
    if args.task == 'coding':
        if args.length_cache is None or args.length_key == 'text_chars':
            parser.error('Coding requires --length-cache and --length-key chars|tokens')
        pools = load_coding(args.data, args.length_cache, args.length_key)
    else:
        pools = load_alignment(args.data, args.reward_key, args.length_key)
    comparisons = evaluate(pools, args)
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
