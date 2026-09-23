"""Prespecified, train-selected nonparametric stopping policies on cached alignment.

No distribution is fitted. These are empirical policies, not confidence bounds.
All prices charge recorded characters. No generation or manuscript edits.
"""
from __future__ import annotations

import argparse
from bisect import insort
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment, q99, sigmoid, validate


PRICES = np.array([2e-8, 1e-7, 2e-7, 1e-6, 2e-6, 1e-5])
GENERATORS = ['gemma2_9b', 'llama3.1_8b', 'llama3.2_3b', 'mistral_7b', 'qwen2.5_7b']


@dataclass(frozen=True)
class Policy:
    family: str
    width: int
    multiplier: float
    minimum: int

    @property
    def name(self):
        return f'{self.family}_w{self.width}_a{self.multiplier:g}_n{self.minimum}'


def policies():
    # Freeze this grid before reading the test results. No test-selected winners.
    result = [Policy('spacing', k, a, max(4, k + 1))
              for k in (1, 2, 4) for a in (.125, .25, .5, 1., 2., 4.)]
    result += [Policy('recent_gain', w, a, 2*w)
               for w in (4, 8, 16) for a in (.5, 1., 2.)]
    result += [Policy('remaining_range', 0, a, n)
               for n in (4, 8) for a in (.03, .1, .3, 1.)]
    return result


def prefix_statistics(rewards):
    """Every entry at index n-1 depends only on rewards[:n].

    spacing[k] averages the largest k excesses above the (k+1)st largest.
    recent_gain[w] measures improvement over the preceding w observations,
    evaluating both incumbents against the SAME current-prefix BT reference.
    remaining_range uses the unachieved portion of the unit utility range.
    """
    cap = len(rewards)
    features = {('spacing', k): np.full(cap, np.inf) for k in (1, 2, 4)}
    features.update({('recent_gain', w): np.full(cap, np.inf) for w in (4, 8, 16)})
    features['remaining_range', 0] = np.full(cap, np.inf)
    ordered = []
    incumbent = np.maximum.accumulate(rewards)
    for j, reward in enumerate(rewards):
        insort(ordered, float(reward))
        n = j + 1
        reference = ordered[min(int(.99*n), n-1)]
        best = float(sigmoid(ordered[-1] - reference))
        features['remaining_range', 0][j] = (1-best)/(n+1)
        for k in (1, 2, 4):
            if n > k:
                upper = sigmoid(np.array(ordered[-k:]) - reference)
                floor = float(sigmoid(ordered[-k-1] - reference))
                features['spacing', k][j] = (float(upper.mean()) - floor)/n
        for w in (4, 8, 16):
            if n > w:
                previous = float(sigmoid(incumbent[j-w] - reference))
                features['recent_gain', w][j] = max(0., best-previous)/w
    return features


def stopping_counts(features, lengths, configs, prices=PRICES):
    cap = len(lengths)
    average = np.cumsum(lengths)/np.arange(1, cap+1)
    gain = np.stack([p.multiplier * features[p.family, p.width] for p in configs])
    eligible = np.arange(1, cap+1)[None, :] >= np.array([p.minimum for p in configs])[:, None]
    hit = (gain[None, :, :] <= prices[:, None, None]*average[None, None, :]) & eligible[None, :, :]
    # The cap is a mandatory return, even when no stopping condition fired.
    hit[:, :, -1] = True
    return np.argmax(hit, axis=2)+1


def collect(pools, configs, cap, permutations, seed):
    """Arrays are [prompt, price, policy/count, quality/cost/samples/profit]."""
    adaptive = np.zeros((len(pools), len(PRICES), len(configs), 4))
    fixed = np.zeros((len(pools), len(PRICES), cap, 4))
    rng = np.random.default_rng(seed)
    for i, pool in enumerate(pools):
        reference = q99(pool.rewards)  # Evaluation only: never passed to statistics.
        for _ in range(permutations):
            order = rng.permutation(len(pool.rewards))[:cap]
            rewards, lengths = pool.rewards[order], pool.lengths[order]
            features = prefix_statistics(rewards)
            stops = stopping_counts(features, lengths, configs)
            quality = sigmoid(np.maximum.accumulate(rewards)-reference)
            costs = PRICES[:, None]*np.cumsum(lengths)[None, :]
            fixed[i, :, :, 0] += quality[None, :]
            fixed[i, :, :, 1] += costs
            fixed[i, :, :, 2] += np.arange(1, cap+1)[None, :]
            fixed[i, :, :, 3] += quality[None, :]-costs
            adaptive[i, :, :, 0] += quality[stops-1]
            adaptive[i, :, :, 1] += np.take_along_axis(costs, stops-1, axis=1)
            adaptive[i, :, :, 2] += stops
            adaptive[i, :, :, 3] += quality[stops-1]-np.take_along_axis(costs, stops-1, axis=1)
        if (i+1) % 20 == 0:
            print(f'  replayed {i+1}/{len(pools)} prompts', flush=True)
    return adaptive/permutations, fixed/permutations


def select_on_train(adaptive, fixed, train, configs):
    a = adaptive[train, :, :, 3].mean(axis=0)
    f = fixed[train, :, :, 3].mean(axis=0)
    groups = {family: [i for i, p in enumerate(configs) if p.family == family]
              for family in ('spacing', 'recent_gain', 'remaining_range')}
    groups['all_train_selected'] = list(range(len(configs)))
    groups['original_sequential'] = [next(i for i, p in enumerate(configs)
                                         if p == Policy('spacing', 2, 4., 4))]
    chosen = {name: np.array([idx[int(np.argmax(row[idx]))] for row in a])
              for name, idx in groups.items()}
    return chosen, np.argmax(f, axis=1)


def evaluate_splits(adaptive, fixed, ids, configs, seeds, generator):
    summaries, details = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ids))
        train, test = order[:len(ids)//2], order[len(ids)//2:]
        chosen, fixed_choices = select_on_train(adaptive, fixed, train, configs)
        for pi, price in enumerate(PRICES):
            baseline = fixed[test, pi, fixed_choices[pi]]
            oracle = fixed[test, pi, :, 3].mean(axis=0).max()
            for method, choices in chosen.items():
                ci = int(choices[pi])
                values = adaptive[test, pi, ci]
                delta = values[:, 3]-baseline[:, 3]
                boot = rng.choice(delta, (2000, len(test)), replace=True).mean(axis=1)
                mean, bmean = values.mean(axis=0), baseline.mean(axis=0)
                low, high = np.quantile(boot, [.025, .975])
                summaries.append(dict(generator=generator, split_seed=seed, price=price,
                    method=method, selected_policy=configs[ci].name, fixed_n=int(fixed_choices[pi]+1),
                    profit=mean[3], fixed_profit=bmean[3], delta_profit=delta.mean(),
                    relative_percent=100*delta.mean()/bmean[3] if bmean[3]>0 else '',
                    quality=mean[0], fixed_quality=bmean[0], cost=mean[1], fixed_cost=bmean[1],
                    samples=mean[2], ci_low=low, ci_high=high, delta_test_oracle=mean[3]-oracle))
                for j, idx in enumerate(test):
                    details.append(dict(generator=generator, split_seed=seed, price=price,
                        method=method, id=ids[idx], selected_policy=configs[ci].name,
                        profit=values[j, 3], fixed_profit=baseline[j, 3],
                        quality=values[j, 0], cost=values[j, 1], samples=values[j, 2]))
    return summaries, details


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('dataset/alpaca'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generators', nargs='+', default=GENERATORS)
    parser.add_argument('--cap', type=int, default=512)
    parser.add_argument('--permutations', type=int, default=8)
    parser.add_argument('--replay-seed', type=int, default=20260922)
    parser.add_argument('--split-seeds', type=int, nargs='+', default=[41, 42, 43, 44, 45])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory; existing results are preserved')
    if args.cap < 4 or args.permutations < 1:
        parser.error('Cap must be at least four; permutations must be positive')
    source = Path(__file__).read_bytes()
    configs = policies()
    args.output.mkdir(parents=True)
    metadata = dict(policies=[asdict(p) for p in configs], prices=PRICES.tolist(),
        cap=args.cap, permutations=args.permutations, replay_seed=args.replay_seed,
        split_seeds=args.split_seeds, source_sha256=hashlib.sha256(source).hexdigest(),
        utility='BT against prefix empirical q99 for decisions; full-pool q99 for evaluation only',
        cost='price times recorded text characters; all opened responses charged',
        selection='all policies and all fixed counts selected on training prompts only',
        scope='Exploratory reused prompts; overlapping splits; no independent confirmation',
        theorem_applies=False, datasets={})
    (args.output/'METHOD.json').write_text(json.dumps(metadata, indent=2)+'\n')
    summaries, details = [], []
    for generator in args.generators:
        print(generator, flush=True)
        path = args.data_dir/f'{generator}_output.merged_rm.jsonl.gz'
        pools = load_alignment(path, 'mistral_rm_reward', 'text_chars')
        validate(pools)
        cap = min(args.cap, min(len(p.rewards) for p in pools))
        ids = [p.id for p in pools]
        a, f = collect(pools, configs, cap, args.permutations, args.replay_seed)
        np.savez_compressed(args.output/f'{generator}_replay.npz', adaptive=a, fixed=f, ids=ids)
        s, d = evaluate_splits(a, f, ids, configs, args.split_seeds, generator)
        summaries.extend(s)
        details.extend(d)
        metadata['datasets'][generator] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                               ids=ids, actual_cap=cap)
        write_csv(args.output/'comparisons.csv', summaries)
        write_csv(args.output/'prompt_metrics.csv', details)
        (args.output/'METHOD.json').write_text(json.dumps(metadata, indent=2)+'\n')
        for method in ('spacing', 'recent_gain', 'remaining_range', 'all_train_selected', 'original_sequential'):
            rr = [r for r in s if r['method']==method]
            print(method, 'mean relative:', np.mean([r['relative_percent'] for r in rr]),
                  'positive:', sum(r['delta_profit']>0 for r in rr), '/', len(rr), flush=True)


if __name__ == '__main__':
    main()
