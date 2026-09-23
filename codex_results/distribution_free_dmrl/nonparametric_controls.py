"""Cost-only control on exactly the nonparametric study's replay streams.

Select a response-length budget on training prompts. Never inspect reward to
decide when to stop. This separates length adaptation from reward adaptation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment, q99, sigmoid
from nonparametric_study import PRICES, write_csv


def collect_budget(pools, cap, permutations, seed, budgets):
    result = np.zeros((len(pools), len(PRICES), len(budgets), 4))
    rng = np.random.default_rng(seed)
    for i, pool in enumerate(pools):
        reference = q99(pool.rewards)
        for _ in range(permutations):
            order = rng.permutation(len(pool.rewards))[:cap]
            cumulative = np.cumsum(pool.lengths[order])
            stops = np.minimum(np.searchsorted(cumulative, budgets)+1, cap)
            quality = sigmoid(np.maximum.accumulate(pool.rewards[order])-reference)[stops-1]
            cost = PRICES[:, None]*cumulative[stops-1][None, :]
            result[i, :, :, 0] += quality[None, :]
            result[i, :, :, 1] += cost
            result[i, :, :, 2] += stops[None, :]
            result[i, :, :, 3] += quality[None, :]-cost
    return result/permutations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    args = parser.parse_args()
    output = args.study/'cost_only_control'
    if output.exists():
        parser.error('Control outputs already exist; refusing overwrite')
    meta = json.loads((args.study/'METHOD.json').read_text())
    budgets = np.r_[np.geomspace(256, 2_000_000, 64), np.inf]
    output.mkdir()
    rows = []
    for generator, data in meta['datasets'].items():
        pools = load_alignment(Path(data['path']), 'mistral_rm_reward', 'text_chars')
        a = collect_budget(pools, data['actual_cap'], meta['permutations'], meta['replay_seed'], budgets)
        with np.load(args.study/f'{generator}_replay.npz') as cache:
            fixed = cache['fixed']
        np.savez_compressed(output/f'{generator}_replay.npz', adaptive=a)
        for seed in meta['split_seeds']:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(pools))
            train, test = order[:len(pools)//2], order[len(pools)//2:]
            ac = a[train, :, :, 3].mean(axis=0).argmax(axis=1)
            fc = fixed[train, :, :, 3].mean(axis=0).argmax(axis=1)
            for pi, price in enumerate(PRICES):
                values, baseline = a[test, pi, ac[pi]], fixed[test, pi, fc[pi]]
                delta = values[:, 3]-baseline[:, 3]
                boot = rng.choice(delta, (2000, len(test)), replace=True).mean(axis=1)
                low, high = np.quantile(boot, [.025, .975])
                mean, bmean = values.mean(axis=0), baseline.mean(axis=0)
                rows.append(dict(generator=generator, split_seed=seed, price=price,
                    selected_length_budget=budgets[ac[pi]], fixed_n=int(fc[pi]+1),
                    profit=mean[3], fixed_profit=bmean[3], delta_profit=delta.mean(),
                    relative_percent=100*delta.mean()/bmean[3] if bmean[3]>0 else '',
                    quality=mean[0], fixed_quality=bmean[0], cost=mean[1], fixed_cost=bmean[1],
                    samples=mean[2], ci_low=low, ci_high=high))
        print(generator, 'control complete', flush=True)
    write_csv(output/'comparisons.csv', rows)
    metadata = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    budgets=[float(v) if np.isfinite(v) else 'infinity' for v in budgets],
                    study='..', stopping='first cumulative character count at least the fixed budget, or cap',
                    selection='training mean profit only; reward never used for stopping')
    (output/'METHOD.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
