"""Online averaging of the n-times-improvement statistic, without training.

Development uses existing Alpaca caches. Any additional dataset confirmation
must use a frozen method list; no per-generator/price selection is permitted.
This is a empirical modification of DMRL-inspired stopping, not a new theorem.
"""
import argparse
from bisect import insort
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment, q99, sigmoid
from nonparametric_study import GENERATORS, PRICES, write_csv
import training_free_spacing as comparison


CONFIGS = [dict(base=base, smoothing=smooth, minimum=minimum)
           for base in ('odds', 'spacing')
           for smooth in ('current', 'mean', 'recent_half')
           for minimum in (4, 6, 8)]
NAMES = [f"{c['base']}_{c['smoothing']}_n{c['minimum']}" for c in CONFIGS]
PRIMARY = 'odds_mean_n4'


def smooth_numerator(values, mode):
    """Average scale estimates, not one-step gains with different denominators."""
    n = np.arange(1, len(values)+1)
    if mode == 'current':
        return values.copy()
    sums = np.cumsum(np.where(n >= 4, values, 0.))
    if mode == 'mean':
        return sums/np.maximum(n-3, 1)
    if mode == 'recent_half':
        # Use prefixes max(4, floor(n/2)+1), ..., n.
        before = np.maximum(3, n//2)
        return (sums-sums[np.minimum(before-1, len(values)-1)])/np.maximum(n-before, 1)
    raise ValueError(mode)


def statistics(rewards):
    """Both statistics depend only on the observed prefix."""
    cap = len(rewards)
    odds = np.zeros(cap); spacing = np.zeros(cap)
    ordered = []
    for j, reward in enumerate(rewards):
        insort(ordered, float(reward))
        n = j+1
        if n < 4:
            continue
        k = n//2
        z = np.exp(np.clip(np.array(ordered[-k-1:])-ordered[-1], -745, 0))
        xi = z[0]; m = float(np.mean(z[1:]-xi))
        benchmark = xi+m*(1+np.log((k/n)/.01))
        odds[j] = n/(n+1)*benchmark*m/((1+benchmark)*(1+benchmark+m))
        k = min(n-1, 32, max(2, int(np.ceil(n/4))))
        ref = ordered[min(int(.99*n), n-1)]
        utilities = sigmoid(np.array(ordered[-k-1:])-ref)
        spacing[j] = np.mean(utilities[1:])-utilities[0]
    return dict(odds=odds, spacing=spacing)


def collect(pools, cap, permutations, seed, configs):
    result = np.zeros((len(pools), len(configs), len(comparison.RATES), 3))
    fixed = np.zeros((len(pools), cap, 3))
    rng = np.random.default_rng(seed)
    for i, pool in enumerate(pools):
        for rep in range(permutations):
            order = rng.permutation(len(pool.rewards))[:cap]
            r, lengths = pool.rewards[order], pool.lengths[order]
            costs = np.cumsum(lengths)
            quality = sigmoid(np.maximum.accumulate(r)-q99(pool.rewards))
            fixed[i] += np.stack([quality, costs, np.arange(1,cap+1)], axis=1)
            stats = statistics(r)
            smoothed = {(base, mode): smooth_numerator(stats[base], mode)
                        for base, mode in {(c['base'],c['smoothing']) for c in configs}}
            for ci, c in enumerate(configs):
                stops = comparison.stopping_indices(smoothed[c['base'],c['smoothing']],
                    costs, comparison.RATES, c['minimum'], 'sequential')
                result[i,ci] += np.stack([quality[stops], costs[stops], stops+1], axis=1)
        if (i+1)%25==0:
            print('  replayed',i+1,'prompts',flush=True)
    return result/permutations, fixed/permutations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('dataset/alpaca'))
    parser.add_argument('--generators', nargs='+', default=GENERATORS)
    parser.add_argument('--methods', nargs='+', choices=NAMES, default=NAMES)
    parser.add_argument('--cap', type=int, default=960)
    parser.add_argument('--permutations', type=int, default=8)
    parser.add_argument('--replay-seed', type=int, default=20260923)
    parser.add_argument('--split-seeds', nargs='+', type=int, default=[71,72,73,74,75])
    parser.add_argument('--stage', choices=['development','confirmation'],default='development')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite results')
    args.output.mkdir(parents=True)
    configs = [CONFIGS[NAMES.index(name)] for name in args.methods]
    meta = dict(stage=args.stage, configs=configs, names=args.methods, primary=PRIMARY,
        data_dir=str(args.data_dir), training_free=True,
        no_parametric_fit=True, selection='No runtime fitting across prompts; global development selection must be disclosed',
        rates=comparison.RATES.tolist(), prices=PRICES.tolist(),
        split_seeds=args.split_seeds, replay_seed=args.replay_seed,
        permutations=args.permutations, cap=args.cap, cost_unit='characters',
        matched_quality='Attained quality; post-hoc fixed-N mixture',
        frontier='Post-hoc mixtures of both classes; not training-free target control',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), datasets={})
    (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
    comparison.NAMES = args.methods
    all_p, all_f = [], []
    for gen in args.generators:
        print(gen,flush=True)
        path = args.data_dir/f'{gen}_output.merged_rm.jsonl.gz'
        pools = load_alignment(path,'mistral_rm_reward','text_chars')
        cap = min(args.cap,min(len(p.rewards) for p in pools))
        adaptive, fixed = collect(pools,cap,args.permutations,args.replay_seed,configs)
        np.savez_compressed(args.output/f'{gen}.npz',adaptive=adaptive,fixed=fixed,ids=[p.id for p in pools])
        p,f = comparison.summarize(adaptive,fixed,args.split_seeds,gen)
        all_p.extend(p);all_f.extend(f)
        write_csv(args.output/'profit.csv',all_p)
        write_csv(args.output/'frontier_diagnostic.csv',all_f)
        meta['datasets'][gen] = dict(path=str(path),actual_cap=cap,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
        for name in args.methods:
            rs=[r for r in p if r['method']==name]
            print(name,'profit',round(np.mean([r['relative_percent'] for r in rs]),3),
                  'saving',round(np.mean([r['matched_saving_percent'] for r in rs
                    if r['matched_saving_percent'] is not None]),3),flush=True)


if __name__=='__main__':main()
