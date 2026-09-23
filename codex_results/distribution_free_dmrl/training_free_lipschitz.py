"""No training or benchmark estimate: raw-reward DMRL and BT Lipschitz bound.

Population identity: if X=(R-xi | R>xi) is DMRL with mean m, then
E[(R-M)+] <= P(R>xi)*m*exp(-(M-xi)/m). Since sigmoid is 1/4-Lipschitz,
one quarter of this bounds expected utility improvement for ANY fixed BT
reference. Empirical tail moments are NOT confidence bounds. DMRL here is
on raw reward residuals, not on the paper's observed utility variable.
"""
import argparse
from bisect import insort
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment, q99, sigmoid
from nonparametric_study import GENERATORS, PRICES, write_csv
import training_free_spacing as spacing


CONFIGS = [dict(width=w, exclude_record=e)
           for w in ('half', 'quarter32', 'sqrt32') for e in (False, True)]
NAMES = [f"{c['width']}_{'exclude' if c['exclude_record'] else 'include'}" for c in CONFIGS]
PRIMARY = 'sqrt32_include'


def gains(rewards):
    cap = len(rewards)
    out = np.full((len(CONFIGS), cap), np.inf)
    ordered = []
    for j, reward in enumerate(rewards):
        insort(ordered, float(reward))
        n = j+1
        if n < 4:
            continue
        for ci, config in enumerate(CONFIGS):
            w = config['width']
            k = (n//2 if w=='half' else min(32, max(2,
                 int(np.ceil(np.sqrt(n) if w=='sqrt32' else n/4)))))
            k = min(k, n-1-int(config['exclude_record']))
            upper = np.array(ordered[-k-1-int(config['exclude_record']):])
            xi, incumbent = upper[0], ordered[-1]
            residuals = upper[1:-1] if config['exclude_record'] else upper[1:]
            m = float(np.mean(residuals-xi))
            p = (k+int(config['exclude_record']))/n
            out[ci,j] = .25*p*m*np.exp(-(incumbent-xi)/m) if m>0 else 0.
    return out


def collect(pools, cap, permutations, seed):
    adaptive = np.zeros((len(pools), len(CONFIGS), len(spacing.RATES), 3))
    fixed = np.zeros((len(pools), cap, 3))
    rng = np.random.default_rng(seed)
    n = np.arange(1,cap+1)
    for i,pool in enumerate(pools):
        for rep in range(permutations):
            order = rng.permutation(len(pool.rewards))[:cap]
            rewards, lengths = pool.rewards[order], pool.lengths[order]
            cumulative = np.cumsum(lengths)
            quality = sigmoid(np.maximum.accumulate(rewards)-q99(pool.rewards))
            fixed[i] += np.stack([quality,cumulative,n],axis=1)
            numerator = gains(rewards)*n
            stops = spacing.stopping_indices(numerator,cumulative,spacing.RATES,4,'sequential')
            adaptive[i] += np.stack([quality[stops],cumulative[stops],stops+1],axis=-1)
        if (i+1)%25==0:
            print('  replayed',i+1,'prompts',flush=True)
    return adaptive/permutations, fixed/permutations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--data-dir', type=Path, default=Path('dataset/alpaca'))
    parser.add_argument('--generators', nargs='+', default=GENERATORS)
    parser.add_argument('--cap', type=int, default=960)
    parser.add_argument('--permutations', type=int, default=8)
    parser.add_argument('--replay-seed', type=int, default=20260923)
    parser.add_argument('--split-seeds', nargs='+', type=int, default=[71,72,73,74,75])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; refusing overwrite')
    args.output.mkdir(parents=True)
    meta = dict(primary=PRIMARY, configs=CONFIGS, rates=spacing.RATES.tolist(),
        training_free=True, prices=PRICES.tolist(), targets=spacing.TARGETS.tolist(),
        cap=args.cap, permutations=args.permutations, replay_seed=args.replay_seed,
        split_seeds=args.split_seeds, cost_unit='characters',
        adaptive_selection='None; actual price, no training and no benchmark estimate',
        assumptions='Population DMRL bound for raw-reward residuals; empirical plug-ins uncertified',
        target_frontier='Post-hoc mixtures of both policy classes; not a target controller',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), datasets={})
    (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
    all_p,all_f = [],[]
    # Reuse identical comparison accounting with these six method names.
    spacing.NAMES = NAMES
    for gen in args.generators:
        print(gen,flush=True)
        path = args.data_dir/f'{gen}_output.merged_rm.jsonl.gz'
        pools = load_alignment(path,'mistral_rm_reward','text_chars')
        cap = min(args.cap,min(len(p.rewards) for p in pools))
        adaptive,fixed = collect(pools,cap,args.permutations,args.replay_seed)
        np.savez_compressed(args.output/f'{gen}.npz',adaptive=adaptive,fixed=fixed,ids=[p.id for p in pools])
        p,f = spacing.summarize(adaptive,fixed,args.split_seeds,gen)
        all_p.extend(p);all_f.extend(f)
        write_csv(args.output/'profit.csv',all_p)
        write_csv(args.output/'frontier_diagnostic.csv',all_f)
        meta['datasets'][gen] = dict(path=str(path),actual_cap=cap,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
        for name in NAMES:
            rows=[r for r in p if r['method']==name]
            print(name,'profit',round(np.mean([r['relative_percent'] for r in rows]),3),
                  'matched saving',round(np.mean([r['matched_saving_percent'] for r in rows
                    if r['matched_saving_percent'] is not None]),3),flush=True)


if __name__=='__main__':
    main()
