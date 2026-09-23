"""Paired cost-only budget and target-quality controls for order refinement."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment
from nonparametric_controls import collect_budget
from nonparametric_study import PRICES, write_csv
from order_statistic_refinement import TARGETS, target_mix, mix_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    args = parser.parse_args()
    output = args.study/'cost_control'
    if output.exists():
        parser.error('Output exists; refusing overwrite')
    output.mkdir()
    meta = json.loads((args.study/'METHOD.json').read_text())
    budgets = np.r_[np.geomspace(256,4_000_000,64),np.inf]
    profits, targets = [], []
    for generator, data in meta['datasets'].items():
        pools = load_alignment(Path(data['path']),'mistral_rm_reward','text_chars')
        replay = collect_budget(pools,data['actual_cap'],meta['permutations'],meta['replay_seed'],budgets)
        a = np.stack([replay[:,0,:,0],replay[:,0,:,1]/PRICES[0],replay[:,0,:,2]],axis=2)
        with np.load(args.study/f'{generator}_fixed.npz') as cache:
            f = cache['fixed']
        np.savez_compressed(output/f'{generator}.npz',adaptive=a)
        for seed in meta['split_seeds']:
            order = np.random.default_rng(seed).permutation(len(pools))
            train,test = order[:len(pools)//2],order[len(pools)//2:]
            amean,fmean,ftest = a[train].mean(axis=0),f[train].mean(axis=0),f[test].mean(axis=0)
            for price in PRICES:
                ai = np.argmax(amean[:,0]-price*amean[:,1])
                fi = np.argmax(fmean[:,0]-price*fmean[:,1])
                pa = a[test,ai,0]-price*a[test,ai,1]
                pf = f[test,fi,0]-price*f[test,fi,1]
                profits.append(dict(generator=generator,split_seed=seed,price=price,
                    selected_budget=budgets[ai],fixed_n=int(fi+1),profit=pa.mean(),fixed_profit=pf.mean(),
                    relative_percent=100*(pa.mean()-pf.mean())/pf.mean()))
            for target in TARGETS:
                mixture = target_mix(amean[:,0],amean[:,1],target)
                v = mix_values(a[test],mixture)
                matched = target_mix(ftest[:,0],ftest[:,1],v[:,0].mean(),exact=True)
                cost = mix_values(f[test],matched)[:,1].mean() if matched else None
                targets.append(dict(generator=generator,split_seed=seed,target=float(target),
                    attained_quality=v[:,0].mean(),mean_length=v[:,1].mean(),
                    matched_feasible=matched is not None,
                    matched_saving_percent=100*(1-v[:,1].mean()/cost) if cost else None,
                    train_feasible=mixture[3]))
        print(generator,'paired budget control finished',flush=True)
    write_csv(output/'profit.csv',profits)
    write_csv(output/'target_quality.csv',targets)
    (output/'METHOD.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        budgets=[float(b) if np.isfinite(b) else 'infinity' for b in budgets],
        selection='training-only fixed cumulative length budget; target mixture also training-only',
        streams='identical to parent study',cost_unit='characters'),indent=2)+'\n')


if __name__ == '__main__':
    main()
