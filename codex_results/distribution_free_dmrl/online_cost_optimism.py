"""Nonparametric online reward estimates with variance-aware cost optimism.

For sample mean ell_bar and standard error se, use the positive estimate
ell_bar/(1+b*se/ell_bar). b=0 recovers the empirical mean. This is a
sensitivity study, NOT a distribution-free or sequential confidence bound.
All stopping uses only prefix information and the actual price.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment,q99,sigmoid
from nonparametric_study import GENERATORS,PRICES,write_csv
from online_smoothed_spacing import statistics,smooth_numerator
import training_free_spacing as comparison


CONFIGS=[dict(smoothing=s,cost_optimism=b) for s in ('current','mean') for b in (0.,1.,2.)]
NAMES=[f"odds_{c['smoothing']}_costse{c['cost_optimism']:g}" for c in CONFIGS]
PRIMARY='odds_mean_costse1'


def effective_cost(lengths, optimism):
    """n times a positive optimistic next-cost estimate, in length units."""
    n=np.arange(1,len(lengths)+1);s=np.cumsum(lengths);mean=s/n
    variance=np.maximum((np.cumsum(lengths**2)-s*s/n)/np.maximum(n-1,1),0.)
    se=np.sqrt(variance/n)
    relative=np.divide(se,mean,out=np.zeros_like(se),where=mean>0)
    return s/(1+optimism*relative)


def collect(pools,cap,permutations,seed,configs):
    a=np.zeros((len(pools),len(configs),len(comparison.RATES),3))
    f=np.zeros((len(pools),cap,3));rng=np.random.default_rng(seed)
    for i,pool in enumerate(pools):
        for rep in range(permutations):
            order=rng.permutation(len(pool.rewards))[:cap]
            r,l=pool.rewards[order],pool.lengths[order]
            cost=np.cumsum(l);quality=sigmoid(np.maximum.accumulate(r)-q99(pool.rewards))
            f[i]+=np.stack([quality,cost,np.arange(1,cap+1)],axis=1)
            stat=statistics(r)['odds']
            smoothed={s:smooth_numerator(stat,s) for s in {c['smoothing'] for c in configs}}
            cost_est={b:effective_cost(l,b) for b in {c['cost_optimism'] for c in configs}}
            for ci,c in enumerate(configs):
                stop=comparison.stopping_indices(smoothed[c['smoothing']],cost_est[c['cost_optimism']],
                    comparison.RATES,4,'sequential')
                a[i,ci]+=np.stack([quality[stop],cost[stop],stop+1],axis=1)
        if (i+1)%25==0:print('  replayed',i+1,'prompts',flush=True)
    return a/permutations,f/permutations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--data-dir',type=Path,default=Path('dataset/alpaca'))
    p.add_argument('--methods',nargs='+',choices=NAMES,default=NAMES)
    p.add_argument('--stage',choices=['development','confirmation'],default='development')
    p.add_argument('--replay-seed',type=int,default=20260923)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing overwrite')
    args.output.mkdir(parents=True)
    configs=[CONFIGS[NAMES.index(name)] for name in args.methods]
    meta=dict(stage=args.stage,configs=configs,names=args.methods,primary=PRIMARY,
        training_free=True,no_parametric_fit=True,prices=PRICES.tolist(),rates=comparison.RATES.tolist(),
        split_seeds=[71,72,73,74,75],replay_seed=args.replay_seed,permutations=8,cap=960,
        cost_unit='characters',cost_bound='Empirical optimistic estimate, not a confidence bound',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['online_smoothed_spacing.py','training_free_spacing.py']},datasets={})
    (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
    comparison.NAMES=args.methods;profits=[];frontiers=[]
    for gen in GENERATORS:
        print(gen,flush=True)
        path=args.data_dir/f'{gen}_output.merged_rm.jsonl.gz'
        pools=load_alignment(path,'mistral_rm_reward','text_chars')
        cap=min(960,min(len(pool.rewards) for pool in pools))
        a,f=collect(pools,cap,8,args.replay_seed,configs)
        np.savez_compressed(args.output/f'{gen}.npz',adaptive=a,fixed=f,ids=[pool.id for pool in pools])
        pr,fr=comparison.summarize(a,f,meta['split_seeds'],gen)
        profits.extend(pr);frontiers.extend(fr)
        write_csv(args.output/'profit.csv',profits);write_csv(args.output/'frontier_diagnostic.csv',frontiers)
        meta['datasets'][gen]=dict(path=str(path),actual_cap=cap,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
        for name in args.methods:
            rs=[r for r in pr if r['method']==name]
            print(name,'profit',round(np.mean([r['relative_percent'] for r in rs]),3),
                'saving',round(np.mean([r['matched_saving_percent'] for r in rs if r['matched_saving_percent'] is not None]),3),flush=True)


if __name__=='__main__':main()
