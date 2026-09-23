"""Development/confirmation of small-start online residual-life stopping.

Investigates the cost of requiring four samples before the first decision.
Earlier starts are pragmatic changes, NOT covered by the manuscript theorem.
No fitted distribution or offline training enters any stopping decision.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment,q99,sigmoid
from nonparametric_study import GENERATORS,PRICES,write_csv
from online_smoothed_spacing import statistics
from online_cost_optimism import effective_cost
import training_free_spacing as comparison


CONFIGS=[dict(minimum=n,smoothing=s,cost_optimism=b)
         for n in (2,3,4) for s in ('current','mean') for b in (0.,1.)]
NAMES=[f"odds_{c['smoothing']}_n{c['minimum']}_costse{c['cost_optimism']:g}" for c in CONFIGS]
PRIMARY='odds_mean_n3_costse1'


def small_start_statistics(rewards):
    out=statistics(rewards)['odds']
    for n in (2,3):
        if len(rewards)<n:continue
        r=np.sort(rewards[:n]);k=n//2
        z=np.exp(np.clip(r[-k-1:]-r[-1],-745,0));xi=z[0];m=float(np.mean(z[1:]-xi))
        benchmark=xi+m*(1+np.log((k/n)/.01))
        out[n-1]=n/(n+1)*benchmark*m/((1+benchmark)*(1+benchmark+m))
    return out


def average_from(values,minimum):
    n=np.arange(1,len(values)+1)
    return np.cumsum(np.where(n>=minimum,values,0.))/np.maximum(n-minimum+1,1)


def collect(pools,cap,seed,configs):
    a=np.zeros((len(pools),len(configs),len(comparison.RATES),3))
    f=np.zeros((len(pools),cap,3));rng=np.random.default_rng(seed)
    for i,pool in enumerate(pools):
        for rep in range(8):
            order=rng.permutation(len(pool.rewards))[:cap];r,l=pool.rewards[order],pool.lengths[order]
            cost=np.cumsum(l);q=sigmoid(np.maximum.accumulate(r)-q99(pool.rewards))
            f[i]+=np.stack([q,cost,np.arange(1,cap+1)],axis=1)
            stat=small_start_statistics(r)
            means={n:average_from(stat,n) for n in {c['minimum'] for c in configs}}
            costs={b:effective_cost(l,b) for b in {c['cost_optimism'] for c in configs}}
            for ci,c in enumerate(configs):
                g=stat if c['smoothing']=='current' else means[c['minimum']]
                stop=comparison.stopping_indices(g,costs[c['cost_optimism']],comparison.RATES,c['minimum'],'sequential')
                a[i,ci]+=np.stack([q[stop],cost[stop],stop+1],axis=1)
        if (i+1)%25==0:print('  replayed',i+1,'prompts',flush=True)
    return a/8,f/8


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('dataset/alpaca'))
    p.add_argument('--methods',nargs='+',choices=NAMES,default=NAMES)
    p.add_argument('--stage',choices=['development','confirmation'],default='development')
    p.add_argument('--replay-seed',type=int,default=20260923)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing overwrite')
    args.output.mkdir(parents=True)
    configs=[CONFIGS[NAMES.index(name)] for name in args.methods]
    meta=dict(primary=PRIMARY,stage=args.stage,configs=configs,names=args.methods,
        rates=comparison.RATES.tolist(),prices=PRICES.tolist(),split_seeds=[71,72,73,74,75],
        training_free=True,no_parametric_fit=True,permutations=8,replay_seed=args.replay_seed,cap=960,
        cost_unit='characters',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['online_smoothed_spacing.py','online_cost_optimism.py','training_free_spacing.py']},
        caveat='Empirical smoothing, early start, and cost optimism are not covered by the theorem',datasets={})
    (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
    comparison.NAMES=args.methods;profits=[];frontiers=[]
    for gen in GENERATORS:
        print(gen,flush=True);path=args.data_dir/f'{gen}_output.merged_rm.jsonl.gz'
        pools=load_alignment(path,'mistral_rm_reward','text_chars')
        cap=min(960,min(len(pool.rewards) for pool in pools));a,f=collect(pools,cap,args.replay_seed,configs)
        np.savez_compressed(args.output/f'{gen}.npz',adaptive=a,fixed=f,ids=[pool.id for pool in pools])
        pr,fr=comparison.summarize(a,f,meta['split_seeds'],gen);profits.extend(pr);frontiers.extend(fr)
        write_csv(args.output/'profit.csv',profits);write_csv(args.output/'frontier_diagnostic.csv',frontiers)
        meta['datasets'][gen]=dict(path=str(path),actual_cap=cap,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
        for name in args.methods:
            rs=[r for r in pr if r['method']==name]
            print(name,'profit',round(np.mean([r['relative_percent'] for r in rs]),3),
                'saving',round(np.mean([r['matched_saving_percent'] for r in rs if r['matched_saving_percent'] is not None]),3),flush=True)


if __name__=='__main__':main()
