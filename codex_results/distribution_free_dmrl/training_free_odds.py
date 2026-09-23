"""Training-free empirical residual-life stopping in reward-odds coordinates.

Primary rule: rank_jensen_bound. No calibration table, fitted predictor, fitted
response family, training-selected multiplier, or learned stopping price.
Empirical moments and order statistics use only the current response prefix.
Shape inequalities motivate estimates but do not certify the plug-in policy.
"""
import argparse
from bisect import insort
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment,q99,sigmoid
from nonparametric_study import GENERATORS,PRICES,write_csv
from order_statistic_refinement import TARGETS,target_mix,mix_values


VARIANTS=tuple(f'{mass}_{gain}_{reference}' for reference in ('bound','moment')
               for mass in ('rank','shape') for gain in ('jensen','empirical'))
PRIMARY='rank_jensen_bound'


def prefix_statistics(rewards):
    cap=len(rewards)
    gains={v:np.zeros(cap) for v in VARIANTS}
    utility={v:np.zeros(cap) for v in VARIANTS}
    ordered=[]
    for j,r in enumerate(rewards):
        n=j+1;insort(ordered,float(r))
        if n<4:continue
        k=n//2
        # Odds are rescaled by the current maximum; no future scale is used.
        odds=np.exp(np.clip(np.array(ordered[-k-1:])-ordered[-1],-745,0))
        xi=odds[0];excess=odds[1:]-xi;m=float(excess.mean());tail_mass=k/n
        rank_prob=1/(n+1)
        shape_prob=tail_mass*min(1.,np.exp(min(0.,1-(1-xi)/m))) if m>0 else 0.
        for ref in ('bound','moment'):
            # With known DMRL residual mean, +1 yields a quantile upper bound;
            # omitting it is an exponential-tail moment approximation, not a fit.
            benchmark=xi+m*(np.log(tail_mass/.01)+(1 if ref=='bound' else 0))
            incumbent=1/(1+benchmark)
            jensen=benchmark*m/((1+benchmark)*(1+benchmark+m))
            empirical=float(np.mean((1+excess)/(1+excess+benchmark)-incumbent))
            for mass,p in [('rank',rank_prob),('shape',shape_prob)]:
                for estimate,g in [('jensen',jensen),('empirical',empirical)]:
                    name=f'{mass}_{estimate}_{ref}'
                    gains[name][j]=p*max(0.,g)
                    utility[name][j]=incumbent
    return gains,utility


def first_stop(gain,utility,lengths,price=None,target=None):
    if (price is None)==(target is None):raise ValueError('Specify exactly one objective')
    cap=len(lengths);n=np.arange(1,cap+1)
    hit=(gain<=price*np.cumsum(lengths)/n) if price is not None else (utility>=target)
    hit[:min(3,cap)]=False;hit[-1]=True
    return int(np.argmax(hit)+1)


def collect(pools,cap,permutations,seed):
    p=np.zeros((len(pools),len(VARIANTS),len(PRICES),3))
    t=np.zeros((len(pools),len(VARIANTS),len(TARGETS),3))
    fixed=np.zeros((len(pools),cap,3));rng=np.random.default_rng(seed)
    for i,pool in enumerate(pools):
        for rep in range(permutations):
            order=rng.permutation(len(pool.rewards))[:cap]
            r,l=pool.rewards[order],pool.lengths[order]
            gains,utility=prefix_statistics(r)
            costs=np.cumsum(l);quality=sigmoid(np.maximum.accumulate(r)-q99(pool.rewards))
            fixed[i]+=np.stack([quality,costs,np.arange(1,cap+1)],axis=1)
            for vi,name in enumerate(VARIANTS):
                for pi,price in enumerate(PRICES):
                    stop=first_stop(gains[name],utility[name],l,price=price)
                    p[i,vi,pi]+=[quality[stop-1],costs[stop-1],stop]
                for ti,target in enumerate(TARGETS):
                    stop=first_stop(gains[name],utility[name],l,target=target)
                    t[i,vi,ti]+=[quality[stop-1],costs[stop-1],stop]
        if (i+1)%25==0:print('  replayed',i+1,'prompts',flush=True)
    return p/permutations,t/permutations,fixed/permutations


def summaries(p,t,fixed,seeds,generator):
    profits,targets=[],[]
    for seed in seeds:
        ids=np.random.default_rng(seed).permutation(len(fixed));train,test=np.array_split(ids,2)
        fm,ft=fixed[train].mean(axis=0),fixed[test].mean(axis=0)
        for vi,name in enumerate(VARIANTS):
            for pi,price in enumerate(PRICES):
                fi=np.argmax(fm[:,0]-price*fm[:,1]);a=p[test,vi,pi]
                value=(a[:,0]-price*a[:,1]).mean()
                baseline=(fixed[test,fi,0]-price*fixed[test,fi,1]).mean()
                oracle=np.max(ft[:,0]-price*ft[:,1])
                profits.append(dict(generator=generator,split_seed=seed,method=name,price=float(price),
                    profit=value,fixed_profit=baseline,fixed_n=int(fi+1),relative_percent=100*(value-baseline)/baseline,
                    relative_vs_test_oracle=100*(value-oracle)/oracle,quality=a[:,0].mean(),
                    cost=price*a[:,1].mean(),samples=a[:,2].mean()))
            for ti,target in enumerate(TARGETS):
                a=t[test,vi,ti];mean=a.mean(axis=0)
                matched=target_mix(ft[:,0],ft[:,1],mean[0],exact=True)
                cost=mix_values(fixed[test],matched)[:,1].mean() if matched else None
                targets.append(dict(generator=generator,split_seed=seed,method=name,target=float(target),
                    attained_quality=mean[0],mean_length=mean[1],samples=mean[2],
                    matched_feasible=matched is not None,matched_length=cost,
                    matched_saving_percent=100*(1-mean[1]/cost) if cost else None))
    return profits,targets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=Path('dataset/alpaca'))
    parser.add_argument('--generators',nargs='+',default=GENERATORS)
    parser.add_argument('--cap',type=int,default=960)
    parser.add_argument('--permutations',type=int,default=8)
    parser.add_argument('--replay-seed',type=int,default=20260923)
    parser.add_argument('--split-seeds',nargs='+',type=int,default=[71,72,73,74,75])
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; refusing overwrite')
    args.output.mkdir(parents=True)
    meta=dict(training_free=True,primary=PRIMARY,variants=list(VARIANTS),actual_prices=PRICES.tolist(),
        targets=TARGETS.tolist(),cap=args.cap,permutations=args.permutations,replay_seed=args.replay_seed,
        split_seeds=args.split_seeds,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        minimum_samples=4,threshold='lower empirical median in exponentiated reward',cost_unit='characters',
        selection='No selection or fitting for adaptive rules. Training prompts select only fixed-N comparators.',
        scope='Exploratory fixed-parameter variants; primary declared before execution; no certified theorem transfer',datasets={})
    (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
    all_p,all_t=[],[]
    for generator in args.generators:
        print(generator,flush=True)
        path=args.data_dir/f'{generator}_output.merged_rm.jsonl.gz'
        pools=load_alignment(path,'mistral_rm_reward','text_chars')
        cap=min(args.cap,min(len(p.rewards) for p in pools))
        p,t,fixed=collect(pools,cap,args.permutations,args.replay_seed)
        np.savez_compressed(args.output/f'{generator}.npz',profit=p,target=t,fixed=fixed,ids=[p.id for p in pools])
        pr,tr=summaries(p,t,fixed,args.split_seeds,generator);all_p.extend(pr);all_t.extend(tr)
        write_csv(args.output/'profit.csv',all_p);write_csv(args.output/'target_quality.csv',all_t)
        meta['datasets'][generator]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),actual_cap=cap)
        (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
        for name in VARIANTS:
            print(name,'profit',round(np.mean([r['relative_percent'] for r in pr if r['method']==name]),3),
                  'saving',round(np.mean([r['matched_saving_percent'] for r in tr if r['method']==name and r['matched_feasible']]),3),flush=True)


if __name__=='__main__':main()
