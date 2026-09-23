"""Learn expected improvement from prefix order statistics, not a response law.

All supervised labels come from training prompts' response pools. A full-pool
gain policy is reported ONLY as a nonimplementable oracle diagnostic and is
never eligible for the deployable selector. Parameters are prespecified.
"""
from bisect import insort
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from evaluate import load_alignment, q99, sigmoid
from nonparametric_study import PRICES, write_csv
from order_statistic_refinement import RATES, TARGETS, target_mix, mix_values


FEATURES = ['log_n','maximum','median','q25','q75','q90','prefix_mean','prefix_std',
            'best_minus_second','best_minus_fifth','best_minus_seventeenth',
            'mean_excess_top2','mean_excess_top4','mean_excess_top8','mean_excess_top16',
            'standardized_maximum','log_mean_length']
MODEL_PARAMETERS = dict(n_estimators=80,max_leaf_nodes=128,min_samples_leaf=30,
                        max_features=1.,n_jobs=2,random_state=20260925)


def prefix_features(rewards,lengths):
    cap=len(rewards)
    x=np.zeros((cap,len(FEATURES)),dtype=np.float32)
    ordered=[];total=0.;squared=0.;length_sum=0.
    for j,r in enumerate(rewards):
        n=j+1;insort(ordered,float(r));total+=r;squared+=r*r;length_sum+=lengths[j]
        maximum=ordered[-1];median=ordered[n//2]
        q25=ordered[int(.25*n)];q75=ordered[min(int(.75*n),n-1)]
        q90=ordered[min(int(.9*n),n-1)]
        gaps=[maximum-ordered[max(0,n-k)] for k in (2,5,17)]
        excess=[]
        for k in (2,4,8,16):
            k=min(k,n-1)
            excess.append(float(np.mean(ordered[-k:]))-ordered[-k-1] if k else 0.)
        x[j]=[np.log(n),maximum,median,q25,q75,q90,total/n,
              np.sqrt(max(0.,squared/n-(total/n)**2)),*gaps,*excess,
              (maximum-median)/max(q75-q25,.1),np.log(max(length_sum/n,1.))]
    return x


def full_pool_gain(rewards,maximum):
    """Empirical iid one-response improvement: training label/oracle ONLY."""
    ref=q99(rewards)
    utility=np.sort(sigmoid(rewards-ref))
    incumbent=sigmoid(maximum-ref)
    indices=np.searchsorted(utility,incumbent,side='right')
    tail=np.r_[np.cumsum(utility[::-1])[::-1],0.]
    return np.maximum(tail[indices]-(len(utility)-indices)*incumbent,0.)/len(utility)


def prepare(pools,cap,permutations,seed):
    shape=(len(pools),permutations,cap)
    x=np.zeros(shape+(len(FEATURES),),dtype=np.float32)
    lengths=np.zeros(shape);quality=np.zeros(shape);gain=np.zeros(shape)
    rng=np.random.default_rng(seed)
    for i,pool in enumerate(pools):
        for rep in range(permutations):
            order=rng.permutation(len(pool.rewards))[:cap]
            r,l=pool.rewards[order],pool.lengths[order]
            x[i,rep]=prefix_features(r,l)
            best=np.maximum.accumulate(r)
            lengths[i,rep]=np.cumsum(l)
            quality[i,rep]=sigmoid(best-q99(pool.rewards))
            gain[i,rep]=full_pool_gain(pool.rewards,best)
        if (i+1)%25==0:print('  prepared',i+1,'prompts',flush=True)
    return x,lengths,quality,gain


def replay(gain,lengths,quality):
    count,reps,cap=gain.shape
    threshold=gain*np.arange(1,cap+1)/np.maximum(lengths,1e-12)
    threshold[...,:3]=np.inf
    crossing=np.minimum.accumulate(threshold,axis=-1).reshape(-1,cap)
    stops=np.array([np.minimum(np.searchsorted(-row,-RATES),cap-1) for row in crossing]).reshape(count,reps,len(RATES))
    return np.stack([np.take_along_axis(quality,stops,axis=2).mean(axis=1),
                     np.take_along_axis(lengths,stops,axis=2).mean(axis=1),
                     (stops+1).mean(axis=1)],axis=2)


def fit_predict(x,gain,train_ids,evaluate_ids,kind):
    cap=x.shape[2]
    checkpoints=np.unique(np.rint(np.geomspace(4,cap,64)).astype(int))-1
    train_x=x[train_ids][:,:,checkpoints].reshape(-1,len(FEATURES))
    scaled=gain[train_ids][:,:,checkpoints]*(checkpoints+1)
    target=np.log(np.maximum(scaled,1e-8)) if kind=='log_gain' else scaled
    model=ExtraTreesRegressor(**MODEL_PARAMETERS).fit(train_x,target.reshape(-1))
    flat=x[evaluate_ids].reshape(-1,len(FEATURES))
    prediction=model.predict(flat).reshape(x[evaluate_ids].shape[:3])
    scaled_prediction=np.exp(prediction) if kind=='log_gain' else np.maximum(prediction,0.)
    return scaled_prediction/np.arange(1,cap+1)


def summaries(a,fixed,train,test,groups,model,seed):
    p,t=[],[]
    am,fm,ft=a[train].mean(axis=0),fixed[train].mean(axis=0),fixed[test].mean(axis=0)
    for method,eligible in groups.items():
        for price in PRICES:
            ai=eligible[np.argmax(am[eligible,0]-price*am[eligible,1])]
            fi=int(np.argmax(fm[:,0]-price*fm[:,1]))
            v=a[test,ai];f=fixed[test,fi]
            profit=(v[:,0]-price*v[:,1]).mean();baseline=(f[:,0]-price*f[:,1]).mean()
            p.append(dict(generator=model,split_seed=seed,method=method,price=float(price),
                selected=int(ai),fixed_n=fi+1,profit=profit,fixed_profit=baseline,
                relative_percent=100*(profit-baseline)/baseline,quality=v[:,0].mean(),
                cost=price*v[:,1].mean(),samples=v[:,2].mean(),
                delta_test_oracle=profit-np.max(ft[:,0]-price*ft[:,1])))
        for target in TARGETS:
            lm=target_mix(am[eligible,0],am[eligible,1],target)
            lo,hi,w,feasible=lm
            mix=(int(eligible[lo]),int(eligible[hi]),w,feasible)
            v=mix_values(a[test],mix);mean=v.mean(axis=0)
            matched=target_mix(ft[:,0],ft[:,1],mean[0],exact=True)
            cost=mix_values(fixed[test],matched)[:,1].mean() if matched else None
            t.append(dict(generator=model,split_seed=seed,method=method,target=float(target),
                attained_quality=mean[0],mean_length=mean[1],samples=mean[2],
                train_feasible=feasible,matched_feasible=matched is not None,matched_length=cost,
                matched_saving_percent=100*(1-mean[1]/cost) if cost else None,
                adaptive_low=mix[0],adaptive_high=mix[1],adaptive_high_weight=w))
    return p,t


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,default=Path('codex_results/distribution_free_dmrl/order_refinement_results'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--generators',nargs='+')
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; refusing overwrite')
    previous=json.loads((args.previous/'METHOD.json').read_text())
    args.output.mkdir(parents=True)
    configurations=[dict(family=c['reference']+'_spacing',**c) for c in previous['configurations']]
    old_count=len(configurations)
    for kind in ('direct_gain','log_gain','oracle_gain'):
        configurations.extend(dict(family=kind,stopping_price=float(rate)) for rate in RATES)
    groups={'spacing':np.arange(old_count),'direct_gain':np.arange(old_count,old_count+49),
        'log_gain':np.arange(old_count+49,old_count+98),'learned_gain':np.arange(old_count,old_count+98),
        'all_train_selected':np.arange(old_count+98),'oracle_gain':np.arange(old_count+98,old_count+147)}
    meta=dict(previous,configurations=configurations,groups={k:v.tolist() for k,v in groups.items()},
        previous=str(args.previous),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        predictor_parameters=MODEL_PARAMETERS,features=FEATURES,
        labels='n times empirical full-training-pool expected utility improvement; train prompt labels only',
        fitting='two-fold prompt cross-fitting inside training; refit on all training prompts for test',
        target_profit_percent=10,target_cost_saving_percent=35,
        oracle='nonimplementable diagnostic; uses test full pools; never selected by deployable groups')
    (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
    all_p,all_t=[],[]
    for generator,d in previous['datasets'].items():
        if args.generators and generator not in args.generators:continue
        print(generator,flush=True)
        pools=load_alignment(Path(d['path']),'mistral_rm_reward','text_chars')
        x,lengths,quality,gain=prepare(pools,d['actual_cap'],previous['permutations'],previous['replay_seed'])
        oracle=replay(gain,lengths,quality)
        with np.load(args.previous/f'{generator}_fixed.npz') as z:fixed=z['fixed']
        for seed in previous['split_seeds']:
            with np.load(args.previous/f'{generator}_seed{seed}.npz') as z:
                old=z['adaptive'];train=z['train'];test=z['test']
            halves=np.array_split(train,2)
            candidates=[]
            for kind in ('direct_gain','log_gain'):
                outcomes=np.zeros((len(pools),len(RATES),3))
                for evaluate_ids,fit_ids in [(halves[0],halves[1]),(halves[1],halves[0]),(test,train)]:
                    prediction=fit_predict(x,gain,fit_ids,evaluate_ids,kind)
                    outcomes[evaluate_ids]=replay(prediction,lengths[evaluate_ids],quality[evaluate_ids])
                candidates.append(outcomes)
            a=np.concatenate([old,*candidates,oracle],axis=1)
            np.savez_compressed(args.output/f'{generator}_seed{seed}.npz',adaptive=a,train=train,test=test)
            p,t=summaries(a,fixed,train,test,groups,generator,seed)
            all_p.extend(p);all_t.extend(t)
            for method in ('direct_gain','log_gain','all_train_selected','oracle_gain'):
                print(' ',seed,method,'profit',round(np.mean([r['relative_percent'] for r in p if r['method']==method]),3),
                    'saving',round(np.mean([r['matched_saving_percent'] for r in t if r['method']==method and r['matched_feasible']]),3),flush=True)
        write_csv(args.output/'profit.csv',all_p);write_csv(args.output/'target_quality.csv',all_t)


if __name__=='__main__':main()
