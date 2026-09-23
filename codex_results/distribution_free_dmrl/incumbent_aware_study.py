"""Incumbent-aware tail-moment stopping, without a fitted response law.

The DMRL envelope uses L_X(d)<=m exp(-d/m), a shape-restriction bound, but
plug-in estimation makes the policy heuristic. It is algebraically similar to
an exponential-tail plug-in and is NOT advertised as a new confidence bound.
The second-moment envelope uses (X-d)+ <= X^2/(4d), with empirical moments.
Exclude the current best from tail-scale estimation to avoid inflating the
continuation estimate precisely when an unusually good answer arrives.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment, sigmoid, validate
from nonparametric_study import GENERATORS, PRICES, write_csv
from order_statistic_refinement import (RATES, TARGETS, WIDTHS, prepare,
    reference_bias, width_sequence, summarize)


FAMILIES = ('spacing', 'dmrl_envelope', 'moment_envelope')


def configs():
    return [dict(family=family, reference=reference, width=width, stopping_price=float(rate))
            for family in FAMILIES for reference in ('empirical','corrected')
            for width in WIDTHS for rate in RATES]


def tail_numerators(top, reference, width):
    """n times gain estimates, all based solely on the observed prefix."""
    cap = reference.shape[-1]
    utility = sigmoid(top.astype(float)-reference[...,None])
    sums = np.cumsum(utility,axis=-1)
    squares = np.cumsum(utility**2,axis=-1)
    k,minimum = width_sequence(width,cap)
    def get(values,index):
        ix = np.broadcast_to(index,reference.shape)[...,None]
        return np.take_along_axis(values,ix,axis=-1)[...,0]
    spacing = np.maximum(get(sums,np.maximum(k-1,0))/np.maximum(k,1)-get(utility,k),0)
    n = np.arange(1,cap+1)
    k = np.minimum(k,np.minimum(31,np.maximum(n-2,0)))
    floor = get(utility,k+1)
    mean = (get(sums,k)-utility[...,0])/np.maximum(k,1)
    residual_mean = np.maximum(mean-floor,0)
    second = np.maximum((get(squares,k)-utility[...,0]**2)/np.maximum(k,1)
                        -2*floor*mean+floor**2,0)
    d = np.maximum(utility[...,0]-floor,0)
    ratio = np.divide(d,residual_mean,out=np.full_like(d,np.inf),where=residual_mean>0)
    dmrl = (k+1)*residual_mean*np.exp(-ratio)
    bound = np.divide(second,4*d,out=np.zeros_like(d),where=d>0)
    moment = (k+1)*np.minimum(residual_mean,bound)
    env_min = max(4,int(width[5:])+2) if width.startswith('fixed') else minimum
    return {'spacing':(spacing,minimum),'dmrl_envelope':(dmrl,env_min),
            'moment_envelope':(moment,env_min)}


def evaluate(top, reference, lengths, quality, bias=None):
    count,reps,cap = reference.shape
    ref = reference if bias is None else reference+bias[None,None,:]
    outputs = {f:np.zeros((count,len(WIDTHS)*len(RATES),3)) for f in FAMILIES}
    for wi,width in enumerate(WIDTHS):
        estimates = tail_numerators(top,ref,width)
        for family,(numerator,minimum) in estimates.items():
            threshold = np.divide(numerator,lengths,out=np.full_like(reference,np.inf),where=lengths>0)
            threshold[...,:minimum-1] = np.inf
            crossing = np.minimum.accumulate(threshold,axis=-1).reshape(-1,cap)
            stops = np.array([np.minimum(np.searchsorted(-row,-RATES),cap-1)
                              for row in crossing]).reshape(count,reps,len(RATES))
            v = np.stack([np.take_along_axis(quality,stops,axis=2).mean(axis=1),
                          np.take_along_axis(lengths,stops,axis=2).mean(axis=1),
                          (stops+1).mean(axis=1)],axis=2)
            outputs[family][:,wi*len(RATES):(wi+1)*len(RATES)] = v
    return outputs


def summarize_groups(adaptive,fixed,train,test,generator,seed,configuration):
    ps,ts,prompts = [],[],[]
    groups = {f:np.array([i for i,c in enumerate(configuration) if c['family']==f]) for f in FAMILIES}
    groups['all_train_selected'] = np.arange(len(configuration))
    for group,indices in groups.items():
        local = [configuration[i] for i in indices]
        p,t,pr = summarize(adaptive[:,indices],fixed,train,test,generator,seed,local)
        for row in p:
            if row['method']!='train_selected_spacing':continue
            row['method']=group
            row['selected']=int(indices[row['selected']])
            ps.append(row)
        for row in t:
            if row['method']!='train_selected_spacing':continue
            row['method']=group
            row['adaptive_low']=int(indices[row['adaptive_low']])
            row['adaptive_high']=int(indices[row['adaptive_high']])
            ts.append(row)
        for row in pr:
            if row['method']!='train_selected_spacing':continue
            row['method']=group
            prompts.append(row)
    return ps,ts,prompts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,default=Path('codex_results/distribution_free_dmrl/order_refinement_results'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a fresh directory')
    prior=json.loads((args.previous/'METHOD.json').read_text())
    args.output.mkdir(parents=True)
    configuration=configs()
    meta=dict(prior,configurations=configuration,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              previous=str(args.previous),hypothesis='A large incumbent should reduce continuation estimates; scale excludes incumbent',
              caveat='Plug-in shape/moment envelopes, not guaranteed confidence bounds; reused prompts and frozen pre-evaluation grid')
    (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
    all_p,all_t,all_pr=[],[],[]
    for generator,d in prior['datasets'].items():
        print(generator,flush=True)
        pools=load_alignment(Path(d['path']),'mistral_rm_reward','text_chars');validate(pools)
        top,ref,lengths,quality,truth=prepare(pools,d['actual_cap'],prior['permutations'],prior['replay_seed'])
        with np.load(args.previous/f'{generator}_fixed.npz') as z:fixed=z['fixed']
        raw=evaluate(top,ref,lengths,quality)
        for seed in prior['split_seeds']:
            with np.load(args.previous/f'{generator}_seed{seed}.npz') as z:
                train,test=z['train'],z['test']; previous_a=z['adaptive']
            halves=np.array_split(train,2)
            corrected={f:np.empty_like(raw[f]) for f in FAMILIES}
            for evaluate_ids,calibrate_ids in [(halves[0],halves[1]),(halves[1],halves[0]),(test,train)]:
                bias=reference_bias(ref,truth,calibrate_ids)
                output=evaluate(top[evaluate_ids],ref[evaluate_ids],lengths[evaluate_ids],quality[evaluate_ids],bias)
                for family in FAMILIES:corrected[family][evaluate_ids]=output[family]
            a=np.concatenate([part for family in FAMILIES for part in (raw[family],corrected[family])],axis=1)
            np.testing.assert_allclose(a[:,:previous_a.shape[1]],previous_a,rtol=1e-10,atol=1e-10)
            np.savez_compressed(args.output/f'{generator}_seed{seed}.npz',adaptive=a,train=train,test=test)
            p,t,pr=summarize_groups(a,fixed,train,test,generator,seed,configuration)
            all_p.extend(p);all_t.extend(t);all_pr.extend(pr)
            for method in FAMILIES+('all_train_selected',):
                profit=np.mean([r['relative_percent'] for r in p if r['method']==method])
                saving=np.mean([r['matched_saving_percent'] for r in t if r['method']==method and r['matched_feasible']])
                print(' ',seed,method,'profit',round(profit,3),'saving',round(saving,3),flush=True)
        write_csv(args.output/'profit.csv',all_p)
        write_csv(args.output/'target_quality.csv',all_t)
        write_csv(args.output/'target_prompt_metrics.csv',all_pr)


if __name__=='__main__':main()
