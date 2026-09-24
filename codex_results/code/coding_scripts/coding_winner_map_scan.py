"""Choose a winner-calibrated fixed isotonic map and rerun the coding price scan.

Map mixture selection uses four-fold problem-grouped CV inside calibration.
Frozen-parameter and proxy-retuned arms isolate map changes. A correctness-tuned
arm is diagnostic, with the known limitation that 24 tuning orderings can be
overfit. Final test outcomes never select maps/settings. No manuscript changes.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import coding_winner_calibration as calibration
import coding_top3_price_scan as scan
from algorithm.adaptive_coding import AdaptiveCoding

old, bridge = scan.old, scan.bridge
METHODS = ("selected_map_frozen", "selected_map_proxy_tuned", "selected_map_observed_tuned")


def tune_both(batch, labels, gain, price):
    best={name:(-np.inf,np.inf,None) for name in ("selected_map_proxy_tuned","selected_map_observed_tuned")}
    minimums=np.arange(10,513)
    rows=np.arange(batch.trials)[:,None]
    for alpha in old.MULTIPLIERS:
        raw=scan.first_crossing(gain,batch.mean_tokens,alpha,price)
        index=np.maximum(raw[:,None],minimums)-1
        tokens=batch.cumulative_tokens[rows,index].mean(axis=0)
        predictions=batch.best_probabilities[rows,index].mean(axis=0)
        actual=labels[rows,index].mean(axis=0)
        for name in best:
            scores=(predictions if 'proxy' in name else actual)-price*tokens
            peak=scores.max()
            ties=np.flatnonzero(scores>=peak-1e-15)
            j=ties[np.argmin(tokens[ties])]
            previous,previous_tokens,_=best[name]
            if scores[j]>previous+1e-15 or (abs(scores[j]-previous)<=1e-15 and tokens[j]<previous_tokens):
                rule=bridge.Rule(width=2,alpha=alpha,beta=0,minimum=int(minimums[j]),cap=512,
                    smoothing="mean",guard=False,latch=True)
                best[name]=(float(scores[j]),float(tokens[j]),rule)
    return best


def run(args):
    output,source,reference=args.output,args.source,args.reference
    output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((source/'manifest.json').read_text())
    data=args.data.resolve() if args.data is not None else Path(meta['data_path'])
    assert old.sha256(data)==meta['data_sha256']
    saved=json.loads((reference/'selected_policies.json').read_text())
    reference_rows=list(csv.DictReader((reference/'mean_retuned'/'split_metrics.csv').open()))
    prices=sorted({float(r['price']) for r in reference_rows})
    old.atomic_json(output/'manifest.json',dict(scope=__doc__,source=str(source.resolve()),
        reference=str(reference.resolve()),data_path=str(data),data_sha256=meta['data_sha256'],
        script_sha256=old.sha256(Path(__file__)),calibration_script_sha256=old.sha256(Path(calibration.__file__)),
        winner_counts=calibration.COUNTS,weight_mixtures=calibration.MIXTURES,
        map_selection='four-fold problem CV winner-mixture Brier score',splits=args.splits,
        calibration_permutations=24,test_permutations=48,prices=prices,methods=METHODS,
        minimum_grid=list(range(10,513)),cap=512,cost='unadjusted prefix mean',
        statistic='top three, full history from n=4, zero cutoffs included, latched crossing',
        bootstrap='1000 shared problem-identity resamples; conditional, pointwise, not selection-adjusted',
        scientific_scope='exploratory reused cohort; no independent final validation'))
    problems=old.load_coding_problems(data)
    weights={key:calibration.winner_weights(problem) for key,problem in problems.items()}
    rows,records={name:[] for name in METHODS},{name:[] for name in METHODS}
    cv_records,quality,policies,map_choices,stop_calibration=[],[],[],[],[]
    public_checks=0
    total_start=time.monotonic()
    for split in range(args.splits):
        tic=time.monotonic()
        train_ids,test_ids=old.split_problem_ids(problems,meta['outer_seed']+split)
        train_problems={key:problems[key] for key in train_ids}
        original_profile=old.CodingProfile.load(source/'profiles'/f'split_{split:02d}.json')
        assert set(original_profile.metadata['fit_problem_ids'])==set(train_ids)
        assert not set(train_ids)&set(test_ids)
        uniform=calibration.fit_profile(train_problems,weights,0)
        reward=np.concatenate([np.asarray(p.rewards) for p in train_problems.values()])
        np.testing.assert_allclose(np.interp(reward,uniform.reward_knots,uniform.probability_knots),
            np.interp(reward,original_profile.reward_knots,original_profile.probability_knots),atol=1e-10)
        profile,cv=calibration.select_profile(train_problems,weights,meta['outer_seed']+2700000+split,
            metadata={'outer_split':split,'holdout_problem_ids':list(test_ids)})
        profile.save(output/'profiles'/f'split_{split:02d}.json')
        cv_records.extend(dict(split=split,**r) for r in cv)
        map_choices.append(dict(split=split,mixture=profile.metadata['winner_weight_mixture'],
            cv_scores=profile.metadata['cv_scores']))
        for group,ids in (('calibration',train_ids),('test',test_ids)):
            subset={key:problems[key] for key in ids}
            for name,p in (('original',original_profile),('selected',profile)):
                quality.extend(dict(split=split,group=group,map=name,**r)
                               for r in calibration.evaluate(p,subset,weights))
        probabilities=old.calibrated_rewards(problems,profile)
        parts=[]
        for ids,count,offset in ((train_ids,24,100000),(test_ids,48,900000)):
            subset={key:problems[key] for key in ids}
            orders=old.make_permutations(subset,count,meta['outer_seed']+offset+10000*split)
            batch=old.build_trajectory_batch(subset,probabilities,orders,width=2)
            labels=old.selected_correctness(batch,subset,orders)
            parts.append((batch,labels,orders))
        (train,train_y,_),(test,test_y,test_orders)=parts
        gain=scan.gain_base(train)
        for price in prices:
            ref=next(r for r in reference_rows if int(r['split'])==split and float(r['price'])==price)
            original=bridge.Rule(**next(r for r in saved if r['split']==split and r['price']==price
                                      and r['method']=='mean_retuned')['rule'])
            fixed_n=int(float(ref['fixed_calls']))
            fixed=bridge.take(test,test_y,np.full(test.trials,fixed_n),price)
            np.testing.assert_allclose(fixed['profit'].mean(),float(ref['fixed_profit']),atol=1e-12)
            tuned=tune_both(train,train_y,gain,price)
            variants={'selected_map_frozen':original,**{name:val[2] for name,val in tuned.items()}}
            for name,rule in variants.items():
                count=bridge.stops(test,None,price,rule)
                metrics=bridge.take(test,test_y,count,price)
                key=str(test.problem_ids[0])
                policy=AdaptiveCoding.from_paper_settings(profile,price,multiplier=rule.alpha,
                    cost_adjustment=0,minimum=rule.minimum,cap=512)
                for index in test_orders[key][0]:
                    decision=policy.observe(problems[key].rewards[index],problems[key].lengths[index])
                    if decision.should_stop:break
                assert decision.count==count[0]
                assert decision.total_length==metrics['tokens'][0]
                assert decision.best_index==test.best_indices[0,count[0]-1]
                public_checks+=1
                rows[name].append(dict(split=split,price=price,minimum=rule.minimum,cap=512,
                    accuracy=float(metrics['correct'].mean()),tokens=float(metrics['tokens'].mean()),
                    calls=float(count.mean()),profit=float(metrics['profit'].mean()),
                    fixed_accuracy=float(fixed['correct'].mean()),fixed_tokens=float(fixed['tokens'].mean()),
                    fixed_calls=fixed_n,fixed_profit=float(fixed['profit'].mean()),
                    cap_rate=float(np.mean(count==512)),minimum_rate=float(np.mean(count==rule.minimum))))
                stop_calibration.append(dict(split=split,price=price,method=name,
                    predicted=float(metrics['predicted'].mean()),correct=float(metrics['correct'].mean()),
                    brier=float(np.mean((metrics['predicted']-metrics['correct'])**2))))
                policies.append(dict(split=split,price=price,method=name,rule=asdict(rule),
                    calibration_score=tuned[name][0] if name in tuned else None))
                for j,identity in enumerate(sorted(test_ids)):
                    sl=slice(48*j,48*(j+1))
                    records[name].append(dict(split=split,price=price,problem_id=identity,
                        profit=float(metrics['profit'][sl].mean()),fixed_profit=float(fixed['profit'][sl].mean()),
                        accuracy=float(metrics['correct'][sl].mean()),tokens=float(metrics['tokens'][sl].mean()),
                        calls=float(count[sl].mean()),fixed_calls=fixed_n,
                        fixed_accuracy=float(fixed['correct'][sl].mean()),fixed_tokens=float(fixed['tokens'][sl].mean())))
        for name in METHODS:
            folder=output/name
            folder.mkdir(exist_ok=True)
            old.atomic_csv(folder/'split_metrics.csv',rows[name])
            old.atomic_csv(folder/'problem_metrics.csv',records[name])
        old.atomic_csv(output/'map_cv_metrics.csv',cv_records)
        old.atomic_csv(output/'map_quality_metrics.csv',quality)
        old.atomic_csv(output/'stopping_calibration.csv',stop_calibration)
        old.atomic_json(output/'selected_policies.json',policies)
        old.atomic_json(output/'map_choices.json',map_choices)
        print(f"split {split}: mixture={profile.metadata['winner_weight_mixture']}, done in {time.monotonic()-tic:.1f}s",flush=True)
    for name in METHODS:
        print(name,flush=True)
        scan.summarize(rows[name],records[name],output/name)
    old.atomic_json(output/'validation.json',dict(completed_splits=args.splits,public_replay_checks=public_checks,
        original_map_reproduced=True,fixed_baseline_reproduced=True,
        map_selection_test_label_access=False,elapsed_seconds=time.monotonic()-total_start))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--splits',type=int,default=10)
    parser.add_argument('--data',type=Path,help='Server path to the matching token-counted JSONL cache')
    run(parser.parse_args())
