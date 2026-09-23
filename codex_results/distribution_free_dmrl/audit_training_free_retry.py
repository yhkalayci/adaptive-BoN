"""Independent arithmetic and LP checks for training-free retry experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_refinement import optimal_cost, interpolated_cost


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path)
    parser.add_argument('--bootstrap',type=int,default=300)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--primary')
    parser.add_argument('--skip-previous-cache-check',action='store_true',
                        help='Use only for a different dataset/replay, not the original paired stream')
    args=parser.parse_args()
    out=args.study/'AUDIT.json'
    if out.exists():parser.error('Refusing to overwrite audit')
    meta=json.loads((args.study/'METHOD.json').read_text())
    source=args.source or args.study.parent/(args.study.name.replace('_results','')+'.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
    with (args.study/'profit.csv').open() as h:rows=list(csv.DictReader(h))
    with (args.study/'frontier_diagnostic.csv').open() as h:front=list(csv.DictReader(h))
    names=list(dict.fromkeys(r['method'] for r in rows))
    rates=np.array(meta['rates']);primary=args.primary or meta['primary'];pi=names.index(primary)
    checked=lp_checked=frontier_lp=0
    bootstrap_profit=np.zeros(args.bootstrap)
    bootstrap_saving=np.zeros(args.bootstrap)
    groups=0;ids_ref=None
    rng=np.random.default_rng(20260926)
    weights=None
    for gen,info in meta['datasets'].items():
        assert hashlib.sha256(Path(info['path']).read_bytes()).hexdigest()==info['sha256']
        z=np.load(args.study/f'{gen}.npz');a=z['adaptive'];f=z['fixed'];ids=z['ids']
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(f))
        minimum=min(c.get('minimum',4) for c in meta['configs'])
        assert np.all(a[...,2]>=minimum) and np.all(a[...,2]<=info['actual_cap'])
        if not args.skip_previous_cache_check:
            previous=np.load(args.study.parent/'training_free_odds_results'/f'{gen}.npz')
            np.testing.assert_allclose(f,previous['fixed'],atol=1e-9,rtol=1e-12)
            np.testing.assert_array_equal(ids,previous['ids'])
        if ids_ref is None:
            ids_ref=ids
            weights=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=args.bootstrap)
        else:np.testing.assert_array_equal(ids,ids_ref)
        for seed in meta['split_seeds']:
            train,test=np.array_split(np.random.default_rng(seed).permutation(len(f)),2)
            fm,ft=f[train].mean(0),f[test].mean(0)
            am=a[test].mean(0)
            subset=[r for r in rows if r['generator']==gen and int(r['split_seed'])==seed]
            for r in subset:
                vi=names.index(r['method']);price=float(r['price']);ri=int(np.flatnonzero(rates==price)[0])
                fi=int(np.argmax(fm[:,0]-price*fm[:,1]));item=am[vi,ri]
                val=item[0]-price*item[1];base=ft[fi,0]-price*ft[fi,1]
                assert fi+1==int(r['fixed_n'])
                np.testing.assert_allclose([val,base,100*(val-base)/base,*item],
                    [float(r[k]) for k in ('profit','fixed_profit','relative_percent','quality','mean_length','samples')],atol=1e-10)
                checked+=1
                if r['method']==primary:
                    optimum=optimal_cost(ft[:,0],ft[:,1],item[0],exact=True)
                    np.testing.assert_allclose(optimum,float(r['matched_length']),rtol=1e-7)
                    np.testing.assert_allclose(100*(1-item[1]/optimum),float(r['matched_saving_percent']),atol=1e-6)
                    lp_checked+=1
            for r in front:
                if r['generator']!=gen or int(r['split_seed'])!=seed or r['method']!=primary:continue
                if r['feasible']!='True':continue
                target=float(r['target'])
                ac=optimal_cost(am[pi,:,0],am[pi,:,1],target,exact=True)
                fc=optimal_cost(ft[:,0],ft[:,1],target,exact=True)
                np.testing.assert_allclose([ac,fc],[float(r['adaptive_length']),float(r['fixed_length'])],rtol=1e-7)
                frontier_lp+=2
            w=weights[:,test].astype(float);w/=w.sum(1,keepdims=True)
            fq=w@f[test,:,0];fc=w@f[test,:,1]
            for price in meta['prices']:
                ri=int(np.flatnonzero(rates==price)[0]);fi=int(np.argmax(fm[:,0]-price*fm[:,1]))
                aq=w@a[test,pi,ri,0];ac=w@a[test,pi,ri,1]
                base=fq[:,fi]-price*fc[:,fi]
                bootstrap_profit+=100*(aq-price*ac-base)/base
                for b in range(args.bootstrap):
                    matched=interpolated_cost(fq[b],fc[b],np.array([aq[b]]))[0]
                    bootstrap_saving[b]+=100*(1-ac[b]/matched)
                groups+=1
    summary={}
    for name in names:
        p=[r for r in rows if r['method']==name]
        fr=[r for r in front if r['method']==name]
        feasible=[r for r in fr if r['feasible']=='True']
        cells={}
        for r in p:cells.setdefault((r['generator'],r['price']),[]).append(float(r['relative_percent']))
        summary[name]=dict(mean_profit_percent=float(np.mean([float(r['relative_percent']) for r in p])),
            mean_profit_vs_test_oracle_percent=float(np.mean([float(r['relative_vs_test_oracle']) for r in p])),
            mean_matched_saving_percent=float(np.mean([float(r['matched_saving_percent']) for r in p if r['matched_saving_percent']])),
            positive_profit_cells=int(sum(np.mean(v)>0 for v in cells.values())),profit_cells=len(cells),
            frontier_feasible=len(feasible),frontier_total=len(fr),
            descriptive_frontier_saving_percent=float(np.mean([float(r['frontier_saving_percent']) for r in feasible])))
    report=dict(primary=primary,arithmetic_checks=checked,primary_profit_matched_lp_checks=lp_checked,
        primary_frontier_lp_checks=frontier_lp,
        paired_streams_match_previous=None if args.skip_previous_cache_check else True,
        primary_overridden=args.primary is not None,
        bootstrap_repetitions=args.bootstrap,bootstrap_seed=20260926,
        conditional_profit_interval=np.quantile(bootstrap_profit/groups,[.025,.975]).tolist(),
        conditional_matched_saving_interval=np.quantile(bootstrap_saving/groups,[.025,.975]).tolist(),
        uncertainty='Shared prompt weights across splits and generators; fixed comparator selections; descriptive conditional sensitivity, not independent confirmation',
        results=summary)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
