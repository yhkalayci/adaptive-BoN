"""Independently audit selections, accounting, and quality-matched comparisons.

Linear programs verify mixture optimality independently of the experiment's
convex-hull code. Shared prompt weights measure conditional sensitivity without
treating overlapping splits or generators as independent datasets.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def optimal_cost(q, c, target, exact):
    scale = max(float(np.max(c)), 1.)
    constraints = (dict(A_eq=np.stack([np.ones(len(q)), q]), b_eq=[1., target])
                   if exact else dict(A_eq=np.ones((1,len(q))), b_eq=[1.],
                                      A_ub=-q[None,:], b_ub=[-target]))
    # Tight constraints matter when nearly equal qualities correspond to
    # materially different costs; default feasibility tolerances can obscure
    # sub-unit differences in costs of order 1e5.
    solution = linprog(c/scale, bounds=(0,None), method='highs',
                      options={'primal_feasibility_tolerance':1e-10,
                               'dual_feasibility_tolerance':1e-10}, **constraints)
    assert solution.success, solution.message
    return float(solution.fun*scale)


def interpolated_cost(q, c, targets):
    """Independent lower-envelope implementation for bootstrap fixed-N curves."""
    vertices = []
    for i in range(len(q)):
        if vertices and q[i] == q[vertices[-1]]:
            if c[i] >= c[vertices[-1]]:
                continue
            vertices.pop()
        while len(vertices)>1:
            a,b = vertices[-2:]
            if (c[b]-c[a])*(q[i]-q[b]) < (c[i]-c[b])*(q[b]-q[a]):
                break
            vertices.pop()
        vertices.append(i)
    assert np.all(targets >= q[vertices[0]]-1e-10)
    assert np.all(targets <= q[vertices[-1]]+1e-10)
    return np.interp(targets, q[vertices], c[vertices])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path)
    parser.add_argument('--bootstrap',type=int,default=400)
    args = parser.parse_args()
    output = args.study/'AUDIT.json'
    if output.exists():
        parser.error('Audit already exists; refusing overwrite')
    meta = json.loads((args.study/'METHOD.json').read_text())
    incumbent_study = 'family' in meta['configurations'][0]
    source = args.study.parent/('incumbent_aware_study.py' if incumbent_study else 'order_statistic_refinement.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
    p = read_csv(args.study/'profit.csv')
    t = read_csv(args.study/'target_quality.csv')
    configs = meta['configurations']
    groups = {'empirical_spacing':np.array([i for i,v in enumerate(configs) if v['reference']=='empirical']),
              'corrected_spacing':np.array([i for i,v in enumerate(configs) if v['reference']=='corrected']),
              'train_selected_spacing':np.arange(len(configs))}
    if incumbent_study:
        groups = {family:np.array([i for i,v in enumerate(configs) if v['family']==family])
                  for family in ('spacing','dmrl_envelope','moment_envelope')}
        groups['all_train_selected'] = np.arange(len(configs))
    rng = np.random.default_rng(20260924)
    ids = next(iter(meta['datasets'].values()))['ids']
    weights = rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=args.bootstrap)
    boot_profit = {m:[] for m in groups}
    boot_saving = {m:[] for m in groups}
    checks = {'profit_selections':0,'target_training_optima':0,'test_matched_optima':0}
    for model,d in meta['datasets'].items():
        assert d['ids']==ids
        assert hashlib.sha256(Path(d['path']).read_bytes()).hexdigest()==d['sha256']
        fixed_root = Path(meta['previous']) if incumbent_study else args.study
        with np.load(fixed_root/f'{model}_fixed.npz') as cache:
            fixed = cache['fixed']
        np.testing.assert_allclose(fixed[:,:,2],np.broadcast_to(np.arange(1,961),fixed[:,:,2].shape))
        for seed in meta['split_seeds']:
            with np.load(args.study/f'{model}_seed{seed}.npz') as cache:
                a,train,test = cache['adaptive'],cache['train'],cache['test']
            assert not set(train)&set(test)
            assert set(train)|set(test)==set(range(len(ids)))
            assert np.isfinite(a).all() and np.isfinite(fixed).all()
            am,fm = a[train].mean(axis=0),fixed[train].mean(axis=0)
            ft = fixed[test].mean(axis=0)
            w = weights[:,test].astype(float)
            w /= w.sum(axis=1,keepdims=True)
            fq,fc = w@fixed[test,:,0],w@fixed[test,:,1]
            for method,eligible in groups.items():
                pp = [r for r in p if r['generator']==model and int(r['split_seed'])==seed and r['method']==method]
                tt = [r for r in t if r['generator']==model and int(r['split_seed'])==seed and r['method']==method]
                for row in pp:
                    price = float(row['price'])
                    selected = int(eligible[np.argmax(am[eligible,0]-price*am[eligible,1])])
                    fi = int(np.argmax(fm[:,0]-price*fm[:,1]))
                    assert selected==int(row['selected']) and fi+1==int(row['fixed_n'])
                    pa = a[test,selected,0]-price*a[test,selected,1]
                    pf = fixed[test,fi,0]-price*fixed[test,fi,1]
                    np.testing.assert_allclose([pa.mean(),pf.mean(),100*(pa.mean()-pf.mean())/pf.mean()],
                        [float(row['profit']),float(row['fixed_profit']),float(row['relative_percent'])],atol=1e-10)
                    boot_profit[method].append(100*((w@pa)-(w@pf))/(w@pf))
                    checks['profit_selections']+=1
                test_mixtures = []
                for row in tt:
                    target = float(row['target'])
                    lo,hi,weight = int(row['adaptive_low']),int(row['adaptive_high']),float(row['adaptive_high_weight'])
                    assert lo in eligible and hi in eligible and 0<=weight<=1
                    train_mix = (1-weight)*am[lo]+weight*am[hi]
                    if row['train_feasible']=='True':
                        assert train_mix[0]>=target-1e-10
                        optimum = optimal_cost(am[eligible,0],am[eligible,1],target,exact=False)
                        np.testing.assert_allclose(train_mix[1],optimum,rtol=2e-6,atol=.02)
                    else:
                        assert am[eligible,0].max()<target
                        np.testing.assert_allclose(train_mix[0],am[eligible,0].max())
                    v = (1-weight)*a[test,lo]+weight*a[test,hi]
                    mean = v.mean(axis=0)
                    np.testing.assert_allclose(mean[:2],[float(row['attained_quality']),float(row['mean_length'])])
                    matched = optimal_cost(ft[:,0],ft[:,1],mean[0],exact=True)
                    np.testing.assert_allclose(matched,float(row['matched_length']),rtol=2e-6,atol=.02)
                    np.testing.assert_allclose(100*(1-mean[1]/matched),float(row['matched_saving_percent']),atol=.002)
                    test_mixtures.append(v)
                    checks['target_training_optima']+=1
                    checks['test_matched_optima']+=1
                values = np.stack(test_mixtures,axis=1)
                aq,ac = w@values[:,:,0],w@values[:,:,1]
                for b in range(args.bootstrap):
                    matched = interpolated_cost(fq[b],fc[b],aq[b])
                    # Store [bootstrap,target] once rather than counting replays.
                    if b==0:
                        savings = np.empty_like(aq)
                    savings[b] = 100*(1-ac[b]/matched)
                boot_saving[method].extend(savings.T)
            print(model,seed,'audited',flush=True)
    result = dict(checks=checks,source_and_data_hashes_verified=True,
        bootstrap=dict(replicates=args.bootstrap,seed=20260924,
            interpretation='Descriptive conditional prompt-cluster sensitivity; selections and training fits fixed. Same prompt weights across generators and overlapping splits. Not independent confirmation, no adjustment for model development.'),
        methods={})
    for method in groups:
        pr = np.array([float(r['relative_percent']) for r in p if r['method']==method])
        sr = np.array([float(r['matched_saving_percent']) for r in t if r['method']==method])
        bp = np.mean(boot_profit[method],axis=0)
        bs = np.mean(boot_saving[method],axis=0)
        result['methods'][method] = dict(mean_profit_improvement_percent=float(pr.mean()),
            profit_conditional_95_percentile=np.quantile(bp,[.025,.975]).tolist(),
            mean_quality_matched_saving_percent=float(sr.mean()),
            saving_conditional_95_percentile=np.quantile(bs,[.025,.975]).tolist())
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
