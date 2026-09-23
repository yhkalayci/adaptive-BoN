"""Evaluate attained-quality savings of the fixed primary odds profit policy.

This does not replace or repair its unsuccessful direct target controller.
Independent LP matching; no adaptive model/threshold selection.
"""
import csv
import json
from pathlib import Path

import numpy as np
from audit_refinement import optimal_cost
from nonparametric_study import write_csv
from training_free_odds import PRIMARY, VARIANTS


def main():
    root=Path(__file__).parent/'training_free_odds_results'
    destination=root/'PROFIT_MATCHED_AUDIT.json'
    if destination.exists():raise ValueError('Refusing to overwrite audit')
    meta=json.loads((root/'METHOD.json').read_text())
    with (root/'profit.csv').open() as h:previous=list(csv.DictReader(h))
    rows=[];vi=VARIANTS.index(PRIMARY)
    for gen in meta['datasets']:
        z=np.load(root/f'{gen}.npz');a=z['profit'];f=z['fixed']
        for seed in meta['split_seeds']:
            train,test=np.array_split(np.random.default_rng(seed).permutation(len(f)),2)
            fm,ft=f[train].mean(0),f[test].mean(0)
            for pi,price in enumerate(meta['actual_prices']):
                item=a[test,vi,pi].mean(0);fi=np.argmax(fm[:,0]-price*fm[:,1])
                baseline=ft[fi,0]-price*ft[fi,1];profit=item[0]-price*item[1]
                row=next(r for r in previous if r['generator']==gen and int(r['split_seed'])==seed
                         and r['method']==PRIMARY and float(r['price'])==price)
                np.testing.assert_allclose([profit,baseline],
                    [float(row['profit']),float(row['fixed_profit'])],atol=1e-12)
                cost=optimal_cost(ft[:,0],ft[:,1],item[0],exact=True)
                rows.append(dict(generator=gen,split_seed=seed,price=price,
                    quality=item[0],mean_length=item[1],samples=item[2],
                    matched_length=cost,matched_saving_percent=100*(1-item[1]/cost),
                    relative_profit_percent=100*(profit-baseline)/baseline,
                    relative_vs_test_oracle=float(row['relative_vs_test_oracle'])))
    cells={}
    for r in rows:cells.setdefault((r['generator'],r['price']),[]).append(r)
    by_gen={}
    for gen in meta['datasets']:
        rs=[r for r in rows if r['generator']==gen]
        by_gen[gen]={key:float(np.mean([r[key] for r in rs])) for key in
                     ['relative_profit_percent','matched_saving_percent','relative_vs_test_oracle']}
    report=dict(primary=PRIMARY,lp_checks=len(rows),
        average_profit_percent=float(np.mean([r['relative_profit_percent'] for r in rows])),
        average_matched_saving_percent=float(np.mean([r['matched_saving_percent'] for r in rows])),
        average_profit_vs_test_oracle_percent=float(np.mean([r['relative_vs_test_oracle'] for r in rows])),
        positive_profit_cells=int(sum(np.mean([r['relative_profit_percent'] for r in c])>0 for c in cells.values())),
        positive_saving_cells=int(sum(np.mean([r['matched_saving_percent'] for r in c])>0 for c in cells.values())),
        total_cells=len(cells),by_generator=by_gen,
        interpretation='Matched at each price-based policy attained quality, not successful control of a requested target. The fixed-N mixture is chosen retrospectively.',
        scope='Exploratory reused Alpaca prompts; no independent confirmation; no coding replay')
    write_csv(root/'profit_matched_quality.csv',rows)
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
