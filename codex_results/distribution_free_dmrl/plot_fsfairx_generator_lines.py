"""FSFairX-only generator lines with shared-prompt bootstrap CI whiskers.

Bootstrap samples prompt IDs, not split rows or response permutations. The same
weights are used across overlapping splits and generation models. Fixed-N
selections and adaptive configurations remain fixed; the quality-matched
fixed-count frontier is recomputed in each resample. Pointwise percentile
intervals are conditional, not post-selection or simultaneous guarantees.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_refinement import interpolated_cost
from nonparametric_study import GENERATORS,write_csv


ROOT=Path(__file__).parent
STUDY=ROOT/'frozen_alpaca_fsfairx'
LABELS=['Gemma-2-9B','Llama-3.1-8B','Llama-3.2-3B','Mistral-7B','Qwen-2.5-7B']
COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']
MARKERS=['o','s','^','D','P']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=20260930)
    parser.add_argument('--study',type=Path,default=STUDY)
    parser.add_argument('--output',type=Path,default=ROOT/'fsfairx_generator_lines')
    args=parser.parse_args()
    study=args.study
    out=args.output
    if out.exists():parser.error('Refusing to overwrite generated figures or intervals')
    if args.bootstrap < 1:parser.error('--bootstrap must be positive')
    meta=json.loads((study/'METHOD.json').read_text())
    if meta.get('reward_key') != 'fsfairx_rm_reward':
        parser.error('This figure requires an FSFairX study')
    cost_unit=meta.get('cost_unit','characters')
    out.mkdir(parents=True)
    with (study/'profit.csv').open() as h:source_rows=list(csv.DictReader(h))
    primary=meta['primary'];vi=meta['names'].index(primary)
    prices=np.array(meta['prices']);rates=np.array(meta['rates'])
    ri=np.array([np.flatnonzero(rates==p)[0] for p in prices])
    rng=np.random.default_rng(args.seed);ids_ref=None;weights=None
    summaries=[];bootstrap_arrays={};source_hashes={}
    for gen in GENERATORS:
        print('Bootstrap:',gen,flush=True)
        path=study/f'{gen}.npz';z=np.load(path)
        a=z['adaptive'][:,vi,ri,:];f=z['fixed'];ids=z['ids']
        source_hashes[gen]=hashlib.sha256(path.read_bytes()).hexdigest()
        if ids_ref is None:
            ids_ref=ids
            weights=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=args.bootstrap)
        else:np.testing.assert_array_equal(ids,ids_ref)
        boot_profit=np.zeros((args.bootstrap,len(prices)))
        boot_saving=np.zeros_like(boot_profit)
        point_profit=[];point_saving=[]
        for seed in meta['split_seeds']:
            train,test=np.array_split(np.random.default_rng(seed).permutation(len(f)),2)
            fm=f[train].mean(0);ft=f[test].mean(0);am=a[test].mean(0)
            fixed_index=np.argmax(fm[:,0,None]-fm[:,1,None]*prices[None,:],axis=0)
            baseline=ft[fixed_index,0]-prices*ft[fixed_index,1]
            point_profit.append(100*(am[:,0]-prices*am[:,1]-baseline)/baseline)
            matched=interpolated_cost(ft[:,0],ft[:,1],am[:,0])
            point_saving.append(100*(1-am[:,1]/matched))
            w=weights[:,test].astype(float)
            assert np.all(w.sum(axis=1)>0)
            w/=w.sum(axis=1,keepdims=True)
            fq=w@f[test,:,0];fc=w@f[test,:,1]
            aq=w@a[test,:,0];ac=w@a[test,:,1]
            base=fq[:,fixed_index]-prices[None,:]*fc[:,fixed_index]
            assert np.all(base>0)
            boot_profit+=100*(aq-prices[None,:]*ac-base)/base
            # One hull per resample serves all six price-specific quality levels.
            for b in range(args.bootstrap):
                mc=interpolated_cost(fq[b],fc[b],aq[b])
                boot_saving[b]+=100*(1-ac[b]/mc)
        boot_profit/=len(meta['split_seeds']);boot_saving/=len(meta['split_seeds'])
        means={'profit':np.mean(point_profit,axis=0),'saving':np.mean(point_saving,axis=0)}
        for metric,boots in [('profit',boot_profit),('saving',boot_saving)]:
            intervals=np.quantile(boots,[.025,.975],axis=0)
            key='relative_percent' if metric=='profit' else 'matched_saving_percent'
            for pi,price in enumerate(prices):
                recorded=np.mean([float(r[key]) for r in source_rows if r['method']==primary
                    and r['generator']==gen and float(r['price'])==price])
                np.testing.assert_allclose(means[metric][pi],recorded,atol=1e-9)
                summaries.append(dict(generator=gen,reward_model='FSFairX',method=primary,
                    metric=metric,price=float(price),mean=float(means[metric][pi]),
                    ci_low=float(intervals[0,pi]),ci_high=float(intervals[1,pi]),
                    confidence=.95,bootstrap_repetitions=args.bootstrap))
        bootstrap_arrays[gen+'_profit']=boot_profit
        bootstrap_arrays[gen+'_saving']=boot_saving
    write_csv(out/'pointwise_intervals.csv',summaries)
    np.savez_compressed(out/'bootstrap_replicates.npz',**bootstrap_arrays)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'pdf.fonttype':42,'savefig.dpi':200})
    for metric,title,ylabel,filename in [
        ('profit','Profit improvement · FSFairX rewards','Profit improvement over best-of-N (%)','profit_improvement'),
        ('saving','Matched-quality cost savings · FSFairX rewards','Generation-cost saving (%)','matched_quality_cost_saving')]:
        fig,ax=plt.subplots(figsize=(10,6.3))
        fig.subplots_adjust(left=.105,right=.97,top=.73,bottom=.24)
        fig.suptitle(title,fontsize=17,fontweight='bold',y=.98)
        for gi,gen in enumerate(GENERATORS):
            rows=sorted([r for r in summaries if r['generator']==gen and r['metric']==metric],
                        key=lambda r:r['price'])
            y=np.array([r['mean'] for r in rows]);lo=np.array([r['ci_low'] for r in rows]);hi=np.array([r['ci_high'] for r in rows])
            # Slight multiplicative offsets keep five whiskers at one price legible.
            x=prices*1e6*np.exp(.045*(gi-2))
            ax.plot(x,y,color=COLORS[gi],marker=MARKERS[gi],markersize=5,
                    linewidth=1.8,label=LABELS[gi],zorder=3)
            ax.errorbar(x,(lo+hi)/2,yerr=(hi-lo)/2,fmt='none',color=COLORS[gi],
                        elinewidth=1.05,capsize=3,capthick=1.05,alpha=.8,zorder=2)
        ax.set_xscale('log');ax.set_xticks(prices*1e6,[f'{p:g}' for p in prices*1e6])
        ax.set_xlabel(f'Generation price ($ per million recorded {cost_unit})',labelpad=9)
        ax.set_ylabel(ylabel);ax.axhline(0,color='#555555',linewidth=.85)
        ax.grid(axis='y',color='#DDE3E8',linewidth=.7);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False);ax.margins(x=.06,y=.10)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.92),
                   ncol=3,frameon=False,title='Generation model',fontsize=9.5,title_fontsize=9.5)
        fig.text(.105,.15,f'Whiskers: pointwise 95% prompt-bootstrap intervals ({args.bootstrap:,} resamples); slight x-offsets for readability.',fontsize=8.3)
        fig.text(.105,.105,'Shared prompt resampling across overlapping splits; policies and baseline selections held fixed.',fontsize=8.3)
        if metric=='profit':
            note='Baseline: fixed N chosen on training prompts. Conditional intervals do not account for policy-development selection.'
        else:
            note='Baseline: retrospective fixed-N mixture matching attained quality, recomputed per resample; not target control.'
        fig.text(.105,.06,note,fontsize=8.1,color='#444444')
        for ext in ('png','pdf'):fig.savefig(out/f'{filename}.{ext}',facecolor='white')
        plt.close(fig)
    report=dict(reward_model='FSFairX',method=primary,generators=GENERATORS,
        cost_unit=cost_unit,length_key=meta.get('length_key','text_chars'),
        bootstrap_repetitions=args.bootstrap,seed=args.seed,confidence=.95,
        resampling='Multinomial prompt weights shared across splits and generators; eight orders averaged within prompt',
        aggregation='Mean of splitwise relative improvements, preserving original plotted estimand',
        conditional_on='Adaptive policy, fixed-N selections, existing prompt splits, cached replay averages',
        reestimated='Quality-matched fixed-N frontier in each resample',
        limitation='Pointwise conditional percentile intervals; not simultaneous and not adjusted for policy-development selection',
        plotted_values_verified_against_saved_csv=True,x_offsets='exp(.045*(generator_index-2)) only for visual separation',
        source_hashes=source_hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'METHOD.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',out,flush=True)


if __name__=='__main__':main()
