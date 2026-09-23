"""Reproducible descriptive plots for the frozen, training-free practical rule.

Only FSFairX and Mistral reward models. No data generation or manuscript edits.
Output includes plotted source values, scope notes, and source-data hashes.
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

from nonparametric_study import GENERATORS,PRICES,write_csv


ROOT=Path(__file__).parent
SOURCES={
    'FSFairX':(ROOT/'frozen_alpaca_fsfairx','mean_costse2','#006F9B'),
    'Mistral':(ROOT/'online_cost_optimism_results','odds_mean_costse2','#C75A17'),
}
LABELS=['Gemma-2\n9B','Llama-3.1\n8B','Llama-3.2\n3B','Mistral\n7B','Qwen-2.5\n7B']


def read_rows(path):
    with path.open() as h:return list(csv.DictReader(h))


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#DDE3E8',linewidth=.65)
    ax.set_axisbelow(True)
    ax.axhline(0,color='#555555',linewidth=.8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-render',action='store_true',
                        help='Replace this script\'s generated figures/plot tables only')
    args=parser.parse_args()
    out=ROOT/'smoothed_policy_report'
    if out.exists() and not args.refresh_render:raise ValueError('Refusing overwrite')
    out.mkdir(exist_ok=args.refresh_render)
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,
        'figure.dpi':120,'savefig.dpi':190,'font.family':'DejaVu Sans',
        'pdf.fonttype':42,'ps.fonttype':42})
    records={};frontiers={};hashes={};summary={};plot_rows=[]
    for label,(folder,method,color) in SOURCES.items():
        records[label]=[r for r in read_rows(folder/'profit.csv') if r['method']==method]
        frontiers[label]=[r for r in read_rows(folder/'frontier_diagnostic.csv') if r['method']==method]
        assert len(records[label])==150
        hashes[label]={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                       for name in ('profit.csv','frontier_diagnostic.csv','METHOD.json')}
        summary[label]={key:float(np.mean([float(r[key]) for r in records[label]]))
                        for key in ['relative_percent','relative_vs_test_oracle','matched_saving_percent']}
    specifications=[('profit_improvement','relative_percent',
        'Profit improvement over fixed-count best-of-N','Relative profit improvement (%)'),
        ('matched_quality_cost_saving','matched_saving_percent',
        'Generation cost saved at the same attained quality','Generation-cost saving (%)')]
    for filename,key,title,ylabel in specifications:
        fig,axes=plt.subplots(1,2,figsize=(12.5,5.3),gridspec_kw={'width_ratios':[1.1,1]})
        fig.subplots_adjust(top=.72,bottom=.27,left=.075,right=.985,wspace=.26)
        fig.suptitle(title,y=.97,fontsize=17,fontweight='semibold')
        for li,(label,(_,_,color)) in enumerate(SOURCES.items()):
            rows=records[label]
            by_price=[];by_gen=[]
            for price in PRICES:
                selected=[r for r in rows if float(r['price'])==price]
                value=float(np.mean([float(r[key]) for r in selected]));by_price.append(value)
                plot_rows.append(dict(figure=filename,reward_model=label,group='price',
                    setting=float(price),value=value,cases=len(selected)))
            for gen in GENERATORS:
                selected=[r for r in rows if r['generator']==gen]
                value=float(np.mean([float(r[key]) for r in selected]));by_gen.append(value)
                plot_rows.append(dict(figure=filename,reward_model=label,group='generator',
                    setting=gen,value=value,cases=len(selected)))
            axes[0].plot(PRICES*1e6,by_price,color=color,marker='o' if li==0 else 's',
                         linewidth=2.1,markersize=5,label=label+' reward model')
            x=np.arange(len(GENERATORS))+(li-.5)*.36
            bars=axes[1].bar(x,by_gen,width=.34,color=color,zorder=3)
            axes[1].bar_label(bars,labels=[f'{v:.1f}' for v in by_gen],padding=3,fontsize=8,color=color)
        axes[0].set_xscale('log');axes[0].set_xticks(PRICES*1e6,[f'{p:g}' for p in PRICES*1e6])
        axes[0].set_xlabel('Price ($ per million recorded characters)')
        axes[0].set_ylabel(ylabel);axes[0].set_title('By generation price · mean over five generators',pad=10)
        axes[1].set_xticks(np.arange(len(GENERATORS)),LABELS)
        axes[1].set_title('By generation model · mean over six prices',pad=10)
        axes[1].margins(y=.18)
        for ax in axes:style(ax)
        if key=='relative_percent':
            for ax in axes:ax.axhline(5,color='#777777',linestyle=':',linewidth=1)
            foot='Profit = BT utility − all generation costs. Baseline: N selected on training prompts from N=1,…,960.'
            avg=f"Overall means: FSFairX {summary['FSFairX'][key]:.2f}%  |  Mistral {summary['Mistral'][key]:.3f}%"
        else:
            foot='Comparator: cheapest retrospective fixed-N mixture matching each price-based policy’s attained quality.'
            avg=f"Overall means: FSFairX {summary['FSFairX'][key]:.2f}%  |  Mistral {summary['Mistral'][key]:.2f}%"
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.91),
                   ncol=2,frameon=False)
        fig.text(.5,.825,avg,ha='center',fontsize=10,color='#303B46')
        fig.text(.075,.13,foot,fontsize=9,color='#394653')
        fig.text(.075,.075,'Alpaca · 100 prompts/generator · 8 replay orders · 5 overlapping splits · descriptive means, not independent trials.',
                 fontsize=8.5,color='#53606D')
        if key=='matched_saving_percent':
            fig.text(.075,.035,'Savings at attained quality do not imply reliable control of a requested quality target.',
                     fontsize=8.5,color='#53606D')
        for ext in ('png','pdf'):fig.savefig(out/f'{filename}.{ext}',facecolor='white')
        plt.close(fig)
    # Supplementary target-quality plot: explicitly separate from the deployable rule.
    fig,ax=plt.subplots(figsize=(8.7,5.5));fig.subplots_adjust(left=.11,right=.97,top=.78,bottom=.28)
    fig.suptitle('Target-quality frontier: retrospective diagnostic',fontsize=16,fontweight='semibold',y=.97)
    for li,(label,(_,_,color)) in enumerate(SOURCES.items()):
        rows=frontiers[label];targets=sorted({float(r['target']) for r in rows})
        values=[];counts=[]
        for t in targets:
            selected=[r for r in rows if float(r['target'])==t and r['feasible']=='True']
            value=float(np.mean([float(r['frontier_saving_percent']) for r in selected]))
            values.append(value);counts.append(len(selected))
            plot_rows.append(dict(figure='target_quality_frontier',reward_model=label,group='target',
                setting=t,value=value,cases=len(selected)))
        ax.plot(targets,values,color=color,marker='o' if li==0 else 's',linewidth=2,label=label+' reward model')
        for t,value,count in zip(targets,values,counts):
            if count<25:
                ax.annotate(f'{count}/25',xy=(t,value),xytext=(10,-16 if li==0 else 10),
                            textcoords='offset points',ha='left',fontsize=8,color=color)
    style(ax);ax.set_xlabel('Target mean Bradley–Terry utility');ax.set_ylabel('Generation-cost saving (%)')
    ax.set_xticks(targets);ax.margins(y=.25)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.91),ncol=2,frameon=False)
    fig.text(.11,.16,'Both adaptive-price mixtures and fixed-N mixtures are chosen retrospectively.',fontsize=9)
    fig.text(.11,.11,'Not a deployable training-free target controller. Means exclude infeasible cases.',fontsize=9)
    fig.text(.11,.06,'Labels show feasible generator–split cases where fewer than 25/25; splits overlap.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'target_quality_frontier.{ext}',facecolor='white')
    plt.close(fig)
    write_csv(out/'plot_values.csv',plot_rows)
    (out/'SUMMARY.json').write_text(json.dumps(dict(results=summary,source_hashes=hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cost_unit='characters',scope='Only FSFairX and Mistral rewards; manuscript unchanged',
        selection='Cost optimism coefficient 2 selected on Alpaca/Mistral development and frozen for FSFairX',
        uncertainty='No error bars: displayed quantities are descriptive means over overlapping splits'),indent=2)+'\n')
    print(json.dumps(summary,indent=2));print('Figures saved to',out)


if __name__=='__main__':main()
