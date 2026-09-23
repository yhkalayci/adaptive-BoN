"""Clean two-panel FSFairX figure with translucent pointwise bootstrap bands.

Rendering only: reuse saved means/intervals, place all curves at actual prices,
and move evaluation/provenance details into the accompanying caption note.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,NullLocator
import numpy as np

from plot_fsfairx_generator_lines import GENERATORS,LABELS,COLORS,MARKERS,ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'fsfairx_generator_lines')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    folder=args.source
    output=args.output if args.output is not None else folder/'paper_bands'
    if output.exists():raise ValueError('Refusing to overwrite paper-band figure')
    output.mkdir(parents=True)
    with (folder/'pointwise_intervals.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    meta=json.loads((folder/'METHOD.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
        'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':7.5,
        'ytick.labelsize':7.5,'axes.linewidth':.65,'pdf.fonttype':42,
        'ps.fonttype':42,'savefig.dpi':320})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
    fig.subplots_adjust(left=.08,right=.99,top=.76,bottom=.19,wspace=.30)
    layout=[('profit','(a) Profit improvement','Improvement (%)'),
            ('saving','(b) Cost saving at matched quality','Saving (%)')]
    for ax,(metric,title,ylabel) in zip(axes,layout):
        for gi,gen in enumerate(GENERATORS):
            selected=sorted([r for r in rows if r['generator']==gen and r['metric']==metric],
                            key=lambda r:float(r['price']))
            assert len(selected)==6
            price=np.array([float(r['price']) for r in selected])*1e6
            mean=np.array([float(r['mean']) for r in selected])
            low=np.array([float(r['ci_low']) for r in selected])
            high=np.array([float(r['ci_high']) for r in selected])
            assert np.all(low<=high)
            ax.fill_between(price,low,high,color=COLORS[gi],alpha=.12,linewidth=0,zorder=1)
            ax.plot(price,mean,color=COLORS[gi],marker=MARKERS[gi],markersize=2.9,
                    markeredgewidth=.4,linewidth=1.35,label=LABELS[gi],zorder=3)
        ax.set_title(title,pad=8)
        ax.set_xscale('log');ax.set_xticks(price,[f'{p:g}' for p in price])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r'Generation price ($10^6 p$)',labelpad=5)
        ax.set_ylabel(ylabel,labelpad=4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.axhline(0,color='#777777',linewidth=.65,zorder=2)
        ax.grid(axis='y',color='#D8DEE5',linewidth=.5)
        ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        ax.tick_params(length=3,width=.65);ax.margins(x=.035,y=.075)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.535,.995),
               ncol=5,frameon=False,fontsize=7.4,handlelength=1.5,
               handletextpad=.45,columnspacing=1.0)
    for ext in ('png','pdf'):
        fig.savefig(output/f'fsfairx_profit_cost_bands.{ext}',facecolor='white')
    plt.close(fig)
    provenance=dict(reward_model='FSFairX',policy=meta['method'],
        source_data=str(folder/'pointwise_intervals.csv'),
        source_sha256=hashlib.sha256((folder/'pointwise_intervals.csv').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interval='Same pointwise 95% shared-prompt bootstrap intervals as the whisker plot',
        bootstrap_repetitions=meta['bootstrap_repetitions'],
        rendering='Five generators; alpha .12; actual x positions, no offsets; linearly connected bounds on log-price axis',
        cost_unit=meta.get('cost_unit','characters'),
        caveat='Conditional intervals, not simultaneous or adjusted for policy development',
        manuscript_modified=False)
    (output/'METHOD.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(output/'fsfairx_profit_cost_bands.png')
    print(output/'fsfairx_profit_cost_bands.pdf')


if __name__=='__main__':main()
