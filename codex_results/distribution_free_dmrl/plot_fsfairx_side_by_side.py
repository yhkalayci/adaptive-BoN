"""Render existing prompt-bootstrap intervals as a single two-panel figure."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_fsfairx_generator_lines import GENERATORS,LABELS,COLORS,MARKERS,ROOT


def main():
    folder=ROOT/'fsfairx_generator_lines'
    png=folder/'profit_and_cost_saving.png';pdf=folder/'profit_and_cost_saving.pdf'
    if png.exists() or pdf.exists():raise ValueError('Refusing to overwrite combined figure')
    with (folder/'pointwise_intervals.csv').open() as handle:rows=list(csv.DictReader(handle))
    meta=json.loads((folder/'METHOD.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'savefig.dpi':210})
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.8))
    fig.subplots_adjust(left=.065,right=.985,top=.73,bottom=.25,wspace=.22)
    fig.suptitle('FSFairX reward model · adaptive stopping versus best-of-N',
                 fontsize=17,fontweight='bold',y=.975)
    settings=[('profit','(a) Profit improvement','Relative profit improvement (%)'),
              ('saving','(b) Cost savings at matched quality','Generation-cost saving (%)')]
    for ax,(metric,title,ylabel) in zip(axes,settings):
        for gi,gen in enumerate(GENERATORS):
            selected=sorted([r for r in rows if r['generator']==gen and r['metric']==metric],
                            key=lambda r:float(r['price']))
            assert len(selected)==6
            price=np.array([float(r['price']) for r in selected])*1e6
            x=price*np.exp(.045*(gi-2))
            mean=np.array([float(r['mean']) for r in selected])
            lo=np.array([float(r['ci_low']) for r in selected]);hi=np.array([float(r['ci_high']) for r in selected])
            ax.plot(x,mean,color=COLORS[gi],marker=MARKERS[gi],linewidth=1.8,markersize=4.8,
                    label=LABELS[gi],zorder=3)
            ax.errorbar(x,(hi+lo)/2,yerr=(hi-lo)/2,fmt='none',color=COLORS[gi],
                        elinewidth=1.05,capsize=3,capthick=1.05,alpha=.82,zorder=2)
        ax.set_title(title,fontsize=12,pad=12)
        ax.set_xscale('log');ax.set_xticks(price,[f'{p:g}' for p in price])
        ax.set_xlabel('Price ($ per million recorded characters)',labelpad=8)
        ax.set_ylabel(ylabel);ax.axhline(0,color='#555555',linewidth=.8)
        ax.grid(axis='y',color='#DDE3E8',linewidth=.7);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False);ax.margins(x=.07,y=.1)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.9),
               ncol=5,frameon=False,title='Generation model',fontsize=9.5,title_fontsize=9.5)
    fig.text(.065,.145,f"Whiskers: pointwise 95% prompt-bootstrap intervals ({meta['bootstrap_repetitions']:,} resamples); small x-offsets separate generators.",fontsize=8.5)
    fig.text(.065,.10,'Profit baseline: training-selected fixed N. Cost baseline: retrospective fixed-N mixture matching the policy’s attained quality.',fontsize=8.5)
    fig.text(.065,.055,'Shared prompt resampling respects overlapping splits. Intervals condition on fixed policies/selections; not simultaneous or selection-adjusted.',fontsize=8.2,color='#444444')
    fig.savefig(png,facecolor='white');fig.savefig(pdf,facecolor='white');plt.close(fig)
    (folder/'COMBINED_FIGURE.json').write_text(json.dumps(dict(
        data='pointwise_intervals.csv',data_sha256=hashlib.sha256((folder/'pointwise_intervals.csv').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        layout='profit left, attained-quality cost savings right; FSFairX only; five generator lines; pointwise CI whiskers'),indent=2)+'\n')
    print(png);print(pdf)


if __name__=='__main__':main()
