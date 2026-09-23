"""Near-direct top-three DMRL policies, restricted to FSFairX/Mistral rewards.

Preserve multiplier four, positivity guard, and doubling checkpoints; also
report changing checkpoints alone to every response. Replace true utilities
with current-prefix BT estimates and known cost with observed mean length.
No manuscript changes or inherited theorem guarantee.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment
from nonparametric_study import GENERATORS,PRICES,write_csv
from order_statistic_refinement import prepare
import training_free_spacing as spacing


CONFIGS=[dict(width='fixed2',factor=4.,schedule=s) for s in ('doubling','sequential')]
NAMES=['theory_doubling','theory_sequential']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reward-key',choices=['fsfairx_rm_reward','mistral_rm_reward'],required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing overwrite')
    args.output.mkdir(parents=True)
    meta=dict(primary=NAMES[0],configs=CONFIGS,names=NAMES,reward_key=args.reward_key,
        rates=spacing.RATES.tolist(),prices=PRICES.tolist(),split_seeds=[71,72,73,74,75],
        replay_seed=20260923,permutations=8,cap=960,cost_unit='characters',
        training_free=True,no_parametric_fit=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['training_free_spacing.py','order_statistic_refinement.py']},
        scope='Near-direct practical plug-in, not exact theorem setting',datasets={})
    (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
    spacing.CONFIGS=CONFIGS;spacing.NAMES=NAMES;profits=[];frontiers=[]
    for gen in GENERATORS:
        print(gen,args.reward_key,flush=True)
        path=Path('dataset/alpaca')/f'{gen}_output.merged_rm.jsonl.gz'
        pools=load_alignment(path,args.reward_key,'text_chars')
        cap=min(960,min(len(pool.rewards) for pool in pools))
        top,ref,length,quality,_=prepare(pools,cap,8,20260923)
        a=spacing.evaluate(top,ref,length,quality)
        f=np.stack([quality.mean(axis=1),length.mean(axis=1),
            np.broadcast_to(np.arange(1,cap+1),(len(pools),cap))],axis=2)
        np.savez_compressed(args.output/f'{gen}.npz',adaptive=a,fixed=f,ids=[pool.id for pool in pools])
        pr,fr=spacing.summarize(a,f,meta['split_seeds'],gen);profits.extend(pr);frontiers.extend(fr)
        write_csv(args.output/'profit.csv',profits);write_csv(args.output/'frontier_diagnostic.csv',frontiers)
        meta['datasets'][gen]=dict(path=str(path),actual_cap=cap,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
        for name in NAMES:
            rs=[r for r in pr if r['method']==name]
            print(name,'profit',round(np.mean([r['relative_percent'] for r in rs]),3),
                'saving',round(np.mean([r['matched_saving_percent'] for r in rs if r['matched_saving_percent'] is not None]),3),flush=True)
        del top,ref,length,quality


if __name__=='__main__':main()
