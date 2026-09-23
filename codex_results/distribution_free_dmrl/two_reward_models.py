"""Frozen-policy comparison restricted to FSFairX and Mistral reward models.

Policy list frozen after Alpaca/Mistral development, before observing FSFairX
performance or this run's HH results. All five generation models retained.
The previous reward-only rule is retained as a paired control. No offline
training/fitting is required by an adaptive policy; the fixed-N comparator is
chosen from training prompts. These are empirical variants, not certified UCBs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment, validate
from nonparametric_study import GENERATORS,PRICES,write_csv
from online_early_start import collect
import training_free_spacing as comparison


CONFIGS=[dict(minimum=4,smoothing='current',cost_optimism=0.),
         dict(minimum=4,smoothing='mean',cost_optimism=0.),
         dict(minimum=4,smoothing='mean',cost_optimism=1.),
         dict(minimum=4,smoothing='mean',cost_optimism=2.)]
NAMES=['previous','mean_costse0','mean_costse1','mean_costse2']
PRIMARY='mean_costse2'


def load_pools(path, reward_key, length_key, length_unit):
    """Validate recorded lengths; never infer tokens from character counts."""
    if length_unit not in ('characters', 'tokens'):
        raise ValueError('length_unit must be characters or tokens')
    if length_key == 'text_chars' and length_unit != 'characters':
        raise ValueError('text_chars measures characters, not tokens')
    pools = load_alignment(path, reward_key, length_key)
    validate(pools)
    if len(pools) < 2:
        raise ValueError('At least two prompts are required for train/test splitting')
    for pool in pools:
        if len(pool.rewards) < 4:
            raise ValueError(f'Prompt {pool.id} needs at least four responses')
        if np.any(pool.lengths <= 0) or np.any(pool.lengths != np.floor(pool.lengths)):
            raise ValueError(f'Prompt {pool.id} needs positive integer generation lengths')
    return pools


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--reward-key',choices=['fsfairx_rm_reward','mistral_rm_reward'],required=True)
    p.add_argument('--replay-seed',type=int,default=20260929)
    p.add_argument('--length-key',default='text_chars',
                   help='text_chars for historical runs, or a recorded field such as token_count')
    p.add_argument('--length-unit',choices=['characters','tokens'],
                   help='Required for a recorded length field; text_chars defaults to characters')
    args=p.parse_args()
    if args.length_unit is None:
        if args.length_key != 'text_chars':
            p.error('Specify --length-unit for a recorded length field')
        args.length_unit='characters'
    if args.length_key == 'text_chars' and args.length_unit != 'characters':
        p.error('text_chars cannot be labeled as tokens; supply a recorded token-count field')
    paths={gen:args.data_dir/f'{gen}_output.merged_rm.jsonl.gz' for gen in GENERATORS}
    for path in paths.values():
        if not path.is_file():p.error(f'Missing input cache: {path}')
    if args.output.exists():p.error('Refusing overwrite')
    args.output.mkdir(parents=True)
    meta=dict(primary=PRIMARY,configs=CONFIGS,names=NAMES,stage='frozen transfer evaluation',
        reward_key=args.reward_key,data_dir=str(args.data_dir),
        rates=comparison.RATES.tolist(),prices=PRICES.tolist(),split_seeds=[71,72,73,74,75],
        training_free=True,no_parametric_fit=True,permutations=8,replay_seed=args.replay_seed,cap=960,
        cost_unit=args.length_unit,length_key=args.length_key,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['online_early_start.py','online_smoothed_spacing.py','online_cost_optimism.py','training_free_spacing.py']},
        caveat='Frozen primary selected on Alpaca/Mistral development; not a theorem-backed policy; per-reward results not pooled or selected',datasets={})
    (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
    comparison.NAMES=NAMES;profits=[];frontiers=[]
    for gen in GENERATORS:
        print(gen,args.reward_key,flush=True);path=paths[gen]
        pools=load_pools(path,args.reward_key,args.length_key,args.length_unit)
        cap=min(960,min(len(pool.rewards) for pool in pools));a,f=collect(pools,cap,args.replay_seed,CONFIGS)
        np.savez_compressed(args.output/f'{gen}.npz',adaptive=a,fixed=f,ids=[pool.id for pool in pools])
        pr,fr=comparison.summarize(a,f,meta['split_seeds'],gen);profits.extend(pr);frontiers.extend(fr)
        write_csv(args.output/'profit.csv',profits);write_csv(args.output/'frontier_diagnostic.csv',frontiers)
        meta['datasets'][gen]=dict(path=str(path),actual_cap=cap,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (args.output/'METHOD.json').write_text(json.dumps(meta,indent=2)+'\n')
        for name in NAMES:
            rs=[r for r in pr if r['method']==name]
            print(name,'profit',round(np.mean([r['relative_percent'] for r in rs]),3),
                'saving',round(np.mean([r['matched_saving_percent'] for r in rs if r['matched_saving_percent'] is not None]),3),flush=True)


if __name__=='__main__':main()
