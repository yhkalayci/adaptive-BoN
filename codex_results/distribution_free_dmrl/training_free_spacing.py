"""Training-free order-statistic stopping at the actual deployment price.

No adaptive selection uses training prompts. A fixed, disclosed sensitivity
grid separates smoothing, optimism, and checkpoint frequency. The primary
variant is declared in source before replay. Frontier interpolation is an
evaluation diagnostic, NOT a training-free target-quality controller.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment, sigmoid
from nonparametric_study import GENERATORS, PRICES, write_csv
from order_statistic_refinement import prepare, width_sequence, target_mix, TARGETS


CONFIGS = [dict(width=w, factor=a, schedule=s)
           for w in ('fixed2', 'sqrt32', 'quarter32')
           for a in (1., 2., 4.) for s in ('sequential', 'doubling')]
NAMES = [f"{c['width']}_a{c['factor']:g}_{c['schedule']}" for c in CONFIGS]
PRIMARY = 'sqrt32_a1_sequential'
RATES = np.unique(np.concatenate([PRICES, np.geomspace(2e-10, 2e-4, 49)]))


def stopping_indices(numerator, cumulative_lengths, rates, minimum, schedule):
    """First eligible crossing of a gain/cost ratio, returning zero-based n."""
    rates = np.asarray(rates)
    if np.any(cumulative_lengths <= 0):
        raise ValueError('Positive cumulative lengths required')
    cap = numerator.shape[-1]
    threshold = numerator / cumulative_lengths
    n = np.arange(1, cap + 1)
    eligible = n >= minimum
    if schedule == 'doubling':
        eligible &= (n >= 4) & ((n & (n - 1)) == 0)
    elif schedule != 'sequential':
        raise ValueError(schedule)
    threshold = np.where(eligible, threshold, np.inf)
    crossing = np.minimum.accumulate(threshold, axis=-1)
    return np.array([np.minimum(np.searchsorted(-row, -rates), cap - 1)
                     for row in crossing.reshape(-1, cap)]).reshape(
                         numerator.shape[:-1] + (len(rates),))


def evaluate(top, reference, lengths, quality, rates=RATES):
    """Utilities/reference used to decide; evaluation quality used only to score."""
    utility = sigmoid(top.astype(float) - reference[..., None])
    sums = np.cumsum(utility, axis=-1)
    output = np.zeros((len(top), len(CONFIGS), len(rates), 3))
    cap = reference.shape[-1]
    for ci, config in enumerate(CONFIGS):
        k, minimum = width_sequence(config['width'], cap)
        ix = np.broadcast_to(k, reference.shape)[..., None]
        upper = np.take_along_axis(sums, np.maximum(ix - 1, 0), axis=-1)[..., 0]
        floor = np.take_along_axis(utility, ix, axis=-1)[..., 0]
        numerator = config['factor'] * np.maximum(upper / np.maximum(k, 1) - floor, 0)
        # As in the theorem, do not stop at a zero order-statistic threshold.
        numerator = np.where(floor > 0, numerator, np.inf)
        stop = stopping_indices(numerator, lengths, rates, minimum, config['schedule'])
        output[:, ci] = np.stack([
            np.take_along_axis(quality, stop, axis=2).mean(axis=1),
            np.take_along_axis(lengths, stop, axis=2).mean(axis=1),
            (stop + 1).mean(axis=1)], axis=2)
    return output


def matched_cost(values, target):
    mix = target_mix(values[:, 0], values[:, 1], target, exact=True)
    if mix is None:
        return None
    a, b, w, _ = mix
    return float((1-w)*values[a, 1] + w*values[b, 1])


def summarize(adaptive, fixed, seeds, generator):
    profits, frontiers = [], []
    for seed in seeds:
        train, test = np.array_split(np.random.default_rng(seed).permutation(len(fixed)), 2)
        fm, ft = fixed[train].mean(axis=0), fixed[test].mean(axis=0)
        am = adaptive[test].mean(axis=0)
        for vi, name in enumerate(NAMES):
            for price in PRICES:
                pi = int(np.flatnonzero(RATES == price)[0])
                fi = int(np.argmax(fm[:, 0]-price*fm[:, 1]))
                a = am[vi, pi]
                value = a[0]-price*a[1]
                baseline = ft[fi, 0]-price*ft[fi, 1]
                oracle = np.max(ft[:, 0]-price*ft[:, 1])
                matched = matched_cost(ft, a[0])
                profits.append(dict(generator=generator, split_seed=seed, method=name,
                    price=float(price), profit=value, fixed_profit=baseline, fixed_n=fi+1,
                    relative_percent=100*(value-baseline)/baseline,
                    relative_vs_test_oracle=100*(value-oracle)/oracle,
                    quality=a[0], mean_length=a[1], samples=a[2],
                    matched_length=matched, matched_saving_percent=
                    100*(1-a[1]/matched) if matched else None))
            for target in TARGETS:
                # BOTH mixtures selected retrospectively. No deployable target claim.
                ac = matched_cost(am[vi], target)
                fc = matched_cost(ft, target)
                frontiers.append(dict(generator=generator, split_seed=seed, method=name,
                    target=float(target), adaptive_length=ac, fixed_length=fc,
                    feasible=ac is not None and fc is not None,
                    frontier_saving_percent=100*(1-ac/fc) if ac and fc else None))
    return profits, frontiers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--data-dir', type=Path, default=Path('dataset/alpaca'))
    parser.add_argument('--generators', nargs='+', default=GENERATORS)
    parser.add_argument('--cap', type=int, default=960)
    parser.add_argument('--permutations', type=int, default=8)
    parser.add_argument('--replay-seed', type=int, default=20260923)
    parser.add_argument('--split-seeds', nargs='+', type=int, default=[71,72,73,74,75])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; refusing overwrite')
    args.output.mkdir(parents=True)
    meta = dict(primary=PRIMARY, configs=CONFIGS, rates=RATES.tolist(),
        training_free=True, prices=PRICES.tolist(), targets=TARGETS.tolist(),
        cap=args.cap, permutations=args.permutations, replay_seed=args.replay_seed,
        split_seeds=args.split_seeds, cost_unit='characters',
        adaptive_selection='None: actual deployment price; no trained multiplier or reference',
        matched_quality='Profit policy attained quality matched by test-oracle fixed-N mixture',
        target_frontier='Both price mixtures and fixed-N mixtures are post-hoc diagnostics, not target controllers',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), datasets={})
    (args.output/'METHOD.json').write_text(json.dumps(meta, indent=2)+'\n')
    all_p, all_f = [], []
    for gen in args.generators:
        print(gen, flush=True)
        path = args.data_dir/f'{gen}_output.merged_rm.jsonl.gz'
        pools = load_alignment(path, 'mistral_rm_reward', 'text_chars')
        cap = min(args.cap, min(len(p.rewards) for p in pools))
        top, ref, costs, quality, _ = prepare(pools, cap, args.permutations, args.replay_seed)
        adaptive = evaluate(top, ref, costs, quality)
        fixed = np.stack([quality.mean(axis=1), costs.mean(axis=1),
            np.broadcast_to(np.arange(1,cap+1), (len(pools),cap))], axis=2)
        np.savez_compressed(args.output/f'{gen}.npz', adaptive=adaptive, fixed=fixed,
                            ids=[p.id for p in pools])
        p, f = summarize(adaptive, fixed, args.split_seeds, gen)
        all_p.extend(p); all_f.extend(f)
        write_csv(args.output/'profit.csv', all_p)
        write_csv(args.output/'frontier_diagnostic.csv', all_f)
        meta['datasets'][gen] = dict(path=str(path), actual_cap=cap,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (args.output/'METHOD.json').write_text(json.dumps(meta, indent=2)+'\n')
        for name in NAMES:
            rows = [r for r in p if r['method']==name]
            print(name, 'profit', round(np.mean([r['relative_percent'] for r in rows]),3),
                  'matched saving', round(np.mean([r['matched_saving_percent'] for r in rows
                      if r['matched_saving_percent'] is not None]),3), flush=True)
        del top, ref, costs, quality


if __name__ == '__main__':
    main()
