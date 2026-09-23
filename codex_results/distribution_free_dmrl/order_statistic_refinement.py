"""Refined order-statistic stopping: profit and target-quality cached replay.

No parametric utility distribution is fitted. Optional reference correction is
an empirical mean quantile-bias table estimated on training prompts only.
Training evaluation cross-fits this table; test decisions never access test
pool quantiles. All online statistics are prefix-measurable.
"""
from __future__ import annotations

import argparse
from bisect import insort
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import load_alignment, q99, sigmoid, validate
from nonparametric_study import GENERATORS, PRICES, write_csv


TARGETS = np.arange(30, 61, 5)/100
WIDTHS = ('fixed2', 'fixed4', 'fixed8', 'sqrt32', 'quarter32')
RATES = np.geomspace(2e-10, 2e-4, 49)


def configurations():
    return [dict(reference=reference, width=width, stopping_price=float(rate))
            for reference in ('empirical', 'corrected') for width in WIDTHS for rate in RATES]


def prepare(pools, cap, permutations, seed):
    """Retain top 33 raw scores at every prefix and evaluation curves separately."""
    shape = (len(pools), permutations, cap)
    top = np.full(shape+(33,), -np.inf, dtype=np.float32)
    reference = np.zeros(shape)
    costs = np.zeros(shape)
    quality = np.zeros(shape)
    true_reference = np.array([q99(p.rewards) for p in pools])
    rng = np.random.default_rng(seed)
    for i, pool in enumerate(pools):
        for rep in range(permutations):
            order = rng.permutation(len(pool.rewards))[:cap]
            rewards = pool.rewards[order]
            ordered = []
            for j, reward in enumerate(rewards):
                insort(ordered, float(reward))
                values = ordered[-33:][::-1]
                top[i, rep, j, :len(values)] = values
                reference[i, rep, j] = ordered[min(int(.99*(j+1)), j)]
            costs[i, rep] = np.cumsum(pool.lengths[order])
            quality[i, rep] = sigmoid(np.maximum.accumulate(rewards)-true_reference[i])
        if (i+1) % 25 == 0:
            print(f'  prepared {i+1}/{len(pools)} prompts', flush=True)
    return top, reference, costs, quality, true_reference


def reference_bias(prefix_reference, true_reference, calibration_ids):
    """No distribution fit: average quantile-estimation error at each n."""
    return (true_reference[calibration_ids, None, None]
            - prefix_reference[calibration_ids]).mean(axis=(0, 1))


def width_sequence(name, cap):
    n = np.arange(1, cap+1)
    if name.startswith('fixed'):
        k = np.full(cap, int(name[5:]))
        minimum = max(4, int(name[5:])+1)
    elif name == 'sqrt32':
        k = np.minimum(32, np.maximum(2, np.ceil(np.sqrt(n)).astype(int)))
        minimum = 4
    elif name == 'quarter32':
        k = np.minimum(32, np.maximum(2, np.ceil(n/4).astype(int)))
        minimum = 4
    else:
        raise ValueError(name)
    return np.minimum(k, np.maximum(0, n-1)), minimum


def evaluate_reference(top, reference, cumulative_lengths, quality, bias=None):
    """Return [prompt, width*rate, quality/length/count], averaged over orders."""
    count, reps, cap = reference.shape
    ref = reference if bias is None else reference+bias[None, None, :]
    utility = sigmoid(top.astype(np.float64)-ref[..., None])
    sums = np.cumsum(utility, axis=-1)
    n = np.arange(1, cap+1)
    output = np.zeros((count, len(WIDTHS)*len(RATES), 3))
    for wi, name in enumerate(WIDTHS):
        k, minimum = width_sequence(name, cap)
        kk = np.broadcast_to(k, reference.shape)[..., None]
        mean = np.take_along_axis(sums, np.maximum(kk-1, 0), axis=-1)[..., 0]/np.maximum(k, 1)
        floor = np.take_along_axis(utility, kk, axis=-1)[..., 0]
        # gain=(mean excess)/n, next cost=stopping_price*cumulative_length/n.
        threshold = np.divide(np.maximum(mean-floor, 0), cumulative_lengths,
                              out=np.full_like(reference, np.inf), where=cumulative_lengths>0)
        threshold[..., :minimum-1] = np.inf
        # Prefix minimum supports exact first-crossing searches, not smoothing.
        crossing = np.minimum.accumulate(threshold, axis=-1).reshape(-1, cap)
        stops = np.array([np.minimum(np.searchsorted(-row, -RATES), cap-1)
                          for row in crossing]).reshape(count, reps, len(RATES))
        q = np.take_along_axis(quality, stops, axis=2).mean(axis=1)
        length = np.take_along_axis(cumulative_lengths, stops, axis=2).mean(axis=1)
        target = slice(wi*len(RATES), (wi+1)*len(RATES))
        output[:, target] = np.stack([q, length, (stops+1).mean(axis=1)], axis=2)
    return output


def lower_hull(quality, cost):
    """Indices of the lower convex envelope; duplicate qualities keep least cost."""
    quality, cost = np.asarray(quality), np.asarray(cost)
    ordered = sorted(range(len(quality)), key=lambda i: (quality[i], cost[i], i))
    unique = []
    for i in ordered:
        if not unique or quality[i] != quality[unique[-1]]:
            unique.append(i)
    hull = []
    for i in unique:
        while len(hull) >= 2:
            a, b = hull[-2:]
            cross = ((quality[b]-quality[a])*(cost[i]-cost[b])
                     - (cost[b]-cost[a])*(quality[i]-quality[b]))
            if cross > 0:
                break
            hull.pop()
        hull.append(i)
    return hull


def target_mix(quality, cost, target, exact=False):
    """Cheapest two-point mixture for quality >= target (or exactly target).

    Returns indices, high-index weight, and feasibility. Unreachable requested
    targets return maximum quality with feasible=False; exact comparisons return
    None rather than extrapolate beyond the available quality range.
    """
    q, c = np.asarray(quality), np.asarray(cost)
    hull = lower_hull(q, c)
    if target > q[hull[-1]]+1e-12 or (exact and target < q[hull[0]]-1e-12):
        return None if exact else (hull[-1], hull[-1], 0., False)
    candidates = []
    for i in hull:
        if (abs(q[i]-target)<=1e-12) or (not exact and q[i]>=target):
            candidates.append((c[i], i, i, 0.))
    for a, b in zip(hull, hull[1:]):
        if q[a] <= target <= q[b]:
            weight = (target-q[a])/(q[b]-q[a])
            candidates.append(((1-weight)*c[a]+weight*c[b], a, b, weight))
    if not candidates:
        return None if exact else (hull[0], hull[0], 0., True)
    _, a, b, weight = min(candidates)
    return int(a), int(b), float(weight), True


def mix_values(values, mixture):
    a, b, weight, _ = mixture
    return (1-weight)*values[:, a]+weight*values[:, b]


def summarize(adaptive, fixed, train, test, generator, seed, configs):
    profit_rows, target_rows, prompt_rows = [], [], []
    groups = {'empirical_spacing': np.array([i for i,p in enumerate(configs) if p['reference']=='empirical']),
              'corrected_spacing': np.array([i for i,p in enumerate(configs) if p['reference']=='corrected']),
              'train_selected_spacing': np.arange(len(configs))}
    amean, fmean = adaptive[train].mean(axis=0), fixed[train].mean(axis=0)
    ftest = fixed[test].mean(axis=0)
    rng = np.random.default_rng(seed+20260922)
    for method, ids in groups.items():
        for price in PRICES:
            ai = ids[np.argmax(amean[ids, 0]-price*amean[ids, 1])]
            fi = np.argmax(fmean[:, 0]-price*fmean[:, 1])
            a, f = adaptive[test, ai], fixed[test, fi]
            pa, pf = a[:, 0]-price*a[:, 1], f[:, 0]-price*f[:, 1]
            delta = pa-pf
            boot = rng.choice(delta, (2000, len(test)), replace=True).mean(axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            profit_rows.append(dict(generator=generator, split_seed=seed, method=method,
                price=price, selected=int(ai), fixed_n=int(fi+1), profit=pa.mean(),
                fixed_profit=pf.mean(), relative_percent=100*delta.mean()/pf.mean(),
                delta_profit=delta.mean(), ci_low=lo, ci_high=hi, quality=a[:, 0].mean(),
                fixed_quality=f[:, 0].mean(), cost=price*a[:, 1].mean(),
                fixed_cost=price*f[:, 1].mean(), samples=a[:, 2].mean(),
                delta_test_oracle=pa.mean()-np.max(ftest[:, 0]-price*ftest[:, 1])))
        for target in TARGETS:
            local_mix = target_mix(amean[ids, 0], amean[ids, 1], target)
            ai, bi, w, feasible = local_mix
            amix = (int(ids[ai]), int(ids[bi]), w, feasible)
            fmix = target_mix(fmean[:, 0], fmean[:, 1], target)
            a, f = mix_values(adaptive[test], amix), mix_values(fixed[test], fmix)
            achieved, length = a[:, 0].mean(), a[:, 1].mean()
            matched = target_mix(ftest[:, 0], ftest[:, 1], achieved, exact=True)
            matched_cost = float(mix_values(fixed[test], matched)[:, 1].mean()) if matched else None
            saving = 100*(1-length/matched_cost) if matched_cost else None
            target_rows.append(dict(generator=generator, split_seed=seed, method=method,
                target=float(target), attained_quality=achieved, fixed_attained_quality=f[:, 0].mean(),
                mean_length=length, fixed_mean_length=f[:, 1].mean(), samples=a[:, 2].mean(),
                train_feasible=feasible, fixed_train_feasible=fmix[3],
                matched_feasible=matched is not None, matched_length=matched_cost,
                matched_saving_percent=saving, unmatched_train_baseline_saving=100*(1-length/f[:, 1].mean()),
                adaptive_low=amix[0], adaptive_high=amix[1], adaptive_high_weight=w,
                fixed_low=int(fmix[0]+1), fixed_high=int(fmix[1]+1), fixed_high_weight=fmix[2]))
            for j, prompt_index in enumerate(test):
                prompt_rows.append(dict(generator=generator, split_seed=seed, method=method,
                    target=float(target), prompt_index=int(prompt_index), quality=a[j, 0],
                    length=a[j, 1], samples=a[j, 2]))
    return profit_rows, target_rows, prompt_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('dataset/alpaca'))
    parser.add_argument('--generators', nargs='+', default=GENERATORS)
    parser.add_argument('--cap', type=int, default=960)
    parser.add_argument('--permutations', type=int, default=8)
    parser.add_argument('--split-seeds', type=int, nargs='+', default=[71,72,73,74,75])
    parser.add_argument('--replay-seed', type=int, default=20260923)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a fresh directory')
    if args.cap<33 or args.permutations<1:
        parser.error('Cap must be at least 33; permutations must be positive')
    args.output.mkdir(parents=True)
    configs = configurations()
    meta = dict(configurations=configs, prices=PRICES.tolist(), targets=TARGETS.tolist(),
        split_seeds=args.split_seeds, replay_seed=args.replay_seed, permutations=args.permutations,
        cap=args.cap, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cost_unit='recorded text characters', theorem_applies=False,
        selection='training only; reference correction cross-fitted on training halves',
        target='training-selected lowest-cost mixture; test fixed-N lower hull matches attained quality retrospectively',
        scope='exploratory reused prompts; new splits and orders are not independent confirmation', datasets={})
    (args.output/'METHOD.json').write_text(json.dumps(meta, indent=2)+'\n')
    all_profit, all_target, all_prompts = [], [], []
    for generator in args.generators:
        print(generator, flush=True)
        path = args.data_dir/f'{generator}_output.merged_rm.jsonl.gz'
        pools = load_alignment(path, 'mistral_rm_reward', 'text_chars')
        validate(pools)
        cap = min(args.cap, min(len(p.rewards) for p in pools))
        top, reference, lengths, quality, truth = prepare(pools, cap, args.permutations, args.replay_seed)
        fixed = np.stack([quality.mean(axis=1), lengths.mean(axis=1),
                          np.broadcast_to(np.arange(1,cap+1), (len(pools), cap))], axis=2)
        raw = evaluate_reference(top, reference, lengths, quality)
        np.savez_compressed(args.output/f'{generator}_fixed.npz', fixed=fixed, ids=[p.id for p in pools])
        for seed in args.split_seeds:
            order = np.random.default_rng(seed).permutation(len(pools))
            train, test = order[:len(pools)//2], order[len(pools)//2:]
            halves = np.array_split(train, 2)
            corrected = np.empty_like(raw)
            bias_test = reference_bias(reference, truth, train)
            for evaluate_ids, calibration_ids in [(halves[0], halves[1]), (halves[1], halves[0]), (test, train)]:
                bias = reference_bias(reference, truth, calibration_ids)
                corrected[evaluate_ids] = evaluate_reference(top[evaluate_ids], reference[evaluate_ids],
                    lengths[evaluate_ids], quality[evaluate_ids], bias)
            adaptive = np.concatenate([raw, corrected], axis=1)
            np.savez_compressed(args.output/f'{generator}_seed{seed}.npz', adaptive=adaptive,
                                train=train, test=test, test_reference_bias=bias_test)
            p, t, pr = summarize(adaptive, fixed, train, test, generator, seed, configs)
            all_profit.extend(p); all_target.extend(t); all_prompts.extend(pr)
            print('  split', seed, 'profit %', round(np.mean([r['relative_percent'] for r in p
                  if r['method']=='train_selected_spacing']),3), 'matched saving %',
                  round(np.mean([r['matched_saving_percent'] for r in t
                  if r['method']=='train_selected_spacing' and r['matched_feasible']]),3), flush=True)
        meta['datasets'][generator] = dict(path=str(path), actual_cap=cap,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), ids=[p.id for p in pools])
        write_csv(args.output/'profit.csv', all_profit)
        write_csv(args.output/'target_quality.csv', all_target)
        write_csv(args.output/'target_prompt_metrics.csv', all_prompts)
        (args.output/'METHOD.json').write_text(json.dumps(meta, indent=2)+'\n')


if __name__ == '__main__':
    main()
