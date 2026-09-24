"""Fixed score-to-utility maps fitted to an exact mixture of selected winners.

The mixture is over predefined counts, with equal weight for each problem and
count. Raw reward determines selection; equal raw-score ties are averaged over
random order. Fitting uses a weighted isotonic regression, not a count-dependent
online map. No final-test outcomes may enter map selection.
"""
import numpy as np
from scipy.optimize import isotonic_regression

import coding_high_price_diagnosis as diagnosis

old = diagnosis.old
COUNTS = (10, 16, 32, 64, 128, 256, 512)
MIXTURES = (0.0, 0.25, 0.5, 0.75, 1.0)


def winner_weights(problem, counts=COUNTS):
    """Row n gives the exact probability each candidate wins best-of-count[n]."""
    size = len(problem.rewards)
    if any(n < 1 or n > size for n in counts):
        raise ValueError("counts must be between one and the cache size")
    order = np.argsort(problem.rewards, kind="stable")
    rewards = np.asarray(problem.rewards)[order]
    starts = np.r_[0, 1 + np.flatnonzero(rewards[1:] != rewards[:-1])]
    ends = np.r_[starts[1:], size]
    cdf = diagnosis.max_cdf(size)[np.asarray(counts)-1]
    group = (cdf[:, ends] - cdf[:, starts]) / (ends-starts)
    sorted_weights = np.repeat(group, ends-starts, axis=1)
    result = np.empty_like(sorted_weights)
    result[:, order] = sorted_weights
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-12)
    assert np.all(result >= 0)
    return result


def fit_profile(problems, weights, mixture, metadata=None):
    """Blend uniform-candidate and selected-winner weighting; no test selection."""
    if not 0 <= mixture <= 1:
        raise ValueError("mixture must be in [0,1]")
    ids = sorted(problems)
    reward = np.concatenate([np.asarray(problems[k].rewards, dtype=float) for k in ids])
    correct = np.concatenate([np.asarray(problems[k].correct, dtype=float) for k in ids])
    weight = np.concatenate([(1-mixture)/len(problems[k].rewards) + mixture*weights[k].mean(axis=0)
                             for k in ids])
    order = np.argsort(reward, kind="stable")
    reward, correct, weight = reward[order], correct[order], weight[order]
    unique, starts = np.unique(reward, return_index=True)
    total = np.add.reduceat(weight, starts)
    positive = total > 0
    label = np.add.reduceat(weight*correct, starts)[positive]/total[positive]
    fit = isotonic_regression(label, weights=total[positive], increasing=True).x
    return old.CodingProfile(tuple(unique[positive]), tuple(np.clip(fit,0,1)),
        mean_length=float(np.mean([length for k in ids for length in problems[k].lengths])),
        metadata={"fit_problem_ids":ids, "winner_weight_mixture":float(mixture),
                  "calibration_method":"weighted isotonic; fixed map", **(metadata or {})})


def evaluate(profile, problems, weights, counts=COUNTS):
    """Proper scoring and mean calibration at each count; exact order average."""
    rows = []
    for key in sorted(problems):
        p = np.interp(problems[key].rewards, profile.reward_knots, profile.probability_knots)
        y = np.asarray(problems[key].correct, dtype=float)
        for j,n in enumerate(counts):
            w = weights[key][j]
            rows.append(dict(problem_id=key, n=int(n), predicted=float(w@p),
                             correct=float(w@y), brier=float(w@((p-y)**2))))
    return rows


def select_profile(problems, weights, seed, metadata=None):
    """Four-fold problem-grouped CV; select mixture by held-out winner Brier loss."""
    ids = np.asarray(sorted(problems))
    folds = np.array_split(np.random.default_rng(seed).permutation(ids),4)
    records = []
    scores = {}
    for mixture in MIXTURES:
        values = []
        for fold, validation in enumerate(folds):
            heldout = set(validation)
            train = {k:problems[k] for k in ids if k not in heldout}
            check = {k:problems[k] for k in validation}
            assert not (set(train)&set(check))
            profile = fit_profile(train, weights, mixture)
            for row in evaluate(profile,check,weights):
                values.append(row["brier"])
                records.append(dict(fold=fold,mixture=mixture,**row))
        scores[mixture] = float(np.mean(values))
    best = min(MIXTURES,key=lambda value:(scores[value],value))
    profile = fit_profile(problems,weights,best,metadata={
        "selection":"four-fold grouped CV; winner-mixture Brier loss", "selection_seed":seed,
        "winner_counts":list(COUNTS), "cv_scores":scores, **(metadata or {})})
    return profile, records
