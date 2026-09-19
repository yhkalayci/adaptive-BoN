"""Leakage-free UCB Pandora experiments for alignment with character costs.

This runner deliberately implements only Pandora reservation stopping.  At an
observed prefix, a fitted upper-confidence reward law ``F_ucb`` induces the
one-box expected improvement

    EI_ucb(v) = E_F_ucb[(u(X) - u(v))_+].

The Pandora fair-cap rule stops iff ``EI_ucb(v) <= E[next_chars]/divisor``.
This is exactly equivalent to computing the reservation value tau satisfying
``E[(u(X)-tau)_+] = cost`` and stopping when ``u(v) >= tau``.

Only the observed reward/character prefix and a training-fitted character-cost
prior are available to the policy.  The prompt's full reward quantile is used
after stopping solely to score quality.  Evaluation cost is the exact sum of
characters of all opened generations, divided by the requested divisor.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit
from scipy.stats import norm, t as student_t


DEFAULT_DIVISORS = (1e5, 5e5, 1e6, 5e6, 5e7)
MODEL_LABELS = {
    "gaussian_raw": "Gaussian UCB (raw reward)",
    "exp_tail_raw": "Shifted-exp UCB (raw reward)",
    "halfnormal_tail_raw": "Shifted half-normal UCB (raw reward)",
    "rayleigh_tail_raw": "Shifted Rayleigh UCB (raw reward)",
    "exp_tail_exponentiated": "Shifted-exp UCB (exp reward)",
}


@dataclass(frozen=True)
class PandoraConfig:
    model: str
    confidence_scale: float
    reward_prior_strength: float
    cost_prior_strength: float
    threshold_quantile: float = 0.5
    delta: float = 0.05
    benchmark_calibration: float = 1.0
    ei_bonus_scale: float = 0.0


@dataclass
class PriorFit:
    raw_sigma: float
    gaussian_quantile_z: float
    raw_tail_scales: dict[float, float]
    exp_tail_ratios: dict[float, float]
    raw_tail_quantile_k: dict[float, float]
    exp_tail_quantile_k: dict[float, float]
    cost_beta: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray


def _stable_alpha_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    index = min(max(int(alpha * len(values)), 0), len(values) - 1)
    return float(np.sort(values)[index])


def load_data(path: Path, reward_key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rewards, chars, prompts = [], [], []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            generations = record["generations"]
            rewards.append([float(g[reward_key]) for g in generations])
            chars.append([len(g["text"]) for g in generations])
            prompts.append(str(record.get("prompt", "")))
    reward_matrix = np.asarray(rewards, dtype=np.float64)
    char_matrix = np.asarray(chars, dtype=np.float64)
    if reward_matrix.shape != char_matrix.shape:
        raise ValueError("reward and character matrices must have identical shape")
    return reward_matrix, char_matrix, prompts


def prompt_features(prompts: Iterable[str]) -> np.ndarray:
    rows = []
    for prompt in prompts:
        n_chars = len(prompt)
        n_words = len(prompt.split())
        n_lines = prompt.count("\n") + 1
        n_code = sum(prompt.count(token) for token in ("{", "}", "(", ")", "```", ";"))
        n_digits = sum(ch.isdigit() for ch in prompt)
        rows.append([
            math.log1p(n_chars), math.log1p(n_words), math.log1p(n_lines),
            math.log1p(n_code), math.log1p(n_digits),
        ])
    return np.asarray(rows, dtype=np.float64)


def fit_priors(
    rewards: np.ndarray,
    chars: np.ndarray,
    features: np.ndarray,
    indices: np.ndarray,
    threshold_quantiles: tuple[float, ...],
) -> PriorFit:
    train_rewards = rewards[indices]
    prompt_sigmas = np.std(train_rewards, axis=1, ddof=1)
    raw_sigma = float(np.median(prompt_sigmas))
    prompt_q99 = np.asarray([_stable_alpha_quantile(row, 0.99) for row in train_rewards])
    gaussian_quantile_z = float(np.median((prompt_q99 - train_rewards.mean(axis=1)) /
                                         np.maximum(prompt_sigmas, 1e-8)))
    raw_tail_scales: dict[float, float] = {}
    exp_tail_ratios: dict[float, float] = {}
    raw_tail_quantile_k: dict[float, float] = {}
    exp_tail_quantile_k: dict[float, float] = {}
    for q in threshold_quantiles:
        raw_scales, exp_ratios, raw_ks, exp_ks = [], [], [], []
        for row, q99 in zip(train_rewards, prompt_q99):
            raw_loc = float(np.quantile(row, q))
            raw_scale = max(float(np.mean(row[row >= raw_loc]) - raw_loc), 1e-8)
            raw_scales.append(raw_scale)
            raw_ks.append(max((q99 - raw_loc) / raw_scale, 1e-8))
            exp_row = np.exp(np.clip(row, -50.0, 50.0))
            exp_loc = float(np.quantile(exp_row, q))
            exp_scale = max(float(np.mean(exp_row[exp_row >= exp_loc]) - exp_loc), 1e-8)
            exp_ratios.append(exp_scale / max(exp_loc, 1e-8))
            exp_ks.append(max((math.exp(float(np.clip(q99, -50.0, 50.0))) - exp_loc) / exp_scale, 1e-8))
        raw_tail_scales[q] = float(np.median(raw_scales))
        exp_tail_ratios[q] = float(np.median(exp_ratios))
        raw_tail_quantile_k[q] = float(np.median(raw_ks))
        exp_tail_quantile_k[q] = float(np.median(exp_ks))

    x = features[indices]
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    xz = (x - x_mean) / x_scale
    design = np.column_stack([np.ones(len(xz)), xz])
    target = np.log(np.maximum(chars[indices].mean(axis=1), 1.0))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + 3.0 * penalty, design.T @ target)
    return PriorFit(raw_sigma, gaussian_quantile_z, raw_tail_scales, exp_tail_ratios,
                    raw_tail_quantile_k, exp_tail_quantile_k, beta, x_mean, x_scale)


def predict_mean_chars(prior: PriorFit, features: np.ndarray) -> np.ndarray:
    xz = (features - prior.feature_mean) / prior.feature_scale
    design = np.column_stack([np.ones(len(xz)), xz])
    return np.exp(np.clip(design @ prior.cost_beta, 0.0, 20.0))


_GH_X, _GH_W = hermgauss(24)
_GH_W = _GH_W / np.sqrt(np.pi)
_EXP_PROBS = (np.arange(48, dtype=np.float64) + 0.5) / 48.0
_EXP_UNIT = -np.log1p(-_EXP_PROBS)
_HALFNORMAL_MEAN = math.sqrt(2.0 / math.pi)
_HALFNORMAL_UNIT = norm.ppf((1.0 + _EXP_PROBS) / 2.0)
_RAYLEIGH_MEAN = math.sqrt(math.pi) / 2.0
_RAYLEIGH_UNIT = np.sqrt(_EXP_UNIT)


def _gaussian_ucb_ei_stats(n: int, reward_sum: float, reward_sumsq: float,
                           best: float, prior: PriorFit, cfg: PandoraConfig) -> float:
    """Gaussian UCB expected improvement from O(1) prefix moments."""
    mu = reward_sum / n
    centered_ss = max(reward_sumsq - reward_sum * reward_sum / n, 0.0)
    sample_var = centered_ss / (n - 1) if n > 1 else prior.raw_sigma**2
    weight = cfg.reward_prior_strength
    sigma = math.sqrt(max(((n - 1) * sample_var + weight * prior.raw_sigma**2) /
                          max(n - 1 + weight, 1e-8), 1e-12))
    radius = math.sqrt(math.log(1.0 / cfg.delta) / max(n, 1))
    mu_ucb = mu + cfg.confidence_scale * sigma * radius
    sigma_ucb = sigma * (1.0 + cfg.confidence_scale * radius)
    # The scoring benchmark is unknown online.  A lower confidence estimate
    # mirrors the original UCB-Pandora implementation and avoids treating an
    # uncertain estimated 99th percentile as known ground truth.
    mu_lcb = mu - cfg.confidence_scale * sigma * radius
    sigma_lcb = max(sigma * (1.0 - cfg.confidence_scale * radius), 0.15 * sigma)
    z_value = ((1.0 - cfg.benchmark_calibration) * norm.ppf(0.99) +
               cfg.benchmark_calibration * prior.gaussian_quantile_z)
    benchmark = mu_lcb + z_value * sigma_lcb
    current_u = float(expit(best - benchmark))
    future = mu_ucb + math.sqrt(2.0) * sigma_ucb * _GH_X
    point_ei = float(np.sum(_GH_W * np.maximum(expit(future - benchmark) - current_u, 0.0)))
    bonus = cfg.ei_bonus_scale * math.sqrt(math.log(1.0 / cfg.delta) / (2.0 * n))
    return min(point_ei + bonus, 1.0)


def _gaussian_stop_vectorized(perm_rewards: np.ndarray, perm_chars: np.ndarray,
                              divisor: float, prior_mean_chars: float,
                              prior: PriorFit, cfg: PandoraConfig,
                              min_open_count: int) -> tuple[int, float, float]:
    """Vectorized evaluation of all observable-prefix Pandora decisions."""
    n_total = len(perm_rewards)
    prefix_sum = np.cumsum(perm_rewards)
    prefix_sumsq = np.cumsum(perm_rewards * perm_rewards)
    prefix_best = np.maximum.accumulate(perm_rewards)
    prefix_chars = np.cumsum(perm_chars)
    ns = np.arange(1, n_total + 1, dtype=np.float64)
    mu = prefix_sum / ns
    centered_ss = np.maximum(prefix_sumsq - prefix_sum * prefix_sum / ns, 0.0)
    sample_var = np.full(n_total, prior.raw_sigma**2)
    sample_var[1:] = centered_ss[1:] / (ns[1:] - 1.0)
    weight = cfg.reward_prior_strength
    denom = np.maximum(ns - 1.0 + weight, 1e-8)
    sigma = np.sqrt(np.maximum(((ns - 1.0) * sample_var + weight * prior.raw_sigma**2) / denom, 1e-12))
    radius = np.sqrt(np.log(1.0 / cfg.delta) / ns)
    mu_ucb = mu + cfg.confidence_scale * sigma * radius
    sigma_ucb = sigma * (1.0 + cfg.confidence_scale * radius)
    mu_lcb = mu - cfg.confidence_scale * sigma * radius
    sigma_lcb = np.maximum(sigma * (1.0 - cfg.confidence_scale * radius), 0.15 * sigma)
    z_value = ((1.0 - cfg.benchmark_calibration) * norm.ppf(0.99) +
               cfg.benchmark_calibration * prior.gaussian_quantile_z)
    benchmark = mu_lcb + z_value * sigma_lcb
    current_u = expit(prefix_best - benchmark)
    future = mu_ucb[:, None] + math.sqrt(2.0) * sigma_ucb[:, None] * _GH_X[None, :]
    ei_ucb = np.sum(_GH_W[None, :] * np.maximum(expit(future - benchmark[:, None]) -
                                                current_u[:, None], 0.0), axis=1)
    ei_ucb = np.minimum(
        ei_ucb + cfg.ei_bonus_scale * np.sqrt(np.log(1.0 / cfg.delta) / (2.0 * ns)),
        1.0,
    )
    expected_next_chars = (cfg.cost_prior_strength * prior_mean_chars + prefix_chars) / (cfg.cost_prior_strength + ns)
    eligible = np.arange(n_total) >= min(min_open_count, n_total) - 1
    eligible[-1] = True
    stops = eligible & (ei_ucb <= expected_next_chars / divisor)
    stop_indices = np.flatnonzero(stops)
    index = int(stop_indices[0]) if len(stop_indices) else n_total - 1
    return index + 1, float(prefix_best[index]), float(prefix_chars[index])


def _exp_tail_ucb_ei_stats(loc: float, point_scale: float, n_tail: int,
                           best_raw: float, prior: PriorFit,
                           cfg: PandoraConfig) -> float:
    q = cfg.threshold_quantile
    tail_mass = 1.0 - q
    if cfg.model == "exp_tail_exponentiated":
        prior_scale = loc * prior.exp_tail_ratios[q]
    else:
        prior_scale = prior.raw_tail_scales[q]
    weight = cfg.reward_prior_strength
    scale = (n_tail * point_scale + weight * prior_scale) / max(n_tail + weight, 1e-8)
    radius = math.sqrt(math.log(1.0 / cfg.delta) / max(n_tail, 1))
    scale_ucb = scale * (1.0 + cfg.confidence_scale * radius)
    scale_lcb = max(scale * (1.0 - cfg.confidence_scale * radius), 1e-8)
    if cfg.model == "exp_tail_exponentiated":
        calibrated_k = prior.exp_tail_quantile_k[q]
    else:
        calibrated_k = prior.raw_tail_quantile_k[q]
    theoretical_k = -math.log((1.0 - 0.99) / tail_mass)
    benchmark_k = ((1.0 - cfg.benchmark_calibration) * theoretical_k +
                   cfg.benchmark_calibration * calibrated_k)
    benchmark_value = loc + scale_lcb * benchmark_k

    if cfg.model == "exp_tail_exponentiated":
        benchmark_raw = math.log(max(benchmark_value, 1e-12))
        future_tail_raw = np.log(np.maximum(loc + scale_ucb * _EXP_UNIT, 1e-12))
    else:
        benchmark_raw = benchmark_value
        future_tail_raw = loc + scale_ucb * _EXP_UNIT
    current_u = float(expit(best_raw - benchmark_raw))
    tail_ei = float(np.mean(np.maximum(expit(future_tail_raw - benchmark_raw) - current_u, 0.0)))
    # Every fitted body value is <= the observed maximum, so its positive
    # improvement over the incumbent is identically zero.  Keeping its mass
    # explicit is important; evaluating the zero-valued terms is unnecessary.
    return tail_mass * tail_ei


def _halfnormal_tail_ucb_ei_stats(
    loc: float,
    point_scale: float,
    n_tail: int,
    best_raw: float,
    prior: PriorFit,
    cfg: PandoraConfig,
) -> float:
    """Optimistic EI for a shifted half-normal tail in raw-score space.

    Conditional on being above the empirical threshold ``loc``, the fitted
    score is ``loc + scale * abs(N(0, 1))``.  The mean-excess estimate is
    converted to the half-normal scale before this helper is called.
    """
    q = cfg.threshold_quantile
    tail_mass = 1.0 - q
    prior_scale = prior.raw_tail_scales[q] / _HALFNORMAL_MEAN
    weight = cfg.reward_prior_strength
    scale = (
        n_tail * point_scale + weight * prior_scale
    ) / max(n_tail + weight, 1e-8)
    radius = math.sqrt(math.log(1.0 / cfg.delta) / max(n_tail, 1))
    scale_ucb = scale * (1.0 + cfg.confidence_scale * radius)
    scale_lcb = max(scale * (1.0 - cfg.confidence_scale * radius), 1e-8)

    conditional_q99 = 1.0 - 0.01 / tail_mass
    theoretical_k = float(norm.ppf((1.0 + conditional_q99) / 2.0))
    calibrated_k = prior.raw_tail_quantile_k[q] * _HALFNORMAL_MEAN
    benchmark_k = (
        (1.0 - cfg.benchmark_calibration) * theoretical_k
        + cfg.benchmark_calibration * calibrated_k
    )
    benchmark_raw = loc + scale_lcb * benchmark_k
    future_tail_raw = loc + scale_ucb * _HALFNORMAL_UNIT
    current_u = float(expit(best_raw - benchmark_raw))
    tail_ei = float(np.mean(
        np.maximum(expit(future_tail_raw - benchmark_raw) - current_u, 0.0)
    ))
    return tail_mass * tail_ei


def _rayleigh_tail_ucb_ei_stats(
    loc: float,
    point_scale: float,
    n_tail: int,
    best_raw: float,
    prior: PriorFit,
    cfg: PandoraConfig,
) -> float:
    """Optimistic EI for a shifted unit-Weibull(shape=2) raw tail."""
    q = cfg.threshold_quantile
    tail_mass = 1.0 - q
    prior_scale = prior.raw_tail_scales[q] / _RAYLEIGH_MEAN
    weight = cfg.reward_prior_strength
    scale = (
        n_tail * point_scale + weight * prior_scale
    ) / max(n_tail + weight, 1e-8)
    radius = math.sqrt(math.log(1.0 / cfg.delta) / max(n_tail, 1))
    scale_ucb = scale * (1.0 + cfg.confidence_scale * radius)
    scale_lcb = max(scale * (1.0 - cfg.confidence_scale * radius), 1e-8)

    theoretical_k = math.sqrt(-math.log(0.01 / tail_mass))
    calibrated_k = prior.raw_tail_quantile_k[q] * _RAYLEIGH_MEAN
    benchmark_k = (
        (1.0 - cfg.benchmark_calibration) * theoretical_k
        + cfg.benchmark_calibration * calibrated_k
    )
    benchmark_raw = loc + scale_lcb * benchmark_k
    future_tail_raw = loc + scale_ucb * _RAYLEIGH_UNIT
    current_u = float(expit(best_raw - benchmark_raw))
    tail_ei = float(np.mean(
        np.maximum(expit(future_tail_raw - benchmark_raw) - current_u, 0.0)
    ))
    return tail_mass * tail_ei


def pandora_stop(
    perm_rewards: np.ndarray,
    perm_chars: np.ndarray,
    divisor: float,
    prior_mean_chars: float,
    prior: PriorFit,
    cfg: PandoraConfig,
    min_open_count: int = 3,
) -> tuple[int, float, float]:
    """Run a UCB Pandora reservation policy; returns opens, best, exact chars."""
    if cfg.model == "gaussian_raw":
        return _gaussian_stop_vectorized(
            perm_rewards, perm_chars, divisor, prior_mean_chars, prior, cfg, min_open_count,
        )
    n_total = len(perm_rewards)
    n = min(min_open_count, n_total)
    best = float(np.max(perm_rewards[:n]))
    observed_char_sum = float(np.sum(perm_chars[:n]))
    reward_sum = float(np.sum(perm_rewards[:n]))
    reward_sumsq = float(np.dot(perm_rewards[:n], perm_rewards[:n]))

    # Streaming order-statistic fit for the peaks-over-threshold law.  `upper`
    # contains the fitted tail and `lower` the body; both membership and the
    # tail sum update in O(log n), avoiding repeated full-prefix sorts.
    lower: list[float] = []  # negated max heap
    upper: list[float] = []  # min heap, includes the threshold
    upper_sum = 0.0

    def add_model_value(value: float, count_after: int) -> None:
        nonlocal upper_sum
        if upper and value >= upper[0]:
            heapq.heappush(upper, value)
            upper_sum += value
        else:
            heapq.heappush(lower, -value)
        target_lower = max(int(math.ceil(cfg.threshold_quantile * count_after)) - 1, 0)
        while len(lower) > target_lower:
            moved = -heapq.heappop(lower)
            heapq.heappush(upper, moved)
            upper_sum += moved
        while len(lower) < target_lower and upper:
            moved = heapq.heappop(upper)
            upper_sum -= moved
            heapq.heappush(lower, -moved)

    initial_model_values = (np.exp(np.clip(perm_rewards[:n], -50.0, 50.0))
                            if cfg.model == "exp_tail_exponentiated" else perm_rewards[:n])
    for count_after, value in enumerate(initial_model_values, start=1):
        add_model_value(float(value), count_after)
    while n < n_total:
        if cfg.model in (
            "exp_tail_raw",
            "halfnormal_tail_raw",
            "rayleigh_tail_raw",
            "exp_tail_exponentiated",
        ):
            loc = float(upper[0])
            mean_excess = max(upper_sum / len(upper) - loc, 1e-8)
            if cfg.model == "halfnormal_tail_raw":
                point_scale = mean_excess / _HALFNORMAL_MEAN
                ei_ucb = _halfnormal_tail_ucb_ei_stats(
                    loc, point_scale, len(upper), best, prior, cfg
                )
            elif cfg.model == "rayleigh_tail_raw":
                point_scale = mean_excess / _RAYLEIGH_MEAN
                ei_ucb = _rayleigh_tail_ucb_ei_stats(
                    loc, point_scale, len(upper), best, prior, cfg
                )
            else:
                ei_ucb = _exp_tail_ucb_ei_stats(
                    loc, mean_excess, len(upper), best, prior, cfg
                )
            ei_ucb = min(
                ei_ucb + cfg.ei_bonus_scale * math.sqrt(math.log(1.0 / cfg.delta) / (2.0 * n)),
                1.0,
            )
        else:
            raise ValueError(f"unknown Pandora reward model: {cfg.model}")
        expected_next_chars = (
            cfg.cost_prior_strength * prior_mean_chars + observed_char_sum
        ) / (cfg.cost_prior_strength + n)
        # Pandora fair-cap decision: EI at the current value is at most cost.
        if ei_ucb <= expected_next_chars / divisor:
            break
        next_reward = float(perm_rewards[n])
        best = max(best, next_reward)
        reward_sum += next_reward
        reward_sumsq += next_reward * next_reward
        observed_char_sum += float(perm_chars[n])
        n += 1
        model_value = (math.exp(float(np.clip(next_reward, -50.0, 50.0)))
                       if cfg.model == "exp_tail_exponentiated" else next_reward)
        add_model_value(model_value, n)
    return n, best, observed_char_sum


def pandora_stop_many(
    perm_rewards: np.ndarray,
    perm_chars: np.ndarray,
    divisors: tuple[float, ...],
    prior_mean_chars: float,
    prior: PriorFit,
    cfg: PandoraConfig,
    min_open_count: int = 3,
) -> dict[float, tuple[int, float, float]]:
    """Evaluate several Pandora costs from one shared observable-prefix fit.

    The reward UCB and expected next-character estimate do not depend on the
    divisor.  Sharing them is therefore exactly equivalent to independent
    calls to :func:`pandora_stop`, while avoiding repeated prefix fitting.
    """
    divisors = tuple(float(d) for d in divisors)
    if cfg.model == "gaussian_raw":
        return {
            d: _gaussian_stop_vectorized(
                perm_rewards, perm_chars, d, prior_mean_chars, prior, cfg, min_open_count,
            )
            for d in divisors
        }
    if cfg.model not in (
        "exp_tail_raw",
        "halfnormal_tail_raw",
        "rayleigh_tail_raw",
        "exp_tail_exponentiated",
    ):
        raise ValueError(f"unknown Pandora reward model: {cfg.model}")

    n_total = len(perm_rewards)
    n = min(min_open_count, n_total)
    best = float(np.max(perm_rewards[:n]))
    observed_char_sum = float(np.sum(perm_chars[:n]))
    lower: list[float] = []
    upper: list[float] = []
    upper_sum = 0.0

    def add_model_value(value: float, count_after: int) -> None:
        nonlocal upper_sum
        if upper and value >= upper[0]:
            heapq.heappush(upper, value)
            upper_sum += value
        else:
            heapq.heappush(lower, -value)
        target_lower = max(int(math.ceil(cfg.threshold_quantile * count_after)) - 1, 0)
        while len(lower) > target_lower:
            moved = -heapq.heappop(lower)
            heapq.heappush(upper, moved)
            upper_sum += moved
        while len(lower) < target_lower and upper:
            moved = heapq.heappop(upper)
            upper_sum -= moved
            heapq.heappush(lower, -moved)

    initial = (np.exp(np.clip(perm_rewards[:n], -50.0, 50.0))
               if cfg.model == "exp_tail_exponentiated" else perm_rewards[:n])
    for count_after, value in enumerate(initial, start=1):
        add_model_value(float(value), count_after)

    active = set(divisors)
    results: dict[float, tuple[int, float, float]] = {}
    while n < n_total and active:
        loc = float(upper[0])
        mean_excess = max(upper_sum / len(upper) - loc, 1e-8)
        if cfg.model == "halfnormal_tail_raw":
            point_scale = mean_excess / _HALFNORMAL_MEAN
            ei_ucb = _halfnormal_tail_ucb_ei_stats(
                loc, point_scale, len(upper), best, prior, cfg
            )
        elif cfg.model == "rayleigh_tail_raw":
            point_scale = mean_excess / _RAYLEIGH_MEAN
            ei_ucb = _rayleigh_tail_ucb_ei_stats(
                loc, point_scale, len(upper), best, prior, cfg
            )
        else:
            ei_ucb = _exp_tail_ucb_ei_stats(
                loc, mean_excess, len(upper), best, prior, cfg
            )
        ei_ucb = min(
            ei_ucb + cfg.ei_bonus_scale * math.sqrt(math.log(1.0 / cfg.delta) / (2.0 * n)),
            1.0,
        )
        expected_next_chars = (
            cfg.cost_prior_strength * prior_mean_chars + observed_char_sum
        ) / (cfg.cost_prior_strength + n)
        for divisor in tuple(active):
            if ei_ucb <= expected_next_chars / divisor:
                results[divisor] = (n, best, observed_char_sum)
                active.remove(divisor)
        if not active:
            break
        next_reward = float(perm_rewards[n])
        best = max(best, next_reward)
        observed_char_sum += float(perm_chars[n])
        n += 1
        model_value = (math.exp(float(np.clip(next_reward, -50.0, 50.0)))
                       if cfg.model == "exp_tail_exponentiated" else next_reward)
        add_model_value(model_value, n)
    for divisor in active:
        results[divisor] = (n, best, observed_char_sum)
    return results


def make_grid(
    quick: bool = False,
    models: tuple[str, ...] | None = None,
    confidence: tuple[float, ...] | None = None,
    reward_prior: tuple[float, ...] | None = None,
    cost_prior: tuple[float, ...] | None = None,
    threshold_quantiles: tuple[float, ...] | None = None,
    benchmark_calibrations: tuple[float, ...] | None = None,
    ei_bonus_scales: tuple[float, ...] | None = None,
) -> list[PandoraConfig]:
    models = tuple(MODEL_LABELS) if models is None else models
    confidence = confidence or ((0.1, 0.3, 0.6) if quick else (0.05, 0.1, 0.2, 0.35, 0.6, 1.0))
    reward_prior = reward_prior or ((0.0, 5.0) if quick else (0.0, 2.0, 5.0, 12.0))
    cost_prior = cost_prior or ((0.0, 5.0) if quick else (0.0, 2.0, 5.0, 12.0))
    threshold_quantiles = threshold_quantiles or ((0.5,) if quick else (0.4, 0.5, 0.6))
    benchmark_calibrations = benchmark_calibrations or ((0.0, 1.0) if not quick else (0.0,))
    ei_bonus_scales = ei_bonus_scales or (0.0,)
    grid = []
    for c in confidence:
        for rp in reward_prior:
            for cp in cost_prior:
                if "gaussian_raw" in models:
                    for bc in benchmark_calibrations:
                        for eb in ei_bonus_scales:
                            grid.append(PandoraConfig("gaussian_raw", c, rp, cp,
                                                      benchmark_calibration=bc,
                                                      ei_bonus_scale=eb))
                for model in (m for m in ("exp_tail_raw", "exp_tail_exponentiated") if m in models):
                    for q in threshold_quantiles:
                        for bc in benchmark_calibrations:
                            for eb in ei_bonus_scales:
                                grid.append(PandoraConfig(model, c, rp, cp, q,
                                                          benchmark_calibration=bc,
                                                          ei_bonus_scale=eb))
    return grid


def _permutations(indices: np.ndarray, n_trials: int, n_generations: int,
                  seed: int) -> dict[int, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    return {int(i): [rng.permutation(n_generations) for _ in range(n_trials)] for i in indices}


def tune_fixed_n(rewards: np.ndarray, chars: np.ndarray, indices: np.ndarray,
                 divisors: tuple[float, ...], n_trials: int, alpha: float,
                 seed: int) -> dict[float, int]:
    n_gen = rewards.shape[1]
    quality_sum = np.zeros(n_gen)
    char_sum = np.zeros(n_gen)
    count = 0
    for i, perms in _permutations(indices, n_trials, n_gen, seed).items():
        benchmark = _stable_alpha_quantile(rewards[i], alpha)
        for perm in perms:
            quality_sum += expit(np.maximum.accumulate(rewards[i, perm]) - benchmark)
            char_sum += np.cumsum(chars[i, perm])
            count += 1
    mean_quality, mean_chars = quality_sum / count, char_sum / count
    return {d: int(np.argmax(mean_quality - mean_chars / d) + 1) for d in divisors}


def evaluate_fixed(rewards: np.ndarray, chars: np.ndarray, indices: np.ndarray,
                   fixed_n: dict[float, int], divisors: tuple[float, ...],
                   permutations: dict[int, list[np.ndarray]], alpha: float) -> dict[float, dict[str, float]]:
    buckets = {d: {"quality": [], "chars": [], "utility": []} for d in divisors}
    for i in indices:
        i = int(i)
        benchmark = _stable_alpha_quantile(rewards[i], alpha)
        for perm in permutations[i]:
            prefix_best = np.maximum.accumulate(rewards[i, perm])
            prefix_chars = np.cumsum(chars[i, perm])
            for d in divisors:
                n = fixed_n[d]
                quality = float(expit(prefix_best[n - 1] - benchmark))
                total_chars = float(prefix_chars[n - 1])
                buckets[d]["quality"].append(quality)
                buckets[d]["chars"].append(total_chars)
                buckets[d]["utility"].append(quality - total_chars / d)
    return {d: {k: float(np.mean(v)) for k, v in bucket.items()} for d, bucket in buckets.items()}


def evaluate_pandora_config(rewards: np.ndarray, chars: np.ndarray, features: np.ndarray,
                            indices: np.ndarray, divisors: tuple[float, ...], prior: PriorFit,
                            cfg: PandoraConfig, permutations: dict[int, list[np.ndarray]],
                            alpha: float, min_open_count: int) -> dict[float, dict[str, float]]:
    predicted_chars = predict_mean_chars(prior, features)
    buckets = {d: {"quality": [], "chars": [], "utility": [], "opens": []} for d in divisors}
    for i in indices:
        i = int(i)
        benchmark = _stable_alpha_quantile(rewards[i], alpha)
        for perm in permutations[i]:
            perm_rewards, perm_chars = rewards[i, perm], chars[i, perm]
            stops = pandora_stop_many(
                perm_rewards, perm_chars, divisors, float(predicted_chars[i]), prior, cfg,
                min_open_count=min_open_count,
            )
            for d in divisors:
                n, best, total_chars = stops[d]
                quality = float(expit(best - benchmark))
                buckets[d]["quality"].append(quality)
                buckets[d]["chars"].append(total_chars)
                buckets[d]["utility"].append(quality - total_chars / d)
                buckets[d]["opens"].append(n)
    return {d: {k: float(np.mean(v)) for k, v in bucket.items()} for d, bucket in buckets.items()}


def run_split(split: int, rewards: np.ndarray, chars: np.ndarray, features: np.ndarray,
              divisors: tuple[float, ...], grid: list[PandoraConfig], alpha: float,
              min_open_count: int, train_perms: int, tune_perms: int,
              test_perms: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(91001 + split)
    shuffled = rng.permutation(len(rewards))
    train_idx, test_idx = shuffled[:len(shuffled) // 2], shuffled[len(shuffled) // 2:]
    qs = tuple(sorted({cfg.threshold_quantile for cfg in grid}))
    # The requested protocol devotes the entire 50% training half to fitting
    # each practical method.  Hyperparameters are ranked by fresh permutations
    # of those training prompts; the held-out 50% is never consulted.
    full_prior = fit_priors(rewards, chars, features, train_idx, qs)
    train_permutations = _permutations(train_idx, tune_perms, rewards.shape[1], 92001 + split)

    best_cfg: dict[float, PandoraConfig] = {}
    best_val = {d: -np.inf for d in divisors}
    tuning_rows = []
    for cfg_id, cfg in enumerate(grid):
        metrics = evaluate_pandora_config(
            rewards, chars, features, train_idx, divisors, full_prior, cfg,
            train_permutations, alpha, min_open_count,
        )
        for d in divisors:
            row = {"split": split, "config_id": cfg_id, "divisor": d, **asdict(cfg), **metrics[d]}
            tuning_rows.append(row)
            if metrics[d]["utility"] > best_val[d]:
                best_val[d] = metrics[d]["utility"]
                best_cfg[d] = cfg

    fixed_n = tune_fixed_n(rewards, chars, train_idx, divisors, train_perms, alpha, 93001 + split)
    test_permutations = _permutations(test_idx, test_perms, rewards.shape[1], 94001 + split)
    fixed = evaluate_fixed(rewards, chars, test_idx, fixed_n, divisors, test_permutations, alpha)
    rows = []
    for d in divisors:
        cfg = best_cfg[d]
        adaptive = evaluate_pandora_config(
            rewards, chars, features, test_idx, (d,), full_prior, cfg,
            test_permutations, alpha, min_open_count,
        )[d]
        f = fixed[d]
        rows.append({
            "split": split, "divisor": d, "train_prompts": len(train_idx),
            "test_prompts": len(test_idx), "fixed_n": fixed_n[d],
            **{f"selected_{k}": v for k, v in asdict(cfg).items()},
            "adaptive_quality": adaptive["quality"], "adaptive_chars": adaptive["chars"],
            "adaptive_opens": adaptive["opens"], "adaptive_utility": adaptive["utility"],
            "fixed_quality": f["quality"], "fixed_chars": f["chars"],
            "fixed_utility": f["utility"],
            "utility_gap": adaptive["utility"] - f["utility"],
            "relative_utility_gain_pct": 100.0 * (adaptive["utility"] - f["utility"]) / f["utility"],
        })
    return rows, tuning_rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for divisor in sorted({float(r["divisor"]) for r in rows}):
        group = [r for r in rows if float(r["divisor"]) == divisor]
        out = {"divisor": divisor, "splits": len(group)}
        for key in ("fixed_n", "adaptive_quality", "adaptive_chars", "adaptive_opens",
                    "adaptive_utility", "fixed_quality", "fixed_chars", "fixed_utility",
                    "utility_gap", "relative_utility_gain_pct"):
            values = np.asarray([float(r[key]) for r in group])
            mean = float(values.mean())
            if len(values) > 1:
                half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
            else:
                half = 0.0
            out[f"{key}_mean"] = mean
            out[f"{key}_ci_low"] = mean - half
            out[f"{key}_ci_high"] = mean + half
        models = [r["selected_model"] for r in group]
        out["most_selected_model"] = max(set(models), key=models.count)
        out["model_selection_counts"] = json.dumps({m: models.count(m) for m in sorted(set(models))})
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict], out_path: Path) -> None:
    ordered = sorted(summary, key=lambda r: r["divisor"])
    x = np.asarray([r["divisor"] for r in ordered])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for prefix, label, color in (("fixed_utility", "Train-tuned fixed N", "#555555"),
                                 ("adaptive_utility", "UCB Pandora", "#D1495B")):
        mean = np.asarray([r[f"{prefix}_mean"] for r in ordered])
        low = np.asarray([r[f"{prefix}_ci_low"] for r in ordered])
        high = np.asarray([r[f"{prefix}_ci_high"] for r in ordered])
        axes[0].plot(x, mean, marker="o", label=label, color=color)
        axes[0].fill_between(x, low, high, alpha=0.18, color=color)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Character cost divisor")
    axes[0].set_ylabel("Mean utility (quality - exact chars/divisor)")
    axes[0].set_title("Held-out utility; 95% CI across splits")
    axes[0].legend(frameon=False)

    gap = np.asarray([r["relative_utility_gain_pct_mean"] for r in ordered])
    low = np.asarray([r["relative_utility_gain_pct_ci_low"] for r in ordered])
    high = np.asarray([r["relative_utility_gain_pct_ci_high"] for r in ordered])
    axes[1].axhline(0, color="#777777", linewidth=1)
    axes[1].plot(x, gap, marker="o", color="#2E7D32")
    axes[1].fill_between(x, low, high, alpha=0.18, color="#2E7D32")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Character cost divisor")
    axes[1].set_ylabel("Relative utility gain over fixed N (%)")
    axes[1].set_title("UCB Pandora advantage; 95% CI")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_open_counts(summary: list[dict], out_path: Path) -> None:
    """Plot how one frozen UCB-Pandora configuration scales with cost."""
    ordered = sorted(summary, key=lambda r: r["divisor"])
    x = np.asarray([r["divisor"] for r in ordered])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for prefix, label, color in (("fixed_n", "Train-tuned fixed N", "#555555"),
                                 ("adaptive_opens", "UCB Pandora", "#D1495B")):
        mean = np.asarray([r[f"{prefix}_mean"] for r in ordered])
        low = np.asarray([r[f"{prefix}_ci_low"] for r in ordered])
        high = np.asarray([r[f"{prefix}_ci_high"] for r in ordered])
        axes[0].plot(x, mean, marker="o", label=label, color=color)
        axes[0].fill_between(x, low, high, color=color, alpha=0.18)
    axes[0].axhline(960, color="#777777", linestyle="--", linewidth=1, label="Full horizon")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Character cost divisor")
    axes[0].set_ylabel("Mean boxes opened")
    axes[0].set_title("Sampling depth; 95% CI across splits")
    axes[0].legend(frameon=False)

    adaptive = np.asarray([r["adaptive_opens_mean"] for r in ordered])
    fixed = np.asarray([r["fixed_n_mean"] for r in ordered])
    axes[1].plot(x, adaptive / fixed, marker="o", color="#2E7D32")
    axes[1].axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Character cost divisor")
    axes[1].set_ylabel("Adaptive opens / fixed N")
    axes[1].set_title("Relative sampling allocation")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("dataset/alpaca/mistral_7b_output.merged_rm.jsonl.gz"))
    parser.add_argument("--reward-key", default="mistral_rm_reward")
    parser.add_argument("--out-dir", type=Path, default=Path("codex_results/alignment_pandora_ucb"))
    parser.add_argument("--divisors", type=float, nargs="+", default=list(DEFAULT_DIVISORS))
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--split-start", type=int, default=0,
                        help="First outer-split id; use a fresh range for confirmatory runs.")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--min-open-count", type=int, default=3)
    parser.add_argument("--train-perms", type=int, default=12)
    parser.add_argument("--tune-perms", type=int, default=6)
    parser.add_argument("--test-perms", type=int, default=20)
    parser.add_argument("--quick-grid", action="store_true")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_LABELS))
    parser.add_argument("--confidence-scales", type=float, nargs="+")
    parser.add_argument("--reward-prior-strengths", type=float, nargs="+")
    parser.add_argument("--cost-prior-strengths", type=float, nargs="+")
    parser.add_argument("--threshold-quantiles", type=float, nargs="+")
    parser.add_argument("--benchmark-calibrations", type=float, nargs="+")
    parser.add_argument("--ei-bonus-scales", type=float, nargs="+")
    args = parser.parse_args()
    if args.min_open_count != 3:
        print(f"warning: requested min-open-count={args.min_open_count}; prior protocol used 3")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rewards, chars, prompts = load_data(args.input, args.reward_key)
    features = prompt_features(prompts)
    divisors = tuple(float(x) for x in args.divisors)
    grid = make_grid(
        args.quick_grid,
        models=None if args.models is None else tuple(args.models),
        confidence=None if args.confidence_scales is None else tuple(args.confidence_scales),
        reward_prior=None if args.reward_prior_strengths is None else tuple(args.reward_prior_strengths),
        cost_prior=None if args.cost_prior_strengths is None else tuple(args.cost_prior_strengths),
        threshold_quantiles=None if args.threshold_quantiles is None else tuple(args.threshold_quantiles),
        benchmark_calibrations=(None if args.benchmark_calibrations is None
                                else tuple(args.benchmark_calibrations)),
        ei_bonus_scales=None if args.ei_bonus_scales is None else tuple(args.ei_bonus_scales),
    )
    print(f"Loaded {rewards.shape[0]} prompts x {rewards.shape[1]} generations; "
          f"{len(grid)} UCB-Pandora configurations", flush=True)
    all_rows, all_tuning = [], []
    for completed, split in enumerate(range(args.split_start, args.split_start + args.splits), start=1):
        rows, tuning = run_split(
            split, rewards, chars, features, divisors, grid, args.alpha,
            args.min_open_count, args.train_perms, args.tune_perms, args.test_perms,
        )
        all_rows.extend(rows)
        all_tuning.extend(tuning)
        best = max(rows, key=lambda r: r["relative_utility_gain_pct"])
        print(f"split {completed}/{args.splits} (id={split}): best held-out gain "
              f"{best['relative_utility_gain_pct']:+.2f}% at divisor {best['divisor']:g} "
              f"({best['selected_model']})", flush=True)
        write_csv(args.out_dir / "split_results.partial.csv", all_rows)

    result_summary = summarize(all_rows)
    write_csv(args.out_dir / "split_results.csv", all_rows)
    write_csv(args.out_dir / "tuning_results.csv", all_tuning)
    write_csv(args.out_dir / "utility_summary.csv", result_summary)
    plot_summary(result_summary, args.out_dir / "utility_comparison.png")
    plot_open_counts(result_summary, args.out_dir / "sample_counts.png")
    manifest = {
        "algorithm": "UCB Pandora reservation/fair-cap stopping only",
        "decision": "stop iff E_UCB[(u(next)-u(best))_+] <= E[next_chars|observed]/divisor",
        "train_test": "50/50 split; all distribution fitting and hyperparameter selection use only training half",
        "cost_available_to_policy": "training-fitted prompt prior plus characters of already opened generations",
        "reported_cost": "exact cumulative character count of opened generations / divisor",
        "evaluation_only": "full-prompt empirical alpha-quantile, never passed to stopping policy",
        "splits": args.splits, "split_start": args.split_start,
        "min_open_count": args.min_open_count,
        "divisors": divisors, "grid_size": len(grid),
    }
    with open(args.out_dir / "METHOD.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(result_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
