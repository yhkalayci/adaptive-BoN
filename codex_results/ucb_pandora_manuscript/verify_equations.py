"""Numerically check the simplified manuscript equations against the code.

This verifier independently implements the displayed alignment and bounded
shifted-exponential coding equations.  It then compares their prefix expected
improvements and final stopping decisions with the implementations under
``../code``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


ROOT = Path(__file__).resolve().parent
ALIGNMENT_SOURCE = ROOT.parent / "code" / "alignment_scripts"
CODING_SOURCE = ROOT.parent / "code" / "coding_scripts"
sys.path.insert(0, str(ALIGNMENT_SOURCE))
sys.path.insert(0, str(CODING_SOURCE))

from alignment_pandora_ucb import (  # noqa: E402
    PandoraConfig as AlignmentConfig,
    PriorFit,
    pandora_stop_many,
)
from coding_ucb_three_objectives import (  # noqa: E402
    Calibration,
    DistributionPrior,
    PolicyConfig,
    _ei_curve,
    pandora_stop_from_curve,
)


EPSILON = 1e-8
EXP_PROBABILITIES = (np.arange(48, dtype=np.float64) + 0.5) / 48.0
EXP_QUANTILES = -np.log1p(-EXP_PROBABILITIES)
LEGENDRE_X, LEGENDRE_W = leggauss(32)
UNIT_X = 0.5 * (LEGENDRE_X + 1.0)
UNIT_W = 0.5 * LEGENDRE_W


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return float(result) if result.ndim == 0 else result


def manual_alignment_stops(rewards, chars, divisors, prior, config, min_open=5):
    """The calculations collected in Algorithm 2 of the manuscript."""
    rewards = np.asarray(rewards, dtype=np.float64)
    chars = np.asarray(chars, dtype=np.float64)
    results = {}
    for divisor in divisors:
        n = min(min_open, len(rewards))
        while n < len(rewards):
            transformed = np.exp(np.clip(rewards[:n], -50.0, 50.0))
            lower_count = max(int(math.ceil(config.threshold_quantile * n)) - 1, 0)
            tail = np.sort(transformed)[lower_count:]
            location = float(tail[0])
            point_scale = max(float(np.mean(tail) - location), EPSILON)
            prior_scale = location * prior.exp_tail_ratios[config.threshold_quantile]
            scale = (
                len(tail) * point_scale
                + config.reward_prior_strength * prior_scale
            ) / (len(tail) + config.reward_prior_strength)
            radius = math.sqrt(math.log(1.0 / config.delta) / len(tail))
            scale_ucb = scale * (1.0 + config.confidence_scale * radius)
            scale_lcb = max(
                scale * (1.0 - config.confidence_scale * radius), EPSILON
            )
            tail_mass = 1.0 - config.threshold_quantile
            theoretical_k = -math.log((1.0 - 0.99) / tail_mass)
            calibrated_k = prior.exp_tail_quantile_k[config.threshold_quantile]
            benchmark_k = (
                (1.0 - config.benchmark_calibration) * theoretical_k
                + config.benchmark_calibration * calibrated_k
            )
            benchmark = math.log(location + scale_lcb * benchmark_k)
            future = np.log(location + scale_ucb * EXP_QUANTILES)
            incumbent = float(np.max(rewards[:n]))
            expected_improvement = tail_mass * float(
                np.mean(
                    np.maximum(
                        sigmoid(future - benchmark)
                        - sigmoid(incumbent - benchmark),
                        0.0,
                    )
                )
            )
            gain = min(
                expected_improvement
                + config.ei_bonus_scale
                * math.sqrt(math.log(1.0 / config.delta) / (2.0 * n)),
                1.0,
            )
            expected_chars = float(np.mean(chars[:n]))
            if gain <= expected_chars / divisor:
                break
            n += 1
        results[float(divisor)] = (
            n,
            float(np.max(rewards[:n])),
            float(np.sum(chars[:n])),
        )
    return results


def verify_alignment():
    prior = PriorFit(
        raw_sigma=1.0,
        gaussian_quantile_z=2.2,
        raw_tail_scales={0.5: 0.7},
        exp_tail_ratios={0.5: 0.65},
        raw_tail_quantile_k={0.5: 4.2},
        exp_tail_quantile_k={0.5: 4.5},
        cost_beta=np.zeros(6),
        feature_mean=np.zeros(5),
        feature_scale=np.ones(5),
    )
    config = AlignmentConfig(
        "exp_tail_exponentiated",
        confidence_scale=0.6,
        reward_prior_strength=0.0,
        cost_prior_strength=0.0,
        threshold_quantile=0.5,
        delta=0.05,
        benchmark_calibration=0.0,
        ei_bonus_scale=0.002,
    )
    rng = np.random.default_rng(20260812)
    divisors = (1_000.0, 10_000.0, 100_000.0, 10_000_000.0)
    checked = 0
    for length in (17, 64, 127):
        rewards = rng.normal(loc=0.1, scale=1.3, size=length)
        # Repeated values exercise inclusive threshold membership.
        rewards[4:7] = rewards[3]
        chars = rng.integers(25, 800, size=length).astype(np.float64)
        expected = manual_alignment_stops(
            rewards, chars, divisors, prior, config
        )
        actual = pandora_stop_many(
            rewards,
            chars,
            divisors,
            prior_mean_chars=300.0,
            prior=prior,
            cfg=config,
            min_open_count=5,
        )
        for divisor in divisors:
            expected_row = expected[divisor]
            actual_row = actual[divisor]
            assert actual_row[0] == expected_row[0], (divisor, actual_row, expected_row)
            np.testing.assert_allclose(actual_row[1:], expected_row[1:], rtol=0, atol=1e-11)
            checked += 1
    return checked


def manual_coding_ei(rewards, prior, config, delta):
    """The bounded-tail calculations in Algorithm 3 for every prefix."""
    rewards = np.asarray(rewards, dtype=np.float64)
    output = np.empty(len(rewards), dtype=np.float64)
    quantile = float(config.tail_quantile)
    for n in range(1, len(rewards) + 1):
        observed = rewards[:n]
        incumbent = float(np.max(observed))
        local_location = float(np.quantile(observed, quantile))
        tail = observed[observed >= local_location]
        local_scale = max(float(np.mean(tail - local_location)), EPSILON)
        if math.isinf(config.reward_prior_strength):
            location = prior.exp_locations[quantile]
            scale = prior.exp_scales[quantile]
        else:
            location = (
                n * local_location
                + config.reward_prior_strength * prior.exp_locations[quantile]
            ) / (n + config.reward_prior_strength)
            scale = (
                len(tail) * local_scale
                + config.reward_prior_strength * prior.exp_scales[quantile]
            ) / (len(tail) + config.reward_prior_strength)
        radius = math.sqrt(math.log(1.0 / delta) / len(tail))
        location_ucb = float(
            np.clip(
                location + config.confidence_scale * scale * radius,
                0.0,
                1.0,
            )
        )
        scale_ucb = scale * (1.0 + config.confidence_scale * radius)
        room = 1.0 - location_ucb
        truncated_mass = -math.expm1(-room / max(scale_ucb, 1e-12))
        future = location_ucb - scale_ucb * np.log1p(
            -truncated_mass * UNIT_X
        )
        future = np.clip(future, 0.0, 1.0)
        output[n - 1] = (1.0 - quantile) * float(
            np.sum(UNIT_W * np.maximum(future - incumbent, 0.0))
        )
    return output


def manual_coding_stop(
    expected_improvement,
    cumulative_chars,
    reservation_divisor,
    prior_mean_chars,
    config,
    fixed_n,
    min_open=3,
):
    """The stopping comparison in Algorithm 3 of the manuscript."""
    horizon = len(expected_improvement)
    if config.cap_factor is not None:
        horizon = max(min_open, int(math.ceil(config.cap_factor * fixed_n)))
        horizon = min(horizon, len(expected_improvement))
    n = min(min_open, horizon)
    while n < horizon:
        expected_chars = (
            cumulative_chars[n - 1]
            + config.cost_prior_strength * prior_mean_chars
        ) / (n + config.cost_prior_strength)
        gain = expected_improvement[n - 1] * (n / min_open) ** (
            -config.tail_decay
        )
        if gain <= expected_chars / reservation_divisor:
            break
        n += 1
    return n


def verify_coding():
    rng = np.random.default_rng(20260813)
    prior = DistributionPrior(
        gaussian_mean=0.20,
        gaussian_variance=0.025,
        exp_locations={0.5: 0.18, 0.75: 0.31},
        exp_scales={0.5: 0.13, 0.75: 0.09},
        mean_chars=525.0,
    )
    calibration = Calibration("bounded_identity")
    checked_curves = 0
    checked_stops = 0
    checked_rescalings = 0
    config = PolicyConfig(
        "shifted_exponential",
        "bounded_identity",
        0.8,
        math.inf,
        0.0,
        None,
        0.75,
        1.0,
    )
    for replicate in range(4):
        rewards = np.sort(rng.beta(1.8, 5.2, size=73))[rng.permutation(73)]
        rewards[8:12] = rewards[7]  # Isotonic-style plateau.
        chars = rng.integers(40, 1_100, size=len(rewards)).astype(np.float64)
        cumulative_chars = np.cumsum(chars)
        incumbent = np.maximum.accumulate(rewards)
        actual_curve = _ei_curve(
            rewards, incumbent, calibration, prior, config, 0.05
        )
        expected_curve = manual_coding_ei(rewards, prior, config, 0.05)
        np.testing.assert_allclose(
            actual_curve, expected_curve, rtol=2e-13, atol=2e-13
        )
        checked_curves += 1
        for economic_divisor in (100_000.0, 400_000.0, 1_000_000.0):
            reservation_divisor = 0.85 * economic_divisor
            expected_n = manual_coding_stop(
                expected_curve,
                cumulative_chars,
                reservation_divisor,
                prior.mean_chars,
                config,
                fixed_n=1,
            )
            actual_n = pandora_stop_from_curve(
                actual_curve,
                cumulative_chars,
                reservation_divisor,
                prior.mean_chars,
                config,
                fixed_n=1,
                min_open=3,
            )
            assert actual_n == expected_n, (
                replicate,
                reservation_divisor,
                actual_n,
                expected_n,
            )
            checked_stops += 1
            scaled_n = manual_coding_stop(
                expected_curve * 0.85,
                cumulative_chars,
                economic_divisor,
                prior.mean_chars,
                config,
                fixed_n=1,
            )
            assert scaled_n == expected_n
            checked_rescalings += 1
    return checked_curves, checked_stops, checked_rescalings


def main():
    alignment_stops = verify_alignment()
    coding_curves, coding_stops, coding_rescalings = verify_coding()
    print(
        "Equation parity passed: "
        f"{alignment_stops} alignment stops, "
        f"{coding_curves} coding EI curves, "
        f"{coding_stops} coding stops, and "
        f"{coding_rescalings} reservation/objective rescalings."
    )


if __name__ == "__main__":
    main()
