"""Held-out coding evaluation for genuine UCB-Pandora stopping policies.

The cohort is restricted to the 83 problems with at least one correct sample.
For every outer split, calibration, distribution priors, Fixed-N, and all
adaptive hyperparameters are learned from the training problems only.  Test
utility charges the exact cumulative number of output characters opened.

By default, the distribution families are fitted in raw-reward space:

* a complete Gaussian distribution;
* a Gaussian kernel density estimate; and
* an empirical body plus a conditional shifted-exponential upper tail.

Both isotonic P(correct | reward) calibration and a smooth Bradley-Terry
(logistic) calibration are available.  Every adaptive decision is the
Pandora reservation test ``UCB expected improvement > expected next cost``.
An optional held-out experiment instead fits isotonic P(correct | raw reward)
on the outer training half, transforms rewards into success probabilities,
and fits the stopping distribution in that calibrated probability space.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from scipy.special import expit, ndtr, ndtri
from scipy.stats import t as student_t
from sklearn.isotonic import IsotonicRegression


REPORT_DIVISORS = tuple(
    float(divisor) for divisor in range(100_000, 1_000_001, 100_000)
)
FRONTIER_DIVISORS = (5e4,) + REPORT_DIVISORS
# A target-quality curve is a Lagrange path, so four utility reporting points
# are far too coarse.  These values change only the character-price multiplier
# in the same UCB Pandora reservation rule.
TARGET_FRONTIER_DIVISORS = (
    2.5e4, 3.5e4, 5e4, 7e4, 1e5, 1.4e5, 2e5,
    2.8e5, 4e5, 5.6e5, 7e5, 1e6, 1.4e6,
)
TARGETS = (0.25, 0.30, 0.35)
ROBUST_METHOD_BY_DIVISOR = {
    1e5: "gaussian_kde_relative_bt",
    4e5: "gaussian_kde_relative_bt",
    7e5: "shifted_exponential_relative_bt",
    1e6: "shifted_exponential_relative_bt",
}
FROZEN_EXP_CONFIG_BY_DIVISOR = {
    5e4: dict(confidence_scale=0.8, reward_prior_strength=1e6,
              cost_prior_strength=0.0, tail_quantile=0.5),
    1e5: dict(confidence_scale=0.8, reward_prior_strength=1e6,
              cost_prior_strength=0.0, tail_quantile=0.5),
    4e5: dict(confidence_scale=0.0, reward_prior_strength=5.0,
              cost_prior_strength=10.0, tail_quantile=0.5),
    7e5: dict(confidence_scale=0.0, reward_prior_strength=20.0,
              cost_prior_strength=0.0, tail_quantile=0.5),
    1e6: dict(confidence_scale=0.0, reward_prior_strength=20.0,
              cost_prior_strength=0.0, tail_quantile=0.5),
}
FROZEN_CONTEXTUAL_EXP_CONFIG_BY_DIVISOR = {
    5e4: dict(confidence_scale=0.2, reward_prior_strength=5.0,
              cost_prior_strength=0.0, cap_factor=1.3, tail_quantile=0.75),
    1e5: dict(confidence_scale=0.2, reward_prior_strength=5.0,
              cost_prior_strength=0.0, cap_factor=1.3, tail_quantile=0.75),
    4e5: dict(confidence_scale=0.2, reward_prior_strength=20.0,
              cost_prior_strength=0.0, cap_factor=1.0, tail_quantile=0.5),
    7e5: dict(confidence_scale=0.2, reward_prior_strength=5.0,
              cost_prior_strength=0.0, cap_factor=1.3, tail_quantile=0.75),
    1e6: dict(confidence_scale=0.2, reward_prior_strength=1e6,
              cost_prior_strength=0.0, cap_factor=1.0, tail_quantile=0.5),
}
METHODS = (
    "gaussian_bt",
    "gaussian_relative_bt",
    "gaussian_isotonic",
    "gaussian_kde_bt",
    "gaussian_kde_relative_bt",
    "gaussian_kde_isotonic",
    "gaussian_calibrated_probability",
    "shifted_exponential_bt",
    "shifted_exponential_relative_bt",
    "shifted_exponential_contextual_relative_bt",
    "shifted_exponential_isotonic",
    "shifted_exponential_calibrated_probability",
    "selected_ucb",
)
LABELS = {
    "gaussian_bt": "Gaussian raw + BT",
    "gaussian_relative_bt": "Gaussian raw + relative BT",
    "gaussian_isotonic": "Gaussian raw + isotonic",
    "gaussian_kde_bt": "Gaussian KDE raw + BT",
    "gaussian_kde_relative_bt": "Gaussian KDE raw + relative BT",
    "gaussian_kde_isotonic": "Gaussian KDE raw + isotonic",
    "gaussian_calibrated_probability": "Gaussian on isotonic probability",
    "shifted_exponential_bt": "Exp-tail raw + BT",
    "shifted_exponential_relative_bt": "Exp-tail raw + relative BT",
    "shifted_exponential_contextual_relative_bt": "Exp-tail raw + contextual relative BT",
    "shifted_exponential_isotonic": "Exp-tail raw + isotonic",
    "shifted_exponential_calibrated_probability": "Exp-tail on isotonic probability",
    "selected_ucb": "Train-selected UCB Pandora",
}
COLORS = {
    "gaussian_bt": "#3A6EA5",
    "gaussian_relative_bt": "#184E77",
    "gaussian_isotonic": "#7A5195",
    "gaussian_kde_bt": "#008C95",
    "gaussian_kde_relative_bt": "#00A6A6",
    "gaussian_kde_isotonic": "#8E6C8A",
    "gaussian_calibrated_probability": "#3A6EA5",
    "shifted_exponential_bt": "#D1495B",
    "shifted_exponential_relative_bt": "#B23A48",
    "shifted_exponential_contextual_relative_bt": "#6F1D2A",
    "shifted_exponential_isotonic": "#E17C05",
    "shifted_exponential_calibrated_probability": "#C45A00",
    "selected_ucb": "#1B7F3A",
}
_GH_X, _GH_W = hermgauss(32)
_GH_W = _GH_W / math.sqrt(math.pi)
_KDE_X, _KDE_W = hermgauss(12)
_KDE_W = _KDE_W / math.sqrt(math.pi)
_LAG_X, _LAG_W = laggauss(32)
_LEG_X, _LEG_W = leggauss(32)
_UNIT_X = 0.5 * (_LEG_X + 1.0)
_UNIT_W = 0.5 * _LEG_W


@dataclass(frozen=True)
class Calibration:
    kind: str
    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()
    intercept: float = 0.0
    slope: float = 1.0
    context_mean_center: float = 0.0
    context_mean_scale: float = 1.0
    context_mean_coef: float = 0.0

    def __call__(self, values):
        values = np.asarray(values, dtype=np.float64)
        if self.kind in ("identity", "bounded_identity"):
            # Calibrated rewards are probabilities.  The exponential tail has
            # unbounded support, so clip extrapolated quadrature points to the
            # valid probability domain before computing expected improvement.
            result = np.clip(values, 0.0, 1.0)
        elif self.kind in ("bt", "relative_bt"):
            result = expit(self.intercept + self.slope * values)
        elif self.kind == "contextual_relative_bt":
            raise ValueError("contextual relative BT requires an online problem context")
        elif self.kind in ("isotonic", "pilot_isotonic"):
            result = np.interp(values, self.x, self.y, left=self.y[0], right=self.y[-1])
        else:
            raise ValueError(f"unknown calibration: {self.kind}")
        return float(result) if values.ndim == 0 else result


@dataclass(frozen=True)
class DistributionPrior:
    gaussian_mean: float
    gaussian_variance: float
    exp_locations: dict[float, float]
    exp_scales: dict[float, float]
    mean_chars: float


@dataclass(frozen=True)
class PolicyConfig:
    family: str
    calibration: str
    confidence_scale: float
    reward_prior_strength: float
    cost_prior_strength: float
    cap_factor: float | None
    tail_quantile: float | None = None
    tail_decay: float = 0.0

    @property
    def method(self) -> str:
        if self.calibration in ("identity", "bounded_identity"):
            return f"{self.family}_calibrated_probability"
        return f"{self.family}_{self.calibration}"


def load_problems(data_path: Path, char_cache: Path) -> dict[str, tuple[np.ndarray, ...]]:
    scored = {}
    with data_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            samples = sorted(record["samples"], key=lambda item: int(item["idx"]))
            correct = np.asarray([bool(item["correct"]) for item in samples], dtype=np.int8)
            if not correct.any():
                continue
            indices = np.asarray([int(item["idx"]) for item in samples], dtype=np.int64)
            if len(np.unique(indices)) != len(indices):
                raise ValueError(f"duplicate sample index for problem {record['id']}")
            rewards = np.asarray([item["r_score"] for item in samples], dtype=np.float64)
            scored[str(record["id"])] = (indices, rewards, correct)
    if len(scored) != 83:
        raise ValueError(f"expected 83 solvable problems, found {len(scored)}")
    with np.load(char_cache, allow_pickle=False) as cache:
        ids = [str(value) for value in cache["ids"].tolist()]
        indices = np.asarray(cache["indices"], dtype=np.int64)
        chars = np.asarray(cache["chars"], dtype=np.float64)
    if ids != sorted(scored) or chars.shape != (83, 512):
        raise ValueError("character cache does not match the 83-problem cohort")
    result = {}
    for row, problem_id in enumerate(ids):
        if not np.array_equal(indices[row], scored[problem_id][0]):
            raise ValueError(f"character-cache indices do not match {problem_id}")
        _, rewards, correct = scored[problem_id]
        if len(rewards) != 512:
            raise ValueError(f"expected 512 samples for {problem_id}, found {len(rewards)}")
        result[problem_id] = (rewards, correct, chars[row].copy())
    return result


def split_problems(problems, seed: int):
    ids = np.asarray(sorted(problems))
    order = np.random.default_rng(seed).permutation(ids)
    midpoint = len(order) // 2
    train_ids, test_ids = order[:midpoint], order[midpoint:]
    return ({pid: problems[str(pid)] for pid in train_ids},
            {pid: problems[str(pid)] for pid in test_ids})


def fit_calibrations(problems) -> dict[str, Calibration]:
    rewards = np.concatenate([values[0] for values in problems.values()])
    correct = np.concatenate([values[1] for values in problems.values()]).astype(np.float64)

    # Quantile bins reduce endpoint variance; Jeffreys smoothing prevents
    # arbitrary zero/one probabilities before the monotone fit.
    min_bin_count = 256
    n_bins = max(4, int(math.ceil(len(rewards) / min_bin_count)))
    edges = np.unique(np.quantile(rewards, np.linspace(0.0, 1.0, n_bins + 1)))
    bin_id = np.searchsorted(edges[1:-1], rewards, side="right")
    counts = np.bincount(bin_id, minlength=len(edges) - 1).astype(np.float64)
    reward_sum = np.bincount(bin_id, weights=rewards, minlength=len(edges) - 1)
    correct_sum = np.bincount(bin_id, weights=correct, minlength=len(edges) - 1)
    keep = counts > 0
    centers = reward_sum[keep] / counts[keep]
    probability = (correct_sum[keep] + 0.5) / (counts[keep] + 1.0)
    isotonic = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    ).fit(centers, probability, sample_weight=counts[keep])
    iso = Calibration(
        "isotonic",
        tuple(float(x) for x in isotonic.X_thresholds_),
        tuple(float(y) for y in isotonic.y_thresholds_),
    )

    # Smooth monotone logistic calibration is the coding analogue of a
    # Bradley-Terry probability: sigmoid(intercept + slope * raw_reward).
    mean = float(correct.mean())
    initial = np.asarray([math.log(mean / max(1.0 - mean, 1e-12)), 0.25])

    def fit_logistic(feature, kind):
        def objective(theta):
            logits = theta[0] + theta[1] * feature
            return float(np.sum(np.logaddexp(0.0, logits) - correct * logits))

        fitted = minimize(objective, initial, method="L-BFGS-B",
                          bounds=((None, None), (0.0, 20.0)))
        if not fitted.success:
            raise RuntimeError(f"{kind} calibration failed: {fitted.message}")
        return Calibration(kind, intercept=float(fitted.x[0]), slope=float(fitted.x[1]))

    bt = fit_logistic(rewards, "bt")
    standardized = np.concatenate([
        (values[0] - np.mean(values[0])) / max(float(np.std(values[0], ddof=1)), 1e-8)
        for values in problems.values()
    ])
    relative_bt = fit_logistic(standardized, "relative_bt")

    # Relative reward alone assumes that every problem has the same base
    # probability of correctness.  Coding problems violate that assumption:
    # their mean reward is informative about problem difficulty.  Add the
    # standardized problem mean as an intercept feature, while retaining a
    # non-negative coefficient on within-problem relative reward.  A fixed
    # ridge penalty was selected on development problem splits and prevents
    # the contextual intercept from following small-split composition noise.
    problem_means = np.asarray([
        float(np.mean(values[0])) for values in problems.values()
    ])
    context_center = float(np.mean(problem_means))
    context_scale = max(float(np.std(problem_means, ddof=1)), 1e-8)
    context_mean = np.concatenate([
        np.full(len(values[0]), (float(np.mean(values[0])) - context_center) /
                context_scale, dtype=np.float64)
        for values in problems.values()
    ])
    ridge = 1.0 / 0.003

    def contextual_objective(theta):
        logits = theta[0] + theta[1] * standardized + theta[2] * context_mean
        loss = np.sum(np.logaddexp(0.0, logits) - correct * logits)
        return float(loss + 0.5 * ridge * np.dot(theta[1:], theta[1:]))

    contextual_fit = minimize(
        contextual_objective,
        np.asarray([initial[0], 0.5, 0.5]),
        method="L-BFGS-B",
        bounds=((None, None), (0.0, None), (None, None)),
    )
    if not contextual_fit.success:
        raise RuntimeError(
            f"contextual relative BT calibration failed: {contextual_fit.message}"
        )
    contextual_relative_bt = Calibration(
        "contextual_relative_bt",
        intercept=float(contextual_fit.x[0]),
        slope=float(contextual_fit.x[1]),
        context_mean_center=context_center,
        context_mean_scale=context_scale,
        context_mean_coef=float(contextual_fit.x[2]),
    )
    return {
        "identity": Calibration("identity"),
        "bounded_identity": Calibration("bounded_identity"),
        "bt": bt,
        "relative_bt": relative_bt,
        "contextual_relative_bt": contextual_relative_bt,
        "isotonic": iso,
    }


def fit_config_calibrations(problems, configs) -> dict[str, Calibration]:
    """Fit only calibration objects needed by the candidate policies."""
    required = {config.calibration for config in configs}
    if required <= {"identity", "bounded_identity"}:
        return {name: Calibration(name) for name in required}
    return {
        name: calibration
        for name, calibration in fit_calibrations(problems).items()
        if name in required
    }


def fit_reward_space_isotonic(problems) -> Calibration:
    """Fit raw reward -> P(correct) using only the supplied problems."""
    rewards = np.concatenate([values[0] for values in problems.values()])
    correct = np.concatenate([values[1] for values in problems.values()]).astype(np.float64)
    isotonic = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    ).fit(rewards, correct)
    return Calibration(
        "isotonic",
        tuple(float(x) for x in isotonic.X_thresholds_),
        tuple(float(y) for y in isotonic.y_thresholds_),
    )


def fit_pilot_reward_space_isotonic(
    problems, permutations, beta=-0.5, min_open=3,
) -> Calibration:
    """Fit a train-only isotonic map after an online pilot location shift."""
    center = float(np.mean(np.concatenate([
        values[0] for values in problems.values()
    ])))
    scores, outcomes = [], []
    for problem_id, (rewards, correct, _) in problems.items():
        for permutation in permutations[problem_id]:
            pilot_mean = float(np.mean(rewards[permutation[:min_open]]))
            scores.append(rewards + beta * (pilot_mean - center))
            outcomes.append(correct)
    isotonic = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    ).fit(np.concatenate(scores), np.concatenate(outcomes))
    return Calibration(
        "pilot_isotonic",
        tuple(float(x) for x in isotonic.X_thresholds_),
        tuple(float(y) for y in isotonic.y_thresholds_),
        context_mean_center=center,
        context_mean_coef=float(beta),
    )


def pilot_transform_rewards(
    rewards, permutation, calibration, min_open=3,
):
    """Apply a frozen pilot-isotonic map using only the opened pilot prefix."""
    if calibration.kind != "pilot_isotonic":
        raise ValueError("pilot reward transformation requires pilot_isotonic")
    ordered = np.asarray(rewards, dtype=np.float64)[permutation]
    pilot_mean = float(np.mean(ordered[:min_open]))
    adjusted = ordered + calibration.context_mean_coef * (
        pilot_mean - calibration.context_mean_center
    )
    return np.asarray(calibration(adjusted), dtype=np.float64)


def fit_pilot_prior(problems, permutations, calibration, min_open=3):
    """Fit the probability-space prior over train trajectories only."""
    transformed = {}
    for problem_id, (rewards, correct, chars) in problems.items():
        for permutation_id, permutation in enumerate(permutations[problem_id]):
            probability = pilot_transform_rewards(
                rewards, permutation, calibration, min_open
            )
            transformed[f"{problem_id}:{permutation_id}"] = (
                probability, correct[permutation].copy(), chars[permutation].copy()
            )
    return fit_prior(transformed)


def transform_reward_space(problems, calibration: Calibration):
    """Freeze a calibration map while preserving outcomes and character cost."""
    return {
        problem_id: (
            np.asarray(calibration(values[0]), dtype=np.float64),
            values[1].copy(),
            values[2].copy(),
        )
        for problem_id, values in problems.items()
    }


def reward_space_calibration_record(split, train, test, calibration):
    """Audit record proving that calibration is trained before held-out use."""
    def metrics(problems):
        rewards = np.concatenate([values[0] for values in problems.values()])
        correct = np.concatenate([values[1] for values in problems.values()]).astype(np.float64)
        probability = np.asarray(calibration(rewards), dtype=np.float64)
        return {
            "problems": len(problems),
            "samples": len(rewards),
            "accuracy": float(np.mean(correct)),
            "mean_probability": float(np.mean(probability)),
            "brier": float(np.mean((probability - correct) ** 2)),
        }

    train_metrics = metrics(train)
    test_metrics = metrics(test)
    return {
        "split": split,
        "fit_partition": "train_only",
        "train_problems": train_metrics["problems"],
        "test_problems": test_metrics["problems"],
        "train_samples": train_metrics["samples"],
        "test_samples": test_metrics["samples"],
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "train_mean_probability": train_metrics["mean_probability"],
        "test_mean_probability": test_metrics["mean_probability"],
        "train_brier": train_metrics["brier"],
        "test_brier": test_metrics["brier"],
        "knot_count": len(calibration.x),
        "raw_reward_min": calibration.x[0],
        "raw_reward_max": calibration.x[-1],
        "raw_reward_knots": json.dumps(calibration.x),
        "success_probability_knots": json.dumps(calibration.y),
    }


def fit_prior(problems, tail_quantiles=(0.25, 0.5, 0.75)) -> DistributionPrior:
    means, variances = [], []
    locations = {q: [] for q in tail_quantiles}
    scales = {q: [] for q in tail_quantiles}
    all_chars = []
    for rewards, _, chars in problems.values():
        means.append(float(np.mean(rewards)))
        variances.append(float(np.var(rewards, ddof=1)))
        all_chars.append(chars)
        for q in tail_quantiles:
            location = float(np.quantile(rewards, q))
            tail = rewards[rewards >= location]
            locations[q].append(location)
            scales[q].append(max(float(np.mean(tail - location)), 1e-8))
    return DistributionPrior(
        gaussian_mean=float(np.mean(means)),
        gaussian_variance=max(float(np.mean(variances)), 1e-8),
        exp_locations={q: float(np.mean(values)) for q, values in locations.items()},
        exp_scales={q: max(float(np.mean(scales[q])), 1e-8) for q in tail_quantiles},
        mean_chars=float(np.mean(np.concatenate(all_chars))),
    )


def exact_fixed_accuracy(problems) -> np.ndarray:
    """Exact without-replacement accuracy of reward-selected Fixed-N."""
    k = min(len(values[0]) for values in problems.values())
    curve = np.zeros(k, dtype=np.float64)
    for rewards, correct, _ in problems.values():
        order = np.argsort(rewards, kind="stable")
        sorted_rewards = rewards[order]
        sorted_correct = correct[order].astype(np.float64)
        starts = np.r_[0, 1 + np.flatnonzero(sorted_rewards[1:] != sorted_rewards[:-1])]
        ends = np.r_[starts[1:], k]
        group_correct = np.add.reduceat(sorted_correct, starts) / (ends - starts)
        q_values = np.ones(k + 1, dtype=np.float64)
        for n in range(1, k + 1):
            q_values.fill(0.0)
            q_values[k] = 1.0
            for a in range(k, n, -1):
                q_values[a - 1] = q_values[a] * (a - n) / a
            curve[n - 1] += float((q_values[ends] - q_values[starts]) @ group_correct)
    return curve / len(problems)


def fixed_curves(problems, divisors) -> tuple[np.ndarray, np.ndarray, dict[float, int]]:
    accuracy = exact_fixed_accuracy(problems)
    mean_chars = float(np.mean(np.concatenate([values[2] for values in problems.values()])))
    chars = mean_chars * np.arange(1, len(accuracy) + 1, dtype=np.float64)
    fixed_n = {
        float(divisor): int(np.argmax(accuracy - chars / divisor)) + 1
        for divisor in divisors
    }
    return accuracy, chars, fixed_n


def make_permutations(problems, permutations: int, seed: int):
    output = {}
    for offset, problem_id in enumerate(sorted(problems)):
        rng = np.random.default_rng(seed + 1_000_003 * (offset + 1))
        output[problem_id] = [rng.permutation(512) for _ in range(permutations)]
    return output


def candidate_configs(quick: bool = False, focused: bool = False) -> list[PolicyConfig]:
    if focused:
        configs = []
        for family, quantiles in (
            ("gaussian", (None,)),
            ("gaussian_kde", (None,)),
            ("shifted_exponential", (0.25, 0.5)),
        ):
            for quantile, cs, rp, cp in itertools.product(
                quantiles, (0.0, 0.8, 1.6, 3.2, 6.4),
                (5.0, 20.0, 1e6), (0.0, 10.0),
            ):
                configs.append(PolicyConfig(
                    family, "relative_bt", cs, rp, cp, 1.0, quantile
                ))
        return configs
    confidence = (0.0, 0.8) if quick else (0.0, 0.4, 0.8, 1.6)
    reward_prior = (5.0, 1e6) if quick else (0.0, 5.0, 20.0, 1e6)
    cost_prior = (10.0,) if quick else (0.0, 10.0)
    cap_factors = (1.0, None) if quick else (1.0, 2.0, None)
    configs = []
    for calibration in ("bt", "relative_bt", "isotonic"):
        for cs, rp, cp, cap in itertools.product(
            confidence, reward_prior, cost_prior, cap_factors
        ):
            configs.append(PolicyConfig(
                "gaussian", calibration, cs, rp, cp, cap, None
            ))
            configs.append(PolicyConfig(
                "gaussian_kde", calibration, cs, rp, cp, cap, None
            ))
            for q in (0.25, 0.5):
                configs.append(PolicyConfig(
                    "shifted_exponential", calibration, cs, rp, cp, cap, q
                ))
    return configs


def expanded_exponential_configs(
    calibration: str = "relative_bt",
) -> list[PolicyConfig]:
    """Focused search for a less one-sided exponential Pandora policy.

    The original focused grid capped every trajectory at the train-optimal
    Fixed N.  That permits adaptive early stopping, but prevents Pandora from
    reallocating those saved characters to unusually difficult problems.  We
    retain the same reservation test and vary only its distribution/UCB
    hyperparameters and the maximum number of boxes it may open.
    """
    return [
        PolicyConfig(
            "shifted_exponential", calibration, cs, rp, cp, cap, quantile
        )
        for quantile, cs, rp, cp, cap in itertools.product(
            (0.5, 0.75),
            (0.0, 0.2, 0.8),
            (5.0, 20.0, 1e6),
            (0.0, 10.0),
            (1.0, 1.1, 1.2, 1.3, 1.5, 2.0, None),
        )
    ]


def calibrated_reward_space_configs() -> list[PolicyConfig]:
    """Matched Gaussian and exponential-tail search in probability space."""
    common = tuple(itertools.product(
        (0.0, 0.2, 0.8),
        (5.0, 20.0, 1e6),
        (0.0, 10.0),
        (1.0, 1.1, 1.2, 1.3, 1.5, 2.0, None),
    ))
    gaussian = [
        PolicyConfig("gaussian", "identity", cs, rp, cp, cap, None)
        for cs, rp, cp, cap in common
    ]
    exponential = [
        PolicyConfig(
            "shifted_exponential", "identity", cs, rp, cp, cap, quantile
        )
        for quantile in (0.5, 0.75)
        for cs, rp, cp, cap in common
    ]
    return gaussian + exponential


def bounded_calibrated_reward_space_configs() -> list[PolicyConfig]:
    """The same grid with distributions properly truncated to [0, 1]."""
    return [
        PolicyConfig(
            config.family, "bounded_identity", config.confidence_scale,
            config.reward_prior_strength, config.cost_prior_strength,
            config.cap_factor, config.tail_quantile,
        )
        for config in calibrated_reward_space_configs()
    ]


def uncapped_tail_decay_configs(
    tail_decays=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
) -> list[PolicyConfig]:
    """Calibrated Gaussian/exp-tail grid with smooth decay and no N cap."""
    common = tuple(itertools.product(
        (0.0, 0.2, 0.8), (5.0, 20.0, 1e6), (0.0, 10.0)
    ))
    base = [
        PolicyConfig(
            "gaussian", "bounded_identity", cs, rp, cp, None, None
        )
        for cs, rp, cp in common
    ]
    base.extend(
        PolicyConfig(
            "shifted_exponential", "bounded_identity", cs, rp, cp,
            None, quantile,
        )
        for quantile in (0.5, 0.75)
        for cs, rp, cp in common
    )
    configs = [
        PolicyConfig(
            config.family, config.calibration, config.confidence_scale,
            config.reward_prior_strength, config.cost_prior_strength,
            config.cap_factor, config.tail_quantile, float(tail_decay),
        )
        for config in base for tail_decay in tail_decays
    ]
    if any(config.cap_factor is not None for config in configs):
        raise AssertionError("tail-decay utility policies must be uncapped")
    return configs


def _prefix_success(rewards: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best = -np.inf
    best_index = 0
    best_rewards = np.empty(len(rewards), dtype=np.float64)
    success = np.empty(len(rewards), dtype=np.float64)
    for position, reward in enumerate(rewards):
        if reward > best:
            best = float(reward)
            best_index = position
        best_rewards[position] = best
        success[position] = correct[best_index]
    return best_rewards, success


def _exp_prefix_stats(rewards: np.ndarray, quantile: float):
    locations = np.empty(len(rewards), dtype=np.float64)
    scales = np.empty(len(rewards), dtype=np.float64)
    tail_counts = np.empty(len(rewards), dtype=np.float64)
    for end in range(1, len(rewards) + 1):
        observed = rewards[:end]
        location = float(np.quantile(observed, quantile))
        tail = observed[observed >= location]
        locations[end - 1] = location
        scales[end - 1] = max(float(np.mean(tail - location)), 1e-8)
        tail_counts[end - 1] = len(tail)
    return locations, scales, tail_counts


def _ei_curve(rewards: np.ndarray, best_rewards: np.ndarray, calibration: Calibration,
              prior: DistributionPrior, config: PolicyConfig, delta: float) -> np.ndarray:
    n_total = len(rewards)
    ns = np.arange(1, n_total + 1, dtype=np.float64)
    reward_sum = np.cumsum(rewards, dtype=np.float64)
    reward_sumsq = np.cumsum(rewards * rewards, dtype=np.float64)
    local_mean = reward_sum / ns
    local_variance = np.maximum(reward_sumsq / ns - local_mean * local_mean, 1e-8)
    if n_total > 1:
        local_variance[1:] *= ns[1:] / (ns[1:] - 1.0)
    confidence = config.confidence_scale
    prior_strength = config.reward_prior_strength
    if math.isinf(prior_strength):
        reference_mean = np.full(n_total, prior.gaussian_mean, dtype=np.float64)
        reference_variance = np.full(
            n_total, prior.gaussian_variance, dtype=np.float64
        )
    else:
        denominator = ns + prior_strength
        reference_mean = (
            ns * local_mean + prior_strength * prior.gaussian_mean
        ) / denominator
        reference_variance = (
            ns * local_variance + prior_strength * prior.gaussian_variance
        ) / denominator
    reference_sigma = np.sqrt(np.maximum(reference_variance, 1e-8))

    def calibrated(values):
        values = np.asarray(values, dtype=np.float64)
        if calibration.kind not in ("relative_bt", "contextual_relative_bt"):
            return np.asarray(calibration(values), dtype=np.float64)
        extra = values.ndim - 1
        center = reference_mean.reshape((n_total,) + (1,) * extra)
        scale = reference_sigma.reshape((n_total,) + (1,) * extra)
        relative = (values - center) / scale
        if calibration.kind == "relative_bt":
            return np.asarray(calibration(relative), dtype=np.float64)
        context = (
            (center - calibration.context_mean_center) /
            calibration.context_mean_scale
        )
        return np.asarray(expit(
            calibration.intercept + calibration.slope * relative +
            calibration.context_mean_coef * context
        ), dtype=np.float64)

    current = calibrated(best_rewards)
    bounded_probability = calibration.kind == "bounded_identity"

    if config.family == "gaussian":
        mean = reference_mean
        sigma = reference_sigma
        radius = np.sqrt(math.log(1.0 / delta) / ns)
        mean_ucb = mean + confidence * sigma * radius
        sigma_ucb = sigma * (1.0 + confidence * radius)
        if bounded_probability:
            lower = ndtr((0.0 - mean_ucb) / sigma_ucb)
            upper = ndtr((1.0 - mean_ucb) / sigma_ucb)
            mass = upper - lower
            probability = lower[:, None] + mass[:, None] * _UNIT_X
            future = mean_ucb[:, None] + sigma_ucb[:, None] * ndtri(
                np.clip(probability, 1e-15, 1.0 - 1e-15)
            )
            degenerate = mass <= 1e-14
            if np.any(degenerate):
                future[degenerate] = np.clip(
                    mean_ucb[degenerate, None], 0.0, 1.0
                )
            future = np.clip(future, 0.0, 1.0)
            future_value = calibrated(future)
            return np.sum(
                _UNIT_W * np.maximum(future_value - current[:, None], 0.0),
                axis=1,
            )
        future = mean_ucb[:, None] + math.sqrt(2.0) * sigma_ucb[:, None] * _GH_X
        future_value = calibrated(future)
        return np.sum(_GH_W * np.maximum(future_value - current[:, None], 0.0), axis=1)

    if config.family == "gaussian_kde":
        # Prefix KDE represented by equal-mass empirical quantiles.  Each
        # center carries a Gaussian kernel with Scott bandwidth.  The UCB
        # shifts the centers upward and inflates the bandwidth; an optional
        # train prior is mixed in as pseudo-observations rather than being
        # mistaken for locally observed test rewards.
        center_probabilities = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
        centers = np.empty((n_total, len(center_probabilities)), dtype=np.float64)
        for end in range(1, n_total + 1):
            centers[end - 1] = np.quantile(rewards[:end], center_probabilities)
        sigma = np.sqrt(np.maximum(local_variance, 1e-8))
        radius = np.sqrt(math.log(1.0 / delta) / ns)
        bandwidth = 1.06 * sigma * np.power(ns, -0.2)
        bandwidth_ucb = np.maximum(bandwidth * (1.0 + confidence * radius), 1e-8)
        center_ucb = centers + (confidence * sigma * radius)[:, None]
        future = (center_ucb[:, :, None] + math.sqrt(2.0) *
                  bandwidth_ucb[:, None, None] * _KDE_X[None, None, :])
        future_value = calibrated(future)
        local_gain = np.mean(
            np.sum(_KDE_W * np.maximum(
                future_value - current[:, None, None], 0.0
            ), axis=2), axis=1,
        )
        prior_sigma = math.sqrt(prior.gaussian_variance)
        prior_mean_ucb = prior.gaussian_mean + confidence * prior_sigma * radius
        prior_sigma_ucb = prior_sigma * (1.0 + confidence * radius)
        prior_future = (prior_mean_ucb[:, None] + math.sqrt(2.0) *
                        prior_sigma_ucb[:, None] * _GH_X[None, :])
        prior_value = calibrated(prior_future)
        prior_gain = np.sum(
            _GH_W * np.maximum(prior_value - current[:, None], 0.0), axis=1
        )
        local_weight = ns / (ns + prior_strength)
        return local_weight * local_gain + (1.0 - local_weight) * prior_gain

    quantile = float(config.tail_quantile)
    if math.isinf(prior_strength):
        # The simple global policy needs only the number of observations in
        # each inclusive upper tail to set its confidence schedule.  Its
        # location and scale are exactly the training fit, so do not form
        # unused local location/scale estimates.
        tail_counts = np.empty(n_total, dtype=np.float64)
        for end in range(1, n_total + 1):
            observed = rewards[:end]
            threshold = float(np.quantile(observed, quantile))
            tail_counts[end - 1] = np.count_nonzero(observed >= threshold)
        location = np.full(
            n_total, prior.exp_locations[quantile], dtype=np.float64
        )
        scale = np.full(
            n_total, prior.exp_scales[quantile], dtype=np.float64
        )
    else:
        locations, scales, tail_counts = _exp_prefix_stats(rewards, quantile)
        location = (
            ns * locations + prior_strength * prior.exp_locations[quantile]
        ) / (ns + prior_strength)
        scale = (
            tail_counts * scales + prior_strength * prior.exp_scales[quantile]
        ) / (tail_counts + prior_strength)
    radius = np.sqrt(math.log(1.0 / delta) / np.maximum(tail_counts, 1.0))
    location_ucb = location + confidence * scale * radius
    scale_ucb = scale * (1.0 + confidence * radius)
    if bounded_probability:
        location_ucb = np.clip(location_ucb, 0.0, 1.0)
        room = 1.0 - location_ucb
        truncated_mass = -np.expm1(-room / np.maximum(scale_ucb, 1e-12))
        future = location_ucb[:, None] - scale_ucb[:, None] * np.log1p(
            -truncated_mass[:, None] * _UNIT_X
        )
        future_value = calibrated(np.clip(future, 0.0, 1.0))
        tail_gain = np.sum(
            _UNIT_W * np.maximum(future_value - current[:, None], 0.0),
            axis=1,
        )
        return (1.0 - quantile) * tail_gain
    future = location_ucb[:, None] + scale_ucb[:, None] * _LAG_X
    future_value = calibrated(future)
    tail_gain = np.sum(_LAG_W * np.maximum(future_value - current[:, None], 0.0), axis=1)
    return (1.0 - quantile) * tail_gain


def pandora_stop_from_curve(ei: np.ndarray, cumulative_chars: np.ndarray,
                            divisor: float, prior_mean_chars: float,
                            config: PolicyConfig, fixed_n: int,
                            min_open: int = 3) -> int:
    cap = len(ei) if config.cap_factor is None else max(
        min_open, int(math.ceil(config.cap_factor * fixed_n))
    )
    cap = min(cap, len(ei))
    n = min(min_open, cap)
    while n < cap:
        expected_chars = (
            cumulative_chars[n - 1] + config.cost_prior_strength * prior_mean_chars
        ) / (n + config.cost_prior_strength)
        tail_multiplier = (n / max(min_open, 1)) ** (-config.tail_decay)
        if ei[n - 1] * tail_multiplier <= expected_chars / divisor:
            break
        n += 1
    return n


def _evaluate_problem(task):
    (problem_id, rewards, correct, chars, permutations, configs, calibrations,
     prior, divisors, fixed_ns, delta, reward_transform) = task
    result = np.empty((len(configs), len(divisors), len(permutations), 4), dtype=np.float64)
    core_keys = [(
        config.family, config.calibration, config.confidence_scale,
        config.reward_prior_strength, config.tail_quantile,
    ) for config in configs]
    for permutation_id, permutation in enumerate(permutations):
        ordered_rewards = rewards[permutation]
        if reward_transform is not None:
            ordered_rewards = pilot_transform_rewards(
                rewards, permutation, reward_transform, 3
            )
        ordered_correct = correct[permutation]
        ordered_chars = chars[permutation]
        cumulative_chars = np.cumsum(ordered_chars, dtype=np.float64)
        best_rewards, success = _prefix_success(ordered_rewards, ordered_correct)
        curves = {}
        for config, key in zip(configs, core_keys):
            if key not in curves:
                curves[key] = _ei_curve(
                    ordered_rewards, best_rewards, calibrations[config.calibration],
                    prior, config, delta,
                )
        for config_id, (config, key) in enumerate(zip(configs, core_keys)):
            for divisor_id, divisor in enumerate(divisors):
                opened = pandora_stop_from_curve(
                    curves[key], cumulative_chars, divisor, prior.mean_chars,
                    config, fixed_ns[divisor], 3,
                )
                result[config_id, divisor_id, permutation_id] = (
                    success[opened - 1], cumulative_chars[opened - 1], opened,
                    best_rewards[opened - 1],
                )
    return problem_id, result


def evaluate(problems, permutations, configs, calibrations, prior, divisors,
             fixed_ns, delta, workers, reward_transform=None) -> np.ndarray:
    tasks = [(
        problem_id, *problems[problem_id], permutations[problem_id], configs,
        calibrations, prior, divisors, fixed_ns, delta, reward_transform,
    ) for problem_id in sorted(problems)]
    pieces = []
    if workers == 1:
        pieces = [_evaluate_problem(task)[1] for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for _, values in pool.map(_evaluate_problem, tasks, chunksize=1):
                pieces.append(values)
    # config x divisor x (problem * permutation) x metric
    return np.concatenate(pieces, axis=2)


def crossfit_evaluate(problems, permutations, configs, divisors, fixed_ns,
                      delta, workers, seed, folds=4) -> np.ndarray:
    """Out-of-fold train predictions for honest policy selection.

    Each training problem is evaluated exactly once with calibrations and
    distribution priors fitted without that problem.  The full training half
    is used only after selection, when fitting the frozen held-out policy.
    """
    problem_ids = np.asarray(sorted(problems))
    fold_ids = np.array_split(
        np.random.default_rng(seed).permutation(problem_ids),
        min(folds, len(problem_ids)),
    )
    pieces = []
    for fold_number, validation_ids in enumerate(fold_ids, start=1):
        validation_set = set(validation_ids.tolist())
        fit = {pid: problems[pid] for pid in problem_ids if pid not in validation_set}
        validation = {pid: problems[pid] for pid in validation_ids}
        validation_permutations = {pid: permutations[pid] for pid in validation_ids}
        calibrations = fit_config_calibrations(fit, configs)
        prior = fit_prior(fit)
        pieces.append(evaluate(
            validation, validation_permutations, configs, calibrations, prior,
            divisors, fixed_ns, delta, workers,
        ))
        print(f"[coding-ucb] train fold {fold_number}/{len(fold_ids)} complete", flush=True)
    return np.concatenate(pieces, axis=2)


def fixed_trials(problems, permutations) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accuracy, chars, rewards = [], [], []
    for problem_id in sorted(problems):
        problem_rewards, correct, problem_chars = problems[problem_id]
        for permutation in permutations[problem_id]:
            ordered_rewards = problem_rewards[permutation]
            ordered_correct = correct[permutation]
            ordered_chars = problem_chars[permutation]
            best_reward, success = _prefix_success(ordered_rewards, ordered_correct)
            accuracy.append(success)
            chars.append(np.cumsum(ordered_chars, dtype=np.float64))
            rewards.append(best_reward)
    return np.asarray(accuracy), np.asarray(chars), np.asarray(rewards)


def mixed_index(curve: np.ndarray, target: float) -> tuple[int, int, float]:
    curve = np.maximum.accumulate(np.asarray(curve, dtype=np.float64))
    if target <= curve[0]:
        return 0, 0, 0.0
    if target >= curve[-1]:
        return len(curve) - 1, len(curve) - 1, 0.0
    high = int(np.searchsorted(curve, target, side="left"))
    low = high - 1
    if curve[high] <= curve[low] + 1e-15:
        return high, high, 0.0
    weight = (target - curve[low]) / (curve[high] - curve[low])
    return low, high, float(np.clip(weight, 0.0, 1.0))


def mixed_mean(values: np.ndarray, mix: tuple[int, int, float]) -> float:
    low, high, weight = mix
    return float(np.mean((1.0 - weight) * values[..., low] + weight * values[..., high]))


def best_policy_mix(qualities: np.ndarray, costs: np.ndarray, target: float):
    """Minimum-cost randomized mixture of two UCB policies at target quality."""
    qualities = np.asarray(qualities, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    low_ids = np.flatnonzero(qualities <= target + 1e-12)
    high_ids = np.flatnonzero(qualities >= target - 1e-12)
    if not len(low_ids) or not len(high_ids):
        closest = int(np.argmin(np.abs(qualities - target)))
        return closest, closest, 0.0, False
    best = None
    for low in low_ids:
        q_low = qualities[low]
        q_high = qualities[high_ids]
        denominator = q_high - q_low
        weights = np.divide(
            target - q_low, denominator,
            out=np.zeros_like(denominator), where=np.abs(denominator) > 1e-15,
        )
        weights = np.clip(weights, 0.0, 1.0)
        mixed_cost = (1.0 - weights) * costs[low] + weights * costs[high_ids]
        position = int(np.argmin(mixed_cost))
        candidate = (float(mixed_cost[position]), int(low), int(high_ids[position]),
                     float(weights[position]))
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best[1], best[2], best[3], True


def best_single_config_policy_mix(
    qualities: np.ndarray,
    costs: np.ndarray,
    config_ids,
    divisor_ids,
    target: float,
):
    """Minimum-cost target mix along one UCB model's Lagrange path.

    Hyperparameters may be selected on training data, but the two randomized
    endpoints must share every UCB/distribution parameter and differ only in
    the cost divisor.  This prevents an unstable mixture of unrelated models
    that happened to bracket the target on one split.
    """
    qualities = np.asarray(qualities, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    divisor_ids = np.asarray(tuple(divisor_ids), dtype=np.int64)
    best = None
    fallback = None
    for config_id in config_ids:
        local_quality = qualities[config_id, divisor_ids]
        local_cost = costs[config_id, divisor_ids]
        low, high, weight, reached = best_policy_mix(
            local_quality, local_cost, target
        )
        mixed_quality = (
            (1.0 - weight) * local_quality[low] + weight * local_quality[high]
        )
        mixed_cost = (
            (1.0 - weight) * local_cost[low] + weight * local_cost[high]
        )
        candidate = (
            float(mixed_cost), int(config_id), int(divisor_ids[low]),
            int(divisor_ids[high]), float(weight), bool(reached),
        )
        if reached and (best is None or candidate[:2] < best[:2]):
            best = candidate
        fallback_key = (abs(float(mixed_quality) - target), float(mixed_cost),
                        int(config_id))
        if fallback is None or fallback_key < fallback[0]:
            fallback = (fallback_key, candidate)
    if best is not None:
        return best[1], best[2], best[3], best[4], best[5]
    candidate = fallback[1]
    return candidate[1], candidate[2], candidate[3], candidate[4], False


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) /
                 math.sqrt(len(values)))
    return mean, mean - half, mean + half


def summarize(rows, keys, metrics):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        record = dict(zip(keys, key))
        record["splits"] = len(group)
        for metric in metrics:
            mean, low, high = interval([row[metric] for row in group])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        output.append(record)
    return output


def write_csv(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _policy_from_flat(flat_id, n_configs, n_divisors):
    return int(flat_id // n_divisors), int(flat_id % n_divisors)


def run_split(split: int, problems, configs, all_divisors, args):
    raw_train, raw_test = split_problems(problems, args.seed + split)
    calibration_record = None
    if args.calibrated_reward_space:
        reward_calibration = fit_reward_space_isotonic(raw_train)
        calibration_record = reward_space_calibration_record(
            split, raw_train, raw_test, reward_calibration
        )
        train = transform_reward_space(raw_train, reward_calibration)
        test = transform_reward_space(raw_test, reward_calibration)
    else:
        train, test = raw_train, raw_test
    train_fixed_accuracy, train_fixed_chars, fixed_ns = fixed_curves(train, all_divisors)
    train_permutations = make_permutations(
        train, args.train_permutations, args.seed + 101_003 * split + 17
    )
    test_permutations = make_permutations(
        test, args.test_permutations, args.seed + 101_003 * split + 31
    )
    opponent_permutations = make_permutations(
        test, args.test_permutations, args.seed + 101_003 * split + 47
    )
    train_values = crossfit_evaluate(
        train, train_permutations, configs, all_divisors, fixed_ns,
        args.delta, args.workers, args.seed + 17_003 * split + 71,
    )
    train_acc = train_values[..., 0].mean(axis=2)
    train_chars = train_values[..., 1].mean(axis=2)
    train_utility = train_acc - train_chars / np.asarray(all_divisors)[None, :]

    selections = {}
    utility_tuning_rows = []
    active_methods = tuple(
        method for method in METHODS[:-1]
        if any(config.method == method for config in configs)
    ) + ("selected_ucb",)
    selection_divisors = tuple(sorted(set(REPORT_DIVISORS) | set(FRONTIER_DIVISORS)))
    selected_method_by_divisor = {
        divisor: (ROBUST_METHOD_BY_DIVISOR.get(divisor)
                  if ROBUST_METHOD_BY_DIVISOR.get(divisor) in active_methods[:-1]
                  else active_methods[0])
        for divisor in selection_divisors
    }
    for divisor in selection_divisors:
        divisor_id = all_divisors.index(divisor)
        for method in active_methods[:-1]:
            eligible = [i for i, config in enumerate(configs) if config.method == method]
            if args.frozen_exponential_configs and method == "shifted_exponential_relative_bt":
                frozen = FROZEN_EXP_CONFIG_BY_DIVISOR[divisor]
                eligible = [i for i in eligible if all(
                    getattr(configs[i], name) == value for name, value in frozen.items()
                )]
                if len(eligible) != 1:
                    raise ValueError(f"expected one frozen exponential config, found {eligible}")
                selected = eligible[0]
            elif (args.frozen_contextual_configs and
                  method == "shifted_exponential_contextual_relative_bt"):
                frozen = FROZEN_CONTEXTUAL_EXP_CONFIG_BY_DIVISOR[divisor]
                eligible = [i for i in eligible if all(
                    getattr(configs[i], name) == value for name, value in frozen.items()
                )]
                if len(eligible) != 1:
                    raise ValueError(
                        f"expected one frozen contextual exponential config, found {eligible}"
                    )
                selected = eligible[0]
            else:
                selected = max(eligible, key=lambda i: train_utility[i, divisor_id])
            selections[(method, divisor)] = selected
        robust_method = selected_method_by_divisor[divisor]
        eligible = [i for i, config in enumerate(configs) if config.method == robust_method]
        if args.calibrated_reward_space:
            eligible = list(range(len(configs)))
            selected = max(eligible, key=lambda i: train_utility[i, divisor_id])
        elif args.frozen_exponential_configs and robust_method == "shifted_exponential_relative_bt":
            frozen = FROZEN_EXP_CONFIG_BY_DIVISOR[divisor]
            eligible = [i for i in eligible if all(
                getattr(configs[i], name) == value for name, value in frozen.items()
            )]
            selected = eligible[0]
        elif (args.frozen_contextual_configs and
              robust_method == "shifted_exponential_contextual_relative_bt"):
            frozen = FROZEN_CONTEXTUAL_EXP_CONFIG_BY_DIVISOR[divisor]
            eligible = [i for i in eligible if all(
                getattr(configs[i], name) == value for name, value in frozen.items()
            )]
            if len(eligible) != 1:
                raise ValueError(
                    f"expected one frozen contextual exponential config, found {eligible}"
                )
            selected = eligible[0]
        elif not args.calibrated_reward_space:
            selected = max(eligible, key=lambda i: train_utility[i, divisor_id])
        selections[("selected_ucb", divisor)] = selected
        for method in active_methods:
            config_id = selections[(method, divisor)]
            utility_tuning_rows.append({
                "split": split, "method": method, "divisor": divisor,
                "config_id": config_id, **asdict(configs[config_id]),
                "train_accuracy": train_acc[config_id, divisor_id],
                "train_chars": train_chars[config_id, divisor_id],
                "train_utility": train_utility[config_id, divisor_id],
                "fixed_n": fixed_ns[divisor],
            })

    target_choices = {}
    target_train_metrics = {}
    flat_quality = train_acc.reshape(-1)
    flat_chars = train_chars.reshape(-1)
    if args.skip_target_accuracy:
        pass
    elif args.target_specific_search:
        target_divisor_ids = [all_divisors.index(divisor)
                              for divisor in args.target_divisors]
        for method in active_methods:
            if method == "selected_ucb":
                eligible = list(range(len(configs)))
            else:
                eligible = [i for i, config in enumerate(configs)
                            if config.method == method]
            if args.target_upper_tail_only:
                eligible = [i for i in eligible
                            if configs[i].family == "shifted_exponential"
                            and configs[i].tail_quantile == 0.75]
            if not eligible:
                raise ValueError(f"no target-quality configs available for {method}")
            for target in TARGETS:
                config_id, low_divisor, high_divisor, weight, reached = (
                    best_single_config_policy_mix(
                        train_acc, train_chars, eligible, target_divisor_ids, target
                    )
                )
                low_flat = config_id * len(all_divisors) + low_divisor
                high_flat = config_id * len(all_divisors) + high_divisor
                target_choices[(method, target)] = (
                    low_flat, high_flat, weight, reached
                )
                target_train_metrics[(method, target)] = (
                    float((1.0 - weight) * train_acc[config_id, low_divisor] +
                          weight * train_acc[config_id, high_divisor]),
                    float((1.0 - weight) * train_chars[config_id, low_divisor] +
                          weight * train_chars[config_id, high_divisor]),
                )
    else:
        # Backward-compatible target report for the utility-frozen policy.
        method_flat = {}
        for method in active_methods:
            mask = np.zeros(len(configs) * len(all_divisors), dtype=bool)
            for divisor in selection_divisors:
                divisor_id = all_divisors.index(divisor)
                config_id = selections[(method, divisor)]
                mask[config_id * len(all_divisors) + divisor_id] = True
            method_flat[method] = mask
        for method in active_methods:
            ids = np.flatnonzero(method_flat[method])
            for target in TARGETS:
                low_local, high_local, weight, reached = best_policy_mix(
                    flat_quality[ids], flat_chars[ids], target
                )
                low_flat, high_flat = int(ids[low_local]), int(ids[high_local])
                target_choices[(method, target)] = (
                    low_flat, high_flat, weight, reached
                )
                target_train_metrics[(method, target)] = (
                    float((1.0 - weight) * flat_quality[low_flat] +
                          weight * flat_quality[high_flat]),
                    float((1.0 - weight) * flat_chars[low_flat] +
                          weight * flat_chars[high_flat]),
                )

    selected_config_ids = sorted(set(selections.values()) | {
        _policy_from_flat(flat_id, len(configs), len(all_divisors))[0]
        for low, high, _, _ in target_choices.values() for flat_id in (low, high)
    })
    if args.diagnostic_all_configs:
        selected_config_ids = list(range(len(configs)))
    selected_configs = [configs[index] for index in selected_config_ids]
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(selected_config_ids)}
    calibrations = fit_config_calibrations(train, selected_configs)
    prior = fit_prior(train)
    test_values = evaluate(
        test, test_permutations, selected_configs, calibrations, prior,
        all_divisors, fixed_ns, args.delta, args.workers,
    )
    test_fixed_acc, test_fixed_chars, _ = fixed_trials(test, test_permutations)
    opponent_acc, opponent_chars, _ = fixed_trials(test, opponent_permutations)
    opponent_acc_curve = opponent_acc.mean(axis=0)
    opponent_char_curve = opponent_chars.mean(axis=0)
    test_fixed_acc_curve = test_fixed_acc.mean(axis=0)
    test_fixed_char_curve = test_fixed_chars.mean(axis=0)

    utility_rows, budget_rows, target_rows, diagnostic_rows = [], [], [], []
    for divisor in REPORT_DIVISORS:
        divisor_id = all_divisors.index(divisor)
        fixed_n = fixed_ns[divisor]
        fixed_utility_trials = (
            test_fixed_acc[:, fixed_n - 1] - test_fixed_chars[:, fixed_n - 1] / divisor
        )
        fixed_utility = float(np.mean(fixed_utility_trials))
        for method in active_methods:
            global_id = selections[(method, divisor)]
            local_id = global_to_local[global_id]
            values = test_values[local_id, divisor_id]
            adaptive_utility = values[:, 0] - values[:, 1] / divisor
            adaptive_mean = float(np.mean(adaptive_utility))
            utility_rows.append({
                "split": split, "method": method, "divisor": divisor,
                "adaptive_utility": adaptive_mean, "fixed_utility": fixed_utility,
                "utility_gap": adaptive_mean - fixed_utility,
                "relative_utility_gain_pct": (
                    100.0 * (adaptive_mean - fixed_utility) / fixed_utility
                ),
                "adaptive_accuracy": float(np.mean(values[:, 0])),
                "adaptive_chars": float(np.mean(values[:, 1])),
                "adaptive_opens": float(np.mean(values[:, 2])),
                "fixed_n": fixed_n, "config_id": global_id,
            })

            adaptive_chars = float(np.mean(values[:, 1]))
            budget_mix = mixed_index(opponent_char_curve, adaptive_chars)
            matched_accuracy = mixed_mean(opponent_acc, budget_mix)
            matched_chars = mixed_mean(opponent_chars, budget_mix)
            adaptive_accuracy = float(np.mean(values[:, 0]))
            budget_rows.append({
                "split": split, "method": method, "divisor": divisor,
                "adaptive_accuracy": adaptive_accuracy,
                "fixed_accuracy": matched_accuracy,
                "accuracy_gap_pp": 100.0 * (adaptive_accuracy - matched_accuracy),
                "adaptive_chars": adaptive_chars, "fixed_chars": matched_chars,
                "cost_mismatch": adaptive_chars - matched_chars,
                "fixed_n_low": budget_mix[0] + 1,
                "fixed_n_high": budget_mix[1] + 1,
                "fixed_high_weight": budget_mix[2],
            })

        if args.diagnostic_all_configs:
            for global_id, config in enumerate(configs):
                local_id = global_to_local[global_id]
                values = test_values[local_id, divisor_id]
                adaptive_utility = values[:, 0] - values[:, 1] / divisor
                adaptive_mean = float(np.mean(adaptive_utility))
                adaptive_chars = float(np.mean(values[:, 1]))
                budget_mix = mixed_index(opponent_char_curve, adaptive_chars)
                matched_accuracy = mixed_mean(opponent_acc, budget_mix)
                adaptive_accuracy = float(np.mean(values[:, 0]))
                diagnostic_rows.append({
                    "split": split, "divisor": divisor, "config_id": global_id,
                    **asdict(config), "fixed_n": fixed_n,
                    "adaptive_utility": adaptive_mean, "fixed_utility": fixed_utility,
                    "utility_gap": adaptive_mean - fixed_utility,
                    "relative_utility_gain_pct": (
                        100.0 * (adaptive_mean - fixed_utility) / fixed_utility
                    ),
                    "adaptive_accuracy": adaptive_accuracy,
                    "adaptive_chars": adaptive_chars,
                    "adaptive_opens": float(np.mean(values[:, 2])),
                    "equal_budget_fixed_accuracy": matched_accuracy,
                    "equal_budget_accuracy_gap_pp": 100.0 * (
                        adaptive_accuracy - matched_accuracy
                    ),
                })

    for method in active_methods:
        for target in (() if args.skip_target_accuracy else TARGETS):
            low_flat, high_flat, weight, reached = target_choices[(method, target)]
            train_target_accuracy, train_target_chars = target_train_metrics[(method, target)]
            low_config, low_divisor = _policy_from_flat(
                low_flat, len(configs), len(all_divisors)
            )
            high_config, high_divisor = _policy_from_flat(
                high_flat, len(configs), len(all_divisors)
            )
            low_values = test_values[global_to_local[low_config], low_divisor]
            high_values = test_values[global_to_local[high_config], high_divisor]
            adaptive_accuracy = float(np.mean(
                (1.0 - weight) * low_values[:, 0] + weight * high_values[:, 0]
            ))
            adaptive_chars = float(np.mean(
                (1.0 - weight) * low_values[:, 1] + weight * high_values[:, 1]
            ))
            adaptive_opens = float(np.mean(
                (1.0 - weight) * low_values[:, 2] + weight * high_values[:, 2]
            ))
            fixed_low, fixed_high, fixed_weight, fixed_reached = best_policy_mix(
                train_fixed_accuracy, train_fixed_chars, target
            )
            fixed_mix = (fixed_low, fixed_high, fixed_weight)
            train_fixed_target_chars = float(
                (1.0 - fixed_weight) * train_fixed_chars[fixed_low] +
                fixed_weight * train_fixed_chars[fixed_high]
            )
            fixed_accuracy = mixed_mean(test_fixed_acc, fixed_mix)
            fixed_chars = mixed_mean(test_fixed_chars, fixed_mix)
            matched_low, matched_high, matched_weight, matched_reached = best_policy_mix(
                test_fixed_acc_curve, test_fixed_char_curve, adaptive_accuracy
            )
            heldout_mix = (matched_low, matched_high, matched_weight)
            matched_fixed_accuracy = mixed_mean(test_fixed_acc, heldout_mix)
            matched_fixed_chars = mixed_mean(test_fixed_chars, heldout_mix)
            matched_fixed_opens = (
                (1.0 - matched_weight) * (matched_low + 1) +
                matched_weight * (matched_high + 1)
            )
            fixed_opens = (
                (1.0 - fixed_weight) * (fixed_low + 1) +
                fixed_weight * (fixed_high + 1)
            )
            target_config = configs[low_config]
            target_rows.append({
                "split": split, "method": method, "target_accuracy": target,
                "target_reached_train": reached,
                "adaptive_train_accuracy": train_target_accuracy,
                "adaptive_train_chars": train_target_chars,
                "train_fixed_target_chars": train_fixed_target_chars,
                "train_estimated_char_saving_pct": 100.0 * (
                    train_fixed_target_chars - train_target_chars
                ) / train_fixed_target_chars,
                "adaptive_accuracy": adaptive_accuracy,
                "fixed_accuracy": fixed_accuracy,
                "accuracy_gap_pp": 100.0 * (adaptive_accuracy - fixed_accuracy),
                "adaptive_chars": adaptive_chars, "adaptive_opens": adaptive_opens,
                "fixed_chars": fixed_chars, "fixed_opens": fixed_opens,
                "direct_char_saving_pct": 100.0 * (fixed_chars - adaptive_chars) / fixed_chars,
                "matched_fixed_chars": matched_fixed_chars,
                "matched_fixed_opens": matched_fixed_opens,
                "matched_fixed_accuracy": matched_fixed_accuracy,
                "matched_accuracy_error": adaptive_accuracy - matched_fixed_accuracy,
                "matched_target_reached": matched_reached,
                "matched_char_saving_pct": (
                    100.0 * (matched_fixed_chars - adaptive_chars) / matched_fixed_chars
                ),
                "adaptive_low_config": low_config,
                "adaptive_low_divisor": all_divisors[low_divisor],
                "adaptive_high_config": high_config,
                "adaptive_high_divisor": all_divisors[high_divisor],
                "adaptive_high_weight": weight,
                "single_config_target_path": low_config == high_config,
                "target_family": target_config.family,
                "target_calibration": target_config.calibration,
                "target_confidence_scale": target_config.confidence_scale,
                "target_reward_prior_strength": target_config.reward_prior_strength,
                "target_cost_prior_strength": target_config.cost_prior_strength,
                "target_cap_factor": target_config.cap_factor,
                "target_tail_quantile": target_config.tail_quantile,
                "fixed_n_low": fixed_mix[0] + 1,
                "fixed_n_high": fixed_mix[1] + 1,
                "fixed_high_weight": fixed_mix[2],
                "fixed_target_reached_train": fixed_reached,
                "matched_fixed_n_low": heldout_mix[0] + 1,
                "matched_fixed_n_high": heldout_mix[1] + 1,
                "matched_fixed_high_weight": heldout_mix[2],
            })
    print(f"[coding-ucb] split {split} complete", flush=True)
    return (
        utility_rows, budget_rows, target_rows, utility_tuning_rows,
        diagnostic_rows, calibration_record,
    )


def band(ax, x, mean, low, high, method, linewidth=2.0):
    ax.plot(x, mean, marker="o", color=COLORS[method], linewidth=linewidth,
            label=LABELS[method], zorder=3 if method == "selected_ucb" else 2)
    ax.fill_between(x, low, high, color=COLORS[method], alpha=0.12)


def plot_methods(summary):
    """Avoid drawing a family and selected-policy series twice when identical."""
    present = [method for method in METHODS
               if any(row["method"] == method for row in summary)]
    if len(present) == 2 and "selected_ucb" in present:
        return ("selected_ucb",)
    return tuple(present)


def plot_utility(summary, path):
    fig, ax = plt.subplots(figsize=(8.8, 5.5), constrained_layout=True)
    for method in plot_methods(summary):
        group = sorted((row for row in summary if row["method"] == method),
                       key=lambda row: row["divisor"])
        if not group:
            continue
        x = np.asarray([row["divisor"] for row in group])
        band(ax, x,
             np.asarray([row["relative_utility_gain_pct_mean"] for row in group]),
             np.asarray([row["relative_utility_gain_pct_ci_low"] for row in group]),
             np.asarray([row["relative_utility_gain_pct_ci_high"] for row in group]),
             method, 2.8 if method == "selected_ucb" else 1.7)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set(xlabel="Character-cost divisor", ylabel="Utility gain over tuned Fixed-N (%)",
           title="Coding UCB Pandora utility advantage; 95% CI across splits")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_equal_budget(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    for method in plot_methods(summary):
        group = sorted((row for row in summary if row["method"] == method),
                       key=lambda row: row["divisor"])
        if not group:
            continue
        x = np.asarray([row["divisor"] for row in group])
        band(axes[0], x,
             np.asarray([row["accuracy_gap_pp_mean"] for row in group]),
             np.asarray([row["accuracy_gap_pp_ci_low"] for row in group]),
             np.asarray([row["accuracy_gap_pp_ci_high"] for row in group]),
             method, 2.8 if method == "selected_ucb" else 1.7)
        if method == "selected_ucb":
            band(axes[1], x,
                 np.asarray([100.0 * row["adaptive_accuracy_mean"] for row in group]),
                 np.asarray([100.0 * row["adaptive_accuracy_ci_low"] for row in group]),
                 np.asarray([100.0 * row["adaptive_accuracy_ci_high"] for row in group]),
                 method, 2.8)
            axes[1].plot(x, [100.0 * row["fixed_accuracy_mean"] for row in group],
                         marker="s", color="#555555", linewidth=2.0,
                         label="Fixed N at adaptive characters")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_xlabel("Character-cost divisor")
        ax.legend(frameon=False, ncol=2, fontsize=8)
    axes[0].set(ylabel="Adaptive accuracy advantage (percentage points)",
                title="Equal-character accuracy advantage")
    axes[1].set(ylabel="Accuracy (%)", title="Train-selected UCB vs equal-budget Fixed N")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_target(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    for method in plot_methods(summary):
        group = sorted((row for row in summary if row["method"] == method),
                       key=lambda row: row["target_accuracy"])
        if not group:
            continue
        x = np.asarray([row["target_accuracy"] for row in group])
        band(axes[0], x,
             np.asarray([row["matched_char_saving_pct_mean"] for row in group]),
             np.asarray([row["matched_char_saving_pct_ci_low"] for row in group]),
             np.asarray([row["matched_char_saving_pct_ci_high"] for row in group]),
             method, 2.8 if method == "selected_ucb" else 1.7)
        if method == "selected_ucb":
            band(axes[1], x,
                 np.asarray([row["adaptive_accuracy_mean"] for row in group]),
                 np.asarray([row["adaptive_accuracy_ci_low"] for row in group]),
                 np.asarray([row["adaptive_accuracy_ci_high"] for row in group]),
                 method, 2.8)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].plot(TARGETS, TARGETS, color="black", linestyle="--", linewidth=1.0,
                 label="Achieved = target")
    axes[0].set(xlabel="Requested adaptive target accuracy",
                ylabel="Character saving over achieved-accuracy-matched Fixed N (%)",
                title="Cost saving at the adaptive policy's achieved quality")
    axes[1].set(xlabel="Target accuracy", ylabel="Held-out adaptive accuracy",
                title="Train-selected UCB target tracking")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path("algorithm/bestofn_coding/data.jsonl"))
    parser.add_argument("--char-cache", type=Path, default=Path(
        "algorithm/bestofn_coding/practical_algorithm/coding_char_counts_83.npz"
    ))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("codex_results/results/coding/utility_gap"))
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--split-start", type=int, default=40)
    parser.add_argument("--train-permutations", type=int, default=4)
    parser.add_argument("--test-permutations", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--focused", action="store_true")
    parser.add_argument("--expanded-exponential", action="store_true")
    parser.add_argument("--contextual-relative-bt", action="store_true")
    parser.add_argument("--diagnostic-all-configs", action="store_true")
    parser.add_argument("--exponential-only", action="store_true")
    parser.add_argument("--frozen-exponential-configs", action="store_true")
    parser.add_argument("--frozen-contextual-configs", action="store_true")
    parser.add_argument(
        "--calibrated-reward-space", action="store_true",
        help=("fit isotonic P(correct | raw reward) on each outer training half, "
              "transform train/test rewards with the frozen map, and fit the "
              "stopping distribution in calibrated probability space"),
    )
    parser.add_argument(
        "--bounded-probability-models", action="store_true",
        help=("when using calibrated reward space, fit Gaussian/exponential "
              "distributions with probability support truncated to [0, 1]"),
    )
    parser.add_argument(
        "--skip-target-accuracy", action="store_true",
        help=("run only utility-gap and equal-budget accuracy objectives; "
              "do not select, evaluate, or plot target-accuracy policies"),
    )
    parser.add_argument(
        "--target-specific-search", action="store_true",
        help=("select minimum-character target policies on cross-fitted train data "
              "along a dense UCB cost-divisor path"),
    )
    parser.add_argument(
        "--target-upper-tail-only", action="store_true",
        help="restrict target-policy model selection to the 0.75 exponential tail",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    problems = load_problems(args.data, args.char_cache)
    if sum((args.quick, args.focused, args.expanded_exponential)) > 1:
        parser.error("--quick, --focused, and --expanded-exponential are mutually exclusive")
    if args.contextual_relative_bt and not args.expanded_exponential:
        parser.error("--contextual-relative-bt requires --expanded-exponential")
    if args.frozen_contextual_configs and not args.contextual_relative_bt:
        parser.error("--frozen-contextual-configs requires --contextual-relative-bt")
    if args.calibrated_reward_space and args.contextual_relative_bt:
        parser.error(
            "--calibrated-reward-space and --contextual-relative-bt are mutually exclusive"
        )
    if args.bounded_probability_models and not args.calibrated_reward_space:
        parser.error("--bounded-probability-models requires --calibrated-reward-space")
    if args.target_upper_tail_only and not args.target_specific_search:
        parser.error("--target-upper-tail-only requires --target-specific-search")
    expanded_calibration = (
        "identity" if args.calibrated_reward_space else
        ("contextual_relative_bt" if args.contextual_relative_bt else "relative_bt")
    )
    configs = (
        (bounded_calibrated_reward_space_configs()
         if args.bounded_probability_models else calibrated_reward_space_configs())
        if args.calibrated_reward_space else
        (expanded_exponential_configs(expanded_calibration)
         if args.expanded_exponential else candidate_configs(args.quick, args.focused))
    )
    if args.exponential_only:
        configs = [config for config in configs
                   if config.family == "shifted_exponential"
                   and config.calibration in (
                       "relative_bt", "contextual_relative_bt", "identity"
                   )]
    args.target_divisors = (
        TARGET_FRONTIER_DIVISORS if args.target_specific_search else FRONTIER_DIVISORS
    )
    all_divisors = tuple(sorted(
        set(REPORT_DIVISORS) | set(FRONTIER_DIVISORS) | set(args.target_divisors)
    ))
    utility_rows, budget_rows, target_rows = [], [], []
    tuning_rows, diagnostic_rows, calibration_rows = [], [], []
    for split in range(args.split_start, args.split_start + args.splits):
        utility, budget, target, tuning, diagnostic, calibration_record = run_split(
            split, problems, configs, all_divisors, args
        )
        utility_rows.extend(utility)
        budget_rows.extend(budget)
        target_rows.extend(target)
        tuning_rows.extend(tuning)
        diagnostic_rows.extend(diagnostic)
        if calibration_record is not None:
            calibration_rows.append(calibration_record)
        write_csv(args.output_dir / "utility_splits.partial.csv", utility_rows)
        write_csv(args.output_dir / "equal_budget_splits.partial.csv", budget_rows)
        if target_rows:
            write_csv(
                args.output_dir / "target_accuracy_splits.partial.csv",
                target_rows,
            )
        if diagnostic_rows:
            write_csv(args.output_dir / "diagnostic_all_configs.partial.csv", diagnostic_rows)
        if calibration_rows:
            write_csv(
                args.output_dir / "isotonic_calibration_splits.partial.csv",
                calibration_rows,
            )

    utility_summary = summarize(
        utility_rows, ("method", "divisor"),
        ("relative_utility_gain_pct", "utility_gap", "adaptive_utility", "fixed_utility",
         "adaptive_accuracy", "adaptive_chars", "adaptive_opens"),
    )
    budget_summary = summarize(
        budget_rows, ("method", "divisor"),
        ("accuracy_gap_pp", "adaptive_accuracy", "fixed_accuracy", "adaptive_chars",
         "fixed_chars", "cost_mismatch"),
    )
    target_summary = (
        summarize(
            target_rows, ("method", "target_accuracy"),
            ("matched_char_saving_pct", "direct_char_saving_pct",
             "adaptive_accuracy", "fixed_accuracy", "accuracy_gap_pp",
             "adaptive_chars", "fixed_chars", "adaptive_opens", "fixed_opens",
             "matched_fixed_accuracy", "matched_accuracy_error",
             "matched_fixed_chars", "matched_fixed_opens",
             "adaptive_train_accuracy", "adaptive_train_chars",
             "train_fixed_target_chars", "train_estimated_char_saving_pct"),
        ) if target_rows else []
    )
    for name, rows in (
        ("utility_splits.csv", utility_rows), ("utility_summary.csv", utility_summary),
        ("equal_budget_splits.csv", budget_rows),
        ("equal_budget_summary.csv", budget_summary),
        ("tuning_selections.csv", tuning_rows),
    ):
        write_csv(args.output_dir / name, rows)
    if target_rows:
        write_csv(args.output_dir / "target_accuracy_splits.csv", target_rows)
        write_csv(args.output_dir / "target_accuracy_summary.csv", target_summary)
    if diagnostic_rows:
        write_csv(args.output_dir / "diagnostic_all_configs.csv", diagnostic_rows)
    if calibration_rows:
        write_csv(
            args.output_dir / "isotonic_calibration_splits.csv", calibration_rows
        )
    plot_utility(utility_summary, args.output_dir / "coding_utility_gap.png")
    plot_equal_budget(budget_summary, args.output_dir / "coding_equal_budget_accuracy.png")
    if target_summary:
        plot_target(target_summary, args.output_dir / "coding_target_accuracy.png")
    method = {
        "algorithm": "UCB Pandora reservation stopping only",
        "cohort": "83 coding problems with at least one correct generation",
        "distribution_models": (
            ["complete Gaussian on isotonic-calibrated success probability",
             "conditional shifted-exponential tail on isotonic-calibrated success probability"]
            if args.calibrated_reward_space else
            ["complete Gaussian on raw reward", "Gaussian KDE on raw reward",
             "conditional shifted-exponential tail on raw reward"]
        ),
        "calibrations": sorted({config.calibration for config in configs}),
        "reward_space": (
            "isotonic P(correct | raw reward), fitted on outer train problems only"
            if args.calibrated_reward_space else "raw reward"
        ),
        "reward_space_calibration": ({
            "fit": "direct isotonic regression of binary correctness on raw reward",
            "fit_partition": "outer training half only",
            "application": "frozen transform applied to train and untouched test halves",
            "range": [0.0, 1.0],
            "distribution_fit": "after reward transformation",
            "tail_extrapolation": (
                "Gaussian and shifted-exponential laws properly truncated "
                "to probability support [0, 1]"
                if args.bounded_probability_models else
                "unbounded laws clipped to probability range [0, 1]"
            ),
        } if args.calibrated_reward_space else None),
        "minimum_open_count": 3,
        "cost": "exact cumulative output characters divided by divisor",
        "splits": args.splits,
        "split_start": args.split_start,
        "train_test": "41/42 problems",
        "train_permutations_per_problem": args.train_permutations,
        "test_permutations_per_problem": args.test_permutations,
        "delta": args.delta,
        "candidate_configurations": len(configs),
        "search_mode": ("calibrated_gaussian_and_exponential"
                        if args.calibrated_reward_space else
                        ("expanded_exponential" if args.expanded_exponential else
                        ("focused" if args.focused else
                         ("quick" if args.quick else "full")))),
        "focused_grid": ({
            "confidence_scale": [0.0, 0.8, 1.6, 3.2, 6.4],
            "reward_prior_strength": [5.0, 20.0, 1e6],
            "cost_prior_strength": [0.0, 10.0],
            "cap_factor": 1.0,
            "tail_quantile": [0.25, 0.5],
        } if args.focused else None),
        "expanded_exponential_grid": ({
            "calibration": expanded_calibration,
            "confidence_scale": [0.0, 0.2, 0.8],
            "reward_prior_strength": [5.0, 20.0, 1e6],
            "cost_prior_strength": [0.0, 10.0],
            "cap_factor": [1.0, 1.1, 1.2, 1.3, 1.5, 2.0, None],
            "tail_quantile": [0.5, 0.75],
        } if (args.expanded_exponential or args.calibrated_reward_space) else None),
        "robust_family_by_divisor": ROBUST_METHOD_BY_DIVISOR,
        "frozen_exponential_config_by_divisor": (
            FROZEN_EXP_CONFIG_BY_DIVISOR if args.frozen_exponential_configs else None
        ),
        "frozen_contextual_exp_config_by_divisor": (
            FROZEN_CONTEXTUAL_EXP_CONFIG_BY_DIVISOR
            if args.frozen_contextual_configs else None
        ),
        "report_divisors": REPORT_DIVISORS,
        "frontier_divisors": FRONTIER_DIVISORS,
        "target_frontier_divisors": args.target_divisors,
        "targets": TARGETS,
        "objectives": (
            ["utility_gap", "equal_budget_accuracy"]
            if args.skip_target_accuracy else
            ["utility_gap", "equal_budget_accuracy", "target_accuracy"]
        ),
        "equal_budget": "independent fixed-N permutations; adjacent-N randomization exactly matches expected held-out characters",
        "target_accuracy": ("not run" if args.skip_target_accuracy else (
            "minimum-character UCB cost-divisor mixtures selected on cross-fitted "
            "train data; endpoints share one distribution/UCB configuration; fixed-N "
            "is quality-matched descriptively on heldout"
            if args.target_specific_search else
            "utility-frozen UCB policy mixtures selected on train; fixed-N is "
            "quality-matched descriptively on heldout"
        )),
        "target_specific_search": args.target_specific_search,
        "target_upper_tail_only": args.target_upper_tail_only,
        "calibrated_reward_space": args.calibrated_reward_space,
        "bounded_probability_models": args.bounded_probability_models,
    }
    with (args.output_dir / "METHOD.json").open("w") as handle:
        json.dump(method, handle, indent=2)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "utility_rows": len(utility_summary),
        "equal_budget_rows": len(budget_summary),
        "target_accuracy_rows": len(target_summary),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
