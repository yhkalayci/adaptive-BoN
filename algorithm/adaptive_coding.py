"""Non-parametric DMRL stopping for coding generation.

The only offline statistical fit is an increasing isotonic map from a reward
to ``P(correct)``.  Online stopping is distribution-free: after at least five
generations it measures the average excess of the four largest calibrated
utilities over the fifth-largest utility, smooths that observed residual
scale, and divides it by the number of observations.  No exponential or
other parametric reward-tail distribution is fitted.

Correctness labels are used only to fit the isotonic map on a disjoint
calibration set and to audit the final selected response.  They are never
visible to :class:`AdaptiveCoding`.
"""

from __future__ import annotations

import argparse
from bisect import insort
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np


PROFILE_SCHEMA = "adaptive_coding_dmrl_profile"
PROFILE_VERSION = 2
SMOOTHING_MODES = ("current", "mean", "recent_half")


@dataclass(frozen=True)
class CodingProblem:
    """Minimal labeled representation used for offline fitting and auditing."""

    rewards: tuple[float, ...]
    correct: tuple[bool, ...]
    lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        size = len(self.rewards)
        if not size or len(self.correct) != size or len(self.lengths) != size:
            raise ValueError("rewards, correct, and lengths must be nonempty peers")
        if any(not math.isfinite(value) for value in self.rewards):
            raise ValueError("all rewards must be finite")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value <= 0
            for value in self.lengths
        ):
            raise ValueError("all lengths must be positive integer counts")


@dataclass(frozen=True)
class CodingProfile:
    """Frozen monotone reward calibration; deliberately no tail fit."""

    reward_knots: tuple[float, ...]
    probability_knots: tuple[float, ...]
    mean_length: float
    length_unit: str = "output_tokens"
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.reward_knots or len(self.reward_knots) != len(
            self.probability_knots
        ):
            raise ValueError("calibration knots must be nonempty peers")
        if any(not math.isfinite(value) for value in self.reward_knots):
            raise ValueError("reward knots must be finite")
        if any(
            right <= left
            for left, right in zip(self.reward_knots, self.reward_knots[1:])
        ):
            raise ValueError("reward knots must be strictly increasing")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.probability_knots
        ):
            raise ValueError("probability knots must lie in [0, 1]")
        if any(
            right < left
            for left, right in zip(
                self.probability_knots, self.probability_knots[1:]
            )
        ):
            raise ValueError("probability knots must be nondecreasing")
        if not math.isfinite(self.mean_length) or self.mean_length <= 0.0:
            raise ValueError("mean_length must be finite and positive")
        if not self.length_unit:
            raise ValueError("length_unit must be nonempty")

    def calibrate(self, reward: float) -> float:
        """Linearly interpolate the isotonic map and clip outside its range."""
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if len(self.reward_knots) == 1:
            return self.probability_knots[0]
        return float(
            np.interp(
                reward,
                self.reward_knots,
                self.probability_knots,
                left=self.probability_knots[0],
                right=self.probability_knots[-1],
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PROFILE_SCHEMA,
            "version": PROFILE_VERSION,
            "calibration": {
                "kind": "isotonic",
                "reward_knots": list(self.reward_knots),
                "probability_knots": list(self.probability_knots),
                "out_of_bounds": "clip",
            },
            "online_distribution": None,
            "cost_model": {
                "mean_length": self.mean_length,
                "length_unit": self.length_unit,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CodingProfile":
        if value.get("schema") != PROFILE_SCHEMA:
            raise ValueError(f"expected profile schema {PROFILE_SCHEMA!r}")
        if value.get("version") != PROFILE_VERSION:
            raise ValueError(f"unsupported profile version {value.get('version')!r}")
        calibration = value.get("calibration")
        cost_model = value.get("cost_model")
        if not isinstance(calibration, Mapping) or calibration.get("kind") != "isotonic":
            raise ValueError("profile must contain an isotonic calibration")
        if value.get("online_distribution", None) is not None:
            raise ValueError("non-parametric DMRL profiles cannot contain a tail fit")
        if not isinstance(cost_model, Mapping):
            raise ValueError("profile must contain a cost model")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("profile metadata must be an object")
        return cls(
            reward_knots=tuple(float(x) for x in calibration["reward_knots"]),
            probability_knots=tuple(
                float(x) for x in calibration["probability_knots"]
            ),
            mean_length=float(cost_model["mean_length"]),
            length_unit=str(cost_model["length_unit"]),
            metadata=metadata,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CodingProfile":
        with Path(path).open() as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


@dataclass(frozen=True)
class CodingDecision:
    should_stop: bool
    count: int
    best_index: int
    best_reward: float
    best_probability: float
    total_length: float
    total_cost: float
    estimated_improvement: float | None
    estimated_next_cost: float | None
    residual_scale: float | None
    smoothed_residual_scale: float | None


class AdaptiveCoding:
    """Sequential non-parametric top-four DMRL policy for one problem.

    The order-statistic statistic uses the four largest observed calibrated
    probabilities and the fifth largest as a local cutoff.  ``multiplier``,
    ``smoothing`` and ``cost_adjustment`` may be selected offline, but they are
    fixed before any evaluation problem is opened.
    """

    def __init__(
        self,
        profile: CodingProfile,
        price: float,
        cap: int = 512,
        width: int = 4,
        multiplier: float = 1.0,
        smoothing: str = "mean",
        cost_adjustment: float = 2.0,
        minimum: int | None = None,
    ) -> None:
        if not isinstance(profile, CodingProfile):
            raise TypeError("profile must be a CodingProfile")
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError("price must be finite and positive")
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("multiplier must be finite and positive")
        if smoothing not in SMOOTHING_MODES:
            raise ValueError(f"smoothing must be one of {SMOOTHING_MODES}")
        if not math.isfinite(cost_adjustment) or cost_adjustment < 0.0:
            raise ValueError("cost_adjustment must be finite and nonnegative")
        if minimum is None:
            minimum = width + 1
        if isinstance(minimum, bool) or not isinstance(minimum, int):
            raise ValueError("minimum must be an integer")
        if minimum < width + 1:
            raise ValueError("minimum must be at least width + 1")
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < minimum:
            raise ValueError("cap must be an integer at least minimum")

        self.profile = profile
        self.price = float(price)
        self.cap = cap
        self.width = width
        self.multiplier = float(multiplier)
        self.smoothing = smoothing
        self.cost_adjustment = float(cost_adjustment)
        self.minimum = minimum
        self._sorted_probabilities: list[float] = []
        self._residual_history: list[float] = []
        self._best_probability = -math.inf
        self._best_reward = -math.inf
        self._best_index = -1
        self._length_sum = 0.0
        self._length_square_sum = 0.0
        self._stop_latched = False
        self.decision: CodingDecision | None = None

    def _smoothed_residual(self) -> float:
        if self.smoothing == "current":
            return self._residual_history[-1]
        if self.smoothing == "mean":
            return float(np.mean(self._residual_history))
        count = max(1, (len(self._residual_history) + 1) // 2)
        return float(np.mean(self._residual_history[-count:]))

    def observe(self, reward: float, length: int) -> CodingDecision:
        """Charge one generation, observe its reward, and decide whether to stop."""
        if self.decision is not None and self.decision.should_stop:
            raise RuntimeError("the policy has stopped; create a new policy per problem")
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if isinstance(length, bool):
            raise ValueError("length must be a positive integer count")
        try:
            numeric_length = float(length)
        except (TypeError, ValueError) as error:
            raise ValueError("length must be a positive integer count") from error
        if (
            not math.isfinite(numeric_length)
            or numeric_length <= 0.0
            or not numeric_length.is_integer()
        ):
            raise ValueError("length must be a positive integer count")

        probability = self.profile.calibrate(reward)
        insort(self._sorted_probabilities, probability)
        n = len(self._sorted_probabilities)
        if probability > self._best_probability or (
            probability == self._best_probability and reward > self._best_reward
        ):
            self._best_probability = probability
            self._best_reward = reward
            self._best_index = n - 1
        self._length_sum += numeric_length
        self._length_square_sum += numeric_length * numeric_length

        residual = smoothed = gain = next_cost = None
        if n >= self.width + 1:
            cutoff = self._sorted_probabilities[-self.width - 1]
            upper = self._sorted_probabilities[-self.width :]
            residual = float(np.mean([value - cutoff for value in upper]))
            self._residual_history.append(residual)
            smoothed = self._smoothed_residual()
            gain = self.multiplier * smoothed / n

            mean_length = self._length_sum / n
            variance = max(
                0.0, self._length_square_sum / n - mean_length * mean_length
            )
            standard_error = math.sqrt(variance / n)
            optimism = 1.0 + self.cost_adjustment * standard_error / mean_length
            next_cost = self.price * mean_length / optimism
            self._stop_latched = self._stop_latched or gain <= next_cost

        self.decision = CodingDecision(
            should_stop=n == self.cap or (n >= self.minimum and self._stop_latched),
            count=n,
            best_index=self._best_index,
            best_reward=self._best_reward,
            best_probability=self._best_probability,
            total_length=self._length_sum,
            total_cost=self.price * self._length_sum,
            estimated_improvement=gain,
            estimated_next_cost=next_cost,
            residual_scale=residual,
            smoothed_residual_scale=smoothed,
        )
        return self.decision


def load_coding_problems(
    path: str | Path,
    *,
    length_field: str = "output_tokens",
    expected_samples: int | None = 512,
) -> dict[str, CodingProblem]:
    """Load the combined correctness/reward JSONL without response text."""
    problems: dict[str, CodingProblem] = {}
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at line {line_number}")
            record = json.loads(line)
            problem_id = str(record["id"])
            if problem_id in problems:
                raise ValueError(f"duplicate problem id {problem_id!r}")
            samples = sorted(record["samples"], key=lambda item: int(item["idx"]))
            if expected_samples is not None and len(samples) != expected_samples:
                raise ValueError(
                    f"{problem_id}: expected {expected_samples} samples, "
                    f"found {len(samples)}"
                )
            if [int(x["idx"]) for x in samples] != list(range(len(samples))):
                raise ValueError(f"{problem_id}: sample indices are not contiguous")
            problems[problem_id] = CodingProblem(
                rewards=tuple(float(x["r_score"]) for x in samples),
                correct=tuple(bool(x["correct"]) for x in samples),
                lengths=tuple(int(x[length_field]) for x in samples),
            )
    if not problems:
        raise ValueError("coding JSONL contains no problems")
    return problems


def split_problem_ids(
    problem_ids: Iterable[str], seed: int, fit_fraction: float = 0.5
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return a deterministic, disjoint problem-level calibration/test split."""
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie strictly between zero and one")
    ids = np.asarray(sorted(set(str(x) for x in problem_ids)), dtype=object)
    if len(ids) < 2:
        raise ValueError("at least two problems are required")
    order = np.random.default_rng(seed).permutation(ids)
    count = min(max(int(math.floor(len(order) * fit_fraction)), 1), len(order) - 1)
    return tuple(str(x) for x in order[:count]), tuple(str(x) for x in order[count:])


def fit_coding_profile(
    problems: Mapping[str, CodingProblem],
    *,
    fit_problem_ids: Sequence[str] | None = None,
    length_unit: str = "output_tokens",
    metadata: Mapping[str, object] | None = None,
) -> CodingProfile:
    """Fit only the monotone reward-to-correctness map on selected problems."""
    if not problems:
        raise ValueError("at least one coding problem is required")
    ids = tuple(problems) if fit_problem_ids is None else tuple(
        str(x) for x in fit_problem_ids
    )
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("fit_problem_ids must be nonempty and unique")
    missing = sorted(set(ids) - set(problems))
    if missing:
        raise ValueError(f"unknown fit problem ids: {missing[:5]}")
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as error:
        raise RuntimeError("fitting requires scikit-learn") from error

    rewards = np.concatenate(
        [np.asarray(problems[x].rewards, dtype=np.float64) for x in ids]
    )
    correct = np.concatenate(
        [np.asarray(problems[x].correct, dtype=np.float64) for x in ids]
    )
    isotonic = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    ).fit(rewards, correct)
    lengths = np.concatenate(
        [np.asarray(problems[x].lengths, dtype=np.float64) for x in ids]
    )
    return CodingProfile(
        reward_knots=tuple(float(x) for x in isotonic.X_thresholds_),
        probability_knots=tuple(float(x) for x in isotonic.y_thresholds_),
        mean_length=float(np.mean(lengths)),
        length_unit=length_unit,
        metadata=dict(metadata or {}),
    )


def calibration_metrics(
    problems: Mapping[str, CodingProblem],
    problem_ids: Sequence[str],
    profile: CodingProfile,
) -> dict[str, float | int]:
    rewards = np.concatenate(
        [np.asarray(problems[x].rewards, dtype=np.float64) for x in problem_ids]
    )
    correct = np.concatenate(
        [np.asarray(problems[x].correct, dtype=np.float64) for x in problem_ids]
    )
    probability = np.asarray([profile.calibrate(x) for x in rewards])
    return {
        "problems": len(problem_ids),
        "samples": len(rewards),
        "accuracy": float(np.mean(correct)),
        "mean_probability": float(np.mean(probability)),
        "brier": float(np.mean((probability - correct) ** 2)),
    }


def replay_problem(
    problem: CodingProblem,
    profile: CodingProfile,
    price: float,
    **policy_options: object,
) -> tuple[CodingDecision, bool]:
    """Replay one order; reveal correctness only after the policy stops."""
    policy = AdaptiveCoding(profile, price, **policy_options)
    for reward, length in zip(problem.rewards, problem.lengths):
        decision = policy.observe(reward, length)
        if decision.should_stop:
            return decision, problem.correct[decision.best_index]
    raise AssertionError("policy did not stop within the recorded samples")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fit_command(args: argparse.Namespace) -> None:
    data_path = args.data.resolve()
    problems = load_coding_problems(
        data_path, length_field=args.length_field, expected_samples=args.expected_samples
    )
    fit_ids, holdout_ids = split_problem_ids(problems, args.seed, args.fit_fraction)
    metadata: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(data_path),
        "source_sha256": _sha256(data_path),
        "source_problem_count": len(problems),
        "source_sample_count": sum(len(x.rewards) for x in problems.values()),
        "split_seed": args.seed,
        "fit_fraction": args.fit_fraction,
        "fit_problem_ids": list(fit_ids),
        "holdout_problem_ids": list(holdout_ids),
        "fit_partition": "problem_level_calibration_split_only",
        "online_correctness_access": False,
        "online_distribution_fit": False,
        "policy_family": "nonparametric_dmrl_top4_above_fifth",
        "reward_model": args.reward_model,
        "generator_model": args.generator_model,
    }
    profile = fit_coding_profile(
        problems,
        fit_problem_ids=fit_ids,
        length_unit=args.length_field,
        metadata=metadata,
    )
    metadata["calibration_metrics"] = {
        "fit": calibration_metrics(problems, fit_ids, profile),
        "holdout": calibration_metrics(problems, holdout_ids, profile),
    }
    profile = CodingProfile(
        profile.reward_knots,
        profile.probability_knots,
        profile.mean_length,
        profile.length_unit,
        metadata,
    )
    profile.save(args.output)
    print(json.dumps({
        "output": str(args.output),
        "fit_problems": len(fit_ids),
        "holdout_problems": len(holdout_ids),
        "isotonic_knots": len(profile.reward_knots),
        "mean_length": profile.mean_length,
        "online_distribution_fit": False,
        "calibration_metrics": metadata["calibration_metrics"],
    }, indent=2))


def _audit_command(args: argparse.Namespace) -> None:
    profile = CodingProfile.load(args.profile)
    data_sha256 = _sha256(args.data)
    expected = profile.metadata.get("source_sha256")
    if expected is not None and expected != data_sha256:
        raise ValueError("audit data SHA-256 does not match fitting source")
    problems = load_coding_problems(
        args.data, length_field=profile.length_unit, expected_samples=args.expected_samples
    )
    if args.partition == "all":
        problem_ids = tuple(sorted(problems))
    else:
        stored = profile.metadata.get(f"{args.partition}_problem_ids")
        if not isinstance(stored, list) or not stored:
            raise ValueError(f"profile has no {args.partition} problem IDs")
        problem_ids = tuple(str(x) for x in stored)

    trials: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for divisor in args.utility_divisors:
        group: list[dict[str, object]] = []
        for problem_id in problem_ids:
            decision, correct = replay_problem(
                problems[problem_id],
                profile,
                1.0 / divisor,
                cap=args.cap,
                width=args.width,
                multiplier=args.multiplier,
                smoothing=args.smoothing,
                cost_adjustment=args.cost_adjustment,
                minimum=args.minimum,
            )
            row = {
                "problem_id": problem_id,
                "utility_divisor": divisor,
                "correct": bool(correct),
                "opened": decision.count,
                "selected_index": decision.best_index,
                "selected_probability": decision.best_probability,
                "output_tokens": decision.total_length,
                "profit": float(correct) - decision.total_length / divisor,
            }
            group.append(row)
            trials.append(row)
        summaries.append({
            "utility_divisor": divisor,
            "problems": len(group),
            "accuracy": float(np.mean([x["correct"] for x in group])),
            "mean_generations": float(np.mean([x["opened"] for x in group])),
            "mean_output_tokens": float(np.mean([x["output_tokens"] for x in group])),
            "total_generations": int(sum(int(x["opened"]) for x in group)),
            "total_output_tokens": int(sum(int(x["output_tokens"]) for x in group)),
            "mean_profit": float(np.mean([x["profit"] for x in group])),
        })
    output = {
        "schema": "adaptive_coding_dmrl_audit",
        "version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_path": str(args.profile.resolve()),
        "profile_sha256": _sha256(args.profile),
        "data_path": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "partition": args.partition,
        "policy": {
            "family": "nonparametric_dmrl_top4_above_fifth",
            "width": args.width,
            "multiplier": args.multiplier,
            "smoothing": args.smoothing,
            "cost_adjustment": args.cost_adjustment,
            "minimum": args.minimum,
            "cap": args.cap,
        },
        "summary": summaries,
        "trials": trials,
    }
    _atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), "summary": summaries}, indent=2))


def _parse_divisors(value: str) -> tuple[float, ...]:
    values = tuple(float(x) for x in value.split(",") if x.strip())
    if not values or any(not math.isfinite(x) or x <= 0.0 for x in values):
        raise argparse.ArgumentTypeError("divisors must be comma-separated positives")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=20260923)
    fit.add_argument("--fit-fraction", type=float, default=0.5)
    fit.add_argument("--length-field", default="output_tokens")
    fit.add_argument("--expected-samples", type=int, default=512)
    fit.add_argument("--reward-model", default="LARK-Lab/CodeScaler-8B")
    fit.add_argument("--generator-model", default="Qwen/Qwen2.5-Coder-3B")
    fit.set_defaults(function=_fit_command)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--profile", type=Path, required=True)
    audit.add_argument("--data", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--partition", choices=("fit", "holdout", "all"), default="holdout")
    audit.add_argument("--utility-divisors", type=_parse_divisors, default=_parse_divisors("25000,50000,100000,200000,400000"))
    audit.add_argument("--expected-samples", type=int, default=512)
    audit.add_argument("--cap", type=int, default=512)
    audit.add_argument("--width", type=int, default=4)
    audit.add_argument("--multiplier", type=float, default=1.0)
    audit.add_argument("--smoothing", choices=SMOOTHING_MODES, default="mean")
    audit.add_argument("--cost-adjustment", type=float, default=2.0)
    audit.add_argument("--minimum", type=int, default=5)
    audit.set_defaults(function=_audit_command)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
