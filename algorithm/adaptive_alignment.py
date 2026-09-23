"""Online alignment stopping rule used in the updated AISTATS experiments.

No offline training or distribution fit is required. Each call to observe()
receives only the reward and paid length of the response just generated.
Defaults implement the mean_costse2 / odds_mean_costse2 policy.

This empirical policy is motivated by DMRL optimal stopping, but is not the
exact theorem-backed policy. Its benchmark estimate, smoothing, and cost
adjustment do not have a proved confidence guarantee.
"""

from bisect import insort
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AlignmentDecision:
    should_stop: bool
    count: int
    best_index: int  # zero-based, earliest maximum-reward response
    best_reward: float
    total_length: float
    total_cost: float
    estimated_improvement: float | None
    estimated_next_cost: float | None


class AdaptiveAlignment:
    """Sequential gain-versus-cost policy; lengths must use the price's unit.

    Supply actual generated token counts for dollar/token accounting, or
    recorded character counts only when reproducing the historical experiments.
    The class never tokenizes or converts between these units. Utility has
    maximum value one dollar. Retain the actual best response in the caller.
    """

    def __init__(self, price, cap=960, minimum=4, reference_quantile=0.99,
                 cost_adjustment=2.0):
        if not math.isfinite(price) or price <= 0:
            raise ValueError("price must be finite and positive")
        if not isinstance(minimum, int) or minimum < 4:
            raise ValueError("minimum must be an integer of at least four")
        if not isinstance(cap, int) or cap < minimum:
            raise ValueError("cap must be an integer at least minimum")
        if not 0 < reference_quantile < 1:
            raise ValueError("reference_quantile must lie strictly between zero and one")
        if not math.isfinite(cost_adjustment) or cost_adjustment < 0:
            raise ValueError("cost_adjustment must be finite and nonnegative")
        self.price = float(price)
        self.cap = cap
        self.minimum = minimum
        self.reference_quantile = reference_quantile
        self.cost_adjustment = cost_adjustment
        self._ordered_rewards = []
        self._best_reward = -math.inf
        self._best_index = -1
        self._length_sum = 0.0
        self._length_square_sum = 0.0
        self._scaled_gain_sum = 0.0
        self.decision = None

    def observe(self, reward, length):
        """Charge one completed response and return the next stopping decision."""
        if self.decision is not None and self.decision.should_stop:
            raise RuntimeError("The policy has stopped; create a new policy for a new prompt")
        reward, length = float(reward), float(length)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if not math.isfinite(length) or length <= 0 or not length.is_integer():
            raise ValueError("length must be a positive integer count")

        insort(self._ordered_rewards, reward)
        n = len(self._ordered_rewards)
        if reward > self._best_reward:  # preserve the earliest response in a tie
            self._best_reward, self._best_index = reward, n - 1
        self._length_sum += length
        self._length_square_sum += length * length
        gain = cost = None

        if n >= self.minimum:
            # Upper-half excess in normalized exponentiated-score coordinates.
            k = n // 2
            z = [math.exp(max(r - self._best_reward, -745.0))
                 for r in self._ordered_rewards[-k-1:]]
            cutoff = z[0]
            excess = sum(value - cutoff for value in z[1:]) / k
            reference = cutoff + excess * (
                1 + math.log((k / n) / (1 - self.reference_quantile)))

            # n times the BT improvement estimate. The incumbent is z=1.
            scaled_gain = (n / (n + 1) * reference * excess
                           / ((1 + reference) * (1 + reference + excess)))
            self._scaled_gain_sum += scaled_gain
            gain = self._scaled_gain_sum / (n - self.minimum + 1) / n

            mean_length = self._length_sum / n
            variance = max(
                (self._length_square_sum - self._length_sum**2 / n) / (n - 1),
                0.0)
            standard_error = math.sqrt(variance / n)
            cost = self.price * mean_length / (
                1 + self.cost_adjustment * standard_error / mean_length)

        self.decision = AlignmentDecision(
            should_stop=n == self.cap or (gain is not None and gain <= cost),
            count=n, best_index=self._best_index, best_reward=self._best_reward,
            total_length=self._length_sum, total_cost=self.price * self._length_sum,
            estimated_improvement=gain, estimated_next_cost=cost)
        return self.decision
