"""Sample-count window shared by theory-guided coding and alignment policies."""
from collections import deque
import math


class RecentHalfMean:
    """Average eligible nonnegative estimates at ceil(n/2) <= j <= n.

    Call once at each count where a raw estimate is computed, even if that
    estimate is ineligible. Counts before four are never included. Empty
    windows return infinity so they cannot trigger a gain-versus-cost stop.
    """

    def __init__(self):
        self._values = deque()
        self._last_count = 0

    def update(self, count, value, *, eligible=True):
        if isinstance(count, bool) or not isinstance(count, int) or count <= self._last_count:
            raise ValueError("count must be a strictly increasing positive integer")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("value must be finite and nonnegative")
        self._last_count = count
        if count >= 4 and eligible:
            self._values.append((count, value))
        first = max(4, (count + 1)//2)
        while self._values and self._values[0][0] < first:
            self._values.popleft()
        if not self._values:
            return math.inf
        return math.fsum(value for _, value in self._values)/len(self._values)
