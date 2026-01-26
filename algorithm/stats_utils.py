import heapq

import numpy as np


class MedianEstimator:
    """Estimate median and mean of exceedance in a stream."""

    def __init__(self):
        self.low = []   # max heap (negated values)
        self.high = []  # min heap
        self.high_sum = 0

    def add(self, x):
        if not self.high or x >= self.high[0]:
            heapq.heappush(self.high, x)
            self.high_sum += x
        else:
            heapq.heappush(self.low, -x)

        if len(self.high) > len(self.low) + 1:
            val = heapq.heappop(self.high)
            self.high_sum -= val
            heapq.heappush(self.low, -val)
        elif len(self.low) > len(self.high):
            val = -heapq.heappop(self.low)
            heapq.heappush(self.high, val)
            self.high_sum += val

    @property
    def median(self):
        if not self.high:
            return None
        return self.high[0]

    @property
    def exceedance_mean(self):
        if not self.high:
            return None
        return self.high_sum / len(self.high)


def simpson_integral(y, dx):
    n = y.size
    if n < 3:
        return np.sum(y) * dx
    if n % 2 == 0:
        y = y[:-1]
        n -= 1
    return (dx / 3.0) * (y[0] + y[-1] + 4.0 * np.sum(y[1:n-1:2]) + 2.0 * np.sum(y[2:n-2:2]))


def alpha_quantile(values, alpha):
    values = np.asarray(values)
    n_total = values.shape[0]
    alpha = min(max(alpha, 1e-6), 1 - 1e-12)
    sorted_data = np.sort(values)
    quantile_index = int(alpha * n_total)
    quantile_index = min(max(quantile_index, 0), n_total - 1)
    return float(sorted_data[quantile_index])


def acceptance_rate(v, w):
    v_exp = np.exp(v)
    w_exp = np.exp(w)
    return v_exp / (v_exp + w_exp)
