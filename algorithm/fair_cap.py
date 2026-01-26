import numpy as np
from scipy import optimize
from scipy.stats import norm, lognorm

from stats_utils import simpson_integral


def find_tau_cdf(c):
    """
    Find fair-cap tau for CDF transformation.

    For any continuous distribution, F(X) ~ Uniform(0,1).
    Therefore: E[(F(X) - tau)^+] = E[(U - tau)^+] where U ~ Uniform(0,1)
             = integral from tau to 1 of (u - tau) du
             = (1 - tau)^2 / 2

    Setting (1 - tau)^2 / 2 = c gives tau = 1 - sqrt(2c)

    Args:
        c: cost parameter

    Returns:
        fair-cap value tau
    """
    if c >= 0.5:
        return 0.0
    return max(0.0, 1.0 - np.sqrt(2 * c))


def find_tau_lognormal_bt(mu, sigma, benchmark, c, n_points=4000):
    """
    Find fair-cap tau for log-normal distribution with Bradley-Terry transformation.

    Solves: E[(X / (X + M) - tau)^+] = c where X ~ LogNormal(mu, sigma)

    Args:
        mu: log-space mean
        sigma: log-space std
        benchmark: M in the BT transformation
        c: cost parameter
        n_points: number of integration points

    Returns:
        fair-cap value tau
    """
    probs = np.linspace(0.0001, 0.9999, n_points)
    x_points = np.exp(mu + sigma * norm.ppf(probs))
    pdf_values = lognorm.pdf(x_points, s=sigma, scale=np.exp(mu))
    f_values = x_points / (x_points + benchmark)

    def expected_excess(tau):
        if tau >= 1:
            return 0.0
        if tau <= 0:
            return np.mean(f_values) - tau

        excess = np.maximum(f_values - tau, 0)
        return np.mean(excess)

    def objective(tau):
        return expected_excess(tau) - c

    try:
        return optimize.brentq(objective, 1e-12, 0.9999, xtol=1e-8)
    except ValueError:
        max_expected = expected_excess(1e-12)
        min_expected = expected_excess(0.9999)
        if c > max_expected:
            return 1e-12
        if c < min_expected:
            return 0.9999
        return 0.5


def find_tau_discrete_optimized(scale, estimated_max, c, loc=0, n_points=4000, max_tau_search=0.9999, tail_prob=1e-9):
    """Optimized expected excess computation for exponential distribution."""
    lam = 1.0 / scale
    tail_prob = min(max(tail_prob, 1e-16), 1e-3)
    x_max_base = loc - scale * np.log(tail_prob)

    def expected_excess_fast(tau):
        if tau <= 0:
            x_points = np.linspace(loc, x_max_base, n_points)
            dx = (x_max_base - loc) / (x_points.size - 1)
            pdf_values = lam * np.exp(-lam * (x_points - loc))
            f_values = x_points / (x_points + estimated_max)
            return simpson_integral((f_values - tau) * pdf_values, dx)

        if tau >= 1:
            return 0.0

        x0 = tau * estimated_max / (1 - tau)
        x0 = max(x0, loc)
        x_max_tail = x0 - scale * np.log(tail_prob)
        x_max = max(x_max_base, x_max_tail)

        x_points = np.linspace(x0, x_max, n_points)
        dx = (x_max - x0) / (x_points.size - 1)
        pdf_values = lam * np.exp(-lam * (x_points - loc))
        f_values = x_points / (x_points + estimated_max)
        excess_values = f_values - tau
        return simpson_integral(excess_values * pdf_values, dx)

    def objective(tau):
        return expected_excess_fast(tau) - c

    try:
        return optimize.brentq(objective, 1e-12, max_tau_search, xtol=1e-8)
    except ValueError:
        try:
            max_expected = expected_excess_fast(1e-12)
            min_expected = expected_excess_fast(max_tau_search)
            if c > max_expected:
                return 1e-12
            if c < min_expected:
                return max_tau_search
            return optimize.brentq(objective, 1e-12, max_tau_search, xtol=1e-6)
        except Exception:
            return np.nan
