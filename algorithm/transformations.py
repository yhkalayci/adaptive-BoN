import numpy as np
from scipy.stats import norm


def transform_bradley_terry(v, benchmark):
    """
    Bradley-Terry transformation: v / (v + M)

    Args:
        v: value (e^reward)
        benchmark: estimated max (M)

    Returns:
        transformed value in [0, 1]
    """
    return v / (v + benchmark)


def transform_cdf_shifted_exp(v, loc, scale):
    """
    CDF transformation for shifted exponential.
    F(x) = 1 - exp(-(x - loc) / scale) for x >= loc

    Args:
        v: value (e^reward)
        loc: location parameter (shift)
        scale: scale parameter

    Returns:
        CDF value in [0, 1]
    """
    if v < loc:
        return 0.0
    return 1.0 - np.exp(-(v - loc) / scale)


def transform_cdf_lognormal(v, mu, sigma):
    """
    CDF transformation for log-normal.
    F(x) = Phi((ln(x) - mu) / sigma)

    Note: v = e^reward, so ln(v) = reward

    Args:
        v: value (e^reward)
        mu: log-space mean (mean of rewards)
        sigma: log-space std (std of rewards)

    Returns:
        CDF value in [0, 1]
    """
    if v <= 0:
        return 0.0
    return norm.cdf((np.log(v) - mu) / sigma)
