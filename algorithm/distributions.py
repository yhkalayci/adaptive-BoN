import numpy as np


def fit_shifted_exponential(exp_rewards, n, delta):
    """
    Fit shifted exponential to tail of exponentiated rewards.

    Args:
        exp_rewards: array of exponentiated reward values (e^r)
        n: number of samples
        delta: confidence parameter

    Returns:
        dict with keys: loc, scale, scale_ucb, scale_lcb
    """
    median = np.median(exp_rewards)
    tail = exp_rewards[exp_rewards >= median]
    mean_exc = np.mean(tail) - median
    mean_exc = max(mean_exc, 1e-8)

    ucb_factor = 1 + np.sqrt(np.log(n) * np.log(1 / delta) / n)
    lcb_factor = 1 - np.sqrt(np.log(n) * np.log(1 / delta) / n)

    return {
        "loc": median,
        "scale": mean_exc,
        "scale_ucb": mean_exc * ucb_factor,
        "scale_lcb": max(mean_exc * lcb_factor, 1e-8),
    }


def fit_lognormal(rewards, n, delta):
    """
    Fit log-normal to exponentiated rewards (full distribution, not just tail).

    Since X = e^r follows log-normal if r ~ Normal, we fit:
    - mu = mean(rewards)  [log-space mean]
    - sigma = std(rewards) [log-space std]

    Args:
        rewards: array of raw reward values (r, not e^r)
        n: number of samples
        delta: confidence parameter

    Returns:
        dict with keys: mu, sigma, mu_ucb, mu_lcb, sigma_ucb, sigma_lcb
    """
    mu = np.mean(rewards)
    sigma = np.std(rewards, ddof=1)
    sigma = max(sigma, 1e-8)

    r_delta = np.sqrt(np.log(n) * np.log(1 / delta) / n)
    se_mu = sigma / np.sqrt(n)

    return {
        "mu": mu,
        "sigma": sigma,
        "mu_ucb": mu + se_mu * (1 + r_delta),
        "mu_lcb": mu - se_mu * (1 + r_delta),
        "sigma_ucb": sigma * (1 + r_delta),
        "sigma_lcb": max(sigma * (1 - r_delta), 1e-8),
    }
