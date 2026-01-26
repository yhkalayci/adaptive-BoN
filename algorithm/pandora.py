import numpy as np
from scipy.stats import norm

from distributions import fit_lognormal, fit_shifted_exponential
from fair_cap import find_tau_cdf, find_tau_discrete_optimized, find_tau_lognormal_bt
from stats_utils import acceptance_rate, alpha_quantile
from transformations import (
    transform_bradley_terry,
    transform_cdf_lognormal,
    transform_cdf_shifted_exp,
)


def pandoras_box(data, cost, delta, min_open_count=10, alpha=0.99,
                 distribution="shifted_exponential", transformation="bradley_terry",
                 batch_size=1):
    """
    Run Pandora's Box algorithm on reward data.

    Args:
        data: array of reward values
        cost: sampling cost
        delta: confidence parameter
        min_open_count: minimum samples before stopping
        alpha: quantile for benchmark
        distribution: "shifted_exponential" or "lognormal"
        transformation: "bradley_terry" or "cdf"
        batch_size: number of boxes to open per step

    Returns:
        dict with results including win_rate, open_count, utility, etc.
    """
    n_total = data.shape[0]
    max_until = {
        "value": -10,
        "exp_value": 0,
        "generator": 0,
        "index": 0,
    }

    observed_rewards = []
    observed_exp_rewards = []

    min_open_count = min(min_open_count, n_total)
    open_count = min_open_count
    for i in range(min_open_count):
        observed_rewards.append(data[i])
        observed_exp_rewards.append(np.exp(data[i]))
        if data[i] > max_until["value"]:
            max_until["value"] = data[i]
            max_until["index"] = i
            max_until["exp_value"] = np.exp(data[i])

    quantile_value = alpha_quantile(data, alpha)
    global_opt = {
        "value": quantile_value,
        "exp_value": np.exp(quantile_value),
        "generator": 0,
    }

    final_estimated_max = None

    while open_count < n_total:
        n = open_count
        v = max_until["exp_value"]
        rewards_arr = np.array(observed_rewards)
        exp_rewards_arr = np.array(observed_exp_rewards)

        if distribution == "shifted_exponential":
            dist_params = fit_shifted_exponential(exp_rewards_arr, n, delta)
            loc = dist_params["loc"]
            scale_ucb = dist_params["scale_ucb"]
            scale_lcb = dist_params["scale_lcb"]

            estimated_max = loc + (-scale_lcb * np.log(1 - alpha))
            final_estimated_max = estimated_max

            if transformation == "bradley_terry":
                v_transformed = transform_bradley_terry(v, estimated_max)
                threshold = find_tau_discrete_optimized(scale_ucb, estimated_max, cost, loc=loc)
            else:
                v_transformed = transform_cdf_shifted_exp(v, loc, scale_ucb)
                threshold = find_tau_cdf(cost)

        else:
            dist_params = fit_lognormal(rewards_arr, n, delta)
            mu = dist_params["mu"]
            sigma = dist_params["sigma"]
            mu_ucb = dist_params["mu_ucb"]
            mu_lcb = dist_params["mu_lcb"]
            sigma_ucb = dist_params["sigma_ucb"]
            sigma_lcb = dist_params["sigma_lcb"]

            estimated_max = np.exp(mu_lcb + sigma_lcb * norm.ppf(alpha))
            final_estimated_max = estimated_max

            if transformation == "bradley_terry":
                v_transformed = transform_bradley_terry(v, estimated_max)
                threshold = find_tau_lognormal_bt(mu_ucb, sigma_ucb, estimated_max, cost)
            else:
                v_transformed = transform_cdf_lognormal(v, mu_ucb, sigma_ucb)
                threshold = find_tau_cdf(cost)

        if v_transformed > threshold:
            break

        next_end = min(open_count + batch_size, n_total)
        for i in range(open_count, next_end):
            next_val = data[i]
            observed_rewards.append(next_val)
            observed_exp_rewards.append(np.exp(next_val))
            if next_val > max_until["value"]:
                max_until["value"] = next_val
                max_until["index"] = i
                max_until["exp_value"] = np.exp(next_val)
        open_count = next_end

    win_rate = acceptance_rate(max_until["value"], global_opt["value"])
    utility = win_rate - cost * open_count

    dist_name = "LogNormal" if distribution == "lognormal" else "Exponential"

    return {
        "dist_name": dist_name,
        "transformation": transformation,
        "exp_score": max_until["exp_value"],
        "open_count": open_count,
        "score": max_until["value"],
        "opt": global_opt["value"],
        "exp_opt": global_opt["exp_value"],
        "win_rate": win_rate,
        "acceptance_rate": win_rate,
        "max_until": max_until,
        "global_opt": global_opt,
        "revenue": utility,
        "utility": utility,
        "final_estimation": final_estimated_max / global_opt["exp_value"] if final_estimated_max else None,
    }


def build_permutations(rewards, epoch, rng):
    return [rng.permutation(rewards) for _ in range(epoch)]


def compute_global_opt(rewards, alpha):
    quantile_value = alpha_quantile(rewards, alpha)
    return {
        "value": float(quantile_value),
        "exp_value": float(np.exp(quantile_value)),
    }


def compute_adaptive_results(permutations, costs, delta, alpha,
                             distribution="shifted_exponential",
                             transformation="bradley_terry",
                             batch_size=1,
                             on_cost_done=None):
    """
    Compute adaptive Pandora's Box results.

    Args:
        permutations: list of permuted reward arrays
        costs: list of cost values to evaluate
        delta: confidence parameter
        alpha: quantile for benchmark
        distribution: "shifted_exponential" or "lognormal"
        transformation: "bradley_terry" or "cdf"
        batch_size: number of boxes to open per step
        on_cost_done: callback called after each cost is processed

    Returns:
        list of result dicts, one per cost
    """
    adaptive = []
    for cost in costs:
        outs = []
        for perm in permutations:
            outs.append(pandoras_box(
                perm, cost, delta, alpha=alpha,
                distribution=distribution,
                transformation=transformation,
                batch_size=batch_size,
            ))
        result = {
            "cost": cost,
            "distribution": distribution,
            "transformation": transformation,
            "mean": float(np.mean([x["win_rate"] for x in outs])),
            "median": float(np.median([x["win_rate"] for x in outs])),
            "mean_utility": float(np.mean([x["utility"] for x in outs])),
            "median_utility": float(np.median([x["utility"] for x in outs])),
            "sample_count": int(np.mean([x["open_count"] for x in outs])),
            "revenue": float(np.mean([x["utility"] for x in outs])),
            "outs": outs,
        }
        adaptive.append(result)
        if on_cost_done is not None:
            on_cost_done(result, adaptive)
    return adaptive


def pandoras_box_target_wr(data, target_wr, delta, min_open_count=10, alpha=0.99,
                           distribution="shifted_exponential", transformation="bradley_terry",
                           batch_size=1):
    """
    Run Pandora's Box algorithm with a target win-rate threshold.

    Args:
        data: array of reward values
        target_wr: target win-rate threshold in transformed space
        delta: confidence parameter
        min_open_count: minimum samples before stopping
        alpha: quantile for benchmark (used in BT benchmark estimation)
        distribution: "shifted_exponential" or "lognormal"
        transformation: "bradley_terry" or "cdf"
        batch_size: number of boxes to open per step

    Returns:
        dict with results including win_rate, open_count, utility, etc.
    """
    n_total = data.shape[0]
    max_until = {
        "value": -10,
        "exp_value": 0,
        "generator": 0,
        "index": 0,
    }

    observed_rewards = []
    observed_exp_rewards = []

    min_open_count = min(min_open_count, n_total)
    open_count = min_open_count
    for i in range(min_open_count):
        observed_rewards.append(data[i])
        observed_exp_rewards.append(np.exp(data[i]))
        if data[i] > max_until["value"]:
            max_until["value"] = data[i]
            max_until["index"] = i
            max_until["exp_value"] = np.exp(data[i])

    global_quantile = alpha_quantile(data, alpha) if n_total else -np.inf
    global_opt = {
        "value": global_quantile,
        "exp_value": float(np.exp(global_quantile)) if n_total else 0.0,
        "generator": 0,
    }

    final_estimated_max = None

    while open_count < n_total:
        n = open_count
        v = max_until["exp_value"]
        rewards_arr = np.array(observed_rewards)
        exp_rewards_arr = np.array(observed_exp_rewards)

        if distribution == "shifted_exponential":
            dist_params = fit_shifted_exponential(exp_rewards_arr, n, delta)
            loc = dist_params["loc"]
            scale = dist_params["scale"]
            scale_ucb = dist_params["scale_ucb"]
            scale_lcb = dist_params["scale_lcb"]

            estimated_max = loc + (-scale * np.log(1 - alpha))
            final_estimated_max = estimated_max

            if transformation == "bradley_terry":
                v_transformed = transform_bradley_terry(v, estimated_max)
            else:
                v_transformed = transform_cdf_shifted_exp(v, loc, scale)

        else:
            dist_params = fit_lognormal(rewards_arr, n, delta)
            mu = dist_params["mu"]
            mu_ucb = dist_params["mu_ucb"]
            mu_lcb = dist_params["mu_lcb"]
            sigma = dist_params["sigma"]
            sigma_ucb = dist_params["sigma_ucb"]
            sigma_lcb = dist_params["sigma_lcb"]

            estimated_max = np.exp(mu_ucb + sigma * norm.ppf(alpha))
            final_estimated_max = estimated_max

            if transformation == "bradley_terry":
                v_transformed = transform_bradley_terry(v, estimated_max)
            else:
                v_transformed = transform_cdf_lognormal(v, mu_ucb, sigma)

        # print(v_transformed, target_wr, estimated_max, np.exp(global_opt["value"]))

        if v_transformed > target_wr:
            break

        next_end = min(open_count + batch_size, n_total)
        for i in range(open_count, next_end):
            next_val = data[i]
            observed_rewards.append(next_val)
            observed_exp_rewards.append(np.exp(next_val))
            if next_val > max_until["value"]:
                max_until["value"] = next_val
                max_until["index"] = i
                max_until["exp_value"] = np.exp(next_val)
        open_count = next_end

    win_rate = acceptance_rate(max_until["value"], global_opt["value"])
    dist_name = "LogNormal" if distribution == "lognormal" else "Exponential"

    return {
        "dist_name": dist_name,
        "transformation": transformation,
        "exp_score": max_until["exp_value"],
        "open_count": open_count,
        "score": max_until["value"],
        "opt": global_opt["value"],
        "exp_opt": global_opt["exp_value"],
        "win_rate": win_rate,
        "acceptance_rate": win_rate,
        "max_until": max_until,
        "global_opt": global_opt,
        "target_wr": target_wr,
        "final_estimation": final_estimated_max / global_opt["exp_value"] if final_estimated_max else None,
    }


def compute_fixed_n_results(permutations, costs, global_quantile):
    n_total = permutations[0].shape[0]
    results_by_cost = []
    best_by_cost = []
    for cost in costs:
        results = []
        for n in range(1, n_total + 1):
            acceptance_rates = []
            utilities = []
            for perm in permutations:
                sampled_rewards = perm[:n]
                best = np.max(sampled_rewards)
                win_rate = acceptance_rate(best, global_quantile)
                acceptance_rates.append(win_rate)
                utilities.append(win_rate - cost * n)
            results.append(
                {
                    "n": n,
                    "mean": float(np.mean(acceptance_rates)),
                    "median": float(np.median(acceptance_rates)),
                    "mean_utility": float(np.mean(utilities)),
                    "median_utility": float(np.median(utilities)),
                }
            )
        best = max(results, key=lambda x: x["mean_utility"])
        results_by_cost.append({"cost": cost, "results": results})
        best_by_cost.append({"cost": cost, **best})
    return results_by_cost, best_by_cost


def _reward_generator(rewards):
    for reward in rewards:
        yield "", reward


def compute_beacon_results(permutations, costs, global_quantile, grid_size=200, on_cost_done=None):
    from beacon import BeaconSampler

    n_total = permutations[0].shape[0]
    base_sampler = BeaconSampler(costs[0], n_total, grid_size=grid_size, verbose=False)
    h_table = base_sampler.h_table

    results = []
    for cost in costs:
        sampler = BeaconSampler(cost, n_total, grid_size=grid_size, h_table=h_table, verbose=False)
        outs = []
        for perm in permutations:
            gen = _reward_generator(perm)
            best_sample, samples = sampler.run("", gen, verbose=False)
            best_reward = best_sample["reward"]
            win_rate = acceptance_rate(best_reward, global_quantile)
            utility = win_rate - cost * len(samples)
            outs.append(
                {
                    "win_rate": float(win_rate),
                    "open_count": len(samples),
                    "best_reward": float(best_reward),
                    "acceptance_rate": float(win_rate),
                    "utility": float(utility),
                }
            )
        result = {
            "cost": cost,
            "mean": float(np.mean([x["win_rate"] for x in outs])),
            "median": float(np.median([x["win_rate"] for x in outs])),
            "mean_utility": float(np.mean([x["utility"] for x in outs])),
            "median_utility": float(np.median([x["utility"] for x in outs])),
            "sample_count": int(np.mean([x["open_count"] for x in outs])),
            "outs": outs,
        }
        results.append(result)
        if on_cost_done is not None:
            on_cost_done(result, results)
    return results
