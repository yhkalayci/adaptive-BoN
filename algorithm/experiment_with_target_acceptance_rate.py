import json
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import timeit
import tqdm
import argparse
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import time
import heapq

llms = ["gemma2_9b", "llama3.1_8b", "llama3.2_3b", "mistral_7b", "qwen2.5_7b", "qwen3_4b"]
rms = ["armo_rm", "fsfairx_rm", "mistral_rm", "skywork_rm"]

def read_data(input_folder, llm_name, rm_name):
    """
    Reads data from a JSONL file.

    Args:
        input_folder (str): The folder containing the data file.
        llm_name (str): The name of the language model.
        rm_name (str): The name of the reward model.

    Returns:
        list: A list of dictionaries, where each dictionary represents a line in the JSONL file.
    """
    file_name = f"{input_folder}/{llm_name}_output.merged_rm.jsonl"
    lines = []
    with open(file_name, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data

class MedianEstimator:
    """
    An efficient class to estimate the median and the mean of the exceedance of a stream of data.
    """
    def __init__(self):
        """
        Initializes the MedianEstimator.
        """
        self.low = []   # max heap (negated values)
        self.high = []  # min heap
        self.high_sum = 0  # sum of elements in high heap
    
    def add(self, x):
        """
        Adds a new data point to the estimator.

        Args:
            x (float): The data point to add.
        """
        # Add to appropriate heap
        if not self.high or x >= self.high[0]:
            heapq.heappush(self.high, x)
            self.high_sum += x
        else:
            heapq.heappush(self.low, -x)
        
        # Rebalance heaps
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
        """
        Calculates the current median of the data stream.

        Returns:
            float: The median of the data stream.
        """
        if not self.high:
            return None
        # if len(self.low) == len(self.high):
            # return (-self.low[0] + self.high[0]) / 2
        return self.high[0]

    @property
    def exceedance_mean(self):
        """
        Calculates the mean of the data points that are greater than or equal to the median.

        Returns:
            float: The mean of the exceedance.
        """
        if not self.high:
            return None
        # All elements in high heap are >= median
        # (In even case, median is between low and high heap tops)
        return self.high_sum / len(self.high)

def fast_expected_excess(tau, loc, lambda_param, M):
    """Optimized numerical computation of E[max(Y-τ, 0)]."""
    if tau >= 0.5:
        return 0.0
    
    # Precompute constants
    exp_loc = np.exp(lambda_param * loc)
    
    def integrand_optimized(y):
        """Optimized integrand computation."""
        if y <= tau:
            return 0.0
        one_minus_y = 1.0 - y
        x = y * M / one_minus_y
        return (y - tau) * lambda_param * M * np.exp(-lambda_param * x) * exp_loc / (one_minus_y * one_minus_y)
    
    # Smart integration limits and method
    if tau < 0.1:
        # For small tau, integrate over important region
        result, _ = quad(integrand_optimized, tau, 0.5, limit=50, epsabs=1e-10)
    else:
        # Split integration for better accuracy
        mid = min(0.4, (tau + 0.5) / 2)
        result1, _ = quad(integrand_optimized, tau, mid, limit=30, epsabs=1e-10)
        result2, _ = quad(integrand_optimized, mid, 0.5, limit=30, epsabs=1e-10)
        result = result1 + result2
    
    # Add point mass at 0.5
    if tau < 0.5:
        result += 0.01 * (0.5 - tau)
    
    return result

def get_cap_value_optimized(cdf, cost, estimated_max):
    """User's CDF-based implementation."""
    exp_max = estimated_max
    eps = min(1e-2, estimated_max/10)
    
    max_iterations = int(np.log(eps / estimated_max) / np.log(1 - eps)) + 1
    
    iteration_indices = np.arange(max_iterations)
    up_values = estimated_max * (1 - eps)**iteration_indices
    
    valid_mask = up_values > eps
    up_values = up_values[valid_mask]
    
    if len(up_values) <= 1:
        return 0
    
    up_next = up_values * (1 - eps)
    
    cdf_up = cdf(up_values[:-1])
    cdf_up_next = cdf(up_next[:-1])
    
    dy_values = cdf_up - cdf_up_next
    
    up_calc = up_next[:-1]
    v_values = up_calc / (up_calc + exp_max)
    
    v_dy_products = v_values * dy_values
    tot_values = np.cumsum(v_dy_products)
    tot_density_values = np.cumsum(dy_values)
    
    condition_values = tot_values - tot_density_values * v_values
    condition_met = condition_values >= cost
    
    if np.any(condition_met):
        first_valid_idx = np.argmax(condition_met)
        return v_values[first_valid_idx]
        
    return 0

def make_cdf_function(loc, lambda_param):
    """Create CDF function for original X ~ Exponential(λ, loc)."""
    def cdf_x(x):
        if hasattr(x, '__len__'):  # Handle arrays
            result = np.zeros_like(x)
            mask = x >= loc
            result[mask] = 1 - np.exp(-lambda_param * (x[mask] - loc))
            return result
        else:  # Handle scalars
            if x < loc:
                return 0.0
            return 1 - np.exp(-lambda_param * (x - loc))
    
    return cdf_x

def hybrid_fast_solver(loc, lambda_param, c, M, newton_steps=1):
    """Hybrid: CDF initial estimate + Newton refinement for <1% error."""
    # Fast initial estimate using CDF method
    cdf = make_cdf_function(loc, lambda_param)
    tau_init = get_cap_value_optimized(cdf, c, M)
    
    if tau_init == 0:
        return 0
    
    # Newton refinement for precision
    tau = tau_init
    h = 1e-7  # Small step for numerical derivative
    
    for _ in range(newton_steps):
        # Fast excess computation
        excess = fast_expected_excess(tau, loc, lambda_param, M)
        excess_h = fast_expected_excess(tau + h, loc, lambda_param, M)
        
        d_excess = (excess_h - excess) / h
        
        if abs(d_excess) > 1e-12:
            delta = -(excess - c) / d_excess
            tau_new = tau + 0.8 * delta  # Damped step
            tau_new = max(0.001, min(tau_new, 0.499))
            
            if abs(tau_new - tau) < 1e-10:
                break
            tau = tau_new
        else:
            break
    
    return tau


def pandoras_box(data, target_wr, delta, min_open_count=20, divide=3):
    """
    Implements the Pandora's Box algorithm with a target win rate.

    Args:
        data (np.ndarray): An array of reward values.
        target_wr (float): The target win rate.
        delta (float): The confidence level.
        min_open_count (int, optional): The minimum number of boxes to open. Defaults to 20.
        divide (int, optional): A parameter used in the estimation of the maximum value. Defaults to 3.

    Returns:
        dict: A dictionary containing the results of the algorithm.
    """
    estimator = MedianEstimator()
    N = data.shape[0]
    max_until = {
        "value": -10,
        "exp_value": 0,
        "generator": 0,
        "index": 0
    }

    open_count = min_open_count
    
    m = len(data)
    for i in range(min_open_count):
        estimator.add(np.exp(data[i]))
        if data[i] > max_until["value"]:
            max_until["value"] = data[i]
            max_until["index"] = i
            max_until["exp_value"] = np.exp(data[i])

    global_opt = {
        "value": np.max(data),
        "exp_value": np.exp(np.max(data)),
        "generator": 0,
        "index": np.argmax(data)
    }
        
    while open_count < N:
        median = estimator.median
        mean = estimator.exceedance_mean - median

        n = open_count
        factor = 1 + np.power( np.log(np.log(n)) * np.log(1/delta) / n , 1/2)
        # factor = 1 + np.power( np.log(n) * np.log(1/delta) / n , 1/2)
        mean = max(mean, 0.0001)
        mean *= factor
        estimated_max = global_opt["exp_value"]
        c2 = target_wr #hybrid_fast_solver(median, 1/mean, cost, estimated_max)
        
        v = max_until["exp_value"]
        v_transformed = v/(v+estimated_max)
        
        if v_transformed > c2:
            break

        next_val = data[open_count]
        estimator.add(np.exp(next_val))
        if next_val > max_until["value"]:
            max_until["value"] = next_val
            max_until["index"] = open_count
            max_until["exp_value"] = np.exp(next_val)

        open_count += 1
    return {
        "dist_name": "LogNormal",
        "exp_score": max_until["exp_value"],
        "open_count": open_count,
        "score": max_until["value"],
        "opt": global_opt["value"],
        "exp_opt": global_opt["exp_value"],
        "win_rate": max_until["exp_value"]/(max_until["exp_value"] + global_opt["exp_value"]),
        "max_until": max_until,
        "global_opt": global_opt,
        # "revenue": max_until["exp_value"]/(max_until["exp_value"] + global_opt["exp_value"]) - cost*open_count
        "target_wr": target_wr
    }
    
def single_prompt_analysis(data, index, epoch=100, threshold=0.485, log=False, target_wrs=[0.4], delta = 0.05):
    """
    Performs a single prompt analysis for a range of target win rates.

    Args:
        data (list): The input data.
        index (int): The index of the prompt to analyze.
        epoch (int, optional): The number of epochs to run. Defaults to 100.
        threshold (float, optional): The threshold for the win rate. Defaults to 0.485.
        log (bool, optional): Whether to log the results. Defaults to False.
        target_wrs (list, optional): A list of target win rates to use. Defaults to [0.4].
        delta (float, optional): The confidence level. Defaults to 0.05.

    Returns:
        tuple: A tuple containing the win rates and the Pandora's Box output.
    """
    rewards = np.array([x[f"{rm_name}_reward"] for x in data[index]["generations"]])
    all_rewards = [np.copy(rewards) for i in range(epoch)]
    for i in range(epoch):
        np.random.shuffle(all_rewards[i])
        
    max_val = np.max(rewards)
    win_rates = []
    for index in range(rewards.shape[0]):
        wrs = []
        scores = []
        for i in range(epoch):
            max_until = np.max(all_rewards[i][:index+1])
            scores.append(max_until)
            wr = np.exp(max_until)/(np.exp(max_until) + np.exp(max_val))
            wrs.append(wr)
        win_rates.append({
            "mean": np.mean(wrs),
            "median": np.median(wrs),
            "sample_count": index+1,
            "scores": scores,
            "max_score": max_val
        })

    last_winrates = []
    pb_out = []

    
    for wr in target_wrs:
        outs = []
        for i in range(epoch):
            out = pandoras_box(all_rewards[i], wr, delta)
            outs.append(out)

        pb_out.append({
            "target_wr": wr,
            "mean": np.mean([x["win_rate"] for x in outs]),
            "median": np.median([x["win_rate"] for x in outs]),
            "index_average": int(np.mean([x["max_until"]["index"]+1 for x in outs]))+1,
            "sample_count": int(np.mean([x["open_count"] for x in outs])),
            "outs": outs
        })
        
    return win_rates, pb_out

def run(args):
    """
    Runs the experiment.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    import os
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    
    global rm_name
    rm_name = args.rm_name

    np.random.seed(int(args.seed))
    
    output_file = f"{args.output_folder}/target_wr_{args.llm_name}_{args.rm_name}_{args.seed}.jsonl"

    data = read_data(args.input_folder, args.llm_name, args.rm_name)
    
    target_wrs = [0.3, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]
    
    NAs = []
    As = []
    for i in tqdm.tqdm(range(100)):
        non_adaptive, adaptive = single_prompt_analysis(data, i, epoch=100, target_wrs=target_wrs)
        NAs.append(non_adaptive)
        As.append(adaptive)
    
        with open(output_file, "a") as f:
            f.write(json.dumps({"na": non_adaptive, "a": adaptive}, default=float) + "\n")

def main():
    """Main function to run the generation script."""
    parser = argparse.ArgumentParser(
        description="Run LLM generation for multiple prompts using vLLM."
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        required=True,
        choices=llms,
        help="Short name of the LLM used for generation."
    )
    parser.add_argument(
        "--rm_name",
        type=str,
        required=True,
        choices=rms,
        help="Short name of the RM used for generation."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="./",
        help="Path to the input folder."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./res",
        help="Path to the output folder."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=46,
        help="Seed for random number generators."
    )

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
