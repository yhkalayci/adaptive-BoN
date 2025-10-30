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



import numpy as np
from scipy import optimize
from scipy.stats import expon
import timeit

def find_tau_discrete_optimized(scale, estimated_max, c, loc=0, n_points=4000, max_tau_search=0.99):
    """
    Highly optimized version for exponential distribution.
    
    Parameters:
    -----------
    scale : float
        Scale parameter of exponential distribution (1/lambda)
    estimated_max : float
        Parameter M in f(x) = x/(x+M)
    c : float
        Target expected excess value
    loc : float
        Location parameter
    n_points : int
        Number of points for integration
    max_tau_search : float
        Maximum tau to search
        
    Returns:
    --------
    float : The tau value
    """
    
    # Pre-compute constants
    lam = 1.0 / scale  # Lambda parameter
    
    # Pre-compute maximum x value for integration (99.99th percentile)
    x_max_base = loc - scale * np.log(1 - 0.9999)  # Analytical ppf for exponential
    
    def expected_excess_fast(tau):
        """Optimized expected excess computation"""
        if tau <= 0:
            # For tau <= 0, we need E[f(X)] - tau
            # E[f(X)] can be computed more efficiently
            x_points = np.linspace(loc, x_max_base, n_points)
            dx = (x_max_base - loc) / (n_points - 1)
            
            # Vectorized PDF computation for exponential
            pdf_values = lam * np.exp(-lam * (x_points - loc))
            f_values = x_points / (x_points + estimated_max)
            f_values = np.max(f_values, 0.5)
            
            return np.sum((f_values - tau) * pdf_values) * dx
        
        if tau >= 1:
            return 0.0
            
        # Calculate x0 where f(x0) = tau
        x0 = tau * estimated_max / (1 - tau)
        x0 = max(x0, loc)
        
        # Dynamic upper bound
        x_max = max(x_max_base, x0 * 2)
        
        # Create integration points from x0 to x_max
        x_points = np.linspace(x0, x_max, n_points)
        dx = (x_max - x0) / (n_points - 1)
        
        # Vectorized computations
        pdf_values = lam * np.exp(-lam * (x_points - loc))
        f_values = x_points / (x_points + estimated_max)
        excess_values = f_values - tau  # All positive since x >= x0
        
        return np.sum(excess_values * pdf_values) * dx
    
    def objective(tau):
        return expected_excess_fast(tau) - c
    
    try:
        result = optimize.brentq(objective, 0.001, max_tau_search, xtol=1e-8)
        return result
    except ValueError:
        # Fallback logic
        try:
            max_expected = expected_excess_fast(0.001)
            min_expected = expected_excess_fast(max_tau_search)
            
            if c > max_expected:
                return 0.001
            elif c < min_expected:
                return max_tau_search
            else:
                return optimize.brentq(objective, 0.001, max_tau_search, xtol=1e-6)
        except:
            return np.nan


def pandoras_box(data, cost, delta, min_open_count=20, divide=5):
    """
    Implements the Pandora's Box algorithm.

    Args:
        data (np.ndarray): An array of reward values.
        cost (float): The cost of opening a box.
        delta (float): The confidence level.
        min_open_count (int, optional): The minimum number of boxes to open. Defaults to 20.
        divide (int, optional): A parameter used in the estimation of the maximum value. Defaults to 5.

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
        # factor = 1 + np.power( np.log(np.log(n)) * np.log(1/delta) / n , 1/2)
        ucb_factor = 1 + np.power( np.log(n) * np.log(1/delta) / n , 1/2)
        lcb_factor = 1 - np.power( np.log(n) * np.log(1/delta) / n , 1/2)
        mean = max(mean, 0.0001)
        ucb_mean = mean * ucb_factor
        lcb_mean = mean * lcb_factor
        estimated_max = median + np.log(N/divide) * lcb_mean
        
        dist = stats.expon(median, ucb_mean)
        # c2 = find_tau_discrete_optimized(mean, estimated_max, cost, median)
        c2 = find_tau_discrete_optimized(mean, estimated_max, cost, loc=median)
        # c3 = faircap_shifted_exp(1/mean, median, estimated_max, cost, tol=1e-10)
        # print(c2, c3)
        # print(c2, estimated_max, global_opt["exp_value"])
        v = max_until["exp_value"]
        v_transformed = v/(v+estimated_max)
        
        if v_transformed > c2:
            # print(c2, estimated_max/global_opt["exp_value"])
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
        "revenue": max_until["exp_value"]/(max_until["exp_value"] + global_opt["exp_value"]) - cost*open_count,
        "final_estimation": estimated_max/global_opt["exp_value"]
    }
    
    
def single_prompt_analysis(data, index, epoch=100, threshold=0.485, log=False, costs=None, delta = 0.05):
    """
    Performs a single prompt analysis.

    Args:
        data (list): The input data.
        index (int): The index of the prompt to analyze.
        epoch (int, optional): The number of epochs to run. Defaults to 100.
        threshold (float, optional): The threshold for the win rate. Defaults to 0.485.
        log (bool, optional): Whether to log the results. Defaults to False.
        costs (list, optional): A list of costs to use. Defaults to None.
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

    if costs is not None:
        # all_wins = []
        for c in costs:
            outs = []
            for i in range(epoch):
                out = pandoras_box(all_rewards[i], c, delta)
                outs.append(out)
            
            pb_out.append({
                "cost": c,
                "mean": np.mean([x["win_rate"] for x in outs]),
                "median": np.median([x["win_rate"] for x in outs]),
                "index_average": int(np.mean([x["max_until"]["index"]+1 for x in outs]))+1,
                "sample_count": int(np.mean([x["open_count"] for x in outs]))+1,
                "revenue": np.mean([x["revenue"] for x in outs]),
                "outs": outs
            })
        return win_rates, pb_out



import json
import os
import numpy as np
import tqdm
from multiprocessing import Pool, Process, Queue

def worker_function(args_tuple):
    """Worker function that processes a single task"""
    i, data, costs, rm_name_val = args_tuple
    
    # Set global rm_name for this process
    import builtins
    builtins.rm_name = rm_name_val
    
    # Process the task
    non_adaptive, adaptive = single_prompt_analysis(data, i, epoch=100, costs=costs)
    
    return i, non_adaptive, adaptive

def writer_process(result_queue, output_file, total_tasks):
    """Dedicated writer process that writes results as they arrive"""
    completed = 0
    with open(output_file, "w") as f:
        while completed < total_tasks:
            try:
                # Get result from queue
                result = result_queue.get()
                
                if result is None:  # Poison pill to stop writer
                    break
                
                i, non_adaptive, adaptive = result
                
                # Write immediately
                f.write(json.dumps({"task_id": i, "na": non_adaptive, "a": adaptive}, default=float) + "\n")
                f.flush()  # Ensure immediate write to disk
                
                completed += 1
                print(f"Task {i} completed and written ({completed}/{total_tasks})")
                
            except Exception as e:
                print(f"Writer error: {e}")
                break

def run(args):
    """
    Runs the experiment.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    
    global rm_name
    rm_name = args.rm_name
    np.random.seed(int(args.seed))
    
    output_file = f"{args.output_folder}/cost_range_{args.llm_name}_{args.rm_name}_{args.seed}.jsonl"
    data = read_data(args.input_folder, args.llm_name, args.rm_name)
    costs = [0.008, 0.006, 0.004, 0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0002, 0.0001, 0.00008, 0.00006]
    # costs = [0.01, 0.008, 0.006]
    
    # Create queue for results
    result_queue = Queue()
    
    # Start writer process
    writer = Process(target=writer_process, args=(result_queue, output_file, 100))
    writer.start()
    
    # Prepare arguments for parallel processing
    worker_args = [(i, data, costs, rm_name) for i in range(100)]
    
    # Use multiprocessing with 8 processes
    print("Starting parallel processing with 8 workers...")
    with Pool(processes=8) as pool:
        # Process tasks and send results to queue
        for result in tqdm.tqdm(pool.imap_unordered(worker_function, worker_args), total=100, desc="Processing tasks"):
            result_queue.put(result)
    
    # Signal writer to stop and wait for it to finish
    result_queue.put(None)  # Poison pill
    writer.join()
    
    print(f"Completed processing 100 tasks. Results saved to {output_file}")

    
    
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
