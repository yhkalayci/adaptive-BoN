"""
BEACON: Bayesian Efficient Adaptive Criterion for Optimal N-stopping

Implementation based on the paper:
"BEACON: Bayesian Optimal Stopping for Efficient LLM Sampling"

This module provides the BeaconSampler class which implements the Universal Index
Policy (UIP) for optimal stopping in sequential sampling problems via
Backward Induction.
"""

import numpy as np
from scipy.stats import t as student_t
from scipy.interpolate import interp1d
import json
import pickle


def read_data(input_folder, llm_name, _rm_name=None):
    """
    Reads data from a JSONL file.

    Args:
        input_folder (str): The folder containing the data file.
        llm_name (str): The name of the language model.
        _rm_name (str): The name of the reward model (unused, kept for API compatibility).

    Returns:
        list: A list of dictionaries, where each dictionary represents a line in the JSONL file.
    """
    file_name = f"{input_folder}/{llm_name}_output.merged_rm.jsonl"
    with open(file_name, "r") as f:
        return [json.loads(line) for line in f]


class BeaconSampler:
    """
    BEACON sampler implementing the Universal Index Policy (UIP) for optimal stopping.

    The sampler uses Bayesian learning with a Normal-Inverse-Gamma conjugate prior
    to estimate the reward distribution. It precomputes an h-index table using
    backward induction on the Bellman equation to determine the optimal stopping point.
    """

    def __init__(self, cost, max_samples, grid_size=100, h_table=None, verbose=True):
        """
        Initialize the BEACON sampler.

        Args:
            cost (float): The cost per sample (c).
            max_samples (int): Maximum budget for samples (n).
            grid_size (int): Resolution for the precomputed h-table.
            h_table (dict, optional): Precomputed h-table to reuse.
            verbose (bool): Whether to print precomputation logs.
        """
        self.c = cost
        self.n = max_samples
        self.grid_size = grid_size
        self.verbose = verbose

        # Jeffreys' non-informative prior parameters (α₀, ν₀, β₀, μ₀)
        # Paper Algorithm 1, Line 1
        self.alpha0 = -0.5
        self.nu0 = 0
        self.beta0 = 0
        self.mu0 = 0

        if h_table is not None:
            self.h_table = h_table
        else:
            self.h_table = self._precompute_h_table()

    def _compute_student_t_ei(self, z_hat_arr, df):
        """
        Compute Expected Improvement (Marginal Gain) for Standardized Student-t.

        For a standardized variable T ~ t_df(0, 1):
        EI(z) = ∫_{z}^{∞} (u - z) * pdf(u) du

        Closed form for standardized Student-t:
        EI(z) = (df + z²)/(df - 1) * pdf(z) - z * (1 - cdf(z))

        Note: The paper assumes df > 1 (k >= 3 implies df >= 2).
        """
        z_hat_arr = np.atleast_1d(z_hat_arr)

        # Require df > 1 for mean existence in EI
        if df <= 1:
            return np.zeros_like(z_hat_arr)

        rv = student_t(df)
        pdf_z = rv.pdf(z_hat_arr)
        cdf_z = rv.cdf(z_hat_arr)

        # Standardized Student-t EI formula
        term1 = (df + z_hat_arr**2) / (df - 1) * pdf_z
        term2 = z_hat_arr * (1.0 - cdf_z)

        ei = term1 - term2
        return np.maximum(ei, 0.0)

    def _precompute_h_table(self):
        """
        Precompute the h-index table using Backward Induction.

        Implements the recursion based on the Bellman equation (Appendix D.5):

        H_{n,k}(z; c) = EI_k(z) + ∫ σ_u × max{0, H_{n,k+1}(z_next; c/σ_u) - c/σ_u} dF(u)

        Where:
        - EI_k(z) is the myopic expected improvement (one-step gain)
        - The integral represents the expected continuation value
        - We find h(z) as the c where H(z; c) = c (the fixed point)

        State Transition (normalized, Eq 5-7):
        - μ_new = u / (k + 1)
        - σ_new² = (k + 2) / (k + 1)² × (k - 1 + u²)
        - z_new = max(z, u)
        """
        if self.verbose:
            print("Precomputing BEACON h-table (Backward Induction)...")

        # 1. Setup Grids
        # z_grid: Standardized best reward. Paper Appendix D.5 suggests [-30, 30]
        z_grid = np.linspace(-30, 30, self.grid_size)

        # c_grid: Cost grid for solving H(c) = c
        c_grid = np.logspace(-5, 1, 100)
        n_c = len(c_grid)

        # Integration points for u (next standardized observation)
        # u ~ t_{df}. Range [-20, 20] covers >99.9% probability mass for df >= 2
        n_int = 150
        u_nodes = np.linspace(-20, 20, n_int)

        # Initialize Future Value Table: H_{n, n} = 0 (boundary condition)
        # Shape: [cost_idx, z_idx]
        H_future = np.zeros((n_c, self.grid_size))

        h_table = {}

        # 2. Backward Induction Loop: from n-1 down to k₀=3
        for k in range(self.n - 1, 2, -1):
            if self.verbose and k % 5 == 0:
                print(f"  Step k={k}...")

            df = k - 1  # Degrees of freedom = 2α_k = k - 1

            # Precompute integration weights (Trapezoidal rule over Student-t PDF)
            rv = student_t(df)
            u_pdf = rv.pdf(u_nodes)
            du = u_nodes[1] - u_nodes[0]
            weights = u_pdf * du

            # A. Calculate Myopic EI (Immediate Gain): H_{k+1, k}(z)
            myopic_gain = self._compute_student_t_ei(z_grid, df)

            # B. Calculate Continuation Value
            continuation_value = np.zeros((n_c, self.grid_size))

            for i, u in enumerate(u_nodes):
                if weights[i] < 1e-9:
                    continue

                # --- State Transition Logic (Normalized) ---
                # Given current normalized state (μ=0, σ=1, best=z_grid), observe u

                # 1. Update Mean (Eq 6 normalized)
                mu_new = u / (k + 1)

                # 2. Update Sigma (Derived from Eq 7 with Jeffreys' prior)
                # σ_new² = (k+2)/(k+1)² × (k-1 + u²)
                sigma_new_sq = (k + 2) / ((k + 1)**2) * (k - 1 + u**2)
                if sigma_new_sq <= 1e-10:
                    continue
                sigma_new = np.sqrt(sigma_new_sq)

                # 3. Update Best Reward
                z_raw_new = np.maximum(z_grid, u)

                # 4. Standardize for Next Step
                z_hat_next = (z_raw_new - mu_new) / sigma_new
                z_hat_next = np.clip(z_hat_next, z_grid[0], z_grid[-1])

                # --- Interpolate H_future on (c_next, z_next) ---

                # Cost scaling: c_next = c / σ_new
                c_next_vec = c_grid / sigma_new
                c_next_vec = np.clip(c_next_vec, c_grid[0], c_grid[-1])

                # Interpolate z dimension (linear)
                z_pos = (z_hat_next - z_grid[0]) / (z_grid[-1] - z_grid[0]) * (self.grid_size - 1)
                z_lo = np.floor(z_pos).astype(int)
                z_hi = np.minimum(z_lo + 1, self.grid_size - 1)
                z_lo = np.maximum(z_lo, 0)
                z_frac = z_pos - z_lo

                # Gather H values at z_lo and z_hi for all costs
                H_z_lo = H_future[:, z_lo]
                H_z_hi = H_future[:, z_hi]
                H_temp = H_z_lo + z_frac * (H_z_hi - H_z_lo)

                # Interpolate c dimension
                c_pos = np.interp(c_next_vec, c_grid, np.arange(n_c))
                c_lo = np.floor(c_pos).astype(int)
                c_hi = np.minimum(c_lo + 1, n_c - 1)
                c_lo = np.maximum(c_lo, 0)
                c_frac = (c_pos - c_lo).reshape(-1, 1)

                # Bilinear interpolation result
                H_next_val = (
                    np.take_along_axis(H_temp, c_lo.reshape(-1, 1), axis=0) * (1 - c_frac) +
                    np.take_along_axis(H_temp, c_hi.reshape(-1, 1), axis=0) * c_frac
                )

                # Compute continuation gain: max{0, H_{n,k+1} - c/σ}
                # This follows from Bellman: V = z + max{0, H - c}
                marginal_gain = np.maximum(0.0, H_next_val - c_next_vec.reshape(-1, 1))

                # Add to integral: σ_new × gain × weight
                continuation_value += sigma_new * marginal_gain * weights[i]

            # C. Total Expected Marginal Gain: H_{n,k}(z; c) = EI + continuation
            H_total = myopic_gain.reshape(1, -1) + continuation_value

            # Store for next backward iteration
            H_future = H_total.copy()

            # D. Solve for h-index: Find c where H(z; c) = c
            h_values = np.zeros(self.grid_size)
            f_c = H_total - c_grid.reshape(-1, 1)

            for z_i in range(self.grid_size):
                col = f_c[:, z_i]

                # Check bounds
                if col[0] < 0:
                    h_values[z_i] = c_grid[0]
                elif col[-1] > 0:
                    h_values[z_i] = c_grid[-1]
                else:
                    # Find first index where f(c) < 0 (H < c)
                    idx = np.argmax(col < 0)
                    if idx == 0:
                        idx = 1

                    # Linear interpolation for root
                    c1, c2 = c_grid[idx-1], c_grid[idx]
                    f1, f2 = col[idx-1], col[idx]

                    if np.abs(f2 - f1) > 1e-12:
                        h_values[z_i] = c1 - f1 * (c2 - c1) / (f2 - f1)
                    else:
                        h_values[z_i] = c1

            # Enforce Monotonicity: h-index is decreasing in z (Theorem 3)
            h_values = np.maximum.accumulate(h_values[::-1])[::-1]

            # Create interpolator
            h_table[k] = interp1d(
                z_grid, h_values, kind='linear',
                bounds_error=False, fill_value=(h_values[0], h_values[-1])
            )

        if self.verbose:
            print("h-table precomputation complete.")

        return h_table

    def _update_stats(self, rewards):
        """
        Update Bayesian posterior parameters given observed rewards.
        Uses Jeffreys' prior updates (Eq 4).

        Returns (μ_k, σ_k) posterior mean and predictive scale.
        """
        k = len(rewards)
        if k == 0:
            return 0.0, 1.0

        rewards_arr = np.asarray(rewards)
        r_bar = np.mean(rewards_arr)

        # Calculate SSD (Sum of Squared Deviations)
        if k > 1:
            ssd = np.sum((rewards_arr - r_bar) ** 2)
        else:
            ssd = 0.0

        # Posterior Parameters (Eq 4 with Jeffreys' Prior)
        nu_k = k
        alpha_k = self.alpha0 + k / 2.0
        beta_k = ssd / 2.0

        # Predictive Scale (derived from Posterior Predictive t-distribution)
        if alpha_k <= 0 or nu_k <= 0 or beta_k <= 0:
            # Fallback for undefined variance (typically k < 3)
            sigma = 1.0
        else:
            sigma = np.sqrt((beta_k * (nu_k + 1)) / (nu_k * alpha_k))

        return r_bar, max(sigma, 1e-8)

    def run(self, prompt, response_generator, verbose=False):
        """
        Execute the BEACON algorithm.

        Args:
            prompt: The input prompt (can be empty string for reward-only mode).
            response_generator: Generator yielding (response_text, reward_score) tuples.
            verbose (bool): Whether to print progress.

        Returns:
            tuple: (best_sample, all_samples) where best_sample is a dict with
                   'text' and 'reward' keys.
        """
        samples = []
        rewards = []

        # Step 1: Collect initial k₀ = 3 samples (Algorithm 1, Line 3)
        # Required for well-defined posterior with Jeffreys' prior
        if verbose:
            prompt_preview = str(prompt)[:50] if prompt else "(no prompt)"
            print(f"--- Processing: {prompt_preview}... ---")

        for _ in range(3):
            try:
                resp, score = next(response_generator)
                samples.append({'text': resp, 'reward': score})
                rewards.append(score)
            except StopIteration:
                break

        if len(rewards) < 3:
            if samples:
                return max(samples, key=lambda x: x['reward']), samples
            return {'text': '', 'reward': float('-inf')}, samples

        # Step 2: Initialize posterior statistics
        mu, sigma = self._update_stats(rewards)
        z = max(rewards)
        k = 3

        # Step 3: Adaptive sampling loop (Algorithm 1, Line 6)
        while k < self.n:
            # Standardized best reward
            z_hat = (z - mu) / sigma

            # Look up h-index from precomputed UIP table
            if k in self.h_table:
                h = float(self.h_table[k](z_hat))
            else:
                # Fallback: use myopic EI as lower bound approximation
                df = k - 1
                h = float(self._compute_student_t_ei(np.array([z_hat]), df)[0])

            # Stopping condition: h(ẑ) ≤ c/σ  ⟺  h·σ ≤ c (Equation 3)
            if h * sigma <= self.c:
                if verbose:
                    print(f"  Stopping at k={k}: h({z_hat:.2f})={h:.4f}, h·σ={h*sigma:.4f} ≤ c={self.c}")
                break

            # Get next sample
            try:
                resp, score = next(response_generator)
            except StopIteration:
                break

            # Robust filtering (Algorithm 1, Line 12 & Appendix E.4)
            # Winsorize rewards below 1% quantile to protect against left-tail outliers
            df = k - 1
            if df > 0:
                q_01 = student_t.ppf(0.01, df, loc=mu, scale=sigma)
                stats_score = score if score >= q_01 else mu
            else:
                stats_score = score

            samples.append({'text': resp, 'reward': score})
            # Use filtered score for stats update, original score for best tracking
            rewards.append(stats_score)
            z = max(z, score)

            # Update posterior
            mu, sigma = self._update_stats(rewards)
            k += 1

        return max(samples, key=lambda x: x['reward']), samples

    def save_h_table(self, filepath):
        """Save the h-table to disk for reuse."""
        z_grid = np.linspace(-30, 30, self.grid_size)
        h_data = {k: interp(z_grid).tolist() for k, interp in self.h_table.items()}
        with open(filepath, 'wb') as f:
            pickle.dump({'z_grid': z_grid.tolist(), 'h_data': h_data, 'n': self.n}, f)

    @classmethod
    def load_h_table(cls, filepath):
        """Load h-table from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        z_grid = np.array(data['z_grid'])
        h_table = {}
        for k, h_values in data['h_data'].items():
            h_table[int(k)] = interp1d(
                z_grid, h_values, kind='linear',
                bounds_error=False, fill_value=(h_values[0], h_values[-1])
            )
        return h_table, data['n']


def real_data_generator(generations, reward_key="mistral_rm_reward"):
    """Generator that yields (text, reward) pairs from pre-generated data."""
    for gen in generations:
        score = gen.get(reward_key)
        if score is None:
            continue
        yield gen.get("text", ""), score


if __name__ == "__main__":
    import time

    print("=== BEACON Demo (Backward Induction UIP Implementation) ===\n")

    # Test with synthetic data
    np.random.seed(42)
    n_samples = 32
    cost = 0.05

    print(f"Initializing BEACON with n={n_samples}, c={cost}")
    print("Running Backward Induction precomputation...")

    start = time.time()
    beacon = BeaconSampler(cost=cost, max_samples=n_samples, grid_size=100, verbose=True)
    prep_time = time.time() - start
    print(f"Precomputation time: {prep_time:.2f}s")

    # Run sampling simulation
    def make_generator():
        while True:
            r = np.random.normal(0, 1)
            yield "response", r

    print("\nRunning simulated queries...")
    results = []
    for i in range(5):
        gen = make_generator()
        best, samples = beacon.run(f"Query {i+1}", gen, verbose=True)
        results.append((best['reward'], len(samples)))

    avg_reward = np.mean([r[0] for r in results])
    avg_samples = np.mean([r[1] for r in results])
    print(f"\nResults: Avg Reward={avg_reward:.3f}, Avg Samples={avg_samples:.1f}")
