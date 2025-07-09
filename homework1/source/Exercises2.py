"""
Copyright (C) 2024 Fu Tszkok

:module: Exercises2
:function: Monte Carlo integration of x³ over [0,1]
:author: Fu Tszkok
:date: 2024-10-31
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms:
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
import pandas as pd


def monte_carlo_integration(n_samples, n_trials):
    """Estimate the integral of x³ from 0 to 1 using Monte Carlo method

    Parameters:
    n_samples (int): Number of random samples per trial
    n_trials (int): Number of independent trials to perform

    Returns:
    tuple: (mean_estimate, variance_estimate)
           mean_estimate - average integral estimate across trials
           variance_estimate - variance of the estimates
    """
    estimates = []  # Store integral estimates from each trial

    for _ in range(n_trials):
        # Generate uniform random samples in [0,1]
        x_samples = np.random.uniform(0, 1, n_samples)

        # Compute Monte Carlo estimate: E[x³] ≈ mean(x³)
        # Since interval length is 1, no need for scaling
        integral_estimate = np.mean(x_samples ** 3)
        estimates.append(integral_estimate)

    return np.mean(estimates), np.var(estimates)


# Experiment configuration
N_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100]  # Different sample sizes to test
n_trials = 100  # Number of trials per sample size

# Results storage
results = []  # Will store tuples of (N, mean, variance)

# Run integration for each sample size
for N in N_values:
    mean_est, var_est = monte_carlo_integration(N, n_trials)
    results.append((N, mean_est, var_est))

# Create pandas DataFrame for analysis and visualization
results_df = pd.DataFrame(
    results,
    columns=['N', 'Means', 'Variances']  # Descriptive column names
)

# Print formatted results
print("Monte Carlo Integration Results (x³ over [0,1]):")
print(results_df)
