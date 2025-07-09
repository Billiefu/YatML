"""
Copyright (C) 2024 Fu Tszkok

:module: Exercises3
:function: Monte Carlo integration of 2D function over rectangular domain
:author: Fu Tszkok
:date: 2024-10-31
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
import pandas as pd


def f(x, y):
    """Compute the 2D function value at given points

    Parameters:
    x (ndarray): x-coordinates
    y (ndarray): y-coordinates

    Returns:
    ndarray: Computed function values f(x,y) = (y²e^(-y²) + x⁴e^(-x²)) / (xe^(-x²))
    """
    numerator = y ** 2 * np.exp(-y ** 2) + x ** 4 * np.exp(-x ** 2)
    denominator = x * np.exp(-x ** 2)
    return numerator / denominator


def monte_carlo_integration(n_samples, n_trials):
    """Perform Monte Carlo integration of f(x,y) over rectangular domain

    Parameters:
    n_samples (int): Number of random samples per trial
    n_trials (int): Number of independent trials

    Returns:
    tuple: (mean_estimate, variance_estimate)
           mean_estimate - average integral estimate
           variance_estimate - variance across trials
    """
    estimates = []

    for _ in range(n_trials):
        # Generate uniform random samples
        x_samples = np.random.uniform(2, 4, n_samples)  # x ∈ [2,4]
        y_samples = np.random.uniform(-1, 1, n_samples)  # y ∈ [-1,1]

        # Compute function values
        f_values = f(x_samples, y_samples)

        # Integral estimate = average(f) * area
        integral_estimate = np.mean(f_values) * 4  # Area = (4-2)*(1-(-1)) = 4
        estimates.append(integral_estimate)

    return np.mean(estimates), np.var(estimates)


# Experiment parameters
N_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 200]  # Sample sizes to test
n_trials = 100  # Trials per sample size

# Store results as (N, mean, variance)
results = []

# Run integration for each sample size
for N in N_values:
    mean_est, var_est = monte_carlo_integration(N, n_trials)
    results.append((N, mean_est, var_est))

# Create results dataframe
results_df = pd.DataFrame(
    results,
    columns=['N', 'Means', 'Variances']
)

# Display formatted results
print("Function: f(x,y) = (y²e^(-y²) + x⁴e^(-x²)) / (xe^(-x²))")
print("Domain: x ∈ [2,4], y ∈ [-1,1]")
print("==============================")
print("Monte Carlo Integration Results:")
print(results_df)
