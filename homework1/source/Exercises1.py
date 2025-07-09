"""
Copyright (C) 2024 Fu Tszkok

:module: Exercises1
:function: Implementation of Monte Carlo simulation for π estimation
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


def estimate_pi(num_points, num_trials):
    """Estimate π using Monte Carlo simulation

    Parameters:
    num_points (int): Number of random points to generate per trial
    num_trials (int): Number of independent trials to run

    Returns:
    tuple: (mean_pi, variance_pi) 
           mean_pi - average π estimate across all trials
           variance_pi - variance of π estimates
    """
    pi_estimates = []  # Store π estimates from each trial

    for _ in range(num_trials):
        # Generate uniformly distributed random points in unit square [0,1]×[0,1]
        x = np.random.uniform(0, 1, num_points)
        y = np.random.uniform(0, 1, num_points)

        # Check if points lie within quarter circle (x² + y² ≤ 1)
        inside_circle = (x ** 2 + y ** 2) <= 1
        num_inside = np.sum(inside_circle)

        # π estimate formula: (points_in_circle/total_points) * 4
        pi_estimate = (num_inside / num_points) * 4
        pi_estimates.append(pi_estimate)

    return np.mean(pi_estimates), np.var(pi_estimates)


# Experiment parameters
N_values = [50, 100, 200, 300, 500, 1000, 5000]  # Different sample sizes to test
num_trials = 100  # Number of independent trials per sample size

# Store results as list of tuples (N, mean_pi, variance_pi)
results = []

# Run simulation for each sample size
for N in N_values:
    mean_pi, var_pi = estimate_pi(N, num_trials)
    results.append((N, mean_pi, var_pi))

# Convert results to pandas DataFrame for better display and analysis
results_df = pd.DataFrame(
    results,
    columns=['N', 'Means', 'Variances']  # Column names
)

# Display the results table
print("Monte Carlo π Estimation Results:")
print(results_df)
