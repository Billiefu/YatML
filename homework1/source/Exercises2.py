import numpy as np
import pandas as pd


# Function to estimate the integral using Monte Carlo method
def monte_carlo_integration(n_samples, n_trials):
    estimates = []

    for _ in range(n_trials):
        # Sample uniformly from [0, 1]
        x_samples = np.random.uniform(0, 1, n_samples)
        # Compute the estimate of the integral
        integral_estimate = np.mean(x_samples ** 3)  # Since the interval length is 1
        estimates.append(integral_estimate)

    return np.mean(estimates), np.var(estimates)


# Parameters
N_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
n_trials = 100

# Store results
results = []

for N in N_values:
    mean_estimate, variance_estimate = monte_carlo_integration(N, n_trials)
    results.append((N, mean_estimate, variance_estimate))

# Create a DataFrame for better visualization
results_df = pd.DataFrame(results, columns=['N', 'Means', 'Variances'])

# Display the results
print(results_df)
