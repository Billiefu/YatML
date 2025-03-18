import numpy as np
import pandas as pd


# Function to compute f(x, y)
def f(x, y):
    return (y ** 2 * np.exp(-y ** 2) + x ** 4 * np.exp(-x ** 2)) / (x * np.exp(-x ** 2))


# Function to estimate the integral using Monte Carlo method
def monte_carlo_integration(n_samples, n_trials):
    estimates = []

    for _ in range(n_trials):
        # Sample uniformly from [2, 4] for x and [-1, 1] for y
        x_samples = np.random.uniform(2, 4, n_samples)
        y_samples = np.random.uniform(-1, 1, n_samples)

        # Compute f(x, y) for each sample pair
        f_values = f(x_samples, y_samples)

        # Estimate the integral
        integral_estimate = np.mean(f_values) * 4  # Area = (4 - 2) * (1 - (-1)) = 4
        estimates.append(integral_estimate)

    return np.mean(estimates), np.var(estimates)


# Parameters
N_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 200]
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
