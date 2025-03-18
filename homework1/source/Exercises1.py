import numpy as np
import pandas as pd


# Function to estimate pi using the Monte Carlo method
def estimate_pi(num_points, num_trials):
    pi_estimates = []

    for _ in range(num_trials):
        # Generate random points
        x = np.random.uniform(0, 1, num_points)
        y = np.random.uniform(0, 1, num_points)

        # Check how many points fall inside the quarter circle
        inside_circle = (x ** 2 + y ** 2) <= 1
        num_inside = np.sum(inside_circle)

        # Estimate pi
        pi_estimate = (num_inside / num_points) * 4
        pi_estimates.append(pi_estimate)

    return np.mean(pi_estimates), np.var(pi_estimates)


# Parameters
N_values = [50, 100, 200, 300, 500, 1000, 5000]
num_trials = 100

# Store results
results = []

for N in N_values:
    mean_pi, var_pi = estimate_pi(N, num_trials)
    results.append((N, mean_pi, var_pi))

# Create a DataFrame for better visualization
results_df = pd.DataFrame(results, columns=['N', 'Means', 'Variances'])

# Display the results
print(results_df)
