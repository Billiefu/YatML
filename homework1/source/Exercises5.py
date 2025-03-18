import numpy as np


# Succeed rate of passing through A or B and C
def false_simulate_system(num_trials):
    successes = 0

    for _ in range(num_trials):
        if np.random.rand() <= 0.5:  # 50% chance to go through A
            # Simulate component A
            if np.random.rand() <= 0.85:
                # Component A success
                successes += 1
        else:  # 50% chance to go through B and C
            # Simulate component B and C
            b_success = np.random.rand() <= 0.95
            c_success = np.random.rand() <= 0.90
            if b_success and c_success:
                # Both components must succeed
                successes += 1

    return successes / num_trials


# Succeed rate of passing through the whole system
def true_simulate_system(num_trials):
    successes = 0

    for _ in range(num_trials):
        # Simulate component A
        if np.random.rand() <= 0.85:
            # Component A success
            successes += 1
            continue
        # Simulate component B and C
        b_success = np.random.rand() <= 0.95
        c_success = np.random.rand() <= 0.90
        if b_success and c_success:
            # Both components must succeed
            successes += 1
            continue

    return successes / num_trials


# Set the number of trials
num_trials = 100000
false_estimated_reliability = false_simulate_system(num_trials)
true_estimated_reliability = true_simulate_system(num_trials)

print(f"False Estimated Reliability: {false_estimated_reliability:.5f}")
print(f"True Estimated Reliability: {true_estimated_reliability:.5f}")
