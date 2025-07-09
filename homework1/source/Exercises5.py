"""
Copyright (C) 2024 Fu Tszkok

:module: Exercises5
:function: System reliability simulation with parallel components
:author: Fu Tszkok
:date: 2024-10-31
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np


def false_simulate_system(num_trials):
    """Simulate system reliability with incorrect path selection

    Args:
        num_trials (int): Number of Monte Carlo simulations to run

    Returns:
        float: Estimated system reliability (0.0 to 1.0)

    Note:
        This implementation incorrectly assumes 50% probability 
        of choosing path A vs path B+C, which doesn't match the 
        actual system behavior where path A is always tried first.
    """
    successes = 0

    for _ in range(num_trials):
        # Randomly choose path A (50%) or path B+C (50%)
        if np.random.rand() <= 0.5:  # Path A selected
            # Component A succeeds with 85% probability
            if np.random.rand() <= 0.85:
                successes += 1
        else:  # Path B+C selected
            # Both components must succeed
            b_success = np.random.rand() <= 0.95
            c_success = np.random.rand() <= 0.90
            if b_success and c_success:
                successes += 1

    return successes / num_trials


def true_simulate_system(num_trials):
    """Simulate system reliability with correct path selection

    Args:
        num_trials (int): Number of Monte Carlo simulations to run

    Returns:
        float: Estimated system reliability (0.0 to 1.0)

    Note:
        Correct implementation where path A is always tried first,
        and path B+C is only attempted if path A fails.
    """
    successes = 0

    for _ in range(num_trials):
        # First try path A
        if np.random.rand() <= 0.85:  # Component A succeeds
            successes += 1
            continue  # Skip B+C if A succeeded

        # If A failed, try path B+C
        b_success = np.random.rand() <= 0.95
        c_success = np.random.rand() <= 0.90
        if b_success and c_success:  # Both B and C must succeed
            successes += 1

    return successes / num_trials


# Simulation parameters
num_trials = 100000  # Number of Monte Carlo trials

# Run simulations
false_estimated_reliability = false_simulate_system(num_trials)
true_estimated_reliability = true_simulate_system(num_trials)

# Print results with 5 decimal places precision
print(f"False Estimated Reliability: {false_estimated_reliability:.5f}")
print(f"True Estimated Reliability: {true_estimated_reliability:.5f}")
