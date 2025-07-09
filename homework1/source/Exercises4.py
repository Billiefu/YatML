"""
Copyright (C) 2024 Fu Tszkok

:module: Exercises4
:function: Monte Carlo simulation of ant path probability on grid
:author: Fu Tszkok
:date: 2024-10-31
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np


def is_within_bounds(x, y, n):
    """Check if coordinates are within the n×n grid boundaries

    Args:
        x (int): x-coordinate
        y (int): y-coordinate
        n (int): grid size

    Returns:
        bool: True if coordinates are within [1,n] range, False otherwise
    """
    return (1 <= x <= n) and (1 <= y <= n)


def simulate_ant(n):
    """Simulate ant's path from (1,1) to (n,n) with movement constraints

    Args:
        n (int): size of the grid

    Returns:
        bool: True if ant reaches (n,n), False otherwise

    Notes:
        - Center point (4,4) can be visited maximum 2 times
        - Other points can be visited maximum 1 time
    """
    # Initialize starting position
    x, y = 1, 1
    # Dictionary to track visit counts for each point
    visits = {(x, y): 1}

    while (x, y) != (n, n):
        # Possible movement directions: right, left, up, down
        moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_moves = []

        # Check validity of each possible move
        for nx, ny in moves:
            if is_within_bounds(nx, ny, n):
                if (nx, ny) == (4, 4):
                    # Center point allows maximum 2 visits
                    if visits.get((nx, ny), 0) < 2:
                        valid_moves.append((nx, ny))
                else:
                    # Other points allow only 1 visit
                    if visits.get((nx, ny), 0) == 0:
                        valid_moves.append((nx, ny))

        # Terminate if no valid moves available
        if not valid_moves:
            break

        # Randomly select next move from valid options
        x, y = valid_moves[np.random.choice(len(valid_moves))]

        # Update visit count for current position
        visits[(x, y)] = visits.get((x, y), 0) + 1

    return (x, y) == (n, n)


def monte_carlo_simulation(trials, n):
    """Estimate probability of ant reaching (n,n) using Monte Carlo method

    Args:
        trials (int): number of simulations to run
        n (int): size of the grid

    Returns:
        float: estimated probability of reaching destination
    """
    success_count = 0
    for _ in range(trials):
        if simulate_ant(n):
            success_count += 1
    return success_count / trials


# Execute simulation
P = monte_carlo_simulation(20000, 7)
print(f"Estimated probability P: {P:.4f}")
