import numpy as np


def is_within_bounds(x, y, n):
    return (1 <= x <= n) and (1 <= y <= n)


def simulate_ant(n):
    # Starting point
    x, y = 1, 1
    # Track visits to points
    visits = {(x, y): 1}

    while (x, y) != (n, n):
        # Possible movements
        moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_moves = []

        for nx, ny in moves:
            if is_within_bounds(nx, ny, n):
                if (nx, ny) == (4, 4):
                    # Center point can be visited up to 2 times
                    if visits.get((nx, ny), 0) < 2:
                        valid_moves.append((nx, ny))
                else:
                    # Other points can be visited only once
                    if visits.get((nx, ny), 0) == 0:
                        valid_moves.append((nx, ny))

        # No valid moves available, terminate
        if not valid_moves:
            break

        # Randomly choose a valid move
        x, y = valid_moves[np.random.choice(len(valid_moves))]

        # Update visits
        visits[(x, y)] = visits.get((x, y), 0) + 1

    return (x, y) == (n, n)


def monte_carlo_simulation(trials, n):
    success_count = 0
    for _ in range(trials):
        if simulate_ant(n):
            success_count += 1
    return success_count / trials


# Running the simulation
P = monte_carlo_simulation(20000, 7)
print(f"Estimated probability P: {P:.4f}")
