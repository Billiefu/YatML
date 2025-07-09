"""
Copyright (C) 2024 Fu Tszkok

:module: Exercises1(b)
:function: Linear regression with batch gradient descent (multi-configuration)
:author: Fu Tszkok
:date: 2024-11-24
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
train = np.loadtxt('./data/dataForTrainingLinear.txt', delimiter=' ', dtype='float')
test = np.loadtxt('./data/dataForTestingLinear.txt', delimiter=' ', dtype='float')
print('Total number of training data:', len(train))
print('Total number of test data:', len(test))
print()

# Standardize features using training data statistics
scaler = StandardScaler()
scaler.fit(train)  # Fit only on training data to prevent data leakage
train = scaler.transform(train)
test = scaler.transform(test)

# Split into features (x) and target (y)
x_train, y_train = train[:, :-1], train[:, -1].flatten()
x_test, y_test = test[:, :-1], test[:, -1].flatten()


def batch_gradient_descent(learning_rate, num_iterations, k):
    """Perform batch gradient descent for linear regression

    Args:
        learning_rate (float): Step size for parameter updates
        num_iterations (int): Total number of training iterations
        k (int): Interval for calculating and storing losses

    Returns:
        tuple: (theta, train_losses, test_losses)
            theta: Final model parameters (weights + bias)
            train_losses: Training RMSE recorded every k iterations
            test_losses: Testing RMSE recorded every k iterations
    """
    # Add intercept term (bias) to features
    X = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=-1)
    X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)

    # Initialize parameters (weights + bias)
    theta = np.zeros(X.shape[1])
    m = len(y_train)  # Number of training examples

    # Track losses during training
    train_losses = []
    test_losses = []

    for i in trange(num_iterations):
        # Compute gradient (vectorized implementation)
        gradients = 2 / m * X.T @ (X @ theta - y_train)

        # Update parameters
        theta -= learning_rate * gradients

        # Record losses periodically
        if i % k == 0:
            train_loss = np.sqrt(np.square((X @ theta - y_train).mean()))
            train_losses.append(train_loss)

            test_loss = np.sqrt(np.square((X_test @ theta - y_test).mean()))
            test_losses.append(test_loss)

    return theta, train_losses, test_losses


# Training configurations
learning_rate = [0.15, 0.015, 0.0015, 0.00015, 0.000015, 0.0000015, 1.1]
num_iterations = [50, 500, 5000, 50000, 500000, 5000000, 100]
K = [5, 50, 500, 5000, 50000, 500000, 5]

# Run training with different configurations
for i in range(min(len(learning_rate), len(num_iterations), len(K))):
    rate, iteration, k = learning_rate[i], num_iterations[i], K[i]

    # Train model with current configuration
    theta, train_losses, test_losses = batch_gradient_descent(rate, iteration, k)

    # Print training results
    print('Learning rate:', rate)
    print('Number of iterations:', iteration)
    print('Regression coefficients:', theta)
    print()

    # Plot training progress
    plt.plot(np.arange(0, iteration, k), train_losses, color='blue', label='Train Loss')
    plt.plot(np.arange(0, iteration, k), test_losses, color='red', linestyle='--', label='Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title(f'Training Progress (learning rate = {rate})')
    plt.show()
