"""
Copyright (C) 2024 Fu Tszkok

:module: Exercises1(a)
:function: Linear regression implementation with batch gradient descent
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

# Load and inspect dataset
train = np.loadtxt('./data/dataForTrainingLinear.txt', delimiter=' ', dtype='float')
test = np.loadtxt('./data/dataForTestingLinear.txt', delimiter=' ', dtype='float')
print('Total number of training data:', len(train))
print('Total number of test data:', len(test))
print()

# Visualize raw data distribution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train[:, 0], train[:, 1], train[:, 2], s=20, c='blue', marker='o', label='Training')
ax.scatter(test[:, 0], test[:, 1], test[:, 2], s=20, c='red', marker='^', label='Test')
ax.set_xlabel('Square Meters')
ax.set_ylabel('Distance to School (km)')
ax.set_zlabel('Price (Billion RMB)')
ax.set_title('Raw Data Distribution')
ax.legend()
plt.show()

# Standardize features using training data statistics
scaler = StandardScaler()
scaler.fit(train)  # Fit only on training data to avoid data leakage
train = scaler.transform(train)
test = scaler.transform(test)

# Split into features (x) and target (y)
x_train, y_train = train[:, :-1], train[:, -1].flatten()
x_test, y_test = test[:, :-1], test[:, -1].flatten()

# Visualize standardized data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:], s=20, c='blue', marker='o', label='Training')
ax.scatter(x_test[:, 0], x_test[:, 1], y_test[:], s=20, c='red', marker='^', label='Test')
ax.set_xlabel('Square Meters (Standardized)')
ax.set_ylabel('Distance to School (Standardized)')
ax.set_zlabel('Price (Standardized)')
ax.set_title('Standardized Data Distribution')
ax.legend()
plt.show()


def batch_gradient_descent(learning_rate, num_iterations, K):
    """Implement batch gradient descent for linear regression

    Args:
        learning_rate (float): Step size for parameter updates
        num_iterations (int): Total number of training iterations
        K (int): Interval for calculating and storing losses

    Returns:
        tuple: (theta, train_losses, test_losses)
            theta: Final model parameters (weights + bias)
            train_losses: Training RMSE recorded every K iterations
            test_losses: Testing RMSE recorded every K iterations
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
        if i % K == 0:
            train_loss = np.sqrt(np.square((X @ theta - y_train).mean()))
            train_losses.append(train_loss)

            test_loss = np.sqrt(np.square((X_test @ theta - y_test).mean()))
            test_losses.append(test_loss)

    return theta, train_losses, test_losses


# Training configuration
learning_rate = 0.00015  # Step size for gradient updates
num_iterations = 1500000  # Total training iterations
K = 100000  # Loss calculation interval

# Train model
theta, train_losses, test_losses = batch_gradient_descent(learning_rate, num_iterations, K)
print('Learning rate:', learning_rate)
print('Number of iterations:', num_iterations)
print('Regression coefficients:', theta)
print()

# Generate prediction surface
x1_range = np.linspace(min(train[:, 0]), max(train[:, 0]), 100)  # Area range
x2_range = np.linspace(min(train[:, 1]), max(train[:, 1]), 100)  # Distance range
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Create design matrix for surface points
X_grid = np.c_[x1_grid.flatten(), x2_grid.flatten(), np.ones_like(x1_grid.flatten())]
y_grid = X_grid @ theta  # Make predictions
y_grid = y_grid.reshape(x1_grid.shape)  # Reshape for surface plot

# Visualize regression surface with data points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:], s=20, c='blue', marker='o', label='Training')
ax.scatter(x_test[:, 0], x_test[:, 1], y_test[:], s=20, c='red', marker='^', label='Test')
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5)
ax.set_xlabel('Square Meters (Standardized)')
ax.set_ylabel('Distance to School (Standardized)')
ax.set_zlabel('Price (Standardized)')
ax.set_title('Linear Regression Surface')
ax.legend()
plt.show()

# Plot training and testing error curves
plt.plot(np.arange(0, num_iterations, K), train_losses, color='blue', label='Train Loss')
plt.plot(np.arange(0, num_iterations, K), test_losses, color='red', linestyle='--', label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.legend()
plt.title(f'Training Progress (learning rate = {learning_rate})')
plt.show()
