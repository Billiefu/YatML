"""
Copyright (C) 2024 Fu Tszkok

:module: Exercises1(c)
:function: Linear regression with stochastic gradient descent implementation
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

# Visualize raw data distribution in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train[:, 0], train[:, 1], train[:, 2], s=20, c='blue', marker='o', label='Training')
ax.scatter(test[:, 0], test[:, 1], test[:, 2], s=20, c='red', marker='^', label='Test')
ax.set_xlabel('Square Meters')
ax.set_ylabel('Distance to School (km)')
ax.set_zlabel('Price (Billion RMB)')
ax.set_title('Raw Data Distribution')
plt.legend()
plt.show()

# Standardize features using training data statistics
scaler = StandardScaler()
scaler.fit(train)  # Fit only on training data to prevent data leakage
train = scaler.transform(train)
test = scaler.transform(test)

# Split into features (x) and target (y)
x_train, y_train = train[:, :-1], train[:, -1].flatten()
x_test, y_test = test[:, :-1], test[:, -1].flatten()

# Visualize standardized data distribution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:], s=20, c='blue', marker='o', label='Training')
ax.scatter(x_test[:, 0], x_test[:, 1], y_test[:], s=20, c='red', marker='^', label='Test')
ax.set_xlabel('Square Meters (Standardized)')
ax.set_ylabel('Distance to School (Standardized)')
ax.set_zlabel('Price (Standardized)')
ax.set_title('Standardized Data Distribution')
plt.legend()
plt.show()


def batch_generator(x, y, batch_size, shuffle=True):
    """Generate mini-batches for stochastic gradient descent

    Args:
        x (ndarray): Input features
        y (ndarray): Target values
        batch_size (int): Size of each mini-batch
        shuffle (bool): Whether to shuffle data before batching

    Yields:
        tuple: (x_batch, y_batch) for each mini-batch
    """
    batch_count = 0
    if shuffle:
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = y[idx]

    while True:
        start = batch_count * batch_size
        end = min(start + batch_size, len(x))
        if start >= end:
            break
        batch_count += 1
        yield x[start: end], y[start: end]


def SGD(num_epoch, learning_rate, batch_size, K):
    """Implement stochastic gradient descent for linear regression

    Args:
        num_epoch (int): Number of training epochs
        learning_rate (float): Step size for parameter updates
        batch_size (int): Size of mini-batches
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

    # Initialize parameters with normal distribution
    theta = np.random.normal(size=X.shape[1])

    # Track losses during training
    train_losses = []
    test_losses = []

    for i in trange(num_epoch):
        batch_g = batch_generator(X, y_train, batch_size, shuffle=True)
        train_loss = 0

        for x_batch, y_batch in batch_g:
            # Compute gradient for current batch
            grad = x_batch.T @ (x_batch @ theta - y_batch)

            # Update parameters with learning rate
            theta = theta - learning_rate * grad / len(x_batch)

            # Accumulate squared error
            train_loss += np.square(x_batch @ theta - y_batch).sum()

        # Record losses periodically
        if i % K == 0:
            train_loss = np.sqrt(train_loss / len(X))
            train_losses.append(train_loss)

            test_loss = np.sqrt(np.square(X_test @ theta - y_test).mean())
            test_losses.append(test_loss)

    return theta, train_losses, test_losses


# Training configuration
num_epoch = 1500000  # Total training epochs
learning_rate = 0.00015  # Learning rate for gradient updates
batch_size = 32  # Mini-batch size
K = 100000  # Loss calculation interval

# Train model with SGD
theta, train_losses, test_losses = SGD(num_epoch, learning_rate, batch_size, K)
print('Learning rate:', learning_rate)
print('Number of epochs:', num_epoch)
print('Batch size:', batch_size)
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
plt.legend()
plt.show()

# Plot training and testing error curves
plt.plot(np.arange(0, num_epoch, K), train_losses, color='blue', label='Train Loss')
plt.plot(np.arange(0, num_epoch, K), test_losses, color='red', linestyle='--', label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title(f'Training Progress (learning rate = {learning_rate})')
plt.legend()
plt.show()
