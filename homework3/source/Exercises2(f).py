"""
Copyright (C) 2024 Fu Tszkok

:module: Exercises2(f)
:function: Study of training set size impact on logistic regression performance
:author: Fu Tszkok
:date: 2024-11-24
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
import matplotlib.pyplot as plt

# Load and inspect dataset
train = np.loadtxt('./data/dataForTrainingLogistic.txt', delimiter=' ', dtype=float)
test = np.loadtxt('./data/dataForTestingLogistic.txt', delimiter=' ', dtype=float)
print('Total number of training data:', len(train))
print('Total number of test data:', len(test))
print()

# Split into features (first 6 columns) and labels (last column)
x_train, y_train = train[:, 0:6], train[:, 6]
x_test, y_test = test[:, 0:6], test[:, 6]

def acc(y_true, y_pred):
    """Calculate classification accuracy

    Args:
        y_true (ndarray): Ground truth labels (0 or 1)
        y_pred (ndarray): Predicted labels (0 or 1)

    Returns:
        float: Accuracy between 0 and 1
    """
    return np.mean(y_true == y_pred)

def logistic(z):
    """Sigmoid (logistic) function for binary classification

    Args:
        z (ndarray): Linear combination of features and weights

    Returns:
        ndarray: Probability estimates between 0 and 1
    """
    return 1 / (1 + np.exp(-z))

def GD(num_steps, learning_rate, l2_coef, X_train_k, X_test):
    """Perform gradient descent optimization for logistic regression

    Args:
        num_steps (int): Number of training iterations
        learning_rate (float): Step size for parameter updates
        l2_coef (float): L2 regularization coefficient
        X_train_k (ndarray): Subset of training features with bias term
        X_test (ndarray): Test features with bias term

    Returns:
        tuple: (theta, train_losses, test_losses)
            theta: Final model parameters
            train_losses: Training loss history
            test_losses: Test loss history
    """
    # Initialize parameters with normal distribution
    theta = np.random.normal(size=(X_train_k.shape[1],))

    # Track loss history
    train_losses = []
    test_losses = []

    for _ in range(num_steps):
        # Compute predictions and gradient
        pred = logistic(X_train_k @ theta)
        grad = -X_train_k.T @ (y_train_k - pred) + l2_coef * theta

        # Update parameters
        theta -= learning_rate * grad

        # Calculate losses
        train_loss = (-y_train_k.T @ np.log(pred) -
                     (1 - y_train_k).T @ np.log(1 - pred) +
                     l2_coef * np.linalg.norm(theta)**2 / 2)
        train_losses.append(train_loss / len(X_train_k))

        test_pred = logistic(X_test @ theta)
        test_loss = (-y_test.T @ np.log(test_pred) -
                    (1 - y_test).T @ np.log(1 - test_pred))
        test_losses.append(test_loss / len(X_test))

    return theta, train_losses, test_losses

# Add bias term (column of 1s) to features
X_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
X_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

# Training configuration
num_steps = 250        # Number of gradient descent iterations
learning_rate = 0.002  # Learning rate for parameter updates
l2_coef = 1.0          # L2 regularization strength
np.random.seed(0)     # For reproducibility

# Define training set sizes to evaluate
train_sizes = np.arange(10, 400, 10)  # From 10 to 390 in steps of 10
train_errors = []      # Track training misclassifications
test_errors = []       # Track test misclassifications

for k in train_sizes:
    # Randomly select k training samples without replacement
    idx = np.random.choice(len(x_train), k, replace=False)
    X_train_k = X_train[idx]
    y_train_k = y_train[idx]

    # Train model on current subset
    theta, _, _ = GD(num_steps, learning_rate, l2_coef, X_train_k, X_test)
    print('Training set size:', k)
    print('Regression coefficients:', theta)
    print()

    # Calculate misclassifications
    train_pred = np.where(logistic(X_train_k @ theta) >= 0.5, 1, 0)
    test_pred = np.where(logistic(X_test @ theta) >= 0.5, 1, 0)

    train_errors.append(np.sum(train_pred != y_train_k))
    test_errors.append(np.sum(test_pred != y_test))

# Plot error trends
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_errors, color='blue', label='Training Errors')
plt.plot(train_sizes, test_errors, color='red', label='Test Errors')
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Number of Misclassifications', fontsize=12)
plt.title('Learning Curve: Errors vs Training Set Size', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
