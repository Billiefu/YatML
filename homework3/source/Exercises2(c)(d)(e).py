"""
Copyright (C) 2024 Fu Tszkok

:module: Exercises2(c)(d)(e)
:function: Logistic regression with L2 regularization using gradient descent
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
from matplotlib.ticker import MaxNLocator

# Load and inspect dataset
train = np.loadtxt('./data/dataForTrainingLogistic.txt', delimiter=' ', dtype=float)
test = np.loadtxt('./data/dataForTestingLogistic.txt', delimiter=' ', dtype=float)
print('Total number of training data:', len(train))
print('Total number of test data:', len(test))
print()

# Split dataset into features (first 6 columns) and labels (last column)
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
    """Sigmoid (logistic) function

    Args:
        z (ndarray): Input values

    Returns:
        ndarray: Output values between 0 and 1
    """
    return 1 / (1 + np.exp(-z))


def GD(num_steps, learning_rate, l2_coef, X, X_test):
    """Perform gradient descent optimization for logistic regression

    Args:
        num_steps (int): Number of training iterations
        learning_rate (float): Step size for gradient updates
        l2_coef (float): L2 regularization coefficient
        X (ndarray): Training features with bias term
        X_test (ndarray): Test features with bias term

    Returns:
        tuple: (theta, train_losses, test_losses)
            theta: Final model parameters
            train_losses: Training loss history
            test_losses: Test loss history
    """
    # Initialize parameters with normal distribution
    theta = np.random.normal(size=(X.shape[1],))

    # Track loss history
    train_losses = []
    test_losses = []

    for i in range(num_steps):
        # Forward pass
        pred = logistic(X @ theta)

        # Compute gradient with L2 regularization
        grad = -X.T @ (y_train - pred) + l2_coef * theta

        # Update parameters
        theta -= learning_rate * grad

        # Calculate training loss (cross-entropy + L2 regularization)
        train_loss = (-y_train.T @ np.log(pred) -
                      (1 - y_train).T @ np.log(1 - pred) +
                      l2_coef * np.linalg.norm(theta) ** 2 / 2)
        train_losses.append(train_loss / len(X))

        # Calculate test loss (cross-entropy only)
        test_pred = logistic(X_test @ theta)
        test_loss = (-y_test.T @ np.log(test_pred) -
                     (1 - y_test).T @ np.log(1 - test_pred))
        test_losses.append(test_loss / len(X_test))

    return theta, train_losses, test_losses


# Add bias term (column of 1s) to features
X_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
X_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

# Training configuration
num_steps = 500  # Number of training iterations
learning_rate = 0.002  # Learning rate for gradient updates
l2_coef = 1.0  # L2 regularization coefficient
np.random.seed(0)  # Set random seed for reproducibility

# Train logistic regression model
theta, train_losses, test_losses = GD(num_steps, learning_rate, l2_coef, X_train, X_test)

# Make predictions on test set
y_pred = np.where(logistic(X_test @ theta) >= 0.5, 1, 0)

# Evaluate model performance
final_acc = acc(y_test, y_pred)
print('Prediction accuracy:', final_acc)
print('Regression coefficients:', theta)

errors = np.sum(y_pred != y_test)
print(f"Number of misclassified samples: {errors}")

# Plot training progress
xticks = np.arange(num_steps) + 1
plt.plot(xticks, train_losses, color='blue', label='train loss')
plt.plot(xticks, test_losses, color='red', ls='--', label='test loss')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Progress with L2 Regularization')
plt.legend()
plt.show()