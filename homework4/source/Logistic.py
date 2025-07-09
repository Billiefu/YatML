"""
Copyright (C) 2024 Fu Tszkok

:module: Logistic
:function: Implementation of logistic regression for binary classification with SVM performance comparison
:author: Fu Tszkok
:date: 2024-11-24
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load dataset from specified file path"""
    return pd.read_csv(file_path, header=None)


def preprocess_data(data):
    """Handle missing values by replacing '?' with appropriate values"""
    data.replace('?', np.nan, inplace=True)
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].mean(), inplace=True)
    return data


def encode_categorical(data):
    """Convert categorical features to numerical codes"""
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category').cat.codes
    return data


def normalize_data(X):
    """Normalize features using z-score normalization"""
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return (X - means) / stds


def acc(y_true, y_pred):
    """Calculate classification accuracy"""
    return np.mean(y_true == y_pred)


def logistic(z):
    """Sigmoid function for logistic regression"""
    z = np.clip(z, -500, 500)  # Prevent numerical overflow
    return 1 / (1 + np.exp(-z))


def GD(num_steps, learning_rate, l2_coef, X, X_test):
    """Train logistic regression model using gradient descent"""
    theta = np.random.normal(size=(X.shape[1],))
    train_accs = []
    test_accs = []

    for i in range(num_steps):
        # Forward pass and gradient calculation
        pred = logistic(X @ theta)
        grad = -X.T @ (y_train - pred) + l2_coef * theta

        # Parameter update
        theta -= learning_rate * grad

        # Model evaluation
        train_pred = np.where(logistic(X_train @ theta) >= 0.5, 1, 0)
        train_acc = np.mean(y_train == train_pred)
        train_accs.append(train_acc)

        test_pred = np.where(logistic(X_test @ theta) >= 0.5, 1, 0)
        test_acc = np.mean(y_test == test_pred)
        test_accs.append(test_acc)

    return theta, train_accs, test_accs


# Data loading and preprocessing
train_data = load_data('./train.txt')
test_data = load_data('./test_ground_truth.txt')
print(f'Training samples: {len(train_data)}, Test samples: {len(test_data)}\n')

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Feature engineering
train_data = encode_categorical(train_data)
test_data = encode_categorical(test_data)
train_data.iloc[:, -1] = np.where(train_data.iloc[:, -1] == 1, 1, 0)  # Binary labels
test_data.iloc[:, -1] = np.where(test_data.iloc[:, -1] == 1, 1, 0)  # Binary labels

# Prepare training/test sets
X_train = train_data.iloc[:, 1:-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, 1:-1].values
y_test = test_data.iloc[:, -1].values

# Feature normalization
X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

# Model training configuration
num_steps = 200
learning_rate = 0.1
l2_coef = 0.01
np.random.seed(0)

# Train model
theta, train_acc, test_acc = GD(num_steps, learning_rate, l2_coef, X_train, X_test)

# Evaluate model
y_pred = np.where(logistic(X_test @ theta) >= 0.5, 1, 0)
final_acc = acc(y_test, y_pred)
print(f'Test accuracy: {final_acc:.4f}')
print(f'Model coefficients: {theta}')
print(f'Misclassified samples: {np.sum(y_pred != y_test)}')

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(range(num_steps), train_acc, 'b-', label='Train Accuracy')
plt.plot(range(num_steps), test_acc, 'r-', label='Test Accuracy')
plt.xlabel('Training Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.title('Logistic Regression Learning Curve\n'
          f'Learning Rate: {learning_rate}, L2: {l2_coef}',
          fontsize=14)
plt.grid(True)
plt.show()
