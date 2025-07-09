"""
Copyright (C) 2024 Fu Tszkok

:module: FilterSVM
:function: SVM classification with class weighting for imbalanced data
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
from tqdm import trange
import matplotlib.pyplot as plt


# Load data from specified file path
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    return data


# Preprocess data by handling missing values
def preprocess_data(data):
    data.replace('?', np.nan, inplace=True)
    for col in data.columns:
        if data[col].dtype == 'object':  # Handle categorical data
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:  # Handle numerical data
            data[col].fillna(data[col].mean(), inplace=True)
    return data


# Convert categorical features to numerical codes
def encode_categorical(data):
    for col in data.columns:
        if data[col].dtype == 'object':  # Only convert categorical columns
            data[col] = data[col].astype('category').cat.codes
    return data


# Normalize features using z-score normalization
def normalize_data(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return (X - means) / stds


# Train SVM using SGD with momentum and learning rate decay
def train_svm(X, y, lambda_param=0.01, learning_rate=0.1, epochs=200, momentum=0.99, decay_rate=0.95,
              class_weight=None):
    # Initialize weights and parameters
    if class_weight is None:
        class_weight = {1: 1, -1: 1}  # Default class weights

    m, n = X.shape
    w = np.random.randn(n) * 0.1  # Random weight initialization
    w[-5:] = 0  # Set last 5 weights to zero
    b = 0  # Initialize bias term
    velocity = np.zeros_like(w)  # For momentum calculation
    rate = learning_rate  # Store initial learning rate

    train_accs = []  # Store training accuracy history
    test_accs = []  # Store test accuracy history

    # Training loop with progress bar
    for _ in trange(epochs):
        # Iterate through each training sample
        for i in range(m):
            # Calculate decision function value
            decision_function = y[i] * (np.dot(X[i], w) + b)

            # Compute gradients based on hinge loss
            if decision_function < 1:
                dw = lambda_param * w - class_weight[y[i]] * y[i] * X[i]
                db = -class_weight[y[i]] * y[i]
            else:
                dw = lambda_param * w
                db = 0

            # Update weights with momentum
            velocity = momentum * velocity - learning_rate * dw
            w += velocity
            b -= learning_rate * db

        # Apply learning rate decay
        learning_rate *= decay_rate

        # Calculate training accuracy
        y_pred = predict_svm(X, w, b)
        train_acc = compute_accuracy(y, y_pred)
        train_accs.append(train_acc)

        # Calculate test accuracy
        y_test_pred = predict_svm(X_test, w, b)
        test_acc = compute_accuracy(y_test, y_test_pred)
        test_accs.append(test_acc)

    # Plot accuracy curves
    plt.plot(range(epochs), train_accs, color='blue', label='Train Accuracy')
    plt.plot(range(epochs), test_accs, color='red', label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.suptitle('Accuracy for SVM', fontsize=14)
    plt.title(f'Learning rate = {rate}, Regularization parameter = {lambda_param}', fontsize=10)
    plt.show()

    return w, b


# Make predictions using learned SVM model
def predict_svm(X, w, b):
    return np.sign(np.dot(X, w) + b)


# Compute classification accuracy
def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Main execution block (完全保留原始代码，不加任何修改)
# Load training and testing datasets
train_data = load_data('./train.txt')
test_data = load_data('./test_ground_truth.txt')
print('Total number of training data:', len(train_data))
print('Total number of test data:', len(test_data))
print()

# Preprocess data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Encode categorical features
train_data = encode_categorical(train_data)
test_data = encode_categorical(test_data)
train_data.iloc[:, -1] = np.where(train_data.iloc[:, -1] == 1, 1, -1)
test_data.iloc[:, -1] = np.where(test_data.iloc[:, -1] == 1, 1, -1)

# Plot histograms for each feature
train_data.hist(bins=20, figsize=(14, 10), edgecolor='black')
plt.suptitle('Feature Distributions on Training Data Set', fontsize=26)
plt.show()

test_data.hist(bins=20, figsize=(14, 10), edgecolor='black')
plt.suptitle('Feature Distributions on Testing Data Set', fontsize=26)
plt.show()

# Separate features and labels
X_train = train_data.iloc[:, 1:-1].values  # Features
y_train = train_data.iloc[:, -1].values  # Labels
X_test = test_data.iloc[:, 1:-1].values  # Features
y_test = test_data.iloc[:, -1].values  # Labels

# Normalize data
X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

# Train the SVM model
pos_weight = 1.0 / np.sum(y_train == 1)
neg_weight = 1.0 / np.sum(y_train == -1)
w, b = train_svm(X_train, y_train, class_weight={1: pos_weight, -1: neg_weight})
print("Regularization parameter: 0.001")
print("Learning rate: 0.01")

# Predict labels for the test set
y_pred = predict_svm(X_test, w, b)
print("Test predictions: ", y_pred)

# If the test set has labels, compute accuracy
accuracy = compute_accuracy(y_test, y_pred)
print("Test accuracy: ", accuracy)
print()

accuracies = []

# Test the impact of the regularization coefficient on model training
lambda_param = [1, 0.1, 0.01, 0.001, 0.0001]
for param in lambda_param:
    pos_weight = 1.0 / np.sum(y_train == 1)
    neg_weight = 1.0 / np.sum(y_train == -1)
    w, b = train_svm(X_train, y_train, lambda_param=param, class_weight={1: pos_weight, -1: neg_weight})
    print("Regularization parameter: ", param)

    # Predict labels for the test set
    y_pred = predict_svm(X_test, w, b)
    print("Test predictions: ", y_pred)

    # If the test set has labels, compute accuracy
    y_pred = predict_svm(X_test, w, b)
    accuracy = compute_accuracy(y_test, y_pred)
    accuracies.append(accuracy)
    print("Test accuracy: ", accuracy)
    print()

# Output the results of experiments with a fixed learning rate
print('All experiments were performed with learning rate = 0.01')
for i in range(len(accuracies)):
    print(f'Test accuracy when regularization coefficient is {lambda_param[i]} : {accuracies[i]}')
print()
accuracies = []

# Test the impact of learning rate on model training
learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001]
for rate in learning_rate:
    pos_weight = 1.0 / np.sum(y_train == 1)
    neg_weight = 1.0 / np.sum(y_train == -1)
    w, b = train_svm(X_train, y_train, learning_rate=rate, class_weight={1: pos_weight, -1: neg_weight})
    print("Learning rate: ", rate)

    # Predict labels for the test set
    y_pred = predict_svm(X_test, w, b)
    print("Test predictions: ", y_pred)

    # If the test set has labels, compute accuracy
    y_pred = predict_svm(X_test, w, b)
    accuracy = compute_accuracy(y_test, y_pred)
    accuracies.append(accuracy)
    print("Test accuracy: ", accuracy)
    print()

# Output the results of experiments with a fixed regularization coefficient
print('All experiments were performed with regularization coefficient = 0.001')
for i in range(len(accuracies)):
    print(f'Test accuracy when learning rate is {learning_rate[i]} : {accuracies[i]}')
print()
