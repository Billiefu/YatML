"""
Copyright (C) 2024 Fu Tszkok

:module: FeatureSVM
:function: SVM classification with key feature extraction
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

# Data loading function - loads CSV data without header
def load_data(file_path):
    """Load dataset from CSV file without headers"""
    data = pd.read_csv(file_path, header=None)
    return data

# Data preprocessing function - handles missing values
def preprocess_data(data):
    """Clean data by replacing missing values ('?') with appropriate substitutes:
    - Mode for categorical features
    - Mean for numerical features"""
    data.replace('?', np.nan, inplace=True)
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].mean(), inplace=True)
    return data

# Feature encoding function - converts categorical to numerical
def encode_categorical(data):
    """Encode categorical features as numerical codes using pandas category conversion"""
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category').cat.codes
    return data

# Feature normalization function - manual z-score normalization
def normalize_data(X):
    """Normalize features to zero mean and unit variance"""
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return (X - means) / stds

# SVM training function with SGD and momentum
def train_svm(X, y, lambda_param=0.01, learning_rate=0.1, epochs=200, momentum=0.99, decay_rate=0.95,
              class_weight=None, string=None):
    """Train SVM classifier using stochastic gradient descent with:
    - Momentum acceleration
    - Learning rate decay
    - Class weighting for imbalanced data
    - L2 regularization"""
    if class_weight is None:
        class_weight = {1: 1, -1: 1}

    m, n = X.shape
    w = np.random.randn(n) * 0.1
    w[-5:] = 0
    b = 0
    velocity = np.zeros_like(w)
    rate = learning_rate

    train_accs = []
    test_accs = []

    for _ in trange(epochs):
        for i in range(m):
            decision_function = y[i] * (np.dot(X[i], w) + b)

            if decision_function < 1:
                dw = lambda_param * w - class_weight[y[i]] * y[i] * X[i]
                db = -class_weight[y[i]] * y[i]
            else:
                dw = lambda_param * w
                db = 0

            velocity = momentum * velocity - learning_rate * dw
            w += velocity
            b -= learning_rate * db

        learning_rate *= decay_rate

        y_pred = predict_svm(X, w, b)
        train_acc = compute_accuracy(y, y_pred)
        train_accs.append(train_acc)

        y_test_pred = predict_svm(X_test, w, b)
        test_acc = compute_accuracy(y_test, y_test_pred)
        test_accs.append(test_acc)

    plt.plot(range(epochs), train_accs, color='blue', label='Train Accuracy')
    plt.plot(range(epochs), test_accs, color='red', label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.suptitle('Accuracy for SVM', fontsize=14)
    if string is not None:
        plt.title(string, fontsize=10)
    plt.show()

    return w, b

# Prediction function for SVM
def predict_svm(X, w, b):
    """Make binary predictions (-1 or 1) using learned SVM parameters"""
    return np.sign(np.dot(X, w) + b)

# Accuracy calculation function
def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy between true and predicted labels"""
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
X_train = np.column_stack((train_data.iloc[:, 7], train_data.iloc[:, 11]))  # Features
y_train = train_data.iloc[:, -1].values  # Labels
X_test = np.column_stack((test_data.iloc[:, 7], test_data.iloc[:, 11]))  # Features
y_test = test_data.iloc[:, -1].values  # Labels
string = "Feature: Relationship, Capital-loss."

# Normalize data
X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

# Train the SVM model
pos_weight = 1.0 / np.sum(y_train == 1)
neg_weight = 1.0 / np.sum(y_train == -1)
w, b = train_svm(X_train, y_train, class_weight={1: pos_weight, -1: neg_weight}, string=string)
print(string)

# Predict labels for the test set
y_pred = predict_svm(X_test, w, b)
print("Test predictions: ", y_pred)

# If the test set has labels, compute accuracy
accuracy = compute_accuracy(y_test, y_pred)
print("Test accuracy: ", accuracy)
print()

# Experiment 2: Relationship + Capital-loss + Native-country features
X_train = np.column_stack((train_data.iloc[:, 7], train_data.iloc[:, 11], train_data.iloc[:, 13]))
y_train = train_data.iloc[:, -1].values
X_test = np.column_stack((test_data.iloc[:, 7], test_data.iloc[:, 11], test_data.iloc[:, 13]))
y_test = test_data.iloc[:, -1].values
string = "Feature: Relationship, Capital-loss, Native-country"

X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

pos_weight = 1.0 / np.sum(y_train == 1)
neg_weight = 1.0 / np.sum(y_train == -1)
w, b = train_svm(X_train, y_train, class_weight={1: pos_weight, -1: neg_weight}, string=string)
print(string)

y_pred = predict_svm(X_test, w, b)
accuracy = compute_accuracy(y_test, y_pred)
print("Test accuracy: ", accuracy)
print()

# Experiment 3: Fnlwgt + Relationship + Capital-loss + Native-country
X_train = np.column_stack((train_data.iloc[:, 2], train_data.iloc[:, 7],
                         train_data.iloc[:, 11], train_data.iloc[:, 13]))
y_train = train_data.iloc[:, -1].values
X_test = np.column_stack((test_data.iloc[:, 2], test_data.iloc[:, 7],
                        test_data.iloc[:, 11], test_data.iloc[:, 13]))
y_test = test_data.iloc[:, -1].values
string = "Feature: Fnlwgt, Relationship, Capital-loss, Native-country"

X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

pos_weight = 1.0 / np.sum(y_train == 1)
neg_weight = 1.0 / np.sum(y_train == -1)
w, b = train_svm(X_train, y_train, class_weight={1: pos_weight, -1: neg_weight}, string=string)
print(string)

y_pred = predict_svm(X_test, w, b)
accuracy = compute_accuracy(y_test, y_pred)
print("Test accuracy: ", accuracy)
print()

# Experiment 4: Education-num + Marital-status + Sex
X_train = np.column_stack((train_data.iloc[:, 4], train_data.iloc[:, 5], train_data.iloc[:, 9]))
y_train = train_data.iloc[:, -1].values
X_test = np.column_stack((test_data.iloc[:, 4], test_data.iloc[:, 5], test_data.iloc[:, 9]))
y_test = test_data.iloc[:, -1].values
string = "Feature: Education-num, Marital-status, Sex"

X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

pos_weight = 1.0 / np.sum(y_train == 1)
neg_weight = 1.0 / np.sum(y_train == -1)
w, b = train_svm(X_train, y_train, class_weight={1: pos_weight, -1: neg_weight}, string=string)
print(string)

y_pred = predict_svm(X_test, w, b)
accuracy = compute_accuracy(y_test, y_pred)
print("Test accuracy: ", accuracy)
print()

# Experiment 5: Comprehensive feature combination
X_train = np.column_stack((train_data.iloc[:, 2], train_data.iloc[:, 3], train_data.iloc[:, 4],
                         train_data.iloc[:, 6], train_data.iloc[:, 12]))
y_train = train_data.iloc[:, -1].values
X_test = np.column_stack((test_data.iloc[:, 2], test_data.iloc[:, 3], test_data.iloc[:, 4],
                        test_data.iloc[:, 6], test_data.iloc[:, 12]))
y_test = test_data.iloc[:, -1].values
string = "Feature: Fnlwgt, Education, Education-num, Occupation, Hours-per-week"

X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

pos_weight = 1.0 / np.sum(y_train == 1)
neg_weight = 1.0 / np.sum(y_train == -1)
w, b = train_svm(X_train, y_train, class_weight={1: pos_weight, -1: neg_weight}, string=string)
print(string)

y_pred = predict_svm(X_test, w, b)
accuracy = compute_accuracy(y_test, y_pred)
print("Test accuracy: ", accuracy)
print()
