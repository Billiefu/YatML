import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt


# Load data
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    return data


# Handle missing values, replacing '?' with NaN
def preprocess_data(data):
    data.replace('?', np.nan, inplace=True)
    for col in data.columns:
        if data[col].dtype == 'object':  # Categorical data
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:  # Numerical data
            data[col].fillna(data[col].mean(), inplace=True)
    return data


# Convert categorical features to numerical
def encode_categorical(data):
    for col in data.columns:
        if data[col].dtype == 'object':  # Only encode categorical data
            data[col] = data[col].astype('category').cat.codes
    return data


# Normalize data (Manually)
def normalize_data(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return (X - means) / stds


# Train SVM using SGD with momentum and learning rate decay
def train_svm(X, y, lambda_param=0.01, learning_rate=0.1, epochs=200, momentum=0.99, decay_rate=0.95,
              class_weight=None, string=None):
    if class_weight is None:
        class_weight = {1: 1, -1: 1}  # Default weights

    m, n = X.shape
    w = np.random.randn(n) * 0.1  # Initialize weights
    w[-5:] = 0
    b = 0  # Initialize bias
    velocity = np.zeros_like(w)  # Initialize velocity for momentum
    rate = learning_rate

    train_accs = []
    test_accs = []

    # Stochastic gradient descent with momentum
    for _ in trange(epochs):
        for i in range(m):
            # Calculate the decision function for the current sample
            decision_function = y[i] * (np.dot(X[i], w) + b)

            # Update rule based on the hinge loss gradient
            if decision_function < 1:
                dw = lambda_param * w - class_weight[y[i]] * y[i] * X[i]
                db = -class_weight[y[i]] * y[i]
            else:
                dw = lambda_param * w
                db = 0

            # Apply momentum to gradient descent
            velocity = momentum * velocity - learning_rate * dw
            w += velocity
            b -= learning_rate * db

        # Learning rate decay after each epoch
        learning_rate *= decay_rate

        # Evaluate the model
        y_pred = predict_svm(X, w, b)
        train_acc = compute_accuracy(y, y_pred)
        train_accs.append(train_acc)

        y_test_pred = predict_svm(X_test, w, b)
        test_acc = compute_accuracy(y_test, y_test_pred)
        test_accs.append(test_acc)

    # Plot accuracy curve
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


# Prediction function
def predict_svm(X, w, b):
    return np.sign(np.dot(X, w) + b)


# Compute accuracy
def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


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


# Separate features and labels
X_train = np.column_stack((train_data.iloc[:, 7], train_data.iloc[:, 11], train_data.iloc[:, 13]))  # Features
y_train = train_data.iloc[:, -1].values  # Labels
X_test = np.column_stack((test_data.iloc[:, 7], test_data.iloc[:, 11], test_data.iloc[:, 13]))  # Features
y_test = test_data.iloc[:, -1].values  # Labels
string = "Feature: Relationship, Capital-loss, Native-country"

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


# Separate features and labels
X_train = np.column_stack((train_data.iloc[:, 2], train_data.iloc[:, 7],
                           train_data.iloc[:, 11], train_data.iloc[:, 13]))  # Features
y_train = train_data.iloc[:, -1].values  # Labels
X_test = np.column_stack((test_data.iloc[:, 2], test_data.iloc[:, 7],
                          test_data.iloc[:, 11], test_data.iloc[:, 13]))  # Features
y_test = test_data.iloc[:, -1].values  # Labels
string = "Feature: Fnlwgt, Relationship, Capital-loss, Native-country"

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


# Separate features and labels
X_train = np.column_stack((train_data.iloc[:, 4], train_data.iloc[:, 5], train_data.iloc[:, 9]))  # Features
y_train = train_data.iloc[:, -1].values  # Labels
X_test = np.column_stack((test_data.iloc[:, 4], test_data.iloc[:, 5], test_data.iloc[:, 9]))  # Features
y_test = test_data.iloc[:, -1].values  # Labels
string = "Feature: Education-num, Marital-status, Sex"

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


# Separate features and labels
X_train = np.column_stack((train_data.iloc[:, 2], train_data.iloc[:, 3], train_data.iloc[:, 4],
                           train_data.iloc[:, 6], train_data.iloc[:, 12]))  # Features
y_train = train_data.iloc[:, -1].values  # Labels
X_test = np.column_stack((test_data.iloc[:, 2], test_data.iloc[:, 3], test_data.iloc[:, 4],
                          test_data.iloc[:, 6], test_data.iloc[:, 12]))  # Features
y_test = test_data.iloc[:, -1].values  # Labels
string = "Feature: Fnlwgt, Education, Education-num, Occupation, Hours-per-week"

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
