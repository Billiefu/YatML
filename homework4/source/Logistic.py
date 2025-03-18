import numpy as np
import pandas as pd
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
        if data[col].dtype == 'object':  # Encode only categorical data
            data[col] = data[col].astype('category').cat.codes
    return data


# Normalize data (Manually)
def normalize_data(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return (X - means) / stds


# Calculate accuracy
def acc(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Define the logistic function
def logistic(z):
    z = np.clip(z, -500, 500)  # 限制 z 的范围
    return 1 / (1 + np.exp(-z))


# Define the gradient descent training function
def GD(num_steps, learning_rate, l2_coef, X, X_test):
    theta = np.random.normal(size=(X.shape[1],))  # Initialize parameters
    train_accs = []
    test_accs = []
    for i in range(num_steps):
        # Calculate predictions
        pred = logistic(X @ theta)
        grad = -X.T @ (y_train - pred) + l2_coef * theta  # Compute the gradient
        theta -= learning_rate * grad  # Update parameters

        # Calculate training set loss
        pred = np.where(logistic(X_train @ theta) >= 0.5, 1, 0)
        train_acc = np.mean(y_train == pred)
        train_accs.append(train_acc)

        # Calculate test set loss
        test_pred = np.where(logistic(X_test @ theta) >= 0.5, 1, 0)
        test_acc = np.mean(y_test == test_pred)
        test_accs.append(test_acc)

    return theta, train_accs, test_accs


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
train_data.iloc[:, -1] = np.where(train_data.iloc[:, -1] == 1, 1, 0)  # Encode '>50K' as 0, and others as 1.
test_data.iloc[:, -1] = np.where(test_data.iloc[:, -1] == 1, 1, 0)  # Encode '>50K' as 0, and others as 1.

# Separate features and labels
X_train = train_data.iloc[:, 1:-1].values  # Features
# X_train = np.column_stack((train_data.iloc[:, 7], train_data.iloc[:, 11]))  # Features
y_train = train_data.iloc[:, -1].values  # Labels
X_test = test_data.iloc[:, 1:-1].values  # Features
# X_test = np.column_stack((test_data.iloc[:, 7], test_data.iloc[:, 11]))
y_test = test_data.iloc[:, -1].values  # Assuming labels are present

# Normalize data
X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

# Set parameters for gradient descent
num_steps = 200
learning_rate = 0.1
l2_coef = 0.01
np.random.seed(0)

# Train the model
theta, train_acc, test_acc = GD(num_steps, learning_rate, l2_coef, X_train, X_test)

# Make predictions using the trained parameters
y_pred = np.where(logistic(X_test @ theta) >= 0.5, 1, 0)

# Calculate accuracy
final_acc = acc(y_test, y_pred)
print('Prediction accuracy:', final_acc)
print('Regression coefficients:', theta)

errors = np.sum(y_pred != y_test)
print(f"Number of misclassified samples: {errors}")

# Plot accuracy curve
plt.plot(range(num_steps), train_acc, color='blue', label='Train Accuracy')
plt.plot(range(num_steps), test_acc, color='red', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.suptitle('Accuracy for Logistic Regression', fontsize=14)
plt.title(f'Learning rate = {learning_rate}, Regularization parameter = {l2_coef}', fontsize=10)
plt.show()
