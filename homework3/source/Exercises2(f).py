import numpy as np
import matplotlib.pyplot as plt

# Read data from source files and preprocess
train = np.loadtxt('./data/dataForTrainingLogistic.txt', delimiter=' ', dtype=float)
test = np.loadtxt('./data/dataForTestingLogistic.txt', delimiter=' ', dtype=float)
print('Total number of training data:', len(train))
print('Total number of test data:', len(test))
print()

# Split into features and labels
x_train, y_train = train[:, 0:6], train[:, 6]
x_test, y_test = test[:, 0:6], test[:, 6]


# Calculate accuracy
def acc(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Define the logistic function
def logistic(z):
    return 1 / (1 + np.exp(-z))


# Define the gradient descent training function
def GD(num_steps, learning_rate, l2_coef, X_train_k, X_test):
    theta = np.random.normal(size=(X_train_k.shape[1],))  # Initialize parameters
    train_losses = []
    test_losses = []
    for i in range(num_steps):
        # Calculate predictions on the training set
        pred = logistic(X_train_k @ theta)
        grad = -X_train_k.T @ (y_train_k - pred) + l2_coef * theta  # Compute gradient using y_train_k
        theta -= learning_rate * grad  # Update parameters

        # Calculate training set loss
        train_loss = -y_train_k.T @ np.log(pred) - (1 - y_train_k).T @ np.log(1 - pred) + l2_coef * np.linalg.norm(
            theta) ** 2 / 2
        train_losses.append(train_loss / len(X_train_k))

        # Calculate test set loss
        test_pred = logistic(X_test @ theta)
        test_loss = -y_test.T @ np.log(test_pred) - (1 - y_test).T @ np.log(1 - test_pred)
        test_losses.append(test_loss / len(X_test))

    return theta, train_losses, test_losses


# Concatenate 1s for the bias term
X_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
X_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

# Set parameters for gradient descent
num_steps = 250
learning_rate = 0.002
l2_coef = 1.0
np.random.seed(0)

# Set the sizes of the training set
train_sizes = np.arange(10, 400, 10)
train_errors = []
test_errors = []

for k in train_sizes:
    # Randomly choose k training samples
    idx = np.random.choice(len(x_train), k, replace=False)
    X_train_k = X_train[idx]
    y_train_k = y_train[idx]

    # Re-train the logistic regression model
    theta, _, _ = GD(num_steps, learning_rate, l2_coef, X_train_k, X_test)
    print('Training set size:', k)
    print('Regression coefficients:', theta)
    print()

    # Calculate the number of misclassified samples on the training and test sets
    train_pred = np.where(logistic(X_train_k @ theta) >= 0.5, 1, 0)
    test_pred = np.where(logistic(X_test @ theta) >= 0.5, 1, 0)

    train_errors.append(np.sum(train_pred != y_train_k))
    test_errors.append(np.sum(test_pred != y_test))

# Plot the error changes
plt.plot(train_sizes, train_errors, color='blue', label='train error')
plt.plot(train_sizes, test_errors, color='red', label='test error')
plt.xlabel('Training Set Size')
plt.ylabel('Number of Errors')
plt.title('Training and Test Errors As A Function of Training Set Size')
plt.legend()
plt.show()
