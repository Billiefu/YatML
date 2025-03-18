import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
def GD(num_steps, learning_rate, l2_coef, X, X_test):
    theta = np.random.normal(size=(X.shape[1],))  # Initialize parameters
    train_losses = []
    test_losses = []
    for i in range(num_steps):
        # Calculate predictions
        pred = logistic(X @ theta)
        grad = -X.T @ (y_train - pred) + l2_coef * theta  # Compute the gradient
        theta -= learning_rate * grad  # Update parameters

        # Calculate training set loss
        train_loss = -y_train.T @ np.log(pred) - (1 - y_train).T @ np.log(1 - pred) + l2_coef * np.linalg.norm(
            theta) ** 2 / 2
        train_losses.append(train_loss / len(X))

        # Calculate test set loss
        test_pred = logistic(X_test @ theta)
        test_loss = -y_test.T @ np.log(test_pred) - (1 - y_test).T @ np.log(1 - test_pred)
        test_losses.append(test_loss / len(X_test))

    return theta, train_losses, test_losses


# Concatenate 1s for the bias term
X_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
X_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

# Set parameters for gradient descent
num_steps = 500
learning_rate = 0.002
l2_coef = 1.0
np.random.seed(0)

# Train the model
theta, train_losses, test_losses = GD(num_steps, learning_rate, l2_coef, X_train, X_test)

# Make predictions using the trained parameters
y_pred = np.where(logistic(X_test @ theta) >= 0.5, 1, 0)

# Calculate accuracy
final_acc = acc(y_test, y_pred)
print('Prediction accuracy:', final_acc)
print('Regression coefficients:', theta)

errors = np.sum(y_pred != y_test)
print(f"Number of misclassified samples: {errors}")

# Plot the training and testing loss curves
xticks = np.arange(num_steps) + 1
plt.plot(xticks, train_losses, color='blue', label='train loss')
plt.plot(xticks, test_losses, color='red', ls='--', label='test loss')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss during Gradient Descent')
plt.legend()
plt.show()
