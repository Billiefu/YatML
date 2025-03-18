import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data from source files and display the characteristics of the data
train = np.loadtxt('./data/dataForTrainingLinear.txt', delimiter=' ', dtype='float')
test = np.loadtxt('./data/dataForTestingLinear.txt', delimiter=' ', dtype='float')
print('Total number of training data:', len(train))
print('Total number of test data:', len(test))
print()

# Data standardization
scaler = StandardScaler()
scaler.fit(train)  # Use only the training data to calculate the mean and variance
train = scaler.transform(train)
test = scaler.transform(test)

# Split into features and labels
x_train, y_train = train[:, :-1], train[:, -1].flatten()
x_test, y_test = test[:, :-1], test[:, -1].flatten()


def batch_gradient_descent(learning_rate, num_iterations, k):
    # Add intercept term
    X = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=-1)
    X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)

    # Initialize theta parameters
    theta = np.zeros(X.shape[1])
    m = len(y_train)

    # Lists to store training and testing errors every 100000 iterations
    train_losses = []
    test_losses = []

    for i in trange(num_iterations):
        # Compute the gradient
        gradients = 2 / m * X.T @ (X @ theta - y_train)

        # Update theta
        theta -= learning_rate * gradients

        if i % k == 0:
            train_loss = np.sqrt(np.square((X @ theta - y_train).mean()))
            train_losses.append(train_loss)

            test_loss = np.sqrt(np.square((X_test @ theta - y_test).mean()))
            test_losses.append(test_loss)

    return theta, train_losses, test_losses


# Set learning rate and number of iterations
learning_rate = [0.15, 0.015, 0.0015, 0.00015, 0.000015, 0.0000015, 1.1]
num_iterations = [50, 500, 5000, 50000, 500000, 5000000, 100]
K = [5, 50, 500, 5000, 50000, 500000, 5]

for i in range(min(len(learning_rate), len(num_iterations), len(K))):
    rate, iteration, k = learning_rate[i], num_iterations[i], K[i]
    theta, train_losses, test_losses = batch_gradient_descent(rate, iteration, k)
    print('Learning rate:', rate)
    print('Number of iterations:', iteration)
    print('Regression coefficients:', theta)
    print()

    # Plot the error curve
    plt.plot(np.arange(0, iteration, k), train_losses, color='blue', label='Train Loss')
    plt.plot(np.arange(0, iteration, k), test_losses, color='red', linestyle='--', label='Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title(f'RMSE of Linear Regression Model (learning rate = {rate})')
    plt.show()
