import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data from source files and display various features of the data
train = np.loadtxt('./data/dataForTrainingLinear.txt', delimiter=' ', dtype='float')
test = np.loadtxt('./data/dataForTestingLinear.txt', delimiter=' ', dtype='float')
print('Total number of training data:', len(train))
print('Total number of test data:', len(test))
print()

# Plot the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train[:, 0], train[:, 1], train[:, 2], s=20, c='blue', marker='o')
ax.scatter(test[:, 0], test[:, 1], test[:, 2], s=20, c='red', marker='^')
ax.set_xlabel('Square Meters')
ax.set_ylabel('Distance to School (km)')
ax.set_zlabel('Price (Billion RMB)')
ax.set_title('Scatter Plot of Data Distribution')
plt.show()

# Data standardization
scaler = StandardScaler()
scaler.fit(train)  # Use only the training data to calculate the mean and variance
train = scaler.transform(train)
test = scaler.transform(test)

# Split into features and labels
x_train, y_train = train[:, :-1], train[:, -1].flatten()
x_test, y_test = test[:, :-1], test[:, -1].flatten()

# Plot the regression surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:], s=20, c='blue', marker='o')
ax.scatter(x_test[:, 0], x_test[:, 1], y_test[:], s=20, c='red', marker='^')
ax.set_xlabel('Square Meters (StandardScaler)')
ax.set_ylabel('Distance to School (StandardScaler)')
ax.set_zlabel('Price (StandardScaler)')
ax.set_title('Scatter Plot of Data Distribution after Preprocessing')
plt.show()


def batch_gradient_descent(learning_rate, num_iterations, K):
    # Add intercept term
    X = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=-1)
    X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)

    # Initialize theta parameters
    theta = np.zeros(X.shape[1])
    m = len(y_train)

    # Lists to store the training and testing errors every 100000 iterations
    train_losses = []
    test_losses = []

    for i in trange(num_iterations):
        # Compute the gradient
        gradients = 2 / m * X.T @ (X @ theta - y_train)

        # Update theta
        theta -= learning_rate * gradients

        if i % K == 0:
            train_loss = np.sqrt(np.square((X @ theta - y_train).mean()))
            train_losses.append(train_loss)

            test_loss = np.sqrt(np.square((X_test @ theta - y_test).mean()))
            test_losses.append(test_loss)

    return theta, train_losses, test_losses


# Set learning rate and number of iterations
learning_rate = 0.00015
num_iterations = 1500000
K = 100000

theta, train_losses, test_losses = batch_gradient_descent(learning_rate, num_iterations, K)
print('Learning rate:', learning_rate)
print('Number of iterations:', num_iterations)
print('Regression coefficients:', theta)
print()

# Create a grid covering the entire range of area and distance
x1_range = np.linspace(min(train[:, 0]), max(train[:, 0]), 100)  # Range of area
x2_range = np.linspace(min(train[:, 1]), max(train[:, 1]), 100)  # Range of distance

# Create meshgrid
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Convert meshgrid points to 2D array (shape suitable for theta)
X_grid = np.c_[x1_grid.flatten(), x2_grid.flatten(), np.ones_like(x1_grid.flatten())]
# Use theta to calculate predictions for each grid point
y_grid = X_grid @ theta
# Reshape predictions to match the grid shape
y_grid = y_grid.reshape(x1_grid.shape)

# Plot the regression surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:], s=20, c='blue', marker='o')
ax.scatter(x_test[:, 0], x_test[:, 1], y_test[:], s=20, c='red', marker='^')
ax.plot_surface(x1_grid, x2_grid, y_grid)
ax.set_xlabel('Square Meters (StandardScaler)')
ax.set_ylabel('Distance to School (StandardScaler)')
ax.set_zlabel('Price (StandardScaler)')
ax.set_title('Regression Surface Plot')
plt.show()

# Plot the error curve
plt.plot(np.arange(0, num_iterations, K), train_losses, color='blue', label='Train Loss')
plt.plot(np.arange(0, num_iterations, K), test_losses, color='red', linestyle='--', label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.legend()
plt.title(f'RMSE of Linear Regression Model (learning rate = {learning_rate})')
plt.show()
