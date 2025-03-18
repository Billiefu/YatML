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

# Plot 3D scatter plot
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
scaler.fit(train)  # Use only training data to calculate mean and variance
train = scaler.transform(train)
test = scaler.transform(test)

# Split into features and labels
x_train, y_train = train[:, :-1], train[:, -1].flatten()
x_test, y_test = test[:, :-1], test[:, -1].flatten()

# Plot regression surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:], s=20, c='blue', marker='o')
ax.scatter(x_test[:, 0], x_test[:, 1], y_test[:], s=20, c='red', marker='^')
ax.set_xlabel('Square Meters (StandardScaler)')
ax.set_ylabel('Distance to School (StandardScaler)')
ax.set_zlabel('Price (StandardScaler)')
ax.set_title('Scatter Plot of Data Distribution after Preprocessing')
plt.show()


def batch_generator(x, y, batch_size, shuffle=True):
    # x and y are the inputs and labels
    # If shuffle=True, the data will be randomly partitioned at each iteration
    batch_count = 0
    if shuffle:
        # Randomly generate indices from 0 to len(x)-1
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = y[idx]
    while True:
        start = batch_count * batch_size
        end = min(start + batch_size, len(x))
        if start >= end:
            # Completed one full iteration, stop generation
            break
        batch_count += 1
        yield x[start: end], y[start: end]


def SGD(num_epoch, learning_rate, batch_size, K):
    # Concatenate the original matrix
    X = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=-1)
    X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)

    # Randomly initialize parameters
    theta = np.random.normal(size=X.shape[1])

    # Stochastic Gradient Descent
    # To observe the iteration process, we record the root mean square error after each iteration on the training and test sets
    train_losses = []
    test_losses = []

    for i in trange(num_epoch):
        # Initialize batch generator
        batch_g = batch_generator(X, y_train, batch_size, shuffle=True)
        train_loss = 0

        for x_batch, y_batch in batch_g:
            # Compute the gradient
            grad = x_batch.T @ (x_batch @ theta - y_batch)

            # Update parameters
            theta = theta - learning_rate * grad / len(x_batch)

            # Accumulate squared error
            train_loss += np.square(x_batch @ theta - y_batch).sum()

        if i % K == 0:
            train_loss = np.sqrt(train_loss / len(X))
            train_losses.append(train_loss)

            test_loss = np.sqrt(np.square(X_test @ theta - y_test).mean())
            test_losses.append(test_loss)

    return theta, train_losses, test_losses


# Set number of epochs, learning rate, and batch size
num_epoch = 1500000
learning_rate = 0.00015
batch_size = 32
K = 100000

theta, train_losses, test_losses = SGD(num_epoch, learning_rate, batch_size, K)
print('Learning rate:', learning_rate)
print('Number of epochs:', num_epoch)
print('Batch size:', batch_size)
print('Regression coefficients:', theta)
print()

# Create a grid covering the entire area and distance range
x1_range = np.linspace(min(train[:, 0]), max(train[:, 0]), 100)  # Area range
x2_range = np.linspace(min(train[:, 1]), max(train[:, 1]), 100)  # Distance range

# Create grid points
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Convert grid points into a 2D array (shape suitable for theta)
X_grid = np.c_[x1_grid.flatten(), x2_grid.flatten(), np.ones_like(x1_grid.flatten())]
# Use theta to calculate the prediction for each grid point
y_grid = X_grid @ theta
# Reshape the prediction to match the grid shape
y_grid = y_grid.reshape(x1_grid.shape)

# Plot regression surface plot
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

# Plot error curve
plt.plot(np.arange(0, num_epoch, K), train_losses, color='blue', label='Train Loss')
plt.plot(np.arange(0, num_epoch, K), test_losses, color='red', linestyle='--', label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title(f'RMSE of Linear Regression Model (learning rate = {learning_rate})')
plt.legend()
plt.show()
