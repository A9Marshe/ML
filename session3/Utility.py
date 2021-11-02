import numpy as np
from matplotlib import pyplot as plt

# Utility functions


# Cost function J
def cost(x, y, theta):
    m = np.size(y)
    #j = np.sum(np.square(np.matmul(x, theta) - y))/(2 * m)
    j=((np.matmul(x, theta) - y).transpose()*(np.matmul(x, theta) - y))/(2 * m)

    #j = ((x* theta - y).transpose() * (x* theta - y)) / (2 * m)#error
    return j


# Gradient Decent function
def gradient(x, y, theta, alpha, iterations):
    m = np.size(y)
    j_history = np.zeros((1, iterations))
    for i in range(0, iterations):
        h = np.mat(x) * np.mat(theta)
        grad = ((h - y).transpose() * x) / m
        theta = theta - (alpha * grad.transpose())
        j_history[0, i] = cost(x, y, theta)

    return theta, j_history


# Plot 2d points
def plot2d(x, y, mode):
    plt.title("DATA")
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    plt.plot(x, y, mode)
    plt.show()
    return


# Plot DataSet points and h function
def plot_data_h(x, y, theta, mode):
    plt.title("DATA & HYPOTHESIS")
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    plt.plot(x[:, 1], y, mode)
    plt.plot(x[:, 1], x * theta,"r")
    plt.show()
    return


# Feature Scaling function
def normalize(x):
    x_norm = x
    m = np.size(x, 0)
    mu = np.mean(x, axis = 0)
    sigma = np.std(x, axis = 0)
    for i in range(0, m):
        x_norm[i, :] = (x_norm[i, :] - mu) / sigma
    return x_norm, mu, sigma


# Compute Theta using normal equation
def normal_equation(x, y):
    theta = np.linalg.inv(np.mat(x.transpose()) * x) * x.transpose() * y
    return theta
