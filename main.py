import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read data from a CSV file
data = pd.read_csv("data.csv")

# Define a loss function to calculate the error
def loss_function(m, b, points):
    """
    Calculate the mean squared error for a given linear regression model.

    :param m: Slope of the line.
    :param b: Y-intercept of the line.
    :param points: Data points as a DataFrame with columns 'x' and 'y'.
    :return: Mean squared error.
    """
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        total_error += (y - (m * x + b)) ** 2
    total_error /= len(points)
    return total_error

# Define a gradient descent function to optimize the model parameters
def gradient_descent(m_now, b_now, points, L):
    """
    Perform gradient descent to update model parameters for linear regression.

    :param m_now: Current slope of the line.
    :param b_now: Current y-intercept of the line.
    :param points: Data points as a DataFrame with columns 'x' and 'y'.
    :param L: Learning rate.
    :return: Updated slope and y-intercept.
    """
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b

# Initialize slope and y-intercept
m = 0
b = 0

# Set learning rate and the number of epochs
L = 0.0001
epochs = 1000

# Perform gradient descent to optimize the model parameters
for i in range(epochs):
    if i % 50 == 0:
        print(f"Epochs: {i}")
    m, b = gradient_descent(m, b, data, L)

# Print the optimized slope and y-intercept
print("Optimized slope (m):", m)
print("Optimized y-intercept (b):", b)

# Visualize the data points and the linear regression line
plt.scatter(data.x, data.y, color="black")
plt.plot(list(range(0, 100)), [m * x + b for x in range(20, 80)], color="red")
plt.show()
