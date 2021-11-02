import numpy as np
import Utility as util

######### Multi-variant example #########

######### Gradient Decent Method ########

# Read DataSet from csv file
data = np.genfromtxt("E:\Projects\ML\session3\data3.csv", delimiter=',')
print("data dimensions= ",data.shape)
numOfFeatures = np.size(data, 1) - 1

# Split the DataSet into feature vector and answer vector
x = data[:, 0:numOfFeatures]
y = data[:, numOfFeatures]

# Number of examples in DataSet
m = np.size(y)

x = x.reshape((m, numOfFeatures))
y = y.reshape((m, 1))

print('Normalizing Features ...')
x, mu, sigma = util.normalize(x)
print("x normal dim= ",x.shape)
print("mean normal dim= ",mu.shape)
print("sigma normal dim= ",sigma.shape)
# Add x0 to the feature vector
x0 = np.ones((m, 1))
x = (np.column_stack((x0, x)))

# Gradient Decent parameters
alpha =1 #0.3#0.01 #0.1#1 [0.01 , 0.03,0.1,0.3]
iterations =200 #100#200

# Theta [b, w1, w2]
theta = np.zeros((numOfFeatures + 1, 1))

[theta, history] = util.gradient(x, y, theta, alpha, iterations)

print("Theta computed from gradient descent = ", theta)

# Plot the J values based on iterations
x_axis = np.mat(range(0, iterations)).reshape((1, iterations))
util.plot2d(x_axis, history, ".b")

# Prediction
price = 0
example = np.array([1650, 3])
# Normalize the example before prediction
example = (example - mu) / sigma

example = np.array([1, example[0], example[1]])
price = example * theta;

print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) = ", price)


################# Normal Equation ###############

data = np.genfromtxt("E:\Projects\ML\session3\data3.csv", delimiter=',')

x = data[:, 0:numOfFeatures]
y = data[:, numOfFeatures]
x = x.reshape((m, numOfFeatures))
y = y.reshape((m, 1))

x0 = np.ones((m, 1))
x = (np.column_stack((x0, x)))

print("x shape: ",x.shape," y shape: ", y.shape)

theta = util.normal_equation(x, y)
print("Theta computed from normal equation = ", theta)

example = np.array([1, 1650, 3])
price = example * theta
print("Predicted price of a 1650 sq-ft, 3 br house (using normal equation) = ", price)
