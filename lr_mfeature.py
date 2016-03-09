import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn
seaborn.set()

df = pd.read_csv('imports-85.data',
                  header=None,
                  names=['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels',
                         'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders',
                         'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
                         'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'],
                      na_values='?')

# Drop every piece of data that contains NaN
df = df.dropna()


### Linear Regression with Multiple Features
### Standardize the data
feature1 = df['engine-size']
feature2 = df['peak-rpm']
target = df['price']

from sklearn.preprocessing import StandardScaler
X1 = feature1.astype(float)
X1_scaled = StandardScaler().fit_transform(X1.reshape(-1,1))

X2 = feature2.astype(float)
X2_scaled = StandardScaler().fit_transform(X2.reshape(-1,1))

Y = target.astype(float)
Y_scaled = StandardScaler().fit_transform(Y.reshape(-1,1))

### Construct the Y matrix
Y_Matrix = np.matrix(np.array(Y_scaled))

### Construct the X matrix
feat1_Matrix = np.matrix(np.array(X1_scaled))
feat2_Matrix = np.matrix(np.array(X2_scaled))
data_Matrix = np.concatenate((feat1_Matrix,feat2_Matrix), axis = 1)
single_one = np.array([[1]])
n_times_1_Matrix = np.repeat(single_one, len(X1), axis = 0)
X_Matrix = np.concatenate((n_times_1_Matrix,data_Matrix), axis = 1)

### Calculate Theta matrix
x_transpose = X_Matrix.transpose()
bracket = x_transpose.dot(X_Matrix)
bracket_inverse = np.linalg.inv(bracket) 

theta = bracket_inverse.dot(x_transpose)
theta = theta.dot(Y_Matrix)

### h(x) = theta_0 + theta_1 * x_1 + theta_2 * x_2
print 'theta_0: ', theta[0]
print 'theta_1: ', theta[1]
print 'theta_2: ', theta[2]

### Multiple Linear Regression with Gradient Descent
from sklearn import linear_model
clf = linear_model.SGDRegressor(loss='squared_loss')
###clf.fit(X_Matrix, np.ravel(Y_scaled))
clf.fit(data_Matrix, np.ravel(Y_scaled))
print '\nUsing SGDRegressor()'
print clf.coef_
print 'theta_1: ', clf.coef_[0]
print 'theta_2: ', clf.coef_[1]
print 'intercept: ', clf.intercept_