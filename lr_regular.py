import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7.5], [9.1], [13.2], [17.5], [19.3]]

X_test = [[6], [8], [11], [16]]
y_test = [[8.3], [12.5], [15.4], [18.6]]


### LR Regression on polynomial data
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = lr_model.predict(xx.reshape(xx.shape[0], 1))
lr_score = lr_model.score(X_test, y_test)
plt.plot(xx, yy)
plt.plot(X_test, y_test)
plt.title('Linear regression (order 1) result')
print 'Linear Regression'
print 'Linear regression (order 1) model score is: ', lr_score
print 'coeff: ', lr_model.coef_
print 'intercept: ', lr_model.intercept_
plt.show()


### Polynomial regression on training data
poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
pr_model = LinearRegression()
pr_model.fit(X_train_poly, y_train)
print '\nLinear Regression order 5'
### print 'powers:', poly.powers_
print 'coefficients: ', pr_model.coef_
print 'intercept: ', pr_model.intercept_
xx_poly = poly.transform(xx.reshape(xx.shape[0], 1))
yy_poly = pr_model.predict(xx_poly)
### score = reg.score(feature_test.reshape(-1, 1), target_test.reshape(-1,1))
pr_score = pr_model.score(X_test_poly, y_test)
print 'Linear regression (order 5) score is: ', pr_score
plt.plot(xx, yy_poly)
plt.plot(X_test, y_test)
plt.title('Linear regression (order 5) result')
plt.show()

### Ridge Regression
ridge_model = Ridge(alpha=1, normalize=False)
ridge_model.fit(X_train_poly, y_train)
yy_ridge = ridge_model.predict(xx_poly)
ridge_score = ridge_model.score(X_test_poly, y_test)
print '\nRidge Regression'
print 'Ridge regression (order 5) score is: ', ridge_score
print 'coeff: ', ridge_model.coef_
print 'intercept: ', ridge_model.intercept_
plt.plot(xx, yy_ridge)
plt.plot(X_test, y_test)
plt.title('Ridge regression (order 5) result')
plt.show()